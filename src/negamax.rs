use std::{
    ops::{Mul, Neg},
    time::Duration,
};

use crate::{node::Node, Evaluate, Gamestate, Move, NodeAim, SearchExit, SearchResult};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NegamaxAim {
    Minimise = -1,
    Maximise = 1,
}

impl From<NegamaxAim> for f32 {
    fn from(aim: NegamaxAim) -> Self {
        aim as i8 as f32
    }
}

impl From<NodeAim> for NegamaxAim {
    fn from(aim: NodeAim) -> Self {
        match aim {
            NodeAim::Minimise => NegamaxAim::Minimise,
            NodeAim::Maximise => NegamaxAim::Maximise,
        }
    }
}

impl Neg for NegamaxAim {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            NegamaxAim::Minimise => NegamaxAim::Maximise,
            NegamaxAim::Maximise => NegamaxAim::Minimise,
        }
    }
}

impl Mul<f32> for NegamaxAim {
    type Output = f32;

    fn mul(self, rhs: f32) -> Self::Output {
        self as i8 as f32 * rhs
    }
}

/// Negamax search with pruning and timeout
pub struct Negamax<G, M, E> {
    node: Node<G, M>,
    evaluator: E,
    // Options
    /// Maximum depth to search
    pub max_depth: Option<u8>,
    /// Maximum time to search
    pub max_time: Option<Duration>,
    /// Use iterative deepening
    pub iterative: bool,
    /// Use alpha-beta pruning
    pub alpha_beta: bool,
    /// Sort children nodes before searching (when interative deepening)
    pub pre_sort: bool,
}

impl<G, M, E> Negamax<G, M, E> {
    pub fn new(node: Node<G, M>, evaluator: E) -> Self {
        Negamax {
            node,
            evaluator,
            max_depth: None,
            max_time: None,
            iterative: false,
            alpha_beta: false,
            pre_sort: false,
        }
    }

    pub fn replace_gamestate(&mut self, gamestate: G) {
        self.node = Node::new(gamestate);
    }
}

impl<G: Gamestate<M>, M: Move, E: Evaluate<G>> Negamax<G, M, E> {
    pub fn search(&mut self) -> SearchResult<M> {
        let start = std::time::Instant::now();
        let aim = NegamaxAim::from(self.node.gamestate.player_aim());
        let expiration = self
            .max_time
            .map(|duration| std::time::Instant::now() + duration);

        let exit = match self.alpha_beta {
            true => negamax_ab(
                &mut self.node,
                &mut self.evaluator,
                self.max_depth.unwrap_or(u8::MAX),
                expiration,
                aim,
                f32::NEG_INFINITY,
                f32::INFINITY,
            ),
            false => negamax(
                &mut self.node,
                &mut self.evaluator,
                self.max_depth.unwrap_or(u8::MAX),
                aim,
            ),
        };

        SearchResult {
            best: self.node.best.clone().unwrap(),
            value: self.node.value.unwrap(),
            exit,
            nodes: self.node.descendants,
            terminals: self.node.terminals,
            time: start.elapsed(),
            depth: self.node.search_depth,
        }
    }

    /// Play the given move, advancing the tree down a node
    pub fn play_move(&mut self, m: &M) {
        self.node.advance(m);
    }
}

/// Negamax search, no pruning or timeout
pub fn negamax<G: Gamestate<M>, M: Move, E: Evaluate<G>>(
    node: &mut Node<G, M>,
    evaluator: &mut E,
    depth: u8,
    aim: NegamaxAim,
) -> SearchExit {
    if node.get_moves() == 0 {
        // Game end condition, evaluate the gamestate
        node.evaluate(evaluator, aim.into());
        SearchExit::Terminal
    } else if depth == 0 {
        // end of recursion, evaluate the gamestate
        node.evaluate(evaluator, aim.into());
        SearchExit::Depth
    } else {
        // continue recursion
        node.reset_stats();
        // track best value and move
        let mut best = (f32::NEG_INFINITY, None);
        // Assume exhaustive first, any time or depth will override
        let mut exit = SearchExit::Exhaustive;
        // Go through each move and child (iterator creates children on the fly)
        let mut descendants = 0;
        let mut terminals = 0;
        for (m, child) in node.into_iter() {
            // Clone gamestate to make child
            // Recurse with child node and remember best
            match negamax(child, evaluator, depth - 1, -aim) {
                SearchExit::Depth => {
                    exit = SearchExit::Depth;
                }
                SearchExit::Terminal => {
                    terminals += 1;
                }
                SearchExit::Time => {
                    // cleanup whatever and exit
                    return SearchExit::Time;
                }
                SearchExit::Exhaustive => {}
            };
            // Get best out of current and child
            let value = -child.value.unwrap();
            if value > best.0 {
                best = (value, Some(m.clone()));
            }
            // Update parent node details
            descendants += child.descendants + 1;
            terminals += child.terminals;
            // Update parent node with child node
        }
        node.descendants = descendants;
        node.terminals = terminals;
        node.best = Some(best.1.unwrap());
        node.value = Some(best.0);
        node.search_depth = depth;
        exit
    }
}

/// Negamax search, with alpha-beta pruning
pub fn negamax_ab<G: Gamestate<M>, M: Move, E: Evaluate<G>>(
    node: &mut Node<G, M>,
    evaluator: &mut E,
    depth: u8,
    expiration: Option<std::time::Instant>,
    aim: NegamaxAim,
    mut alpha: f32,
    beta: f32,
) -> SearchExit {
    if node.get_moves() == 0 {
        // Game end condition, evaluate the gamestate
        node.evaluate(evaluator, aim.into());
        SearchExit::Terminal
    } else if depth == 0 {
        // end of recursion, evaluate the gamestate
        node.evaluate(evaluator, aim.into());
        SearchExit::Depth
    } else {
        // Check expiration
        if let Some(expire) = expiration {
            if expire < std::time::Instant::now() {
                return SearchExit::Time;
            }
        }
        // continue recursion
        node.reset_stats();
        // track best value and move
        let mut best = (f32::NEG_INFINITY, None);
        // Assume exhaustive first, any time or depth will override
        let mut exit = SearchExit::Exhaustive;

        let mut descendants = 0;
        let mut terminals = 0;
        // Go through each move and child
        for (m, child) in node.into_iter() {
            // Recurse with child node and remember best
            match negamax_ab(child, evaluator, depth - 1, expiration, -aim, -beta, -alpha) {
                SearchExit::Depth => {
                    exit = SearchExit::Depth;
                }
                SearchExit::Terminal => {
                    terminals += 1;
                }
                SearchExit::Time => {
                    // cleanup whatever and exit
                    return SearchExit::Time;
                }
                SearchExit::Exhaustive => {}
            };

            // Get best out of current and child
            let value = -child.value.unwrap();
            if value > best.0 {
                best = (value, Some(m.clone()));
            }
            // Update parent node details
            descendants += child.descendants + 1;
            terminals += child.terminals;

            // check a/b
            alpha = alpha.max(value);
            if alpha >= beta {
                break;
            }
        }
        node.descendants = descendants;
        node.terminals = terminals;
        node.best = Some(best.1.unwrap());
        node.value = Some(best.0);
        node.search_depth = depth;
        exit
    }
}

#[cfg(test)]
mod test {
    use std::result;

    use super::{negamax, Negamax};
    use crate::{
        games::tictactoe::{Ttt, TttEvaluator, TttMove},
        negamax::{negamax_ab, NegamaxAim},
        node::Node,
    };

    #[test]
    fn negamax_ttt() {
        // Create a game
        let ttt = Ttt::new(crate::games::tictactoe::Player::One);
        // Create node
        let mut node = Node::new(ttt);
        // Create evaluator
        let mut evaluator = crate::games::tictactoe::TttEvaluator;
        // Run search
        assert_eq!(
            negamax(&mut node, &mut evaluator, 9, NegamaxAim::Maximise),
            crate::SearchExit::Exhaustive
        );
        assert_eq!(node.best, Some(TttMove::from(0)));
        assert_eq!(node.terminals, 255168);

        // player 2
        let ttt = Ttt::new(crate::games::tictactoe::Player::Two);
        // Create node
        let mut node = Node::new(ttt);
        // Create evaluator
        let mut evaluator = crate::games::tictactoe::TttEvaluator;
        // Run search
        assert_eq!(
            negamax(&mut node, &mut evaluator, 9, NegamaxAim::Minimise),
            crate::SearchExit::Exhaustive
        );
        assert_eq!(node.best, Some(TttMove::from(0)));
        assert_eq!(node.terminals, 255168);

        //  Repeated calls
        let ttt = Ttt::new(crate::games::tictactoe::Player::One);
        let mut node = Node::new(ttt);
        assert_eq!(
            negamax(&mut node, &mut evaluator, 2, NegamaxAim::Maximise),
            crate::SearchExit::Depth
        );
        assert_eq!(node.best, Some(TttMove::from(0)));
        assert_eq!(
            negamax(&mut node, &mut evaluator, 6, NegamaxAim::Maximise),
            crate::SearchExit::Depth
        );
        assert_eq!(node.best, Some(TttMove::from(0)));
        assert_eq!(
            negamax(&mut node, &mut evaluator, 9, NegamaxAim::Maximise),
            crate::SearchExit::Exhaustive
        );
        dbg!(node.children.len());
        dbg!(node.descendants);
        assert_eq!(node.best, Some(TttMove::from(0)));
        assert_eq!(node.terminals, 255168);
    }

    #[test]
    fn ttt_advance() {
        let mut n = Negamax::new(Node::new(Ttt::default()), TttEvaluator);

        for _ in 0..9 {
            let result = n.search();
            println!("{:?}", result);
            n.play_move(&result.best);
        }
    }

    #[test]
    fn ttt_alpha_beta() {
        // Create a game
        let ttt = Ttt::new(crate::games::tictactoe::Player::One);
        // Create node
        let mut node = Node::new(ttt);
        // Create evaluator
        let mut evaluator = crate::games::tictactoe::TttEvaluator;
        // Run search
        assert_eq!(
            negamax_ab(
                &mut node,
                &mut evaluator,
                9,
                None,
                NegamaxAim::Maximise,
                f32::NEG_INFINITY,
                f32::INFINITY
            ),
            crate::SearchExit::Exhaustive
        );
        assert_eq!(node.best, Some(TttMove::from(0)));
        // assert_eq!(node.terminals, 255168);

        let mut n = Negamax::new(Node::new(Ttt::default()), TttEvaluator);
        n.alpha_beta = true;
        // n.max_depth = Some(3);
        let result = n.search();
        println!("{:?}", result);
        n.play_move(&result.best);
        let result = n.search();
        println!("{:?}", result);
        assert_eq!(result.best, TttMove::from(4));
    }
}
