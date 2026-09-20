use atomic_float::{AtomicF32, AtomicF64};
use core::panic;
use log::{debug, trace};
use rayon::prelude::*;
use std::{
    ops::{Mul, Neg},
    sync::atomic::Ordering,
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

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct SearchOptions {
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
    /// Run the search in parallel
    pub parallel: bool,
}

/// Negamax search with pruning and timeout
pub struct Negamax<G, M, E> {
    node: Node<G, M>,
    evaluator: E,
    /// Options
    pub options: SearchOptions,
}

impl<G, M, E> Negamax<G, M, E> {
    pub fn new(node: Node<G, M>, evaluator: E, options: SearchOptions) -> Self {
        Negamax {
            node,
            evaluator,
            options,
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
            .options
            .max_time
            .map(|duration| std::time::Instant::now() + duration);

        // Check if iterative enabled
        if self.options.iterative {
            // Implement iterative deepening
            let mut depth = 1;
            let mut result = None;
            loop {
                match if self.options.alpha_beta {
                    if self.options.parallel {
                        debug!("Running parallel negamax with depth {depth}");
                        negamax_ab_parallel(
                            &mut self.node,
                            &mut self.evaluator,
                            depth,
                            if depth > 1 { expiration } else { None },
                            aim,
                            f32::NEG_INFINITY,
                            f32::INFINITY,
                            self.options.pre_sort,
                        )
                    } else {
                        debug!("Running single threaded negamax with depth {depth}");
                        negamax_ab(
                            &mut self.node,
                            &mut self.evaluator,
                            depth,
                            if depth > 1 { expiration } else { None },
                            aim,
                            f32::NEG_INFINITY,
                            f32::INFINITY,
                            self.options.pre_sort,
                        )
                    }
                } else {
                    negamax(&mut self.node, &mut self.evaluator, depth, aim)
                } {
                    SearchExit::Depth => {
                        // Store result and carry on to next depth
                        result = Some(SearchResult {
                            best: self.node.best.clone().unwrap(),
                            value: self.node.value.unwrap(),
                            exit: SearchExit::Depth,
                            nodes: self.node.descendants,
                            terminals: self.node.terminals,
                            time: start.elapsed(),
                            depth: self.node.search_depth,
                        });
                        // Stop once the requested depth has been completed,
                        // otherwise a search with no time limit never returns.
                        if self.options.max_depth.is_some_and(|max| depth >= max) {
                            return result.unwrap();
                        }
                        depth += 1;
                    }
                    SearchExit::Time => {
                        // This depth failed, return previous depth result
                        let mut result = result.unwrap();
                        result.exit = SearchExit::Time;
                        result.time = start.elapsed();
                        return result;
                    }
                    SearchExit::Exhaustive => {
                        // Full search complete, return result
                        return SearchResult {
                            best: self.node.best.clone().unwrap(),
                            value: self.node.value.unwrap(),
                            exit: SearchExit::Exhaustive,
                            nodes: self.node.descendants,
                            terminals: self.node.terminals,
                            time: start.elapsed(),
                            depth: self.node.search_depth,
                        };
                    }
                    SearchExit::Terminal => {
                        // No moves available, return error
                        panic!("No moves available");
                    }
                }
            }
        }

        let exit = if self.options.alpha_beta {
            if self.options.parallel {
                debug!("Running parallel negamax");
                negamax_ab_parallel(
                    &mut self.node,
                    &mut self.evaluator,
                    self.options.max_depth.unwrap_or(u8::MAX),
                    expiration,
                    aim,
                    f32::NEG_INFINITY,
                    f32::INFINITY,
                    self.options.pre_sort,
                )
            } else {
                debug!("Running single threaded negamax");
                negamax_ab(
                    &mut self.node,
                    &mut self.evaluator,
                    self.options.max_depth.unwrap_or(u8::MAX),
                    expiration,
                    aim,
                    f32::NEG_INFINITY,
                    f32::INFINITY,
                    self.options.pre_sort,
                )
            }
        } else {
            negamax(
                &mut self.node,
                &mut self.evaluator,
                self.options.max_depth.unwrap_or(u8::MAX),
                aim,
            )
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

#[derive(Debug, Clone, Copy)]
struct ParallelResult<M> {
    exit: SearchExit,
    best: f32,
    mov: M,
    descendants: u32,
    terminals: u32,
}

fn negamax_ab_parallel<G: Gamestate<M>, M: Move, E: Evaluate<G>>(
    node: &mut Node<G, M>,
    evaluator: &mut E,
    depth: u8,
    expiration: Option<std::time::Instant>,
    aim: NegamaxAim,
    alpha: f32,
    beta: f32,
    pre_sort: bool,
) -> SearchExit {
    // Run the negamax search in parallel at this depth
    if depth == 0 {
        // End of recursion. Checked before move generation: leaves are the bulk
        // of the tree and generating their moves only to discard them dominates
        // the search when the move generator allocates.
        node.evaluate(evaluator, aim.into());
        if node.gamestate.is_terminal() {
            SearchExit::Terminal
        } else {
            SearchExit::Depth
        }
    } else if node.get_moves() == 0 {
        // Game end condition, evaluate the gamestate
        node.evaluate(evaluator, aim.into());
        SearchExit::Terminal
    } else {
        // Check expiration
        if let Some(expire) = expiration {
            if expire < std::time::Instant::now() {
                return SearchExit::Time;
            }
        }
        // continue recursion
        node.reset_stats();
        // Go through each move and child in parallel using rayon
        // Print node move state
        node.create_all_children();

        let alpha = AtomicF32::new(alpha);
        let mut results = Vec::new();
        let evaluators = node
            .children
            .iter_mut()
            .map(|child| evaluator.clone())
            .collect::<Vec<_>>();
        node.children
            .par_iter_mut()
            .zip(evaluators)
            .map(|((m, child), mut evaluator)| {
                // Recurse with child node and remember best
                let exit = negamax_ab(
                    child,
                    &mut evaluator,
                    depth - 1,
                    expiration,
                    -aim,
                    -beta,
                    -alpha.load(Ordering::Relaxed),
                    pre_sort,
                );

                // Get best out of current and child
                let value = -child.value.unwrap();

                // Store larger of alpha and value in alpha
                let a = alpha.load(Ordering::Acquire);
                alpha.store(a.max(value), Ordering::Release);

                // Return result
                let result = ParallelResult {
                    exit,
                    best: value,
                    mov: m.clone(),
                    descendants: child.descendants + 1,
                    terminals: child.terminals + 1,
                };
                debug!("Result: {:?}", result);
                result
            })
            .collect_into_vec(&mut results);

        // Update node stats
        let mut best = (f32::NEG_INFINITY, None);
        for r in &results {
            node.descendants += r.descendants;
            node.terminals += r.terminals;
            if r.best > best.0 {
                best = (r.best, Some(r.mov.clone()));
            }
        }
        node.best = Some(best.1.unwrap());
        node.value = Some(best.0);
        node.search_depth = depth;
        // Return exit reason
        let mut exit = SearchExit::Exhaustive;
        for r in results {
            match r.exit {
                SearchExit::Depth => {
                    exit = SearchExit::Depth;
                }
                SearchExit::Terminal => {}
                SearchExit::Time => {
                    return SearchExit::Time;
                }
                SearchExit::Exhaustive => {}
            }
        }
        exit
    }
}

/// Negamax search, no pruning or timeout
pub fn negamax<G: Gamestate<M>, M: Move, E: Evaluate<G>>(
    node: &mut Node<G, M>,
    evaluator: &mut E,
    depth: u8,
    aim: NegamaxAim,
) -> SearchExit {
    if depth == 0 {
        // End of recursion. Checked before move generation: leaves are the bulk
        // of the tree and generating their moves only to discard them dominates
        // the search when the move generator allocates.
        node.evaluate(evaluator, aim.into());
        if node.gamestate.is_terminal() {
            SearchExit::Terminal
        } else {
            SearchExit::Depth
        }
    } else if node.get_moves() == 0 {
        // Game end condition, evaluate the gamestate
        node.evaluate(evaluator, aim.into());
        SearchExit::Terminal
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
    pre_sort: bool,
) -> SearchExit {
    if depth == 0 {
        // End of recursion. Checked before move generation: leaves are the bulk
        // of the tree and generating their moves only to discard them dominates
        // the search when the move generator allocates.
        node.evaluate(evaluator, aim.into());
        if node.gamestate.is_terminal() {
            SearchExit::Terminal
        } else {
            SearchExit::Depth
        }
    } else if node.get_moves() == 0 {
        // Game end condition, evaluate the gamestate
        node.evaluate(evaluator, aim.into());
        SearchExit::Terminal
    } else {
        // Check expiration
        if let Some(expire) = expiration {
            if expire < std::time::Instant::now() {
                return SearchExit::Time;
            }
        }
        // continue recursion
        node.reset_stats();
        // Order children by the previous iteration's results before searching.
        // A child stores its value from its own perspective and the parent
        // negates it, so ascending child value puts this node's best move
        // first, which is what makes alpha-beta cut off early.
        if pre_sort && !node.children.is_empty() {
            node.sort_children_ascending();
        }
        // track best value and move
        let mut best = (f32::NEG_INFINITY, None);
        // Assume exhaustive first, any time or depth will override
        let mut exit = SearchExit::Exhaustive;

        let mut descendants = 0;
        let mut terminals = 0;
        // Go through each move and child
        for (m, child) in node.into_iter() {
            trace!("Exploring move {m:?} at depth {depth}");
            // Recurse with child node and remember best
            match negamax_ab(
                child,
                evaluator,
                depth - 1,
                expiration,
                -aim,
                -beta,
                -alpha,
                pre_sort,
            ) {
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
        negamax::{negamax_ab, NegamaxAim, SearchOptions},
        node::Node,
    };

    use test_log::test;

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
        let opts = SearchOptions::default();
        let mut n = Negamax::new(Node::new(Ttt::default()), TttEvaluator, opts);

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
                f32::INFINITY,
                false
            ),
            crate::SearchExit::Exhaustive
        );
        assert_eq!(node.best, Some(TttMove::from(0)));
        // assert_eq!(node.terminals, 255168);

        let mut n = Negamax::new(Node::new(Ttt::default()), TttEvaluator, Default::default());
        n.options.alpha_beta = true;
        // n.options.max_depth = Some(3);
        let result = n.search();
        println!("{:?}", result);
        n.play_move(&result.best);
        let result = n.search();
        println!("{:?}", result);
        assert_eq!(result.best, TttMove::from(4));
    }

    #[test]
    fn ttt_parallel() {
        // Make sure that parallel and single threaded deliver the same results for each depth
        let mut ttt = Ttt::new(crate::games::tictactoe::Player::One);
        for depth in 1..10 {
            let mut par = Negamax::new(
                Node::new(ttt.clone()),
                TttEvaluator,
                SearchOptions {
                    alpha_beta: true,
                    parallel: true,
                    max_depth: Some(depth),
                    ..Default::default()
                },
            );

            let mut single = Negamax::new(
                Node::new(ttt.clone()),
                TttEvaluator,
                SearchOptions {
                    alpha_beta: true,
                    parallel: false,
                    max_depth: Some(depth),
                    ..Default::default()
                },
            );

            let par_result = par.search();
            let single_result = single.search();
            println!("Par: {:?}, Single: {:?}", par_result, single_result);
            // assert_eq!(par_result.best, single_result.best);
            assert_eq!(par_result.value, single_result.value);
        }
    }
}
