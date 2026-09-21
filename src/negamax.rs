use atomic_float::{AtomicF32, AtomicF64};
use core::panic;
use log::{debug, trace};
use rayon::prelude::*;
use std::{
    ops::{Mul, Neg},
    sync::atomic::Ordering,
    time::Duration,
};

use crate::tt::{Probe, Tt};
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
    /// Prefer short winning lines and long losing ones.
    /// Weakens the alpha == beta cutoff, so it costs some nodes.
    pub prune_by_path_length: bool,
    /// Pick uniformly at random between root moves that tie on depth and
    /// value. Also weakens the alpha == beta cutoff.
    pub random_best: bool,
    /// Pick randomly between root moves, weighted towards better values.
    /// `w` makes each +1 of value `w` times more likely. 0 disables.
    pub random_weight: f32,
    /// Size of the transposition table, as bits of index: `2^bits` entries.
    /// 0 disables it, which is the default. Needs `Gamestate::position_key`
    /// to return something other than 0 to be of any use.
    pub tt_bits: u8,
    /// Order children by their own static evaluation the first time a node is
    /// expanded.
    ///
    /// `pre_sort` can only order children a previous iteration has already
    /// given a value to, so the very first visit to a node searches them in
    /// whatever order move generation produced. Alpha-beta only cuts off once
    /// a good move has been found, so that first ordering decides how much of
    /// the subtree is skipped. Costs one evaluation per child, and forces all
    /// children to be created rather than generated lazily.
    pub sort_on_create: bool,
    /// Only pay for `sort_on_create` when at least this much depth remains.
    ///
    /// Creating every child costs a clone and a move application each, which
    /// is wasted on any child alpha-beta then cuts off. Near the leaves the
    /// subtree saved is too small to repay that, so the ordering only earns
    /// its keep higher up. 0 applies it everywhere.
    pub sort_on_create_min_depth: u8,
    /// Start iterative deepening from this depth instead of 1.
    ///
    /// Each deepening pass exists to leave behind the move ordering the next
    /// one relies on. When the tree has already been searched -- the subtree
    /// kept by `Negamax::play_move`, or a warm transposition table -- that
    /// ordering is already there and the early passes only redo it.
    ///
    /// 0 and 1 both mean "start at 1", the unchanged behaviour. Skipping is
    /// only sound as far as the existing ordering reaches: starting cold at a
    /// high depth searches the unordered worst-case tree and is far slower
    /// than deepening into it. So unless the transposition table has
    /// something in it, the request is clamped to one past the depth the
    /// retained tree was actually searched to, which is exactly one deepening
    /// step. `u8::MAX` therefore means "as deep as the retained tree allows".
    /// The request is also clamped to `max_depth`.
    ///
    /// The first pass is run without the time limit, as it always was, so
    /// that there is a result to return; with a high `initial_depth` and
    /// nothing to order by, that first pass can overrun `max_time`.
    pub initial_depth: u8,
    /// Stop cutting off on `alpha == beta`, so equal-valued siblings survive
    /// to be chosen between. Required for `prune_by_path_length` to see
    /// alternatives, and widens `random_best`. Costs a lot of nodes when the
    /// evaluator produces many equal values.
    pub keep_equal_siblings: bool,
}

/// Negamax search with pruning and timeout
pub struct Negamax<G, M, E> {
    node: Node<G, M>,
    evaluator: E,
    /// Options
    pub options: SearchOptions,
    /// Shared across the iterative deepening passes, so a result found at one
    /// depth can save work at the next.
    tt: Tt<M>,
}

impl<G, M: Move, E> Negamax<G, M, E> {
    pub fn new(node: Node<G, M>, evaluator: E, options: SearchOptions) -> Self {
        let tt = Tt::new(options.tt_bits);
        Negamax {
            node,
            evaluator,
            options,
            tt,
        }
    }

    /// Transposition table hits and stores from the last search.
    pub fn tt_stats(&self) -> (u64, u64) {
        (self.tt.hits, self.tt.stores)
    }

    pub fn replace_gamestate(&mut self, gamestate: G) {
        self.node = Node::new(gamestate);
    }
}

impl<G: Gamestate<M>, M: Move, E: Evaluate<G>> Negamax<G, M, E> {
    /// The move to report from the root, honouring the random selection
    /// options. Falls back to the search's own choice.
    fn root_move(&self, aim: NegamaxAim) -> (M, f32) {
        if self.options.random_best || self.options.random_weight > 0.0 {
            if let Some(chosen) = pick_root_move(&self.node, aim, &self.options) {
                return chosen;
            }
        }
        (self.node.best.clone().unwrap(), self.node.value.unwrap())
    }

    /// Depth for the first iterative deepening pass. See
    /// [`SearchOptions::initial_depth`] for why the request is clamped.
    fn start_depth(&self) -> u8 {
        let requested = self.options.initial_depth;
        if requested <= 1 {
            return 1;
        }
        // The retained tree carries ordering down to `search_depth`; one more
        // ply is the normal deepening step. A table with entries in it orders
        // positions the tree no longer holds, so take the caller's word then.
        let supported = if self.tt.stores > 0 {
            requested
        } else {
            self.node.search_depth.saturating_add(1)
        };
        requested
            .min(supported)
            .min(self.options.max_depth.unwrap_or(u8::MAX))
            .max(1)
    }

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
            let mut depth = self.start_depth();
            let mut result = None;
            loop {
                match if self.options.alpha_beta {
                    if self.options.parallel {
                        debug!("Running parallel negamax with depth {depth}");
                        negamax_ab_parallel(
                            &mut self.node,
                            &mut self.evaluator,
                            depth,
                            if result.is_some() { expiration } else { None },
                            aim,
                            f32::NEG_INFINITY,
                            f32::INFINITY,
                            self.options,
                            &mut self.tt,
                        )
                    } else {
                        debug!("Running single threaded negamax with depth {depth}");
                        negamax_ab(
                            &mut self.node,
                            &mut self.evaluator,
                            depth,
                            if result.is_some() { expiration } else { None },
                            aim,
                            f32::NEG_INFINITY,
                            f32::INFINITY,
                            self.options,
                            &mut self.tt,
                        )
                    }
                } else {
                    negamax(&mut self.node, &mut self.evaluator, depth, aim)
                } {
                    SearchExit::Depth => {
                        // Store result and carry on to next depth
                        let root = self.root_move(aim);
                        result = Some(SearchResult {
                            best: root.0,
                            value: root.1,
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
                        let root = self.root_move(aim);
                        return SearchResult {
                            best: root.0,
                            value: root.1,
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
                    self.options,
                    &mut self.tt,
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
                    self.options,
                    &mut self.tt,
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

        let root = self.root_move(aim);
        SearchResult {
            best: root.0,
            value: root.1,
            exit,
            nodes: self.node.descendants,
            terminals: self.node.terminals,
            time: start.elapsed(),
            depth: self.node.search_depth,
        }
    }

    /// Play the given move, advancing the tree down a node.
    ///
    /// Returns whether the subtree for that move existed and was kept; see
    /// [`Node::advance`].
    pub fn play_move(&mut self, m: &M) -> bool {
        self.node.advance(m)
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
    opts: SearchOptions,
    // Accepted for signature parity with the sequential search. The table is
    // not shared across rayon tasks: each gets its own, so a shared one here
    // would go unused and hide that.
    _tt: &mut Tt<M>,
) -> SearchExit {
    // Run the negamax search in parallel at this depth
    if depth == 0 {
        // End of recursion. Checked before move generation: leaves are the bulk
        // of the tree and generating their moves only to discard them dominates
        // the search when the move generator allocates.
        node.evaluate(evaluator, aim.into());
        if node.gamestate.is_terminal() {
            node.path_length = 0;
            SearchExit::Terminal
        } else {
            node.path_length = 1;
            SearchExit::Depth
        }
    } else if node.get_moves() == 0 {
        // Game end condition, evaluate the gamestate
        node.evaluate(evaluator, aim.into());
        node.path_length = 0;
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
                // Only flip the window when the turn actually passes over.
                let child_aim = NegamaxAim::from(child.gamestate.player_aim());
                let a_now = alpha.load(Ordering::Relaxed);
                let (c_aim, c_alpha, c_beta) = if child_aim == aim {
                    (aim, a_now, beta)
                } else {
                    (-aim, -beta, -a_now)
                };
                // Recurse with child node and remember best
                let exit = negamax_ab(
                    child,
                    &mut evaluator,
                    depth - 1,
                    expiration,
                    c_aim,
                    c_alpha,
                    c_beta,
                    opts,
                    &mut Tt::new(0),
                );

                // Get best out of current and child
                let value = parent_view(child, aim);

                // Raise alpha atomically; a load/store pair loses concurrent
                // updates and silently weakens pruning.
                alpha.fetch_max(value, Ordering::AcqRel);

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

/// A child's value as the parent sees it.
///
/// Most games alternate turns, so the child's value is negated. Games that
/// can grant a repeat turn (Kalah landing its last stone in its own store)
/// produce children with the same player to move, and those must not be
/// flipped.
fn parent_view<G: Gamestate<M>, M: Move>(child: &Node<G, M>, parent_aim: NegamaxAim) -> f32 {
    match child.value {
        Some(v) => {
            if NegamaxAim::from(child.gamestate.player_aim()) == parent_aim {
                v
            } else {
                -v
            }
        }
        None => f32::NEG_INFINITY,
    }
}

/// Re-pick the root move among equally good alternatives.
///
/// The search keeps the first move that beats the incumbent, so a
/// deterministic engine replays the same game forever. That is poor value as
/// a training opponent, hence the option to spread the choice over moves the
/// search cannot separate.
fn pick_root_move<G: Gamestate<M>, M: Move>(
    node: &Node<G, M>,
    aim: NegamaxAim,
    opts: &SearchOptions,
) -> Option<(M, f32)> {
    use rand::Rng;
    if node.children.is_empty() {
        return None;
    }
    let best_value = node.value?;
    // Depth of the child the search actually settled on. Children left behind
    // by a cutoff have a staler search_depth and are not real alternatives.
    let best_depth = node
        .children
        .iter()
        .find(|(m, _)| Some(m) == node.best.as_ref())
        .map(|(_, c)| c.search_depth)?;

    let mut rng = rand::thread_rng();

    if opts.random_best {
        let tied: Vec<&M> = node
            .children
            .iter()
            .filter(|(_, c)| c.search_depth == best_depth && parent_view(c, aim) == best_value)
            .map(|(m, _)| m)
            .collect();
        if tied.is_empty() {
            return None;
        }
        // Every candidate ties the best value exactly, so the reported value
        // is unchanged by the choice.
        return Some((tied[rng.gen_range(0..tied.len())].clone(), best_value));
    }

    if opts.random_weight > 0.0 {
        // weight^(value - best) so each +1 of value is `weight` times likelier.
        let mut total = 0.0f32;
        let weighted: Vec<(&M, f32, f32)> = node
            .children
            .iter()
            .filter(|(_, c)| c.search_depth == best_depth)
            .map(|(m, c)| {
                let v = parent_view(c, aim);
                let w = opts
                    .random_weight
                    .powf(v - best_value)
                    .clamp(f32::MIN_POSITIVE, f32::MAX);
                total += w;
                (m, w, v)
            })
            .collect();
        if weighted.is_empty() || !total.is_finite() || total <= 0.0 {
            return None;
        }
        let mut pick = rng.gen_range(0.0..total);
        for (m, w, v) in weighted {
            pick -= w;
            if pick <= 0.0 {
                // Report the value of the move actually chosen. This can be
                // below the root's best, which is the point of the option.
                return Some((m.clone(), v));
            }
        }
    }
    None
}

/// Order children best-first for the player to move at `aim`.
///
/// Primary key is how deeply the child was actually searched: a child left
/// with a stale value by a beta cutoff should not displace one with a fresh
/// result. Value breaks that tie, and path length breaks the value tie when
/// `prune_by_path_length` is on.
fn order_children<G: Gamestate<M>, M: Move>(
    node: &mut Node<G, M>,
    aim: NegamaxAim,
    prune_by_path_length: bool,
) {
    node.children.sort_by(|(_, a), (_, b)| {
        b.search_depth
            .cmp(&a.search_depth)
            .then_with(|| {
                parent_view(b, aim)
                    .partial_cmp(&parent_view(a, aim))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                if !prune_by_path_length {
                    return std::cmp::Ordering::Equal;
                }
                if parent_view(a, aim) > 0.0 {
                    a.path_length.cmp(&b.path_length)
                } else {
                    b.path_length.cmp(&a.path_length)
                }
            })
    });
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
            node.path_length = 0;
            SearchExit::Terminal
        } else {
            node.path_length = 1;
            SearchExit::Depth
        }
    } else if node.get_moves() == 0 {
        // Game end condition, evaluate the gamestate
        node.evaluate(evaluator, aim.into());
        node.path_length = 0;
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
            let child_aim = NegamaxAim::from(child.gamestate.player_aim());
            match negamax(child, evaluator, depth - 1, child_aim) {
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
            let value = parent_view(child, aim);
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
    mut beta: f32,
    opts: SearchOptions,
    tt: &mut Tt<M>,
) -> SearchExit {
    if depth == 0 {
        // End of recursion. Checked before move generation: leaves are the bulk
        // of the tree and generating their moves only to discard them dominates
        // the search when the move generator allocates.
        node.evaluate(evaluator, aim.into());
        if node.gamestate.is_terminal() {
            node.path_length = 0;
            SearchExit::Terminal
        } else {
            node.path_length = 1;
            SearchExit::Depth
        }
    } else if node.get_moves() == 0 {
        // Game end condition, evaluate the gamestate
        node.evaluate(evaluator, aim.into());
        node.path_length = 0;
        SearchExit::Terminal
    } else {
        // Check expiration
        if let Some(expire) = expiration {
            if expire < std::time::Instant::now() {
                return SearchExit::Time;
            }
        }
        // Ask the table before doing any work: this position may already have
        // been searched down a different move order.
        let key = if tt.enabled() { node.gamestate.position_key() } else { 0 };
        let alpha_orig = alpha;
        let mut tt_best: Option<M> = None;
        if key != 0 {
            match tt.lookup(key, depth, alpha, beta) {
                Probe::Cutoff(value, m) => {
                    node.value = Some(value);
                    node.best = Some(m);
                    node.search_depth = depth;
                    node.descendants = 0;
                    node.terminals = 0;
                    node.path_length = 1;
                    // Deliberately not Exhaustive: a stored value says nothing
                    // about whether that subtree reached its leaves, and
                    // claiming otherwise would stop deepening early.
                    return SearchExit::Depth;
                }
                Probe::Hint { alpha: a, beta: b, best } => {
                    alpha = a;
                    beta = b;
                    tt_best = best;
                }
                Probe::Miss => {}
            }
        }

        // continue recursion
        node.reset_stats();
        // A move that was best for this position elsewhere in the tree is the
        // best guess here too. ChildrenIter pops from the back, so moving it
        // last gets it searched first.
        if let Some(m) = &tt_best {
            if node.children.is_empty() && !node.moves.is_empty() {
                if let Some(i) = node.moves.iter().position(|x| x == m) {
                    let last = node.moves.len() - 1;
                    node.moves.swap(i, last);
                }
            }
        }
        // Order children by the previous iteration's results before searching.
        // A child stores its value from its own perspective and the parent
        // negates it, so ascending child value puts this node's best move
        // first, which is what makes alpha-beta cut off early.
        if opts.sort_on_create && node.children.is_empty() && depth >= opts.sort_on_create_min_depth
        {
            // First visit: nothing has an inherited value yet, so fall back to
            // each child's own static evaluation rather than move order.
            node.create_all_children();
            for (_, child) in node.children.iter_mut() {
                let child_aim = NegamaxAim::from(child.gamestate.player_aim());
                child.evaluate(evaluator, child_aim.into());
            }
            order_children(node, aim, opts.prune_by_path_length);
        } else if opts.pre_sort && !node.children.is_empty() {
            order_children(node, aim, opts.prune_by_path_length);
        }
        // track best value and move
        let mut best = (f32::NEG_INFINITY, None);
        let mut best_path: u8 = 0;
        // Assume exhaustive first, any time or depth will override
        let mut exit = SearchExit::Exhaustive;

        let mut descendants = 0;
        let mut terminals = 0;
        // Go through each move and child
        for (m, child) in node.into_iter() {
            trace!("Exploring move {m:?} at depth {depth}");
            // Recurse with child node and remember best
            // Only flip the window when the turn actually passes over.
            let child_aim = NegamaxAim::from(child.gamestate.player_aim());
            let (c_aim, c_alpha, c_beta) = if child_aim == aim {
                (aim, alpha, beta)
            } else {
                (-aim, -beta, -alpha)
            };
            match negamax_ab(
                child,
                evaluator,
                depth - 1,
                expiration,
                c_aim,
                c_alpha,
                c_beta,
                opts,
                tt,
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
            let value = parent_view(child, aim);
            if value > best.0 {
                best = (value, Some(m.clone()));
                best_path = child.path_length;
            } else if opts.prune_by_path_length && value == best.0 && best.1.is_some() {
                // Same value: take the quicker win, or the slower loss.
                let better = if best.0 > 0.0 {
                    child.path_length < best_path
                } else {
                    child.path_length > best_path
                };
                if better {
                    best = (value, Some(m.clone()));
                    best_path = child.path_length;
                }
            }
            // Update parent node details
            descendants += child.descendants + 1;
            terminals += child.terminals;

            // check a/b
            alpha = alpha.max(value);
            if alpha > beta {
                break;
            } else if alpha >= beta && !opts.keep_equal_siblings {
                // Cutting on equality hides equal-valued siblings. Keeping them
                // is what path-length preference needs, and it widens the pool
                // random_best can draw from -- but it is expensive, so it is
                // opt-in rather than implied.
                break;
            }
        }
        node.descendants = descendants;
        node.terminals = terminals;
        node.best = Some(best.1.unwrap());
        node.value = Some(best.0);
        node.search_depth = depth;
        node.path_length = best_path.saturating_add(1);
        if key != 0 {
            tt.record(key, depth, best.0, alpha_orig, beta, node.best.clone());
        }
        exit
    }
}

#[cfg(test)]
mod test {
    use std::result;

    use super::{negamax, Negamax};
    use crate::tt::Tt;
    use crate::{
        games::tictactoe::{Ttt, TttEvaluator, TttMove},
        negamax::{negamax_ab, NegamaxAim, SearchOptions},
        node::Node,
    };

    use test_log::test;

    /// Perfect play from an empty board draws, so every opening move scores
    /// 0.0 and `best` is decided purely by tie-break: the search keeps the
    /// first move it sees (strict `>`), and `ChildrenIter` pops moves off the
    /// back of the list, so move 8 wins. These assertions therefore pin
    /// iteration order, not game-theoretic correctness -- the meaningful
    /// assertion is the terminal count.
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
        assert_eq!(node.best, Some(TttMove::from(8)));
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
        assert_eq!(node.best, Some(TttMove::from(8)));
        assert_eq!(node.terminals, 255168);

        //  Repeated calls
        let ttt = Ttt::new(crate::games::tictactoe::Player::One);
        let mut node = Node::new(ttt);
        assert_eq!(
            negamax(&mut node, &mut evaluator, 2, NegamaxAim::Maximise),
            crate::SearchExit::Depth
        );
        assert_eq!(node.best, Some(TttMove::from(8)));
        assert_eq!(
            negamax(&mut node, &mut evaluator, 6, NegamaxAim::Maximise),
            crate::SearchExit::Depth
        );
        assert_eq!(node.best, Some(TttMove::from(8)));
        assert_eq!(
            negamax(&mut node, &mut evaluator, 9, NegamaxAim::Maximise),
            crate::SearchExit::Exhaustive
        );
        dbg!(node.children.len());
        dbg!(node.descendants);
        assert_eq!(node.best, Some(TttMove::from(8)));
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

    /// See `negamax_ttt`: `best` here is the tie-break winner among nine
    /// drawing opening moves, not a uniquely correct answer.
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
                SearchOptions::default(),
                &mut Tt::new(0)
            ),
            crate::SearchExit::Exhaustive
        );
        assert_eq!(node.best, Some(TttMove::from(8)));
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

    /// The reported value must describe the move actually returned.
    /// `random_best` only ever picks exact ties, so it cannot move the value;
    /// `random_weight` can pick a worse move and must then say so.
    #[test]
    fn random_selection_value_matches_chosen_move() {
        use crate::games::mancala::{Mancala, MancalaEvaluator};

        let base = SearchOptions {
            alpha_beta: true,
            iterative: true,
            pre_sort: true,
            max_depth: Some(4),
            ..Default::default()
        };

        let deterministic = Negamax::new(Node::new(Mancala::new()), MancalaEvaluator, base)
            .search()
            .value;

        // Ties only: the value can never differ from the deterministic best.
        for _ in 0..50 {
            let r = Negamax::new(
                Node::new(Mancala::new()),
                MancalaEvaluator,
                SearchOptions { random_best: true, ..base },
            )
            .search();
            assert_eq!(
                r.value, deterministic,
                "random_best must only choose between equally valued moves"
            );
        }

        // Weighted: sometimes picks a worse move, and must report that move's
        // value rather than the root's best. Never better than the best.
        let mut saw_worse = false;
        for _ in 0..200 {
            let r = Negamax::new(
                Node::new(Mancala::new()),
                MancalaEvaluator,
                SearchOptions { random_weight: 1.2, ..base },
            )
            .search();
            assert!(
                r.value <= deterministic,
                "reported value {} exceeds the best {}",
                r.value,
                deterministic
            );
            if r.value < deterministic {
                saw_worse = true;
            }
        }
        assert!(
            saw_worse,
            "random_weight never reported a below-best value, so value is not tracking the choice"
        );
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

    /// The move played is not always one the search expanded: a time-limited
    /// search can stop before it reaches every root move, and analysing a
    /// recorded game plays the record's move rather than the search's. That
    /// used to panic in `Node::advance`.
    #[test]
    fn play_move_rebuilds_when_move_was_not_expanded() {
        let opts = SearchOptions {
            alpha_beta: true,
            iterative: true,
            max_depth: Some(3),
            ..Default::default()
        };
        let mut n = Negamax::new(Node::new(Ttt::default()), TttEvaluator, opts);

        // Nothing has been searched, so no child exists for this move.
        assert!(!n.play_move(&TttMove::from(0)));
        // The rebuilt node is still searchable.
        let result = n.search();
        assert!(result.nodes > 0);
        // And a move the search did expand keeps its subtree.
        assert!(n.play_move(&result.best));
    }

    /// `initial_depth` may skip the early deepening passes but must not
    /// change the answer: at a fixed depth the value is the exact minimax
    /// value whatever order the moves were searched in.
    #[test]
    fn initial_depth_does_not_change_the_answer() {
        let base = SearchOptions {
            alpha_beta: true,
            iterative: true,
            pre_sort: true,
            max_depth: Some(5),
            ..Default::default()
        };
        let mut from_one = Negamax::new(Node::new(Ttt::default()), TttEvaluator, base);
        let mut skipping = Negamax::new(
            Node::new(Ttt::default()),
            TttEvaluator,
            SearchOptions { initial_depth: u8::MAX, ..base },
        );

        for _ in 0..5 {
            let a = from_one.search();
            let b = skipping.search();
            assert_eq!(a.value, b.value);
            assert!(b.depth <= 5, "start depth must respect max_depth");
            // Same move into both, so the two stay on the same position.
            from_one.play_move(&a.best);
            skipping.play_move(&a.best);
        }
    }

    /// A cold node has no ordering to skip ahead on, so the request is
    /// clamped back to a normal start from depth 1.
    #[test]
    fn initial_depth_is_clamped_on_a_cold_node() {
        let opts = SearchOptions {
            alpha_beta: true,
            iterative: true,
            max_depth: Some(4),
            initial_depth: u8::MAX,
            ..Default::default()
        };
        let n = Negamax::new(Node::new(Ttt::default()), TttEvaluator, opts);
        assert_eq!(n.start_depth(), 1);

        // After a search the tree carries ordering, so one step past what it
        // was searched to is allowed.
        let mut n = n;
        n.search();
        n.play_move(&TttMove::from(0));
        assert_eq!(n.start_depth(), (n.node.search_depth + 1).min(4));
        assert!(n.start_depth() > 1);
    }
}
