use atomic_float::{AtomicF32, AtomicF64};
use core::panic;
use log::{debug, trace};
use rayon::prelude::*;
use std::{
    ops::{Mul, Neg},
    sync::atomic::Ordering,
    time::Duration,
};

use crate::arena::{migrate, Arena, NodeId};
use crate::tt::{Probe, Tt};
use crate::{node::Node, Evaluate, Gamestate, Move, NodeAim, SearchExit, SearchResult};

/// When to discard the parts of the tree a deepening pass no longer needs.
///
/// Between passes the tree holds two quite different things: the move ordering
/// that makes alpha-beta cut off early, and the nodes that ordering was
/// derived from. Only the first is needed by the next pass. Removal writes the
/// ordering back into each node's move list and then throws the nodes away,
/// which is what stops memory growing with the union of every pass rather than
/// with the last one.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum RemovalMethod {
    /// Keep every node. The default, and the behaviour before removal existed.
    #[default]
    None,
    /// Remove after every deepening pass.
    Always,
    /// Remove once the pass just finished was at least
    /// [`SearchOptions::removal_depth`]. The early passes are small enough
    /// that removing after them costs more in regenerated nodes than it saves.
    Depth,
    /// Remove once the tree holds at least [`SearchOptions::removal_count`]
    /// nodes, which is the one that bounds memory directly.
    Count,
}

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
    /// one relies on, so when the tree kept by `Negamax::play_move` already
    /// carries that ordering the early passes look like pure repetition.
    ///
    /// **Measured, it is not.** Analysing three recorded Azul games, 199
    /// positions, cap 6, 2^20-entry table, re-rooting onto the played move
    /// each time: starting at the cap searched 1.5-1.7x the nodes of
    /// deepening from 1 and took 1.6-2.1x the time, for identical answers.
    /// The early passes are cheap -- each ply costs about 4x the one below,
    /// so depths 1..cap-1 together are about a third of the last pass -- and
    /// the ordering they leave behind is worth much more than they cost. The
    /// retained tree is only the ordering of one subtree, one ply stale, and
    /// it does not cover the nodes the deeper search creates. Leave this off
    /// unless a measurement on your game says otherwise; it is here for
    /// parity with the TypeScript original.
    ///
    /// 0 and 1 both mean "start at 1", the unchanged behaviour. The request
    /// is clamped to one past the depth the retained tree was searched to,
    /// and to `max_depth`, so `u8::MAX` means "as far as the tree allows"
    /// and a cold node always starts at 1 -- starting cold at depth 4 on the
    /// same games cost 2.5-2.8x the nodes, and the penalty grows with depth.
    ///
    /// The first pass is run without the time limit, as it always was, so
    /// that there is a result to return; a high `initial_depth` therefore
    /// makes that first pass able to overrun `max_time`.
    pub initial_depth: u8,
    /// Stop cutting off on `alpha == beta`, so equal-valued siblings survive
    /// to be chosen between. Required for `prune_by_path_length` to see
    /// alternatives, and widens `random_best`. Costs a lot of nodes when the
    /// evaluator produces many equal values.
    pub keep_equal_siblings: bool,
    /// When to discard nodes between iterative deepening passes. See
    /// [`RemovalMethod`]. Only applies when `iterative` is set.
    pub removal: RemovalMethod,
    /// Pass depth at or above which [`RemovalMethod::Depth`] removes.
    pub removal_depth: u8,
    /// Tree size at or above which [`RemovalMethod::Count`] removes.
    pub removal_count: u32,
}

/// Negamax search with pruning and timeout
pub struct Negamax<G, M, E> {
    /// Owns every node of the search tree. Re-rooting and node removal both
    /// work by moving what is kept into a fresh arena and dropping this one.
    arena: Arena<G, M>,
    root: NodeId,
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
        let (arena, root) = Arena::with_root(node);
        Negamax {
            arena,
            root,
            evaluator,
            options,
            tt,
        }
    }

    /// Transposition table hits and stores from the last search.
    pub fn tt_stats(&self) -> (u64, u64) {
        (self.tt.hits, self.tt.stores)
    }

    /// The root of the search tree, for inspecting the result of a search.
    pub fn root(&self) -> &Node<G, M> {
        self.arena.get(self.root)
    }

    /// Nodes currently held in the tree. Not the number searched: removal
    /// discards nodes a pass has finished with.
    pub fn tree_size(&self) -> usize {
        self.arena.len()
    }

    #[cfg(test)]
    pub(crate) fn arena(&self) -> &Arena<G, M> {
        &self.arena
    }

    #[cfg(test)]
    pub(crate) fn root_id(&self) -> NodeId {
        self.root
    }

    pub fn replace_gamestate(&mut self, gamestate: G) {
        let (arena, root) = Arena::with_root(Node::new(gamestate));
        self.arena = arena;
        self.root = root;
    }
}

impl<G: Gamestate<M>, M: Move, E: Evaluate<G>> Negamax<G, M, E> {
    /// The move to report from the root, honouring the random selection
    /// options. Falls back to the search's own choice.
    fn root_move(&self, aim: NegamaxAim) -> (M, f32) {
        if self.options.random_best || self.options.random_weight > 0.0 {
            if let Some(chosen) = pick_root_move(&self.arena, self.root, aim, &self.options) {
                return chosen;
            }
        }
        let root = self.arena.get(self.root);
        (root.best.clone().unwrap(), root.value.unwrap())
    }

    /// Depth for the first iterative deepening pass. See
    /// [`SearchOptions::initial_depth`] for why the request is clamped.
    fn start_depth(&self) -> u8 {
        let requested = self.options.initial_depth;
        if requested <= 1 {
            return 1;
        }
        // Only the retained tree licenses a skip. Its ordering reaches
        // `search_depth`, and one more ply is exactly one deepening step, so
        // that is the furthest the jump can land and still be searching an
        // ordered tree. A cold node therefore starts at 1 however high the
        // request.
        //
        // The transposition table is deliberately not counted as ordering.
        // Letting a warm table license the full request instead (cold node,
        // entries from earlier positions) measured worse on Azul than the
        // tree clamp on the same games: 2.0-2.9x the nodes of a normal start
        // against 1.5-1.7x. See `initial_depth`.
        requested
            .min(self.arena.get(self.root).search_depth.saturating_add(1))
            .min(self.options.max_depth.unwrap_or(u8::MAX))
            .max(1)
    }

    pub fn search(&mut self) -> SearchResult<M> {
        let start = std::time::Instant::now();
        let aim = NegamaxAim::from(self.arena.get(self.root).gamestate.player_aim());
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
                            &mut self.arena,
                            self.root,
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
                            &mut self.arena,
                            self.root,
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
                    negamax(&mut self.arena, self.root, &mut self.evaluator, depth, aim)
                } {
                    SearchExit::Depth => {
                        // Store result and carry on to next depth
                        let root = self.root_move(aim);
                        result = Some(SearchResult {
                            best: root.0,
                            value: root.1,
                            exit: SearchExit::Depth,
                            nodes: self.arena.get(self.root).descendants,
                            terminals: self.arena.get(self.root).terminals,
                            time: start.elapsed(),
                            depth: self.arena.get(self.root).search_depth,
                        });
                        // Stop once the requested depth has been completed,
                        // otherwise a search with no time limit never returns.
                        if self.options.max_depth.is_some_and(|max| depth >= max) {
                            return result.unwrap();
                        }
                        // Everything the next pass needs from this one is the
                        // move ordering, which removal writes back into the
                        // move lists before freeing the nodes it came from.
                        if self.should_remove(depth) {
                            self.remove_nodes();
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
                            nodes: self.arena.get(self.root).descendants,
                            terminals: self.arena.get(self.root).terminals,
                            time: start.elapsed(),
                            depth: self.arena.get(self.root).search_depth,
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
                    &mut self.arena,
                    self.root,
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
                    &mut self.arena,
                    self.root,
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
                &mut self.arena,
                self.root,
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
            nodes: self.arena.get(self.root).descendants,
            terminals: self.arena.get(self.root).terminals,
            time: start.elapsed(),
            depth: self.arena.get(self.root).search_depth,
        }
    }

    /// Whether the pass that just finished at `depth` should be followed by
    /// node removal.
    fn should_remove(&self, depth: u8) -> bool {
        match self.options.removal {
            RemovalMethod::None => false,
            RemovalMethod::Always => true,
            RemovalMethod::Depth => depth >= self.options.removal_depth,
            RemovalMethod::Count => self.arena.len() as u64 >= self.options.removal_count as u64,
        }
    }

    /// Collapse the tree to the principal variation plus one level of
    /// siblings, keeping the move ordering.
    ///
    /// Two steps, and the order of them is the whole point. First each node
    /// that is about to lose its children has them sorted and their moves
    /// written back into its move list, so the ordering outlives the nodes
    /// that produced it -- ordering is worth several times the search and must
    /// not be thrown away with them. Then what is left is moved into a fresh
    /// arena and the old one is dropped whole, which is what actually returns
    /// the memory.
    pub fn remove_nodes(&mut self) {
        prune_to_best(
            &mut self.arena,
            self.root,
            true,
            self.options.prune_by_path_length,
        );
        let mut kept = Arena::new();
        let root = migrate(&mut self.arena, &mut kept, self.root);
        self.arena = kept;
        self.root = root;
    }

    /// Play the given move, re-rooting the tree onto the subtree for that move
    /// and dropping the rest.
    ///
    /// Returns whether an existing subtree was found and kept. A search that
    /// ran out of time can stop before expanding every root move, and
    /// analysing a recorded game plays the move from the record rather than
    /// the one the search chose, so the move asked for may never have been
    /// expanded. In that case the tree is rebuilt from the move instead, which
    /// is the same position the caller would have got from a fresh node.
    pub fn play_move(&mut self, m: &M) -> bool {
        let existing = self
            .arena
            .get(self.root)
            .children
            .iter()
            .find(|(mov, _)| mov == m)
            .map(|(_, id)| *id);
        match existing {
            Some(child) => {
                let mut kept = Arena::new();
                let root = migrate(&mut self.arena, &mut kept, child);
                self.arena = kept;
                self.root = root;
                true
            }
            None => {
                let rebuilt = self.arena.get(self.root).play_move(m);
                let (arena, root) = Arena::with_root(rebuilt);
                self.arena = arena;
                self.root = root;
                false
            }
        }
    }
}

/// Expand every move of `id` into a child node.
///
/// Needed by the parallel search, which cannot create children lazily from
/// inside a rayon task, and by `sort_on_create`.
fn create_all_children<G: Gamestate<M>, M: Move>(arena: &mut Arena<G, M>, id: NodeId) {
    loop {
        let Some(m) = arena.get_mut(id).moves.pop() else {
            break;
        };
        let child = arena.get(id).play_move(&m);
        let child = arena.alloc(child);
        arena.get_mut(id).children.push((m, child));
    }
}

/// What a rayon task reports back about the child it searched.
#[derive(Debug, Clone)]
struct ParallelResult<M> {
    /// Index into the parent's children, so the parent can write the result
    /// back without the task holding a reference into the arena.
    index: usize,
    exit: SearchExit,
    /// The child's value as the parent sees it.
    best: f32,
    /// The child's own fields, to be copied back into the arena.
    value: Option<f32>,
    best_move: Option<M>,
    search_depth: u8,
    path_length: u8,
    descendants: u32,
    terminals: u32,
}

fn negamax_ab_parallel<G: Gamestate<M>, M: Move, E: Evaluate<G>>(
    arena: &mut Arena<G, M>,
    id: NodeId,
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
        let node = arena.get_mut(id);
        node.evaluate(evaluator, aim.into());
        if node.gamestate.is_terminal() {
            node.path_length = 0;
            SearchExit::Terminal
        } else {
            node.path_length = 1;
            SearchExit::Depth
        }
    } else if arena.get_mut(id).get_moves() == 0 {
        // Game end condition, evaluate the gamestate
        let node = arena.get_mut(id);
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
        arena.get_mut(id).reset_stats();
        // Go through each move and child in parallel using rayon
        create_all_children(arena, id);

        // Each task searches its child in an arena of its own.
        //
        // Disjoint `&mut` into one arena is not something the borrow checker
        // will hand out, and a shared allocator would need a lock on the
        // hottest operation in the search. The cost is that the subtrees the
        // tasks build are dropped rather than kept for the next deepening
        // pass, so parallel iterative deepening re-searches below the root.
        // The root's children keep their values, so root ordering survives.
        // The sequential search, which is what the deep work actually uses,
        // keeps its whole tree.
        let alpha = AtomicF32::new(alpha);
        let tasks: Vec<(usize, G, NegamaxAim, E)> = arena
            .get(id)
            .children
            .iter()
            .enumerate()
            .map(|(i, (_, child))| {
                let child = arena.get(*child);
                (
                    i,
                    child.gamestate.clone(),
                    NegamaxAim::from(child.gamestate.player_aim()),
                    evaluator.clone(),
                )
            })
            .collect();

        let mut results = Vec::new();
        tasks
            .into_par_iter()
            .map(|(index, gamestate, child_aim, mut evaluator)| {
                // Only flip the window when the turn actually passes over.
                let a_now = alpha.load(Ordering::Relaxed);
                let (c_aim, c_alpha, c_beta) = if child_aim == aim {
                    (aim, a_now, beta)
                } else {
                    (-aim, -beta, -a_now)
                };
                let (mut sub, sub_root) = Arena::with_root(Node::new(gamestate));
                let exit = negamax_ab(
                    &mut sub,
                    sub_root,
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
                let child = sub.get(sub_root);
                let value = parent_view(child, aim);

                // Raise alpha atomically; a load/store pair loses concurrent
                // updates and silently weakens pruning.
                alpha.fetch_max(value, Ordering::AcqRel);

                // Return result
                let result = ParallelResult {
                    index,
                    exit,
                    best: value,
                    value: child.value,
                    best_move: child.best.clone(),
                    search_depth: child.search_depth,
                    path_length: child.path_length,
                    descendants: child.descendants + 1,
                    terminals: child.terminals + 1,
                };
                debug!("Result: {:?}", result);
                result
            })
            .collect_into_vec(&mut results);

        // Copy each child's result back into the tree so the next deepening
        // pass can order by it.
        for r in &results {
            let child = arena.get(id).children[r.index].1;
            let child = arena.get_mut(child);
            child.value = r.value;
            child.best = r.best_move.clone();
            child.search_depth = r.search_depth;
            child.path_length = r.path_length;
            child.descendants = r.descendants.saturating_sub(1);
            child.terminals = r.terminals.saturating_sub(1);
        }

        // Update node stats
        let mut best = (f32::NEG_INFINITY, None);
        let mut descendants = 0;
        let mut terminals = 0;
        for r in &results {
            descendants += r.descendants;
            terminals += r.terminals;
            if r.best > best.0 {
                best = (r.best, Some(r.index));
            }
        }
        let best_move = best.1.map(|i| arena.get(id).children[i].0.clone());
        let node = arena.get_mut(id);
        node.descendants = descendants;
        node.terminals = terminals;
        node.best = Some(best_move.unwrap());
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

/// Re-pick the root move among equally good alternatives.
///
/// The search keeps the first move that beats the incumbent, so a
/// deterministic engine replays the same game forever. That is poor value as
/// a training opponent, hence the option to spread the choice over moves the
/// search cannot separate.
fn pick_root_move<G: Gamestate<M>, M: Move>(
    arena: &Arena<G, M>,
    id: NodeId,
    aim: NegamaxAim,
    opts: &SearchOptions,
) -> Option<(M, f32)> {
    use rand::Rng;
    let node = arena.get(id);
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
        .map(|(_, c)| arena.get(*c).search_depth)?;

    let mut rng = rand::thread_rng();

    if opts.random_best {
        let tied: Vec<&M> = node
            .children
            .iter()
            .filter(|(_, c)| {
                let c = arena.get(*c);
                c.search_depth == best_depth && parent_view(c, aim) == best_value
            })
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
            .filter(|(_, c)| arena.get(*c).search_depth == best_depth)
            .map(|(m, c)| {
                let v = parent_view(arena.get(*c), aim);
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
    arena: &mut Arena<G, M>,
    id: NodeId,
    aim: NegamaxAim,
    prune_by_path_length: bool,
) {
    // Taken out so the comparator can read the children's nodes out of the
    // same arena the list lives in.
    let mut children = std::mem::take(&mut arena.get_mut(id).children);
    children.sort_by(|(_, a), (_, b)| {
        let a = arena.get(*a);
        let b = arena.get(*b);
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
    arena.get_mut(id).children = children;
}

/// The child of `id` at position `index`, creating it from the next unplayed
/// move if it does not exist yet.
///
/// This is the index-based replacement for the old `ChildrenIter`: children
/// are still generated lazily, so a node that alpha-beta cuts off early never
/// pays for the moves it did not look at. A move popped here becomes
/// `children[index]`, which is why `index` can be used to name the best move
/// after the loop rather than cloning one per iteration.
fn child_at<G: Gamestate<M>, M: Move>(
    arena: &mut Arena<G, M>,
    id: NodeId,
    index: usize,
) -> Option<NodeId> {
    {
        let node = arena.get(id);
        if index < node.children.len() {
            return Some(node.children[index].1);
        }
    }
    let m = arena.get_mut(id).moves.pop()?;
    let child = arena.get(id).play_move(&m);
    let child = arena.alloc(child);
    arena.get_mut(id).children.push((m, child));
    Some(child)
}

/// Collapse the subtree at `id` to the principal variation plus one level of
/// siblings, first writing the ordering it is about to lose into the move
/// lists.
///
/// Ported from `removeNonBestNodes` in the TypeScript original. `keep` follows
/// the principal variation: the node it is true at keeps all of its children,
/// and passes it on only to the best of them. Everywhere else the node is
/// reduced to its best child, and the moves of the children being discarded
/// are appended to its move list in order so that the next pass regenerates
/// them best-first.
///
/// This does not free anything by itself -- it decides what the following
/// migration will keep.
fn prune_to_best<G: Gamestate<M>, M: Move>(
    arena: &mut Arena<G, M>,
    id: NodeId,
    keep: bool,
    prune_by_path_length: bool,
) {
    if arena.get(id).children.is_empty() {
        return;
    }
    // A node's aim is always that of the player to move in it: the search
    // flips the window exactly when the turn passes over, so games with
    // repeat turns stay consistent here without being special-cased.
    let aim = NegamaxAim::from(arena.get(id).gamestate.player_aim());
    order_children(arena, id, aim, prune_by_path_length);

    if keep {
        let count = arena.get(id).children.len();
        for i in 0..count {
            let child = arena.get(id).children[i].1;
            prune_to_best(arena, child, i == 0, prune_by_path_length);
        }
        return;
    }

    let children = std::mem::take(&mut arena.get_mut(id).children);
    let mut children = children.into_iter();
    let best = children.next().expect("children checked non-empty");
    // `moves` is popped from the back, so the discarded moves go on reversed:
    // the next pass then plays them in the order this pass ranked them, after
    // the best child which is already expanded. Any move never expanded at all
    // stays in front of them and so is tried last, as before.
    let mut discarded: Vec<M> = children.map(|(m, _)| m).collect();
    discarded.reverse();
    let node = arena.get_mut(id);
    node.moves.append(&mut discarded);
    debug_assert_ne!(
        node.moves.capacity(),
        0,
        "an emptied move list with no capacity would be regenerated, \
         duplicating the child that was kept"
    );
    node.children.push(best.clone());
    prune_to_best(arena, best.1, false, prune_by_path_length);
}

/// Negamax search, no pruning or timeout
pub fn negamax<G: Gamestate<M>, M: Move, E: Evaluate<G>>(
    arena: &mut Arena<G, M>,
    id: NodeId,
    evaluator: &mut E,
    depth: u8,
    aim: NegamaxAim,
) -> SearchExit {
    if depth == 0 {
        // End of recursion. Checked before move generation: leaves are the bulk
        // of the tree and generating their moves only to discard them dominates
        // the search when the move generator allocates.
        let node = arena.get_mut(id);
        node.evaluate(evaluator, aim.into());
        if node.gamestate.is_terminal() {
            node.path_length = 0;
            SearchExit::Terminal
        } else {
            node.path_length = 1;
            SearchExit::Depth
        }
    } else if arena.get_mut(id).get_moves() == 0 {
        // Game end condition, evaluate the gamestate
        let node = arena.get_mut(id);
        node.evaluate(evaluator, aim.into());
        node.path_length = 0;
        SearchExit::Terminal
    } else {
        // continue recursion
        arena.get_mut(id).reset_stats();
        // track best value and the index of the child it came from
        let mut best = (f32::NEG_INFINITY, None);
        // Assume exhaustive first, any time or depth will override
        let mut exit = SearchExit::Exhaustive;
        // Go through each move and child (children are created on the fly)
        let mut descendants = 0;
        let mut terminals = 0;
        let mut index = 0;
        while let Some(child) = child_at(arena, id, index) {
            // Clone gamestate to make child
            // Recurse with child node and remember best
            let child_aim = NegamaxAim::from(arena.get(child).gamestate.player_aim());
            match negamax(arena, child, evaluator, depth - 1, child_aim) {
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
            let child = arena.get(child);
            let value = parent_view(child, aim);
            if value > best.0 {
                best = (value, Some(index));
            }
            // Update parent node details
            descendants += child.descendants + 1;
            terminals += child.terminals;
            index += 1;
        }
        let best_move = best.1.map(|i| arena.get(id).children[i].0.clone());
        let node = arena.get_mut(id);
        node.descendants = descendants;
        node.terminals = terminals;
        node.best = Some(best_move.unwrap());
        node.value = Some(best.0);
        node.search_depth = depth;
        exit
    }
}

/// Negamax search, with alpha-beta pruning
#[allow(clippy::too_many_arguments)]
pub fn negamax_ab<G: Gamestate<M>, M: Move, E: Evaluate<G>>(
    arena: &mut Arena<G, M>,
    id: NodeId,
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
        let node = arena.get_mut(id);
        node.evaluate(evaluator, aim.into());
        if node.gamestate.is_terminal() {
            node.path_length = 0;
            SearchExit::Terminal
        } else {
            node.path_length = 1;
            SearchExit::Depth
        }
    } else if arena.get_mut(id).get_moves() == 0 {
        // Game end condition, evaluate the gamestate
        let node = arena.get_mut(id);
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
        let key = if tt.enabled() {
            arena.get(id).gamestate.position_key()
        } else {
            0
        };
        let alpha_orig = alpha;
        let mut tt_best: Option<M> = None;
        if key != 0 {
            match tt.lookup(key, depth, alpha, beta) {
                Probe::Cutoff(value, m) => {
                    let node = arena.get_mut(id);
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
        arena.get_mut(id).reset_stats();
        // A move that was best for this position elsewhere in the tree is the
        // best guess here too. Moves are popped from the back, so moving it
        // last gets it searched first.
        if let Some(m) = &tt_best {
            let node = arena.get_mut(id);
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
        if opts.sort_on_create
            && arena.get(id).children.is_empty()
            && depth >= opts.sort_on_create_min_depth
        {
            // First visit: nothing has an inherited value yet, so fall back to
            // each child's own static evaluation rather than move order.
            create_all_children(arena, id);
            let count = arena.get(id).children.len();
            for i in 0..count {
                let child = arena.get(id).children[i].1;
                let child_aim = NegamaxAim::from(arena.get(child).gamestate.player_aim());
                arena.get_mut(child).evaluate(evaluator, child_aim.into());
            }
            order_children(arena, id, aim, opts.prune_by_path_length);
        } else if opts.pre_sort && !arena.get(id).children.is_empty() {
            order_children(arena, id, aim, opts.prune_by_path_length);
        }
        // track best value and the index of the child it came from
        let mut best = (f32::NEG_INFINITY, None);
        let mut best_path: u8 = 0;
        // Assume exhaustive first, any time or depth will override
        let mut exit = SearchExit::Exhaustive;

        let mut descendants = 0;
        let mut terminals = 0;
        let mut index = 0;
        // Go through each move and child
        while let Some(child) = child_at(arena, id, index) {
            trace!("Exploring child {index} at depth {depth}");
            // Recurse with child node and remember best
            // Only flip the window when the turn actually passes over.
            let child_aim = NegamaxAim::from(arena.get(child).gamestate.player_aim());
            let (c_aim, c_alpha, c_beta) = if child_aim == aim {
                (aim, alpha, beta)
            } else {
                (-aim, -beta, -alpha)
            };
            match negamax_ab(
                arena,
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
            let child = arena.get(child);
            let value = parent_view(child, aim);
            if value > best.0 {
                best = (value, Some(index));
                best_path = child.path_length;
            } else if opts.prune_by_path_length && value == best.0 && best.1.is_some() {
                // Same value: take the quicker win, or the slower loss.
                let better = if best.0 > 0.0 {
                    child.path_length < best_path
                } else {
                    child.path_length > best_path
                };
                if better {
                    best = (value, Some(index));
                    best_path = child.path_length;
                }
            }
            // Update parent node details
            descendants += child.descendants + 1;
            terminals += child.terminals;
            index += 1;

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
        let best_move = best.1.map(|i| arena.get(id).children[i].0.clone());
        let node = arena.get_mut(id);
        node.descendants = descendants;
        node.terminals = terminals;
        node.best = Some(best_move.unwrap());
        node.value = Some(best.0);
        node.search_depth = depth;
        node.path_length = best_path.saturating_add(1);
        if key != 0 {
            let best_move = node.best.clone();
            tt.record(key, depth, best.0, alpha_orig, beta, best_move);
        }
        exit
    }
}

#[cfg(test)]
mod test {
    use std::result;

    use super::{negamax, Negamax, RemovalMethod};
    use crate::arena::Arena;
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
        let (mut arena, root) = Arena::with_root(Node::new(ttt));
        // Create evaluator
        let mut evaluator = crate::games::tictactoe::TttEvaluator;
        // Run search
        assert_eq!(
            negamax(&mut arena, root, &mut evaluator, 9, NegamaxAim::Maximise),
            crate::SearchExit::Exhaustive
        );
        assert_eq!(arena.get(root).best, Some(TttMove::from(8)));
        assert_eq!(arena.get(root).terminals, 255168);

        // player 2
        let ttt = Ttt::new(crate::games::tictactoe::Player::Two);
        // Create node
        let (mut arena, root) = Arena::with_root(Node::new(ttt));
        // Create evaluator
        let mut evaluator = crate::games::tictactoe::TttEvaluator;
        // Run search
        assert_eq!(
            negamax(&mut arena, root, &mut evaluator, 9, NegamaxAim::Minimise),
            crate::SearchExit::Exhaustive
        );
        assert_eq!(arena.get(root).best, Some(TttMove::from(8)));
        assert_eq!(arena.get(root).terminals, 255168);

        //  Repeated calls
        let ttt = Ttt::new(crate::games::tictactoe::Player::One);
        let (mut arena, root) = Arena::with_root(Node::new(ttt));
        assert_eq!(
            negamax(&mut arena, root, &mut evaluator, 2, NegamaxAim::Maximise),
            crate::SearchExit::Depth
        );
        assert_eq!(arena.get(root).best, Some(TttMove::from(8)));
        assert_eq!(
            negamax(&mut arena, root, &mut evaluator, 6, NegamaxAim::Maximise),
            crate::SearchExit::Depth
        );
        assert_eq!(arena.get(root).best, Some(TttMove::from(8)));
        assert_eq!(
            negamax(&mut arena, root, &mut evaluator, 9, NegamaxAim::Maximise),
            crate::SearchExit::Exhaustive
        );
        dbg!(arena.get(root).children.len());
        dbg!(arena.get(root).descendants);
        assert_eq!(arena.get(root).best, Some(TttMove::from(8)));
        assert_eq!(arena.get(root).terminals, 255168);
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
        let (mut arena, root) = Arena::with_root(Node::new(ttt));
        // Create evaluator
        let mut evaluator = crate::games::tictactoe::TttEvaluator;
        // Run search
        assert_eq!(
            negamax_ab(
                &mut arena,
                root,
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
        assert_eq!(arena.get(root).best, Some(TttMove::from(8)));
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
        let ttt = Ttt::new(crate::games::tictactoe::Player::One);
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
        assert_eq!(n.start_depth(), (n.root().search_depth + 1).min(4));
        assert!(n.start_depth() > 1);
    }

    /// Removal must not change what the search returns.
    ///
    /// At a fixed depth alpha-beta from a full window gives the exact minimax
    /// value whatever order the moves were tried in, so discarding nodes --
    /// which only changes the order the next pass regenerates them in -- has
    /// to leave the value alone. Mancala is the interesting case: it grants
    /// repeat turns, so the player to move does not alternate and removal has
    /// to take each node's aim from its own gamestate.
    #[test]
    fn removal_does_not_change_the_value() {
        use crate::games::mancala::{Mancala, MancalaEvaluator};

        for depth in 1..=9u8 {
            let base = SearchOptions {
                alpha_beta: true,
                iterative: true,
                pre_sort: true,
                max_depth: Some(depth),
                ..Default::default()
            };
            let kept = Negamax::new(Node::new(Mancala::new()), MancalaEvaluator, base)
                .search();
            let removed = Negamax::new(
                Node::new(Mancala::new()),
                MancalaEvaluator,
                SearchOptions { removal: RemovalMethod::Always, ..base },
            )
            .search();
            assert_eq!(
                kept.value, removed.value,
                "removal changed the value at depth {depth}"
            );
            assert_eq!(kept.exit, removed.exit, "removal changed the exit at depth {depth}");
        }

        for depth in 1..=7u8 {
            let base = SearchOptions {
                alpha_beta: true,
                iterative: true,
                pre_sort: true,
                max_depth: Some(depth),
                ..Default::default()
            };
            let kept = Negamax::new(Node::new(Ttt::default()), TttEvaluator, base).search();
            let removed = Negamax::new(
                Node::new(Ttt::default()),
                TttEvaluator,
                SearchOptions { removal: RemovalMethod::Always, ..base },
            )
            .search();
            assert_eq!(kept.value, removed.value, "removal changed the value at depth {depth}");
        }
    }

    /// The point of removal: the tree left standing between passes is a
    /// fraction of the one that was searched.
    #[test]
    fn removal_shrinks_the_tree() {
        use crate::games::mancala::{Mancala, MancalaEvaluator};

        let base = SearchOptions {
            alpha_beta: true,
            iterative: true,
            pre_sort: true,
            max_depth: Some(9),
            ..Default::default()
        };
        let mut n = Negamax::new(Node::new(Mancala::new()), MancalaEvaluator, base);
        n.search();
        let searched = n.tree_size();
        n.remove_nodes();
        let kept = n.tree_size();
        assert!(
            kept * 4 < searched,
            "removal kept {kept} of {searched} nodes, which is not a saving"
        );
        assert!(kept > 1, "removal kept only the root, so the ordering is gone too");
    }

    /// Ordering has to survive the nodes it was derived from, which is what
    /// writing the sorted moves back into the move list is for.
    ///
    /// It only survives at the nodes removal keeps, though. A discarded
    /// subtree is regenerated with nothing but move-generation order to go
    /// on, so removal lands between an ordered search and an unordered one
    /// rather than matching the ordered one. Kalah to depth 11 from the
    /// opening: ordered 23707 nodes, removal 54094, unordered 79709. That
    /// regeneration cost is the price of the memory and it is not small --
    /// see `examples/removal_probe.rs`.
    #[test]
    fn removal_keeps_the_move_ordering() {
        use crate::games::mancala::{Mancala, MancalaEvaluator};

        let opts = SearchOptions {
            alpha_beta: true,
            iterative: true,
            max_depth: Some(11),
            ..Default::default()
        };
        let ordered = Negamax::new(
            Node::new(Mancala::new()),
            MancalaEvaluator,
            SearchOptions { pre_sort: true, ..opts },
        )
        .search();
        let unordered = Negamax::new(Node::new(Mancala::new()), MancalaEvaluator, opts).search();
        let removed = Negamax::new(
            Node::new(Mancala::new()),
            MancalaEvaluator,
            SearchOptions { pre_sort: true, removal: RemovalMethod::Always, ..opts },
        )
        .search();

        assert_eq!(removed.value, ordered.value);
        assert!(
            removed.nodes < unordered.nodes,
            "removal searched {} nodes against an unordered {} -- the move \
             order written back before the nodes were freed is not being used",
            removed.nodes,
            unordered.nodes
        );
    }

    /// Handing moves back to a node that already has a child is the part of
    /// removal that can go quietly wrong: if the move list is regenerated
    /// rather than appended to, the kept child gets expanded a second time and
    /// the tree silently doubles up. Walk the tree and check no node lists the
    /// same move twice.
    #[test]
    fn removal_does_not_duplicate_children() {
        use crate::games::mancala::{Mancala, MancalaEvaluator};

        let mut n = Negamax::new(
            Node::new(Mancala::new()),
            MancalaEvaluator,
            SearchOptions {
                alpha_beta: true,
                iterative: true,
                pre_sort: true,
                max_depth: Some(9),
                removal: RemovalMethod::Always,
                ..Default::default()
            },
        );
        n.search();

        let mut stack = vec![n.root_id()];
        let mut checked = 0;
        while let Some(id) = stack.pop() {
            let node = n.arena().get(id);
            for i in 0..node.children.len() {
                for j in (i + 1)..node.children.len() {
                    assert_ne!(
                        node.children[i].0, node.children[j].0,
                        "move expanded twice under one node"
                    );
                }
                assert!(
                    !node.moves.contains(&node.children[i].0),
                    "a move that was already expanded is queued to be played again"
                );
                stack.push(node.children[i].1);
            }
            checked += 1;
        }
        assert!(checked > 1);
    }
}
