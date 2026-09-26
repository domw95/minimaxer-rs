use crate::arena::NodeId;
use crate::{Gamestate, Move};

/// A position in the search tree.
///
/// Children are [`NodeId`]s into the [`Arena`](crate::arena::Arena) that owns
/// this node rather than nodes inline, so a node is a fixed size whatever its
/// branching factor and the tree can be re-rooted by moving the nodes that are
/// kept instead of freeing the ones that are not.
#[derive(Debug, Clone)]
pub struct Node<G, M> {
    // moves still available from this node.
    // Once a move is played, it is removed from this list
    // and a child node is created with the new gamestate
    pub(crate) moves: Vec<M>,
    // Move / child id pairs
    pub(crate) children: Vec<(M, NodeId)>,
    // The gamestate at this node
    pub(crate) gamestate: G,
    // The value of the gamestate at this node
    pub(crate) value: Option<f32>,
    // Best move and value for this nodes descendants
    pub(crate) best: Option<M>,
    // Count of descendants
    pub(crate) descendants: u32,
    // Count of terminal nodes
    pub(crate) terminals: u32,
    // Depth searched to from this node
    pub(crate) search_depth: u8,
    // Plies from this node to the nearest leaf along the best line.
    // 0 at a terminal, 1 at a depth-limited node.
    pub(crate) path_length: u8,
}

impl<G, M> Node<G, M> {
    /// The best move the deepest completed search found from this node.
    ///
    /// Re-rooting keeps it, so a host that has just played a move can answer
    /// with the move the previous search already settled on rather than
    /// waiting for a fresh pass to finish. That is what makes a bounded
    /// response time possible without interrupting the search.
    pub fn best_move(&self) -> Option<&M> {
        self.best.as_ref()
    }

    /// Plies this node has been searched to.
    pub fn searched_depth(&self) -> u8 {
        self.search_depth
    }


    pub fn new(gamestate: G) -> Self {
        Node {
            moves: Vec::new(),
            children: Vec::new(),
            gamestate,
            value: None,
            best: None,
            descendants: 0,
            terminals: 0,
            search_depth: 0,
            path_length: 0,
        }
    }

    /// The move the search settled on from this position.
    pub fn best(&self) -> Option<&M> {
        self.best.as_ref()
    }

    /// The value of this position from the point of view of the player to
    /// move here.
    pub fn value(&self) -> Option<f32> {
        self.value
    }

    /// Nodes below this one in the last search.
    pub fn descendants(&self) -> u32 {
        self.descendants
    }

    /// Terminal positions below this one in the last search.
    pub fn terminals(&self) -> u32 {
        self.terminals
    }

    /// Depth this node was last searched to.
    pub fn search_depth(&self) -> u8 {
        self.search_depth
    }

    /// Children expanded so far. Nodes are created lazily, so this is not the
    /// number of legal moves.
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    pub fn gamestate(&self) -> &G {
        &self.gamestate
    }

    pub(crate) fn reset_stats(&mut self) {
        self.descendants = 0;
        self.terminals = 0;
    }
}

impl<G: Gamestate<M>, M: Move> Node<G, M> {
    /// Get the moves for the current player
    /// and store them in the moves field
    /// Doesnt get them twice
    ///
    /// The `capacity` test, not `len`, is what distinguishes "never generated"
    /// from "generated and all played". Node removal relies on that: it hands
    /// moves back to a node that already has children, and regenerating them
    /// there would expand every move twice.
    pub(crate) fn get_moves(&mut self) -> usize {
        if self.moves.capacity() == 0 {
            self.moves = self.gamestate.get_moves();
        }
        self.moves.len() + self.children.len()
    }

    /// Play a move, cloning the node.
    /// Returns the child node resulting from the move
    pub(crate) fn play_move(&self, m: &M) -> Node<G, M> {
        let mut gs = self.gamestate.clone();
        gs.play_move(m);
        Node::new(gs)
    }

    /// Evaluate the node and assign to value
    pub(crate) fn evaluate<E: crate::Evaluate<G>>(&mut self, evaluator: &mut E, multiplier: f32) {
        self.value = Some(evaluator.evaluate(&self.gamestate) * multiplier);
    }
}
