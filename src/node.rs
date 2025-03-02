use std::mem;

use crate::{Gamestate, Move};

#[derive(Debug, Clone)]
pub struct Node<G, M> {
    // moves still available from this node.
    // Once a move is played, it is removed from this list
    // and a child node is created with the new gamestate
    pub(crate) moves: Vec<M>,
    // Move / node pairs
    pub(crate) children: Vec<(M, Node<G, M>)>,
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
}

impl<G, M> Node<G, M> {
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
        }
    }

    pub(crate) fn reset_stats(&mut self) {
        self.descendants = 0;
        self.terminals = 0;
    }
}

impl<G: Gamestate<M>, M: Move> Node<G, M> {
    /// Get the moves for the current player
    /// and store them in the moves field
    pub(crate) fn get_moves(&mut self) -> usize {
        self.moves = self.gamestate.get_moves();
        self.moves.len()
    }

    /// Play a move, consuming the node.
    /// Returns the child node resulting from the move
    /// with the rest of the tree in place
    pub fn advance(&mut self, m: &M) {
        let mut child = self.children.drain(..).find(|(mov, _)| mov == m).unwrap().1;
        mem::swap(self, &mut child);
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
