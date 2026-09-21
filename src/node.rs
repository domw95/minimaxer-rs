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
    // Plies from this node to the nearest leaf along the best line.
    // 0 at a terminal, 1 at a depth-limited node.
    pub(crate) path_length: u8,
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
            path_length: 0,
        }
    }

    pub(crate) fn reset_stats(&mut self) {
        self.descendants = 0;
        self.terminals = 0;
    }

    pub fn sort_children_descending(&mut self) {
        self.children.sort_by(|(_, a), (_, b)| {
            a.value
                .partial_cmp(&b.value)
                .unwrap_or(std::cmp::Ordering::Equal)
                .reverse()
        });
    }

    pub fn sort_children_ascending(&mut self) {
        self.children.sort_by(|(_, a), (_, b)| {
            a.value
                .partial_cmp(&b.value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}

impl<G: Gamestate<M>, M: Move> Node<G, M> {
    /// Get the moves for the current player
    /// and store them in the moves field
    /// Doesnt get them twice
    pub(crate) fn get_moves(&mut self) -> usize {
        if self.moves.capacity() == 0 {
            self.moves = self.gamestate.get_moves();
        }
        self.moves.len() + self.children.len()
    }

    /// Play a move, re-rooting this node onto the subtree for that move and
    /// dropping the rest of the tree.
    ///
    /// Returns whether an existing subtree was found and kept. A search that
    /// ran out of time can stop before expanding every root move, and
    /// analysing a recorded game plays the move from the record rather than
    /// the one the search chose, so the move asked for may never have been
    /// expanded. In that case the node is rebuilt from the move instead, with
    /// no children and `search_depth` back at 0, which is the same position
    /// the caller would have got by constructing a fresh node.
    pub fn advance(&mut self, m: &M) -> bool {
        match self.children.iter().position(|(mov, _)| mov == m) {
            Some(i) => {
                let mut child = self.children.swap_remove(i).1;
                mem::swap(self, &mut child);
                true
            }
            None => {
                let mut child = self.play_move(m);
                mem::swap(self, &mut child);
                false
            }
        }
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

    pub fn create_all_children(&mut self) {
        while let Some(m) = self.moves.pop() {
            let mut gs = self.gamestate.clone();
            gs.play_move(&m);
            self.children.push((m, Node::new(gs)));
        }
    }
}

impl<'a, G: Gamestate<M>, M: Move> IntoIterator for &'a mut Node<G, M> {
    type Item = &'a mut (M, Node<G, M>);
    type IntoIter = ChildrenIter<'a, G, M>;

    fn into_iter(self) -> Self::IntoIter {
        let mut_self = unsafe { &mut *(self as *mut Node<G, M>) };

        ChildrenIter {
            node: mut_self,
            child_pos: 0,
        }
    }
}

/// Iterator over the children of a node
/// Generates children from moves if required
pub struct ChildrenIter<'a, G, M: Move> {
    node: &'a mut Node<G, M>,
    child_pos: usize,
}

impl<'a, G: Gamestate<M>, M: Move> Iterator for ChildrenIter<'a, G, M> {
    type Item = &'a mut (M, Node<G, M>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.child_pos < self.node.children.len() {
            // Get a mut ref to child at child_pos
            let child =
                unsafe { &mut *(&mut self.node.children[self.child_pos] as *mut (M, Node<G, M>)) };
            self.child_pos += 1;
            Some(child)
        } else if let Some(m) = self.node.moves.pop() {
            let child = self.node.play_move(&m);
            self.node.children.push((m, child));
            self.child_pos += 1;
            unsafe {
                let last = self.node.children.last_mut().unwrap() as *mut (M, Node<G, M>);
                Some(&mut *last)
            }
        } else {
            None
        }
    }
}
