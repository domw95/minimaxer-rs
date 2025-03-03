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

impl<'a, G: Gamestate<M>, M: Move> IntoIterator for &'a mut Node<G, M> {
    type Item = &'a mut (M, Node<G, M>);
    type IntoIter = ChildrenIter<'a, G, M>;

    fn into_iter(self) -> Self::IntoIter {
        let mut_self = unsafe { &mut *(self as *mut Node<G, M>) };
        let move_iter = self.moves.iter();
        let child_iter = self.children.iter_mut();
        ChildrenIter {
            node: mut_self,
            move_iter,
            child_iter,
        }
    }
}

/// Iterator over the children of a node
/// Generates children from moves if required
pub struct ChildrenIter<'a, G, M: Move> {
    node: &'a mut Node<G, M>,
    move_iter: std::slice::Iter<'a, M>,
    child_iter: std::slice::IterMut<'a, (M, Node<G, M>)>,
}

impl<'a, G: Gamestate<M>, M: Move> Iterator for ChildrenIter<'a, G, M> {
    type Item = &'a mut (M, Node<G, M>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(child) = self.child_iter.next() {
            Some(child)
        } else if let Some(m) = self.move_iter.next() {
            let child = self.node.play_move(m);
            self.node.children.push((m.clone(), child));
            unsafe {
                let last = self.node.children.last_mut().unwrap() as *mut (M, Node<G, M>);
                Some(&mut *last)
            }
        } else {
            // self.node.moves.clear();
            None
        }
    }
}

impl<G, M: Move> Drop for ChildrenIter<'_, G, M> {
    fn drop(&mut self) {
        self.node.moves.clear();

        for m in self.move_iter.by_ref() {
            self.node.moves.push(m.clone());
        }
    }
}
