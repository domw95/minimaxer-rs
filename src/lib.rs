use std::{fmt::Debug, time::Duration};
pub mod games;
pub mod negamax;
pub mod node;

/// Reason for finishing search
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchExit {
    /// Search finished due to depth limit
    Depth,
    /// Search finished due to time limit
    Time,
    /// Search finished due to exploring full tree
    Exhaustive,
    /// No more moves beyond this node
    Terminal,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchResult<M> {
    pub best: M,
    pub value: f32,
    pub exit: SearchExit,
    pub nodes: u32,
    pub terminals: u32,
    pub time: Duration,
    pub depth: u8,
}

pub enum NodeAim {
    Minimise,
    Maximise,
}

pub trait Move: Clone + PartialEq + Send + Sync + Debug {}

pub trait Gamestate<M>: Clone + Send + Sync {
    /// Get the moves for the current player
    fn get_moves(&mut self) -> Vec<M>;
    /// Make a move
    fn play_move(&mut self, m: &M);
    /// Get the aim of the current player
    fn player_aim(&self) -> NodeAim;
    /// Cheap check for whether the game has ended here.
    ///
    /// Called at every leaf, which is the bulk of the search tree. The default
    /// generates the move list and throws it away; override it with a direct
    /// test to avoid that work. For a game whose move generation allocates,
    /// doing so is worth a large constant factor on the whole search.
    fn is_terminal(&mut self) -> bool {
        self.get_moves().is_empty()
    }
}

pub trait Evaluate<G>: Clone + Send + Sync {
    /// Evaluate the gamestate from perspective of first player
    fn evaluate(&mut self, g: &G) -> f32;
}
