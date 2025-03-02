use std::{
    fmt::Debug,
    ops::{Mul, MulAssign},
    time::Duration,
};
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
    best: M,
    value: f32,
    exit: SearchExit,
    nodes: u32,
    terminals: u32,
    time: Duration,
    depth: u8,
}

pub enum NodeAim {
    Minimise,
    Maximise,
}

pub trait Move: Clone + PartialEq {}

pub trait Gamestate<M>: Clone {
    /// Get the moves for the current player
    fn get_moves(&mut self) -> Vec<M>;
    /// Make a move
    fn play_move(&mut self, m: &M);
    /// Get the aim of the current player
    fn player_aim(&self) -> NodeAim;
}

pub trait Evaluate<G> {
    /// Evaluate the gamestate from perspective of first player
    fn evaluate(&mut self, g: &G) -> f32;
}
