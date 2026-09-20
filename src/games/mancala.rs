//! Kalah(6,4) - the standard Mancala variant.
//!
//! Board is a 14 slot ring: 0-5 are player one's pits, 6 is player one's
//! store, 7-12 are player two's pits, 13 is player two's store. Sowing runs
//! in increasing index order, skipping the opponent's store.

use crate::{Gamestate, Move, NodeAim};

pub const P1_STORE: usize = 6;
pub const P2_STORE: usize = 13;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Mancala {
    pub board: [u8; 14],
    pub activeplayer: Player,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Player {
    One,
    Two,
}

impl Player {
    fn store(self) -> usize {
        match self {
            Player::One => P1_STORE,
            Player::Two => P2_STORE,
        }
    }
    /// The store the player must skip when sowing.
    fn opponent_store(self) -> usize {
        match self {
            Player::One => P2_STORE,
            Player::Two => P1_STORE,
        }
    }
    fn pits(self) -> std::ops::RangeInclusive<usize> {
        match self {
            Player::One => 0..=5,
            Player::Two => 7..=12,
        }
    }
    fn other(self) -> Player {
        match self {
            Player::One => Player::Two,
            Player::Two => Player::One,
        }
    }
}

/// A move is the index of the pit to sow from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MancalaMove(pub u8);

impl Move for MancalaMove {}

/// Pit directly across the board from `i`.
fn opposite(i: usize) -> usize {
    12 - i
}

impl Default for Mancala {
    fn default() -> Self {
        Self::new()
    }
}

impl Mancala {
    pub fn new() -> Mancala {
        Self::with_stones(4)
    }

    /// Kalah(6, n) - six pits a side with `n` stones in each.
    pub fn with_stones(n: u8) -> Mancala {
        let mut board = [n; 14];
        board[P1_STORE] = 0;
        board[P2_STORE] = 0;
        Mancala {
            board,
            activeplayer: Player::One,
        }
    }

    /// Pits belonging to `player` that have stones in them.
    pub fn moves(&self) -> Vec<MancalaMove> {
        self.activeplayer
            .pits()
            .filter(|&i| self.board[i] > 0)
            .map(|i| MancalaMove(i as u8))
            .collect()
    }

    /// True once either side has no stones left to sow.
    pub fn is_over(&self) -> bool {
        let p1_empty = (0..=5).all(|i| self.board[i] == 0);
        let p2_empty = (7..=12).all(|i| self.board[i] == 0);
        p1_empty || p2_empty
    }

    /// Sweep the remaining stones into their owner's store. Called once the
    /// game is over so the final scores are complete.
    fn collect_remaining(&mut self) {
        let p1: u8 = (0..=5).map(|i| self.board[i]).sum();
        let p2: u8 = (7..=12).map(|i| self.board[i]).sum();
        for i in 0..=5 {
            self.board[i] = 0;
        }
        for i in 7..=12 {
            self.board[i] = 0;
        }
        self.board[P1_STORE] += p1;
        self.board[P2_STORE] += p2;
    }

    /// Score from player one's point of view.
    pub fn score(&self) -> i16 {
        self.board[P1_STORE] as i16 - self.board[P2_STORE] as i16
    }

    pub fn play(&mut self, m: MancalaMove) {
        let start = m.0 as usize;
        debug_assert!(self.activeplayer.pits().contains(&start));
        debug_assert!(self.board[start] > 0);

        let player = self.activeplayer;
        let skip = player.opponent_store();

        // Pick up the pit and sow one stone at a time, never into the
        // opponent's store.
        let mut stones = self.board[start];
        self.board[start] = 0;
        let mut i = start;
        while stones > 0 {
            i = (i + 1) % 14;
            if i == skip {
                continue;
            }
            self.board[i] += 1;
            stones -= 1;
        }

        // Landing in your own store earns another turn.
        if i == player.store() {
            if self.is_over() {
                self.collect_remaining();
            }
            return;
        }

        // Landing in one of your own empty pits captures it and the pit
        // opposite, provided that one has stones.
        if player.pits().contains(&i) && self.board[i] == 1 {
            let opp = opposite(i);
            if self.board[opp] > 0 {
                self.board[player.store()] += self.board[opp] + 1;
                self.board[opp] = 0;
                self.board[i] = 0;
            }
        }

        self.activeplayer = player.other();

        if self.is_over() {
            self.collect_remaining();
        }
    }
}

impl Gamestate<MancalaMove> for Mancala {
    fn get_moves(&mut self) -> Vec<MancalaMove> {
        if self.is_over() {
            Vec::new()
        } else {
            self.moves()
        }
    }

    fn play_move(&mut self, m: &MancalaMove) {
        self.play(*m);
    }

    fn is_terminal(&mut self) -> bool {
        self.is_over()
    }

    fn player_aim(&self) -> NodeAim {
        match self.activeplayer {
            Player::One => NodeAim::Maximise,
            Player::Two => NodeAim::Minimise,
        }
    }
}

/// Final score differential, from player one's perspective.
#[derive(Debug, Clone, Copy)]
pub struct MancalaEvaluator;

impl crate::Evaluate<Mancala> for MancalaEvaluator {
    fn evaluate(&mut self, g: &Mancala) -> f32 {
        g.score() as f32
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn initial_position() {
        let m = Mancala::new();
        assert_eq!(m.board.iter().map(|&x| x as u32).sum::<u32>(), 48);
        assert_eq!(m.board[P1_STORE], 0);
        assert_eq!(m.board[P2_STORE], 0);
        assert_eq!(m.moves().len(), 6);
        assert_eq!(m.moves()[0], MancalaMove(0));
        assert_eq!(m.moves()[5], MancalaMove(5));
    }

    #[test]
    fn p2_moves_are_its_own_pits() {
        let mut m = Mancala::new();
        m.activeplayer = Player::Two;
        let mv = m.moves();
        assert_eq!(mv.first(), Some(&MancalaMove(7)));
        assert_eq!(mv.last(), Some(&MancalaMove(12)));
    }

    #[test]
    fn sowing_skips_opponent_store() {
        let mut m = Mancala::new();
        m.board = [0; 14];
        m.board[5] = 10; // enough to wrap past p2's store and back round
        m.board[2] = 1; // pre-fill the landing pit so no capture fires
        m.activeplayer = Player::One;
        m.play(MancalaMove(5));
        // 10 stones from pit 5: store(6), 7,8,9,10,11,12, skip 13, 0,1,2
        assert_eq!(m.board[P1_STORE], 1);
        assert_eq!(m.board[P2_STORE], 0, "must never sow into opponent store");
        assert_eq!(&m.board[7..=12], &[1, 1, 1, 1, 1, 1]);
        assert_eq!(&m.board[0..=2], &[1, 1, 2]);
    }

    #[test]
    fn last_stone_in_own_store_grants_another_turn() {
        let mut m = Mancala::new();
        m.board = [0; 14];
        m.board[5] = 1;
        m.board[0] = 1;
        m.board[7] = 1; // keep p2 alive, otherwise the game ends here
        m.activeplayer = Player::One;
        m.play(MancalaMove(5));
        assert_eq!(m.board[P1_STORE], 1);
        assert_eq!(m.activeplayer, Player::One, "extra turn");
    }

    #[test]
    fn capture_on_own_empty_pit() {
        let mut m = Mancala::new();
        m.board = [0; 14];
        m.board[0] = 1; // lands in pit 1, which is empty
        m.board[11] = 5; // opposite(1) == 11
        m.board[7] = 1; // keep p2 alive so the game is not over
        m.activeplayer = Player::One;
        m.play(MancalaMove(0));
        assert_eq!(m.board[P1_STORE], 6, "captured 5 plus the sown stone");
        assert_eq!(m.board[1], 0);
        assert_eq!(m.board[11], 0);
    }

    #[test]
    fn no_capture_when_opposite_pit_empty() {
        let mut m = Mancala::new();
        m.board = [0; 14];
        m.board[0] = 1;
        m.board[11] = 0; // opposite is empty
        m.board[7] = 1;
        m.activeplayer = Player::One;
        m.play(MancalaMove(0));
        assert_eq!(m.board[P1_STORE], 0, "no capture");
        assert_eq!(m.board[1], 1, "stone stays put");
    }

    #[test]
    fn game_end_sweeps_remaining_stones() {
        let mut m = Mancala::new();
        m.board = [0; 14];
        m.board[5] = 1; // p1's last stone, lands in p1 store -> p1 side empty
        m.board[7] = 3;
        m.board[8] = 2;
        m.activeplayer = Player::One;
        m.play(MancalaMove(5));
        assert!(m.is_over());
        assert_eq!(m.board[P1_STORE], 1);
        assert_eq!(m.board[P2_STORE], 5, "p2 sweeps its remaining stones");
        assert_eq!(m.board.iter().map(|&x| x as u32).sum::<u32>(), 6);
    }

    #[test]
    fn stones_are_conserved_over_a_random_playout() {
        let mut m = Mancala::new();
        let mut seed = 12345u64;
        for _ in 0..200 {
            if m.is_over() {
                break;
            }
            let mv = m.moves();
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let pick = mv[(seed >> 33) as usize % mv.len()];
            m.play(pick);
            assert_eq!(
                m.board.iter().map(|&x| x as u32).sum::<u32>(),
                48,
                "stones must be conserved"
            );
        }
    }
}
