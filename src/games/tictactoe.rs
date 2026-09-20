use crate::{Evaluate, Gamestate, Move};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
enum Cell {
    #[default]
    Empty,
    Cross,
    Nought,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Player {
    #[default]
    One,
    Two,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Ttt {
    grid: [Cell; 9],
    activeplayer: Player,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TttMove(pub u8);

impl Move for TttMove {}

impl From<TttMove> for u8 {
    fn from(m: TttMove) -> u8 {
        m.0
    }
}

impl From<TttMove> for usize {
    fn from(m: TttMove) -> usize {
        m.0 as usize
    }
}

impl From<u8> for TttMove {
    fn from(m: u8) -> TttMove {
        TttMove(m)
    }
}

impl Ttt {
    pub fn new(player: Player) -> Self {
        Ttt {
            activeplayer: player,
            ..Default::default()
        }
    }

    pub fn moves(&self) -> Vec<TttMove> {
        self.grid
            .iter()
            .enumerate()
            .filter_map(|(i, cell)| {
                if matches!(cell, Cell::Empty) {
                    Some(TttMove(i as u8))
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn moves_iter(&self) -> impl Iterator<Item = u8> + '_ {
        self.grid.iter().enumerate().filter_map(|(i, cell)| {
            if matches!(cell, Cell::Empty) {
                Some(i as u8)
            } else {
                None
            }
        })
    }

    pub fn play_move(&mut self, mov: TttMove) {
        match self.activeplayer {
            Player::One => {
                self.grid[usize::from(mov)] = Cell::Nought;
                self.activeplayer = Player::Two;
            }
            Player::Two => {
                self.grid[usize::from(mov)] = Cell::Cross;
                self.activeplayer = Player::One;
            }
        }
    }

    fn check_win(&self) -> Option<Player> {
        let win = |a, b, c| {
            matches!(
                (self.grid[a], self.grid[b], self.grid[c]),
                (Cell::Cross, Cell::Cross, Cell::Cross)
            ) || matches!(
                (self.grid[a], self.grid[b], self.grid[c]),
                (Cell::Nought, Cell::Nought, Cell::Nought)
            )
        };

        if win(0, 1, 2)
            || win(3, 4, 5)
            || win(6, 7, 8)
            || win(0, 3, 6)
            || win(1, 4, 7)
            || win(2, 5, 8)
            || win(0, 4, 8)
            || win(2, 4, 6)
        {
            Some(match self.activeplayer {
                Player::One => Player::Two,
                Player::Two => Player::One,
            })
        } else {
            None
        }
    }
}

impl Gamestate<TttMove> for Ttt {
    fn get_moves(&mut self) -> Vec<TttMove> {
        // check if game has ended first
        if self.check_win().is_some() {
            vec![]
        } else {
            self.moves()
        }
    }

    fn play_move(&mut self, m: &TttMove) {
        self.play_move(*m);
    }

    fn is_terminal(&mut self) -> bool {
        self.check_win().is_some() || self.grid.iter().all(|c| !matches!(c, Cell::Empty))
    }

    fn player_aim(&self) -> crate::NodeAim {
        match self.activeplayer {
            Player::One => crate::NodeAim::Maximise,
            Player::Two => crate::NodeAim::Minimise,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TttEvaluator;

impl Evaluate<Ttt> for TttEvaluator {
    fn evaluate(&mut self, g: &Ttt) -> f32 {
        if let Some(player) = g.check_win() {
            match player {
                Player::One => 1.0,
                Player::Two => -1.0,
            }
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod test {
    use super::{Player, Ttt};

    #[test]
    fn test() {
        let mut ttt = Ttt::new(Player::One);
        ttt.play_move(4.into());
    }
}
