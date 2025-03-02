#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Mancala {
    cells: [u8; 12],
    pots: [u8; 2],
    activeplayer: Player,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Player {
    One,
    Two,
}

impl Mancala {
    pub fn new() -> Mancala {
        Mancala {
            cells: [4; 12],
            pots: [0; 2],
            activeplayer: Player::One,
        }
    }

    pub fn moves(&self) -> Vec<u8> {
        match self.activeplayer {
            Player::One => self.cells[0..6]
                .iter()
                .enumerate()
                .filter_map(|(i, count)| if *count > 0 { Some(i as u8) } else { None })
                .collect(),
            Player::Two => self.cells[6..]
                .iter()
                .enumerate()
                .filter_map(|(i, count)| if *count > 0 { Some(i as u8) } else { None })
                .collect(),
        }
    }

    pub fn play_move(&mut self) {
        match self.activeplayer {
            Player::One => todo!(),
            Player::Two => todo!(),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::games::mancala::Mancala;

    #[test]
    fn test() {
        let m = Mancala::new();
        dbg!(m.moves());
    }
}
