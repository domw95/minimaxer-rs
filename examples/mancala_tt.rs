//! Decisive test: bounded memory AND good move ordering.
//!
//! `mancala_solve` gets its ordering from the retained tree, so ordering and
//! RAM are coupled. This decouples them: a fixed-size transposition table
//! supplies both cutoffs and a best-move hint, so memory is a constant the
//! caller picks rather than a function of nodes searched.

use minimaxer::games::mancala::{Mancala, MancalaMove, Player};
use std::time::Instant;

const EXACT: u8 = 0;
const LOWER: u8 = 1;
const UPPER: u8 = 2;

#[derive(Clone, Copy, Default)]
struct TtEntry {
    key: u64,
    value: i16,
    depth: u8,
    flag: u8,
    best: u8,
    has_best: bool,
}

struct Tt {
    entries: Vec<TtEntry>,
    mask: usize,
}

impl Tt {
    fn new(bits: usize) -> Self {
        let n = 1usize << bits;
        Tt { entries: vec![TtEntry::default(); n], mask: n - 1 }
    }
    fn probe(&self, key: u64) -> Option<&TtEntry> {
        let e = &self.entries[(key as usize) & self.mask];
        if e.key == key { Some(e) } else { None }
    }
    fn store(&mut self, key: u64, depth: u8, value: i16, flag: u8, best: Option<u8>) {
        let slot = (key as usize) & self.mask;
        let e = &mut self.entries[slot];
        // Depth-preferred replacement: a deeper result is worth more.
        if e.key != key || depth >= e.depth {
            *e = TtEntry {
                key, value, depth, flag,
                best: best.unwrap_or(0),
                has_best: best.is_some(),
            };
        }
    }
}

fn hash(g: &Mancala) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in g.board.iter() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h ^= if g.activeplayer == Player::One { 0x9E3779B97F4A7C15 } else { 0 };
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58476D1CE4E5B9);
    h ^= h >> 32;
    h
}

fn ab(g: &Mancala, depth: u8, mut alpha: i16, mut beta: i16, tt: &mut Tt, nodes: &mut u64) -> i16 {
    *nodes += 1;
    if depth == 0 || g.is_over() {
        return g.score();
    }
    let key = hash(g);
    let alpha_orig = alpha;
    let mut tt_best: Option<u8> = None;
    if let Some(e) = tt.probe(key) {
        if e.has_best {
            tt_best = Some(e.best);
        }
        if e.depth >= depth {
            match e.flag {
                EXACT => return e.value,
                LOWER => alpha = alpha.max(e.value),
                UPPER => beta = beta.min(e.value),
                _ => {}
            }
            if alpha >= beta {
                return e.value;
            }
        }
    }

    let mut moves = g.moves();
    if moves.is_empty() {
        return g.score();
    }
    // Try the table's best move first - this is the ordering that the
    // tree-retaining search gets from its retained children.
    if let Some(b) = tt_best {
        if let Some(pos) = moves.iter().position(|m| m.0 == b) {
            moves.swap(0, pos);
        }
    }

    let maximizing = g.activeplayer == Player::One;
    let mut best_val = if maximizing { i16::MIN } else { i16::MAX };
    let mut best_move: Option<u8> = None;

    for m in moves {
        let mut c = *g;
        c.play(m);
        let v = ab(&c, depth - 1, alpha, beta, tt, nodes);
        if maximizing {
            if v > best_val {
                best_val = v;
                best_move = Some(m.0);
            }
            alpha = alpha.max(best_val);
        } else {
            if v < best_val {
                best_val = v;
                best_move = Some(m.0);
            }
            beta = beta.min(best_val);
        }
        if alpha >= beta {
            break;
        }
    }

    let flag = if best_val <= alpha_orig { UPPER } else if best_val >= beta { LOWER } else { EXACT };
    tt.store(key, depth, best_val, flag, best_move);
    best_val
}

fn rss_mib() -> f64 {
    let s = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
    let pages: u64 = s.split_whitespace().nth(1).and_then(|v| v.parse().ok()).unwrap_or(0);
    (pages * 4096) as f64 / (1024.0 * 1024.0)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let max_ply: u8 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(60);
    let secs: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(60.0);
    let stones: u8 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);
    let bits: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(23);

    let g = Mancala::with_stones(stones);
    let mut tt = Tt::new(bits);
    println!(
        "Kalah(6,{stones}) bounded memory + TT ordering. table=2^{bits} entries ({:.0}MiB)",
        (tt.entries.len() * std::mem::size_of::<TtEntry>()) as f64 / 1048576.0
    );
    println!("{:>4} {:>16} {:>12} {:>10} {:>8} {:>6}", "ply", "nodes", "ms", "RSS_MiB", "value", "best");
    let mut total = 0.0;
    for ply in 1..=max_ply {
        let mut nodes = 0u64;
        let t = Instant::now();
        let v = ab(&g, ply, i16::MIN + 1, i16::MAX - 1, &mut tt, &mut nodes);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        total += ms / 1000.0;
        let best = tt.probe(hash(&g)).and_then(|e| e.has_best.then_some(e.best));
        println!("{:>4} {:>16} {:>12.1} {:>10.1} {:>8} {:>6?}", ply, nodes, ms, rss_mib(), v, best);
        if g.is_over() {
            break;
        }
        if total > secs {
            println!("\nSTOPPED after {total:.1}s at ply {ply}. RSS {:.1}MiB (flat).", rss_mib());
            return;
        }
    }
    println!("\nCompleted max_ply.");
}
