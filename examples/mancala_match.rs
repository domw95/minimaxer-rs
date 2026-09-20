//! Plays Kalah(6,4) between two fixed-depth searches, to calibrate what
//! search depth is actually worth in this game.
//!
//! Also reports plies vs turn changes: Kalah's repeat turn means one "go"
//! can span several plies, so a ply budget buys fewer real turns than it
//! looks like.

use minimaxer::games::mancala::{Mancala, MancalaEvaluator, Player};
use minimaxer::negamax::{Negamax, SearchOptions};
use minimaxer::node::Node;
use std::time::Instant;

fn best_move(g: &Mancala, depth: u8) -> minimaxer::games::mancala::MancalaMove {
    let mut s = Negamax::new(
        Node::new(*g),
        MancalaEvaluator,
        SearchOptions {
            alpha_beta: true,
            pre_sort: true,
            iterative: true,
            max_depth: Some(depth),
            ..Default::default()
        },
    );
    s.search().best
}

/// Returns (final score from p1, plies, turn changes, seconds).
fn play(d1: u8, d2: u8) -> (i16, usize, usize, f64) {
    let mut g = Mancala::new();
    let mut plies = 0usize;
    let mut turn_changes = 0usize;
    let t = Instant::now();
    while !g.is_over() && plies < 400 {
        let before = g.activeplayer;
        let d = if before == Player::One { d1 } else { d2 };
        let m = best_move(&g, d);
        g.play(m);
        plies += 1;
        if g.activeplayer != before {
            turn_changes += 1;
        }
    }
    (g.score(), plies, turn_changes, t.elapsed().as_secs_f64())
}

fn round_robin(depths: &[u8]) {
    println!("Round robin, every pair both ways. Cell = margin for the ROW depth.");
    print!("{:>6}", "row\\col");
    for d in depths {
        print!("{:>7}", d);
    }
    println!("{:>9}{:>8}", "avg", "wins");
    for &a in depths {
        print!("{:>6}", a);
        let mut total = 0i32;
        let mut games = 0i32;
        let mut wins = 0i32;
        for &b in depths {
            if a == b {
                print!("{:>7}", "-");
                continue;
            }
            // a first, then b first; average a's margin over both
            let (s1, _, _, _) = play(a, b);
            let (s2, _, _, _) = play(b, a);
            let m = (s1 as i32) + (-(s2 as i32));
            print!("{:>7}", m as f32 / 2.0);
            total += m;
            games += 2;
            if s1 > 0 { wins += 1; }
            if s2 < 0 { wins += 1; }
        }
        println!("{:>9.2}{:>8}", total as f32 / games as f32, wins);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s == "rr").unwrap_or(false) {
        let depths: Vec<u8> = args.get(2)
            .map(|s| s.split(',').filter_map(|v| v.parse().ok()).collect())
            .unwrap_or_else(|| vec![2, 4, 6, 8, 10, 12]);
        round_robin(&depths);
        return;
    }
    let strong: u8 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(14);
    let weak_list: Vec<u8> = args
        .get(2)
        .map(|s| s.split(',').filter_map(|v| v.parse().ok()).collect())
        .unwrap_or_else(|| vec![2, 4, 6, 8]);

    println!("Kalah(6,4): depth {strong} vs weaker depths. Score is +ve = that side won by N stones.");
    println!(
        "{:>6} {:>6} {:>10} {:>8} {:>8} {:>10}  {}",
        "strong", "weak", "margin", "plies", "turns", "secs", "outcome"
    );
    for w in weak_list {
        // strong moves first
        let (s1, p1, t1, sec1) = play(strong, w);
        let out1 = if s1 > 0 { "strong wins" } else if s1 < 0 { "WEAK WINS" } else { "draw" };
        println!(
            "{:>6} {:>6} {:>10} {:>8} {:>8} {:>10.1}  {} (strong first)",
            strong, w, s1, p1, t1, sec1, out1
        );
        // weak moves first
        let (s2, p2, t2, sec2) = play(w, strong);
        let margin = -s2; // from the strong side's view
        let out2 = if margin > 0 { "strong wins" } else if margin < 0 { "WEAK WINS" } else { "draw" };
        println!(
            "{:>6} {:>6} {:>10} {:>8} {:>8} {:>10.1}  {} (weak first)",
            strong, w, margin, p2, t2, sec2, out2
        );
    }
}
