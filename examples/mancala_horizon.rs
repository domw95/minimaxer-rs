//! Plays a game and reports, per move, whether the search reached the end of
//! the game (Exhaustive) or hit its depth limit. Once it is Exhaustive the
//! engine is playing perfectly, not heuristically.

use minimaxer::games::mancala::{Mancala, MancalaEvaluator, Player};
use minimaxer::negamax::{Negamax, SearchOptions};
use minimaxer::node::Node;
use minimaxer::SearchExit;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let d1: u8 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let d2: u8 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);

    let mut g = Mancala::new();
    let mut ply = 0;
    let mut first_exact: Option<usize> = None;
    println!("depth {d1} (P1) vs depth {d2} (P2)");
    println!("{:>4} {:>7} {:>12} {:>10} {:>7}", "ply", "mover", "exit", "nodes", "value");
    while !g.is_over() && ply < 200 {
        let mover = g.activeplayer;
        let d = if mover == Player::One { d1 } else { d2 };
        let mut s = Negamax::new(
            Node::new(g),
            MancalaEvaluator,
            SearchOptions {
                alpha_beta: true,
                pre_sort: true,
                iterative: true,
                max_depth: Some(d),
                ..Default::default()
            },
        );
        let r = s.search();
        if mover == Player::One && r.exit == SearchExit::Exhaustive && first_exact.is_none() {
            first_exact = Some(ply);
        }
        if mover == Player::One {
            println!(
                "{:>4} {:>7} {:>12} {:>10} {:>7}",
                ply,
                "P1",
                format!("{:?}", r.exit),
                r.nodes,
                r.value
            );
        }
        g.play(r.best);
        ply += 1;
    }
    println!("\nfinal score (P1 view): {}  after {ply} plies", g.score());
    match first_exact {
        Some(p) => println!("P1's search first saw the whole game at ply {p} of {ply} ({:.0}% in)", 100.0 * p as f64 / ply as f64),
        None => println!("P1's search never reached the end of the game"),
    }
}
