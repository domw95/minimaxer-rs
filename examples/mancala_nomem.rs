//! Control for `mancala_solve`: the same alpha-beta search over Kalah, but
//! plain recursion that keeps no tree. Memory is O(depth) instead of
//! O(nodes searched), so whatever stops this one is compute, not RAM.

use minimaxer::games::mancala::{Mancala, Player};
use std::time::Instant;

fn rss_mib() -> f64 {
    let s = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
    let pages: u64 = s.split_whitespace().nth(1).and_then(|v| v.parse().ok()).unwrap_or(0);
    (pages * 4096) as f64 / (1024.0 * 1024.0)
}

/// Score from player one's point of view. Handles Kalah's repeat turn by
/// branching on whose turn it actually is rather than assuming alternation.
fn ab(g: &Mancala, depth: u8, mut alpha: i16, mut beta: i16, nodes: &mut u64) -> i16 {
    *nodes += 1;
    if depth == 0 || g.is_over() {
        return g.score();
    }
    let moves = g.moves();
    if moves.is_empty() {
        return g.score();
    }
    if g.activeplayer == Player::One {
        let mut best = i16::MIN;
        for m in moves {
            let mut c = *g;
            c.play(m);
            best = best.max(ab(&c, depth - 1, alpha, beta, nodes));
            alpha = alpha.max(best);
            if alpha >= beta {
                break;
            }
        }
        best
    } else {
        let mut best = i16::MAX;
        for m in moves {
            let mut c = *g;
            c.play(m);
            best = best.min(ab(&c, depth - 1, alpha, beta, nodes));
            beta = beta.min(best);
            if alpha >= beta {
                break;
            }
        }
        best
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let max_ply: u8 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(40);
    let secs: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(60.0);
    let stones: u8 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);

    let g = Mancala::with_stones(stones);
    println!("Kalah(6,{stones}) no tree retention. budget={secs}s");
    println!("{:>4} {:>16} {:>12} {:>10} {:>8}", "ply", "nodes", "ms", "RSS_MiB", "value");
    for ply in 1..=max_ply {
        let mut nodes = 0u64;
        let t = Instant::now();
        let v = ab(&g, ply, i16::MIN, i16::MAX, &mut nodes);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        println!("{:>4} {:>16} {:>12.1} {:>10.1} {:>8}", ply, nodes, ms, rss_mib(), v);
        if ms / 1000.0 > secs {
            println!("\nSTOPPED: ply {ply} took {:.1}s, over the {secs}s budget.", ms / 1000.0);
            println!("Memory never grew: RSS {:.1}MiB. The wall here is compute.", rss_mib());
            return;
        }
    }
}
