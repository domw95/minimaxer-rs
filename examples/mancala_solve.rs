//! How deep can the tree-retaining search get on Kalah(6,4) before it runs
//! out of memory?
//!
//! Drives iterative deepening by hand so the tree is retained between depths
//! (which is what costs the memory) and reports nodes, time and RSS at each
//! ply. Stops at a caller-supplied RSS budget so it can't take the machine
//! down.

use minimaxer::games::mancala::{Mancala, MancalaEvaluator, MancalaMove};
use minimaxer::negamax::{Negamax, SearchOptions};
use minimaxer::node::Node;
use minimaxer::SearchExit;
use std::time::Instant;

/// Resident set size in bytes, straight from procfs.
fn rss_bytes() -> u64 {
    let s = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
    let pages: u64 = s.split_whitespace().nth(1).and_then(|v| v.parse().ok()).unwrap_or(0);
    pages * 4096
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // RSS budget in MiB; stop cleanly before the machine notices.
    let budget_mib: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2048.0);
    let max_ply: u8 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(60);
    let pre_sort: bool = args.get(3).map(|s| s == "sorted").unwrap_or(true);
    let stones: u8 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(4);

    println!(
        "Kalah(6,{stones}) first move. budget={budget_mib:.0}MiB max_ply={max_ply} pre_sort={pre_sort}"
    );
    println!("node_bytes={}", std::mem::size_of::<Node<Mancala, MancalaMove>>());
    println!(
        "{:>4} {:>14} {:>12} {:>10} {:>10} {:>8}  {:<11} {}",
        "ply", "nodes", "ms", "RSS_MiB", "B/node", "value", "exit", "best"
    );

    let base = rss_bytes();
    let mut search = Negamax::new(
        Node::new(Mancala::with_stones(stones)),
        MancalaEvaluator,
        SearchOptions {
            alpha_beta: true,
            pre_sort,
            ..Default::default()
        },
    );

    for ply in 1..=max_ply {
        search.options.max_depth = Some(ply);
        let t = Instant::now();
        let r = search.search();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let rss = rss_bytes();
        let grown = rss.saturating_sub(base);
        let per_node = if r.nodes > 0 { grown as f64 / r.nodes as f64 } else { 0.0 };

        println!(
            "{:>4} {:>14} {:>12.1} {:>10.1} {:>10.1} {:>8.1}  {:<11} {:?}",
            ply,
            r.nodes,
            ms,
            mib(rss),
            per_node,
            r.value,
            format!("{:?}", r.exit),
            r.best
        );

        if r.exit == SearchExit::Exhaustive {
            println!("\nSOLVED at ply {ply}: value={} best={:?}", r.value, r.best);
            return;
        }
        if mib(rss) > budget_mib {
            println!(
                "\nSTOPPED: RSS {:.1}MiB exceeded budget {:.0}MiB at ply {ply}.",
                mib(rss),
                budget_mib
            );
            println!("Not solved. Last value={} best={:?}", r.value, r.best);
            return;
        }
    }
    println!("\nReached max_ply without exhausting the tree.");
}
