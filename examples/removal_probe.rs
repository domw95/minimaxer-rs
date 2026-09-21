//! What node removal actually costs and saves, in nodes rather than bytes.
//!
//! Peak memory is reached at the end of the deepest pass, so the question is
//! not "how small is the tree between passes" but "how big is the tree when
//! the last pass finishes". Removal pushes that both ways: it throws away
//! everything the earlier passes left behind, and it makes the last pass
//! regenerate subtrees it would otherwise have found already built.
//!
//!     cargo run --release --example removal_probe [max_depth]

use minimaxer::games::mancala::{Mancala, MancalaEvaluator};
use minimaxer::negamax::{Negamax, RemovalMethod, SearchOptions};
use minimaxer::node::Node;
use std::time::Instant;

fn run(depth: u8, removal: RemovalMethod, removal_depth: u8) -> (usize, u32, f32, f64) {
    let mut n = Negamax::new(
        Node::new(Mancala::new()),
        MancalaEvaluator,
        SearchOptions {
            alpha_beta: true,
            iterative: true,
            pre_sort: true,
            max_depth: Some(depth),
            removal,
            removal_depth,
            ..Default::default()
        },
    );
    let t = Instant::now();
    let r = n.search();
    (n.tree_size(), r.nodes, r.value, t.elapsed().as_secs_f64())
}

fn main() {
    let max: u8 = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(13);

    println!(
        "{:>5} {:>10} {:>14} {:>14} {:>10} {:>8} {:>9}",
        "depth", "removal", "tree_at_end", "last_pass_nodes", "ratio", "value", "secs"
    );
    for depth in 7..=max {
        let (base_tree, base_nodes, base_value, base_secs) = run(depth, RemovalMethod::None, 0);
        println!(
            "{depth:>5} {:>10} {base_tree:>14} {base_nodes:>14} {:>10.2} {base_value:>8} {base_secs:>9.3}",
            "none", 1.0
        );
        for (label, method, rd) in [
            ("always", RemovalMethod::Always, 0),
            ("depth>=4", RemovalMethod::Depth, 4),
            ("depth>=6", RemovalMethod::Depth, 6),
        ] {
            let (tree, nodes, value, secs) = run(depth, method, rd);
            assert_eq!(value, base_value, "removal changed the value at depth {depth}");
            println!(
                "{depth:>5} {label:>10} {tree:>14} {nodes:>14} {:>10.2} {value:>8} {secs:>9.3}",
                tree as f64 / base_tree as f64
            );
        }
    }
}
