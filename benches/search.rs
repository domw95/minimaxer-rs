//! Covers the paths that actually carry the search: alpha-beta, the move
//! ordering that iterative deepening feeds, and a game with repeat turns.
//!
//! Timing here is wall time and will be noisy on a shared machine. The
//! node-count side of these searches is asserted in the unit tests, which is
//! where a pruning regression shows up without any timing noise.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use minimaxer::games::mancala::{Mancala, MancalaEvaluator};
use minimaxer::games::tictactoe::{Ttt, TttEvaluator};
use minimaxer::negamax::{negamax, Negamax, NegamaxAim, SearchOptions};
use minimaxer::node::Node;

/// Full-width tic-tac-toe, no pruning. The worst case.
fn ttt_plain() {
    let mut node = Node::new(Ttt::new(minimaxer::games::tictactoe::Player::One));
    black_box(negamax(&mut node, &mut TttEvaluator, 9, NegamaxAim::Maximise));
}

fn ttt_with(opts: SearchOptions) {
    let mut s = Negamax::new(
        Node::new(Ttt::new(minimaxer::games::tictactoe::Player::One)),
        TttEvaluator,
        opts,
    );
    black_box(s.search());
}

/// Kalah exercises the repeat-turn path, where the player to move does not
/// alternate and the window must not be flipped.
fn mancala_with(depth: u8, opts: SearchOptions) {
    let mut s = Negamax::new(
        Node::new(Mancala::new()),
        MancalaEvaluator,
        SearchOptions { max_depth: Some(depth), ..opts },
    );
    black_box(s.search());
}

pub fn benches(c: &mut Criterion) {
    let ab = SearchOptions { alpha_beta: true, max_depth: Some(9), ..Default::default() };
    let ab_sorted = SearchOptions { iterative: true, pre_sort: true, ..ab };

    let mut g = c.benchmark_group("ttt");
    g.bench_function("plain", |b| b.iter(ttt_plain));
    g.bench_function("alpha_beta", |b| b.iter(|| ttt_with(ab)));
    g.bench_function("alpha_beta+presort", |b| b.iter(|| ttt_with(ab_sorted)));
    g.finish();

    let base = SearchOptions { alpha_beta: true, ..Default::default() };
    let sorted = SearchOptions { iterative: true, pre_sort: true, ..base };
    let mut g = c.benchmark_group("mancala");
    g.bench_function("alpha_beta/d8", |b| b.iter(|| mancala_with(8, base)));
    g.bench_function("alpha_beta+presort/d8", |b| b.iter(|| mancala_with(8, sorted)));
    g.bench_function("alpha_beta+presort/d12", |b| b.iter(|| mancala_with(12, sorted)));
    g.bench_function("presort+random_best/d8", |b| {
        b.iter(|| mancala_with(8, SearchOptions { random_best: true, ..sorted }))
    });
    g.finish();
}

criterion_group!(search, benches);
criterion_main!(search);
