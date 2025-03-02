use criterion::{black_box, criterion_group, criterion_main, Criterion};
use minimaxer::{
    games::tictactoe::{Ttt, TttEvaluator},
    negamax::{negamax, NegamaxAim},
    node::Node,
};

fn ttt_negamax() {
    let mut ttt = Ttt::new(minimaxer::games::tictactoe::Player::One);
    // Create node
    let mut node = Node::new(ttt);
    // Create evaluator
    let mut evaluator = TttEvaluator;
    // Run search
    negamax(&mut node, &mut evaluator, 9, NegamaxAim::Maximise);
}

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("ttt_negamax", |b| b.iter(ttt_negamax));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
