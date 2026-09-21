# Arena-allocated node storage + node removal

You are working in a git worktree at `/home/dom/minimaxer-rs-arena` on branch
`feat/arena-removal`, cut from `minimaxer-rs` master `d340732`.

## Goal

Make deep Azul search memory-viable, so a dataset of *exactly solved* positions
can be generated for PPO to learn from.

Azul rounds end when every factory is empty, and the search treats a round end
as terminal. Measured: from 3+ plies into a round the search reaches
`SearchExit::Exhaustive` — an exact answer to the round boundary — in
milliseconds. Only the first 2-3 plies of each round are expensive. So most
positions in a game can be labelled *exactly* rather than heuristically. The
blocker is memory at the opening, where there are ~90 legal moves.

Peak RSS for one worker generating a game (cap = max_depth, TT 2^18):

    cap 6   1,657 MiB    162,521 positions/hour
    cap 7   4,190 MiB     49,660 positions/hour
    cap 8  31,409 MiB     11,650 positions/hour   <- unusable, box has 32GB

Cap 8 is where labels get good and memory explodes. That is the target.

## What to build

Two changes, in order. The second is the actual win; the first makes it cheap.

### 1. Arena allocation for nodes

Today `Node.children` is `Vec<(M, Node<G,M>)>` — nodes inline, each entry 248
bytes for Azul. Move nodes into a bump arena and make children `(M, u32)`
indices.

Expect roughly 10-20% memory from removing malloc headers, `Vec` capacity slack
and 8-byte pointers. Do **not** expect more: boxing the children (the cheap
approximation) measured 0.96x, i.e. slightly slower and no real memory win.
`perf/box-children` has that if you want to see it. The arena's real purpose is
to make step 2 cheap.

The borrow checker is the hard part: `negamax_ab` recurses with `&mut Node`, and
you cannot hold that while also holding `&mut Arena`. Index-based access
throughout is the usual answer.

### 2. Node removal between iterative-deepening passes

This is the scaling fix. Reference implementation is the TypeScript original at
`/home/dom/minimaxer/src/tree/searchtree.ts`, `removeNonBestNodes` (~line 324),
with options in `src/tree/search.ts` (`RemovalMethod`: NONE / ALWAYS / DEPTH /
COUNT).

It collapses the tree to the principal variation plus one level of siblings and
discards the rest — but **first writes the sorted move order back into
`node.moves`**, so ordering information survives the nodes being freed. That
decoupling is the whole point: ordering is worth 4.4x and must not be lost.

With an arena this becomes: copy the retained PV into a fresh arena, drop the
old one. O(kept), not O(discarded), which is the right way round when you are
discarding 90%+.

## Do not go down these roads

Measured, and they lose:

- **Replacing the retained tree with the transposition table.** Tried in
  `examples/mancala_tt.rs`. The retained tree solved Kalah(6,2) in 901k nodes;
  the TT-only version took 165M at a shallower depth, ~180x worse. A TT stores
  one lossy best-move hint per position; the tree stores every child sorted by
  its exact value. Only ~22% of searched nodes are even in the TT — leaves
  return before the table code, and it is fixed-size and overwrites.
- **`reserve(moves.len())` on the children vec.** Alpha-beta cuts off early, so
  most of the reserve is waste: 1411 -> 2101 bytes/node and slower.
- **Hoisting the row-playability check out of Azul's move generation.** 32%
  slower. There are only 38 `can_play_tile` calls per generation; a 5x5 table
  costs 25 always, and the table machinery exceeds the 13 calls saved.
- **`sort_on_create` for memory.** It cuts searched nodes 3-4x but *increases*
  peak RSS (2031 -> 2613 MiB), because it creates every child eagerly.

## Correctness anchors — use these, they are load-immune

Node counts and value checksums are deterministic. Timings are not.

    cd /home/dom/azul-tiles-rs-perf     # worktree, branch bench/engine-profile
    cargo bench --bench search -- iter_sorted 6 1
    # expect: nodes=275975 checksum=1aea0b04a1000000

Removal changes *which* nodes are searched (discarded subtrees get regenerated),
so node counts may move. **Checksums must not.** If a checksum changes, the
search is returning different answers and something is wrong.

`cargo test` in this worktree: 14 tests, all passing at `d340732`. Keep them
green. Kalah covers the repeat-turn path where the player to move does not
alternate.

## Measuring on this machine — read this before trusting any timing

This is a shared box with other Claude sessions doing ML training. Load has
ranged 1 to 38 during a single afternoon.

- At load 30, comparing a binary **against itself** gave a 276-298% spread. At
  load 4 the same test gives 1.5%. CPU time (`CLOCK_PROCESS_CPUTIME_ID`) does
  not rescue it — it excludes descheduling but not the cache damage.
- **Put a null control inside every run.** Compare A, B, A — the same binary
  twice under different labels. If the two A's disagree, discard the run. This
  turns "is the box quiet enough" into a measurement. `/tmp/final.sh` does this;
  usage `final.sh <depth> <rounds> <bin>:<mode>:<label> ...`.
- **Check hyperthread siblings.** `cat /sys/devices/system/cpu/cpu*/topology/thread_siblings_list`.
  A core reading 98% idle whose sibling is 77% busy is not idle. That mistake
  cost a set of measurements.
- Load *compresses* relative effects while destroying absolute ones, so a ratio
  measured under load understates. A node-size effect measured at ~10% under
  load was 19% when quiet.

Other sessions are working here. `dom-d1` runs long training jobs; `dom-d5` runs
benchmarks. Use `ListAgents` / `SendMessage` to ask for a quiet window rather
than just taking the box — both have been happy to pause. Use **one core**
(`taskset -c <n>`) and `nice -n 19`.

## Pitfalls that have already bitten

- **A counting global allocator must forward `realloc`.** The default
  `GlobalAlloc::realloc` is alloc+copy+dealloc, which discards libc's ability to
  grow in place and penalises exactly the large-vector growth you are measuring.
  Omitting it manufactured a fake 1.55x win for boxing that was really 0.96x.
- **`git add -A` has twice swept a local `[patch]` stanza into a commit**,
  pointing `minimaxer` at a worktree path and stripping the git source line from
  `Cargo.lock`. Check `git show HEAD:Cargo.toml | grep patch` before pushing.
- **`search()` panics on a position with no legal moves** (`negamax.rs:146`,
  `Option::unwrap()` on `None`). Pre-existing. Check `is_round_over()` first.
- The container has restarted twice unprompted, killing background jobs. Commit
  early; prefer foreground or restart-tolerant jobs.

## Definition of done

Cap 8 generation fits comfortably in RAM alongside other users of the box, with
checksums unchanged and tests green. Report the RSS and throughput table above,
re-measured, with a null control in each timing run.
