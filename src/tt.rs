//! Transposition table.
//!
//! Games where move order does not fully determine the position reach the
//! same state down many different lines. Measured on Azul, 30% of nodes at
//! depth 4 and 56% at depth 5 are positions already seen elsewhere in the
//! tree, so re-searching them is the single largest waste in the search.

use crate::Move;

/// What a stored value tells us about the true value of a position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bound {
    /// The search returned within its window, so the value is the true one.
    Exact,
    /// The search cut off high; the true value is at least this.
    Lower,
    /// No move beat alpha; the true value is at most this.
    Upper,
}

#[derive(Debug, Clone)]
struct Entry<M> {
    key: u64,
    value: f32,
    depth: u8,
    bound: Bound,
    best: Option<M>,
    /// Whether the search that produced this value reached the game's end down
    /// every line, rather than stopping at its depth limit.
    ///
    /// Without this a table hit could never be reported as exhaustive, since a
    /// stored value alone says nothing about how it was obtained. That makes a
    /// single hit anywhere in the tree enough to stop the whole search ever
    /// reporting `Exhaustive` — so a caller looking for exactly solved
    /// positions silently gets none of them.
    exhaustive: bool,
}

/// Fixed-size, direct-mapped, depth-preferred replacement.
///
/// Bounded by construction: the caller picks the size, so unlike the retained
/// tree this cannot grow without limit as the search deepens.
pub struct Tt<M> {
    entries: Vec<Option<Entry<M>>>,
    mask: usize,
    pub hits: u64,
    pub stores: u64,
}

impl<M: Move> Tt<M> {
    /// `bits` of index, so `2^bits` entries. 0 disables the table entirely.
    pub fn new(bits: u8) -> Self {
        if bits == 0 {
            return Tt { entries: Vec::new(), mask: 0, hits: 0, stores: 0 };
        }
        let n = 1usize << bits;
        Tt { entries: vec![None; n], mask: n - 1, hits: 0, stores: 0 }
    }

    pub fn enabled(&self) -> bool {
        !self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        for e in self.entries.iter_mut() {
            *e = None;
        }
        self.hits = 0;
        self.stores = 0;
    }

    fn probe(&self, key: u64) -> Option<&Entry<M>> {
        if self.entries.is_empty() || key == 0 {
            return None;
        }
        match &self.entries[(key as usize) & self.mask] {
            Some(e) if e.key == key => Some(e),
            _ => None,
        }
    }

    fn store(
        &mut self,
        key: u64,
        depth: u8,
        value: f32,
        bound: Bound,
        best: Option<M>,
        exhaustive: bool,
    ) {
        if self.entries.is_empty() || key == 0 {
            return;
        }
        let slot = (key as usize) & self.mask;
        // Depth-preferred: a result from a deeper search is worth more than a
        // shallow one that happens to be newer.
        let replace = match &self.entries[slot] {
            Some(e) => e.key != key || depth >= e.depth,
            None => true,
        };
        if replace {
            self.stores += 1;
            self.entries[slot] = Some(Entry { key, value, depth, bound, best, exhaustive });
        }
    }
}

/// What a probe told us to do.
pub(crate) enum Probe<M> {
    /// Nothing usable.
    Miss,
    /// Deep enough and conclusive: this is the answer. The flag carries
    /// whether the stored search reached the game's end down every line.
    Cutoff(f32, M, bool),
    /// Not conclusive, but narrows the window and/or suggests a move.
    Hint {
        alpha: f32,
        beta: f32,
        best: Option<M>,
    },
}

impl<M: Move> Tt<M> {
    pub(crate) fn lookup(&mut self, key: u64, depth: u8, alpha: f32, beta: f32) -> Probe<M> {
        // Copy out what we need so the borrow ends before the hit counter.
        let Some((e_value, e_depth, e_bound, best, e_exhaustive)) = self
            .probe(key)
            .map(|e| (e.value, e.depth, e.bound, e.best.clone(), e.exhaustive))
        else {
            return Probe::Miss;
        };
        if e_depth >= depth {
            let (mut a, mut b) = (alpha, beta);
            match e_bound {
                // An exact value can only be returned outright when there is a
                // move to report with it, since callers rely on having one.
                Bound::Exact => {
                    if let Some(m) = best.clone() {
                        self.hits += 1;
                        return Probe::Cutoff(e_value, m, e_exhaustive);
                    }
                }
                Bound::Lower => a = a.max(e_value),
                Bound::Upper => b = b.min(e_value),
            }
            if a >= b {
                if let Some(m) = best.clone() {
                    self.hits += 1;
                    return Probe::Cutoff(e_value, m, e_exhaustive);
                }
            }
            return Probe::Hint { alpha: a, beta: b, best };
        }
        // Too shallow to trust the value, but the move is still worth trying
        // first: it was good enough somewhere else.
        Probe::Hint { alpha, beta, best }
    }

    pub(crate) fn record(
        &mut self,
        key: u64,
        depth: u8,
        value: f32,
        alpha_orig: f32,
        beta: f32,
        best: Option<M>,
        exhaustive: bool,
    ) {
        let bound = if value <= alpha_orig {
            Bound::Upper
        } else if value >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.store(key, depth, value, bound, best, exhaustive);
    }
}
