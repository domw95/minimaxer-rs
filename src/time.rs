//! A monotonic clock that also exists on wasm.
//!
//! `std::time::Instant::now` panics on `wasm32-unknown-unknown`: there is no
//! clock inside the sandbox, so the host has to supply one. The search reads
//! this on a per-node hot path, so it is a bare `f64` of milliseconds rather
//! than anything that allocates or traps.

#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;

#[cfg(target_arch = "wasm32")]
pub use wasm_clock::{now_millis, Instant};

#[cfg(target_arch = "wasm32")]
mod wasm_clock {
    use core::ops::Add;
    use core::time::Duration;

    #[link(wasm_import_module = "env")]
    extern "C" {
        /// Milliseconds from an arbitrary origin, supplied by the host. In a
        /// browser this is `performance.now`; under node, `perf_hooks`.
        fn now_ms() -> f64;
    }

    /// Host milliseconds, for callers that want the raw reading.
    pub fn now_millis() -> f64 {
        now()
    }

    fn now() -> f64 {
        // Safety: the host must export `env.now_ms`. A module instantiated
        // without it fails at instantiation, not here.
        unsafe { now_ms() }
    }

    #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
    pub struct Instant(f64);

    impl Instant {
        pub fn now() -> Self {
            Instant(now())
        }

        pub fn elapsed(&self) -> Duration {
            Duration::from_secs_f64((now() - self.0).max(0.0) / 1000.0)
        }

        pub fn duration_since(&self, earlier: Self) -> Duration {
            Duration::from_secs_f64((self.0 - earlier.0).max(0.0) / 1000.0)
        }
    }

    impl Add<Duration> for Instant {
        type Output = Instant;

        fn add(self, rhs: Duration) -> Instant {
            Instant(self.0 + rhs.as_secs_f64() * 1000.0)
        }
    }
}
