//! Caller-side wall times including numerical dispatch and worker completion.
//! These are inclusive region timings, not hardware counters or worker CPU time.
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

#[derive(Clone, Debug, serde::Serialize)]
pub struct Region {
    pub phase: &'static str,
    pub stage: &'static str,
    pub calls: usize,
    pub wall_us: f64,
}

#[derive(Default)]
struct State {
    phase: &'static str,
    regions: BTreeMap<(&'static str, &'static str), (usize, Duration)>,
}

thread_local! {
    static ACTIVE: RefCell<Option<State>> = const { RefCell::new(None) };
}

/// Collect this thread's regions. Numerical workers are timed by their caller.
/// Scopes restore previous collection state even if the callback unwinds.
pub fn capture<R>(f: impl FnOnce() -> R) -> (R, Vec<Region>) {
    struct Restore(Option<State>);
    impl Drop for Restore {
        fn drop(&mut self) {
            ACTIVE.with(|s| {
                s.replace(self.0.take());
            });
        }
    }
    let restore = Restore(ACTIVE.with(|s| s.replace(Some(State::default()))));
    let result = f();
    let state = ACTIVE.with(|s| s.take()).unwrap();
    drop(restore);
    let regions = state
        .regions
        .into_iter()
        .map(|((phase, stage), (calls, elapsed))| Region {
            phase,
            stage,
            calls,
            wall_us: elapsed.as_secs_f64() * 1e6,
        })
        .collect();
    (result, regions)
}

pub struct Phase(&'static str);
impl Drop for Phase {
    fn drop(&mut self) {
        ACTIVE.with(|s| {
            if let Some(state) = s.borrow_mut().as_mut() {
                state.phase = self.0;
            }
        });
    }
}

pub fn phase(name: &'static str) -> Phase {
    Phase(ACTIVE.with(|s| {
        s.borrow_mut()
            .as_mut()
            .map(|state| std::mem::replace(&mut state.phase, name))
            .unwrap_or("")
    }))
}

#[inline]
pub fn measure<R>(stage: &'static str, f: impl FnOnce() -> R) -> R {
    let _region = region(stage);
    f()
}

pub struct Scope(Option<(&'static str, &'static str, Instant)>);

pub fn region(stage: &'static str) -> Scope {
    let phase = ACTIVE.with(|s| s.borrow().as_ref().map(|s| s.phase));
    Scope(phase.map(|phase| (phase, stage, Instant::now())))
}

impl Drop for Scope {
    fn drop(&mut self) {
        if let Some((phase, stage, start)) = self.0 {
            let elapsed = start.elapsed();
            ACTIVE.with(|s| {
                if let Some(state) = s.borrow_mut().as_mut() {
                    let entry = state.regions.entry((phase, stage)).or_default();
                    entry.0 += 1;
                    entry.1 += elapsed;
                }
            });
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/cpu/profiling/tests.rs"]
mod tests;
