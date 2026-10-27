//! Work-aware scheduling for CPU numerical kernels.
//!
//! Defaults to the caller's current Rayon pool. This module never configures the
//! global pool or moves a Tensor/graph to another thread. A scoped policy applies
//! to integrated MatVec, CPU matmul/batch-matmul forward and backward, fused
//! projections, argmax, prefill/inference attention and native low-precision
//! optimizer-state kernels; other operators retain their existing scheduling.
use rayon::prelude::*;
use std::cell::{Cell, RefCell};
use std::num::NonZeroUsize;
use std::sync::Arc;

/// Opt-in developer instrumentation; excluded unless explicitly compiled in.
#[cfg(feature = "cpu-profiling")]
pub mod profiling;

/// Measure a caller-side region only in explicitly instrumented builds.
#[doc(hidden)]
#[macro_export]
macro_rules! cpu_profile {
    ($stage:literal, $body:expr) => {{
        #[cfg(feature = "cpu-profiling")]
        {
            $crate::cpu::profiling::measure($stage, || $body)
        }
        #[cfg(not(feature = "cpu-profiling"))]
        {
            $body
        }
    }};
}

/// Reusable numerical-kernel policy. Clone it to share an existing worker pool.
#[derive(Clone, Default)]
pub struct CpuKernelContext {
    pool: Option<Arc<rayon::ThreadPool>>,
    task_limit: Option<NonZeroUsize>,
}

thread_local! {
    static CONTEXT: RefCell<CpuKernelContext> = RefCell::new(CpuKernelContext::default());
    static IN_KERNEL: Cell<bool> = const { Cell::new(false) };
}

impl CpuKernelContext {
    /// Use a caller-owned pool. Create pools once, outside the inference loop.
    pub fn with_pool(pool: Arc<rayon::ThreadPool>) -> Self {
        Self {
            pool: Some(pool),
            task_limit: None,
        }
    }

    /// Limit independent tasks per integrated kernel, not total process threads.
    /// A limit of one executes those kernels on the calling thread.
    pub fn with_task_limit(mut self, limit: NonZeroUsize) -> Self {
        self.task_limit = Some(limit);
        self
    }

    /// Keep the model and thread-local precision/autograd state on this thread.
    /// Only the numerical slice closures are submitted to a worker pool. The
    /// previous policy is restored on return or unwinding. This scope does not
    /// propagate to arbitrary user-spawned Rayon tasks.
    pub fn scope<R>(&self, f: impl FnOnce() -> R) -> R {
        struct Restore(CpuKernelContext);
        impl Drop for Restore {
            fn drop(&mut self) {
                CONTEXT.with(|slot| {
                    slot.replace(self.0.clone());
                });
            }
        }
        let _restore = Restore(CONTEXT.with(|slot| slot.replace(self.clone())));
        f()
    }
}

/// Explicit prefill/decode policies sharing the same worker pool.
/// Only integrated numerical kernels observe these policies. Neither phase
/// creates workers, benchmarks hardware, or moves the model to another thread.
#[derive(Clone, Default)]
pub struct CpuInferenceContext {
    prefill: CpuKernelContext,
    decode: CpuKernelContext,
}

impl CpuInferenceContext {
    pub fn new(base: CpuKernelContext) -> Self {
        Self {
            prefill: base.clone(),
            decode: base,
        }
    }

    pub fn with_pool(pool: Arc<rayon::ThreadPool>) -> Self {
        Self::new(CpuKernelContext::with_pool(pool))
    }

    pub fn with_prefill_task_limit(mut self, limit: NonZeroUsize) -> Self {
        self.prefill = self.prefill.with_task_limit(limit);
        self
    }

    pub fn with_decode_task_limit(mut self, limit: NonZeroUsize) -> Self {
        self.decode = self.decode.with_task_limit(limit);
        self
    }

    /// Enter the prefill policy on the calling thread, restoring it on return.
    pub fn prefill<R>(&self, f: impl FnOnce() -> R) -> R {
        self.prefill.scope(f)
    }

    /// Enter the decode policy on the calling thread, restoring it on return.
    pub fn decode<R>(&self, f: impl FnOnce() -> R) -> R {
        self.decode.scope(f)
    }
}

fn kernel<R>(f: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            IN_KERNEL.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(IN_KERNEL.with(|flag| flag.replace(true)));
    f()
}

// A portable grain heuristic, in scalar multiply/add-sized work units. This is
// not a bandwidth estimate, a CPU-model lookup, or an optimal-thread guarantee.
const MIN_WORK_PER_TASK: usize = 32 * 1024;

fn task_count(units: usize, work_per_unit: usize, workers: usize, limit: usize) -> usize {
    let useful = units.saturating_mul(work_per_unit) / MIN_WORK_PER_TASK;
    useful
        .max(1)
        .min(units.max(1))
        .min(workers.max(1))
        .min(limit.max(1))
}

/// Capacity available to one integrated kernel, before its work/grain checks.
pub(crate) fn parallelism() -> usize {
    if IN_KERNEL.with(Cell::get) {
        return 1;
    }
    CONTEXT.with(|slot| {
        let context = slot.borrow();
        if context.task_limit.is_some_and(|limit| limit.get() == 1) {
            return 1;
        }
        let workers = context
            .pool
            .as_ref()
            .map_or_else(rayon::current_num_threads, |p| p.current_num_threads());
        workers
            .min(context.task_limit.map_or(usize::MAX, NonZeroUsize::get))
            .max(1)
    })
}

pub(crate) fn should_parallelize(units: usize, work: usize) -> bool {
    if IN_KERNEL.with(Cell::get) || units.saturating_mul(work) < 2 * MIN_WORK_PER_TASK {
        return false;
    }
    CONTEXT.with(|slot| {
        let context = slot.borrow();
        if context.task_limit.is_some_and(|limit| limit.get() == 1) {
            return false;
        }
        let workers = context
            .pool
            .as_ref()
            .map_or_else(rayon::current_num_threads, |p| p.current_num_threads());
        task_count(
            units,
            work,
            workers,
            context.task_limit.map_or(usize::MAX, NonZeroUsize::get),
        ) > 1
    })
}

fn dispatch<R: Send>(units: usize, work: usize, f: impl FnOnce(usize) -> R + Send) -> R {
    if IN_KERNEL.with(Cell::get) {
        return f(1);
    }
    let context = CONTEXT.with(|slot| slot.borrow().clone());
    let limit = context.task_limit.map_or(usize::MAX, NonZeroUsize::get);
    if units <= 1 || limit == 1 || units.saturating_mul(work) < 2 * MIN_WORK_PER_TASK {
        return kernel(|| f(1));
    }
    let workers = context
        .pool
        .as_ref()
        .map_or_else(rayon::current_num_threads, |p| p.current_num_threads());
    let tasks = task_count(units, work, workers, limit);
    if tasks == 1 {
        return kernel(|| f(1));
    }
    // Retain Rayon's native tile splitting for unconstrained large jobs.
    // Pool capacity bounds workers; an explicit task limit bounds groups.
    let tasks = if context.task_limit.is_none() {
        units
    } else {
        tasks
    };
    if let Some(pool) = context.pool {
        pool.install(|| f(tasks))
    } else {
        f(tasks)
    }
}

// Group existing microkernel tiles without changing their numerical order or
// tail handling. At most `tasks` groups execute concurrently. Each group marks
// itself as a kernel region, so recursively scheduled numerical work is serial.
pub(crate) fn for_each_indexed<I, F>(iter: I, work: usize, f: F)
where
    I: ExactSizeIterator + Send + IntoParallelIterator<Item = <I as Iterator>::Item>,
    <I as IntoParallelIterator>::Iter: IndexedParallelIterator,
    <I as Iterator>::Item: Send,
    F: Fn((usize, <I as Iterator>::Item)) + Send + Sync,
{
    let units = iter.len();
    dispatch(units, work, |tasks| {
        if tasks == 1 {
            iter.enumerate().for_each(f);
        } else if tasks >= units {
            iter.into_par_iter()
                .enumerate()
                .for_each(|item| kernel(|| f(item)));
        } else {
            // A capped group owns its iterator items, including disjoint mutable
            // ndarray views. No caller-side list is needed on the native path.
            let per_group = units.div_ceil(tasks);
            iter.into_par_iter()
                .enumerate()
                .chunks(per_group)
                .for_each(|items| {
                    kernel(|| items.into_iter().for_each(&f));
                });
        }
    });
}

pub(crate) fn for_each_chunk_mut<T: Send>(
    output: &mut [T],
    chunk: usize,
    work: usize,
    f: impl Fn((usize, &mut [T])) + Send + Sync,
) {
    assert!(chunk > 0);
    let units = output.len().div_ceil(chunk);
    dispatch(units, work, |tasks| {
        if tasks == 1 {
            output.chunks_mut(chunk).enumerate().for_each(f);
        } else if tasks >= units {
            output
                .par_chunks_mut(chunk)
                .enumerate()
                .for_each(|item| kernel(|| f(item)));
        } else {
            let per_group = units.div_ceil(tasks);
            let span = per_group.saturating_mul(chunk);
            output
                .par_chunks_mut(span)
                .enumerate()
                .for_each(|(group, values)| {
                    kernel(|| {
                        for (offset, values) in values.chunks_mut(chunk).enumerate() {
                            f((group * per_group + offset, values));
                        }
                    })
                });
        }
    });
}

pub(crate) fn for_each_chunk2_mut<T: Send, U: Send>(
    left: &mut [T],
    right: &mut [U],
    chunk: usize,
    work: usize,
    f: impl Fn((usize, (&mut [T], &mut [U]))) + Send + Sync,
) {
    assert_eq!(left.len(), right.len());
    assert!(chunk > 0);
    let units = left.len().div_ceil(chunk);
    dispatch(units, work, |tasks| {
        if tasks == 1 {
            left.chunks_mut(chunk)
                .zip(right.chunks_mut(chunk))
                .enumerate()
                .for_each(f);
        } else if tasks >= units {
            left.par_chunks_mut(chunk)
                .zip(right.par_chunks_mut(chunk))
                .enumerate()
                .for_each(|item| kernel(|| f(item)));
        } else {
            let per_group = units.div_ceil(tasks);
            let span = per_group.saturating_mul(chunk);
            left.par_chunks_mut(span)
                .zip(right.par_chunks_mut(span))
                .enumerate()
                .for_each(|(group, (a, b))| {
                    kernel(|| {
                        for (offset, values) in
                            a.chunks_mut(chunk).zip(b.chunks_mut(chunk)).enumerate()
                        {
                            f((group * per_group + offset, values));
                        }
                    })
                });
        }
    });
}

pub(crate) fn for_each_chunk3_mut<T: Send, U: Send, V: Send>(
    a: &mut [T],
    b: &mut [U],
    c: &mut [V],
    chunk: usize,
    work: usize,
    f: impl Fn((usize, ((&mut [T], &mut [U]), &mut [V]))) + Send + Sync,
) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());
    assert!(chunk > 0);
    let units = a.len().div_ceil(chunk);
    dispatch(units, work, |tasks| {
        if tasks == 1 {
            a.chunks_mut(chunk)
                .zip(b.chunks_mut(chunk))
                .zip(c.chunks_mut(chunk))
                .enumerate()
                .for_each(f);
        } else if tasks >= units {
            a.par_chunks_mut(chunk)
                .zip(b.par_chunks_mut(chunk))
                .zip(c.par_chunks_mut(chunk))
                .enumerate()
                .for_each(|item| kernel(|| f(item)));
        } else {
            let per_group = units.div_ceil(tasks);
            let span = per_group.saturating_mul(chunk);
            a.par_chunks_mut(span)
                .zip(b.par_chunks_mut(span))
                .zip(c.par_chunks_mut(span))
                .enumerate()
                .for_each(|(group, ((a, b), c))| {
                    kernel(|| {
                        for (offset, values) in a
                            .chunks_mut(chunk)
                            .zip(b.chunks_mut(chunk))
                            .zip(c.chunks_mut(chunk))
                            .enumerate()
                        {
                            f((group * per_group + offset, values));
                        }
                    })
                });
        }
    });
}

pub(crate) fn map_reduce<R: Send>(
    units: usize,
    work: usize,
    map: impl Fn(usize) -> R + Send + Sync,
    identity: impl Fn() -> R + Send + Sync,
    reduce: impl Fn(R, R) -> R + Send + Sync,
) -> R {
    dispatch(units, work, |tasks| {
        if tasks == 1 {
            return (0..units).map(map).fold(identity(), reduce);
        }
        if tasks >= units {
            return (0..units)
                .into_par_iter()
                .map(|i| kernel(|| map(i)))
                .reduce(identity, reduce);
        }
        let per_group = units.div_ceil(tasks);
        (0..units.div_ceil(per_group))
            .into_par_iter()
            .map(|group| {
                kernel(|| {
                    let start = group * per_group;
                    (start..start.saturating_add(per_group).min(units))
                        .map(&map)
                        .fold(identity(), &reduce)
                })
            })
            .reduce(&identity, &reduce)
    })
}

#[cfg(test)]
#[path = "../../tests/unit/cpu/tests.rs"]
mod tests;
