//! Pre-made composable scorers for locomotion / robotics evolution.
//!
//! Almost every robot or L-system evolution app reinvents the same fitness
//! scorers — displacement, up-alignment, energy efficiency. This module
//! promotes them upstream as composable building blocks.
//!
//! # Architecture
//!
//! - [`Scorer<S, O>`] — generic trait: given state `S`, return a numeric
//!   score of type `O: NumLike`.
//! - [`Trajectory`] — a fixed bundle of physical quantities (start/end
//!   position, up-vector, max height, energy used, final descriptor)
//!   covering the common case. Concrete scorers in this module operate
//!   on `Trajectory`. Users with exotic state types implement `Scorer`
//!   for their own state.
//! - **Combinators** — [`Multiply`], [`Sum`], [`Penalize`], [`Normalize`]
//!   compose simpler scorers into multi-term objectives.
//! - [`CompositeEvaluator`] — bridges scorers into the existing
//!   [`Evaluator`](crate::Evaluator) trait, so the same scorer composition
//!   feeds [`SimpleGA`](crate::algorithms::simple::SimpleGA),
//!   [`Nsga2`](crate::algorithms::nsga2::Nsga2), and the MAP-Elites family.
//!
//! # Why no `BehaviouralDiversity` scorer?
//!
//! Behavioural diversity is a population-level signal, not a per-individual
//! one — measuring the diversity of a single phenotype is meaningless.
//! [`crate::algorithms::novelty_search::NoveltySearch`] already drives
//! selection by population-level behavioural distance; reach for it when
//! you need diversity pressure.
//!
//! # Example
//!
//! ```
//! use symbios_genetics::scorers::{
//!     CompositeEvaluator, Displacement, EnergyEfficiency, Height,
//!     Multiply, Sum, Trajectory, UpAlignment,
//! };
//! # use serde::{Deserialize, Serialize};
//! # use rand::Rng;
//! # #[derive(Clone, Serialize, Deserialize)]
//! # struct Robot;
//! # impl symbios_genetics::Genotype for Robot {
//! #     fn mutate<R: Rng>(&mut self, _: &mut R, _: f32) {}
//! #     fn crossover<R: Rng>(&self, _: &Self, _: &mut R) -> Self { Robot }
//! # }
//!
//! // Locomotion = displacement * up_alignment * (height + 0.5)
//! let locomotion = Multiply(
//!     Multiply(Displacement, UpAlignment),
//!     Sum(Height, Const(0.5_f32)),
//! );
//!
//! let evaluator = CompositeEvaluator::<Robot, Trajectory, _>::new(
//!     // sim: Robot -> Trajectory (user-defined)
//!     |_robot: &Robot| Trajectory::default(),
//!     Box::new(locomotion),
//! )
//! .with_objectives(vec![Box::new(Displacement), Box::new(EnergyEfficiency)])
//! .with_descriptors(vec![Box::new(Height), Box::new(UpAlignment)]);
//! # use symbios_genetics::scorers::Const;
//! ```

use crate::{Evaluator, Genotype};
use std::marker::PhantomData;

// =============================================================================
// NumLike: a minimal numeric trait, locally defined to avoid a num-traits dep.
// =============================================================================

/// Minimal numeric trait covering `f32` and `f64`.
///
/// Implemented locally rather than depending on `num-traits` since the
/// concrete scorers always work with floats. If you need a wider numeric
/// type, implement `NumLike` for it yourself.
pub trait NumLike:
    Copy
    + Send
    + Sync
    + PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::fmt::Debug
{
    /// The additive identity (`0`).
    const ZERO: Self;
    /// The multiplicative identity (`1`).
    const ONE: Self;
    /// Convert from `f32` (used by combinators that need a literal).
    fn from_f32(v: f32) -> Self;
    /// Convert to `f32` (used by [`CompositeEvaluator`] when bridging to
    /// the [`Evaluator`](crate::Evaluator) trait).
    fn to_f32(self) -> f32;

    /// Clamp `self` into `[lo, hi]`, treating NaN-like values per
    /// `PartialOrd`'s rules.
    fn clamp_to(self, lo: Self, hi: Self) -> Self {
        if self < lo {
            lo
        } else if self > hi {
            hi
        } else {
            self
        }
    }
}

impl NumLike for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    fn from_f32(v: f32) -> Self {
        v
    }
    fn to_f32(self) -> f32 {
        self
    }
}

impl NumLike for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    fn from_f32(v: f32) -> Self {
        v as f64
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
}

// =============================================================================
// Scorer trait
// =============================================================================

/// Generic scorer: produces a numeric score of type `O` from state `S`.
///
/// Implementations must be deterministic given the input state and safe
/// to call from parallel threads.
pub trait Scorer<S, O: NumLike>: Send + Sync {
    /// Score `state`. Caller decides what `state` means (often a
    /// [`Trajectory`] from a physics sim).
    fn score(&self, state: &S) -> O;
}

// =============================================================================
// Trajectory: the common case for locomotion/robotics scorers
// =============================================================================

/// Bundle of physical quantities that the concrete scorers in this module
/// operate on.
///
/// Users populate this from their simulation output. Coordinate convention:
/// **Y-up** (the `up` field's Y component is the alignment with the world
/// vertical). Switch axis if your engine uses Z-up.
#[derive(Clone, Debug, Default)]
pub struct Trajectory {
    /// Initial position `[x, y, z]`.
    pub start: [f32; 3],
    /// Final position `[x, y, z]`.
    pub end: [f32; 3],
    /// Final up-vector `[x, y, z]`. Should be unit-length; the
    /// [`UpAlignment`] scorer reads its Y component.
    pub up: [f32; 3],
    /// Peak Y-coordinate during the simulation.
    pub max_height: f32,
    /// Total energy expended (units defined by user).
    pub energy_used: f32,
    /// User-defined behavioural descriptor (e.g. for MAP-Elites).
    pub final_descriptor: Vec<f32>,
}

// =============================================================================
// Concrete scorers
// =============================================================================

/// Euclidean distance between `start` and `end` positions.
#[derive(Clone, Copy, Debug, Default)]
pub struct Displacement;

impl Scorer<Trajectory, f32> for Displacement {
    fn score(&self, t: &Trajectory) -> f32 {
        let dx = t.end[0] - t.start[0];
        let dy = t.end[1] - t.start[1];
        let dz = t.end[2] - t.start[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Alignment of the final up-vector with world-up (Y axis).
///
/// Returns the Y component of `up`, clamped to `[0.0, 1.0]`. A value of
/// `1.0` means perfectly upright; `0.0` means horizontal or upside-down.
#[derive(Clone, Copy, Debug, Default)]
pub struct UpAlignment;

impl Scorer<Trajectory, f32> for UpAlignment {
    fn score(&self, t: &Trajectory) -> f32 {
        t.up[1].clamp(0.0, 1.0)
    }
}

/// Peak height (Y coordinate) reached during the trajectory.
#[derive(Clone, Copy, Debug, Default)]
pub struct Height;

impl Scorer<Trajectory, f32> for Height {
    fn score(&self, t: &Trajectory) -> f32 {
        t.max_height
    }
}

/// Distance covered per unit of energy.
///
/// Returns `displacement / energy_used`, or `0.0` if `energy_used <= 0`.
/// Useful as an objective for NSGA-II alongside displacement (trade off
/// distance vs efficiency).
#[derive(Clone, Copy, Debug, Default)]
pub struct EnergyEfficiency;

impl Scorer<Trajectory, f32> for EnergyEfficiency {
    fn score(&self, t: &Trajectory) -> f32 {
        if t.energy_used > 0.0 {
            Displacement.score(t) / t.energy_used
        } else {
            0.0
        }
    }
}

/// Constant scorer — always returns the wrapped value.
///
/// Useful as a building block in arithmetic combinators (e.g. shifting a
/// scorer's output, or providing a scaling factor).
#[derive(Clone, Copy, Debug)]
pub struct Const<O: NumLike>(pub O);

impl<S, O: NumLike> Scorer<S, O> for Const<O> {
    fn score(&self, _state: &S) -> O {
        self.0
    }
}

// =============================================================================
// Combinators
// =============================================================================

/// Product of two scorers: `score(s) = a.score(s) * b.score(s)`.
#[derive(Clone, Copy, Debug)]
pub struct Multiply<A, B>(pub A, pub B);

impl<S, O, A, B> Scorer<S, O> for Multiply<A, B>
where
    O: NumLike,
    A: Scorer<S, O>,
    B: Scorer<S, O>,
{
    fn score(&self, state: &S) -> O {
        self.0.score(state) * self.1.score(state)
    }
}

/// Sum of two scorers: `score(s) = a.score(s) + b.score(s)`.
#[derive(Clone, Copy, Debug)]
pub struct Sum<A, B>(pub A, pub B);

impl<S, O, A, B> Scorer<S, O> for Sum<A, B>
where
    O: NumLike,
    A: Scorer<S, O>,
    B: Scorer<S, O>,
{
    fn score(&self, state: &S) -> O {
        self.0.score(state) + self.1.score(state)
    }
}

/// Subtract a penalty from a base score: `score(s) = base - penalty`.
///
/// Often used to discourage undesired behaviour while still rewarding the
/// primary objective.
#[derive(Clone, Copy, Debug)]
pub struct Penalize<A, B> {
    /// The primary score.
    pub base: A,
    /// The penalty to subtract.
    pub penalty: B,
}

impl<S, O, A, B> Scorer<S, O> for Penalize<A, B>
where
    O: NumLike,
    A: Scorer<S, O>,
    B: Scorer<S, O>,
{
    fn score(&self, state: &S) -> O {
        self.base.score(state) - self.penalty.score(state)
    }
}

/// Linearly normalise a scorer to `[0, 1]` against an expected maximum.
///
/// Computes `inner / max`, then clamps to `[0, 1]`. Use when composing
/// scorers with very different ranges (e.g. `Height` is unbounded while
/// `UpAlignment` is already in `[0, 1]`).
///
/// # Panics
///
/// Panics if `max == 0` (division by zero).
#[derive(Clone, Copy, Debug)]
pub struct Normalize<A, O: NumLike> {
    /// The inner scorer.
    pub inner: A,
    /// The expected maximum value of `inner` against which to normalise.
    pub max: O,
}

impl<S, O, A> Scorer<S, O> for Normalize<A, O>
where
    O: NumLike,
    A: Scorer<S, O>,
{
    fn score(&self, state: &S) -> O {
        assert!(
            self.max > O::ZERO,
            "Normalize::max must be > 0 (division by zero otherwise)"
        );
        let raw = self.inner.score(state) / self.max;
        raw.clamp_to(O::ZERO, O::ONE)
    }
}

// =============================================================================
// CompositeEvaluator: bridges Scorer composition into Evaluator<G>
// =============================================================================

/// Bridges a [`Scorer`] composition into the [`Evaluator`] trait used by all
/// algorithms.
///
/// Given a simulator function `Fn(&G) -> S`, runs it once per genotype and
/// feeds the resulting state to:
///
/// - one **fitness** scorer (scalar, used by `SimpleGA` / MAP-Elites)
/// - any number of **objective** scorers (used by `Nsga2`)
/// - any number of **descriptor** scorers (used by MAP-Elites variants and
///   novelty search)
///
/// All three are computed against the *same* simulation result, so the
/// sim runs only once per genotype.
///
/// # Type Parameters
///
/// * `G` — the genotype type
/// * `S` — the simulator output (typically [`Trajectory`])
/// * `F` — the simulator function `Fn(&G) -> S`
pub struct CompositeEvaluator<G, S, F>
where
    G: Genotype,
    S: Send + Sync,
    F: Fn(&G) -> S + Send + Sync,
{
    sim: F,
    fitness_scorer: Box<dyn Scorer<S, f32> + Send + Sync>,
    objective_scorers: Vec<Box<dyn Scorer<S, f32> + Send + Sync>>,
    descriptor_scorers: Vec<Box<dyn Scorer<S, f32> + Send + Sync>>,
    _marker: PhantomData<G>,
}

impl<G, S, F> CompositeEvaluator<G, S, F>
where
    G: Genotype,
    S: Send + Sync,
    F: Fn(&G) -> S + Send + Sync,
{
    /// Creates a composite evaluator with a fitness scorer only.
    ///
    /// Use [`with_objectives`](Self::with_objectives) and
    /// [`with_descriptors`](Self::with_descriptors) to add more signals.
    pub fn new(sim: F, fitness_scorer: Box<dyn Scorer<S, f32> + Send + Sync>) -> Self {
        Self {
            sim,
            fitness_scorer,
            objective_scorers: Vec::new(),
            descriptor_scorers: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Sets the objective scorers (consumed for NSGA-II `objectives` vector).
    pub fn with_objectives(mut self, scorers: Vec<Box<dyn Scorer<S, f32> + Send + Sync>>) -> Self {
        self.objective_scorers = scorers;
        self
    }

    /// Sets the descriptor scorers (consumed for MAP-Elites / novelty
    /// search `descriptor` vector).
    pub fn with_descriptors(mut self, scorers: Vec<Box<dyn Scorer<S, f32> + Send + Sync>>) -> Self {
        self.descriptor_scorers = scorers;
        self
    }
}

impl<G, S, F> Evaluator<G> for CompositeEvaluator<G, S, F>
where
    G: Genotype,
    S: Send + Sync,
    F: Fn(&G) -> S + Send + Sync,
{
    fn evaluate(&self, genotype: &G) -> (f32, Vec<f32>, Vec<f32>) {
        let state = (self.sim)(genotype);
        let fitness = self.fitness_scorer.score(&state);
        let objectives = self
            .objective_scorers
            .iter()
            .map(|s| s.score(&state))
            .collect();
        let descriptor = self
            .descriptor_scorers
            .iter()
            .map(|s| s.score(&state))
            .collect();
        (fitness, objectives, descriptor)
    }
}
