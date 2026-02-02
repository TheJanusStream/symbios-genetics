//! # Symbios Genetics
//!
//! A high-performance evolutionary computation library for Rust with deterministic
//! execution and serializable state.
//!
//! ## Features
//!
//! - **Deterministic Execution**: Bit-perfect reproducibility across runs with seeded RNG
//! - **Serializable State**: Save and restore evolution state via Serde
//! - **Parallel Evaluation**: Optional parallel fitness evaluation via Rayon
//! - **Multiple Algorithms**: Simple GA, NSGA-II, and MAP-Elites
//!
//! ## Quick Start
//!
//! ```rust
//! use rand::Rng;
//! use serde::{Deserialize, Serialize};
//! use symbios_genetics::{Evaluator, Evolver, Genotype, algorithms::simple::SimpleGA};
//!
//! // Define your genome
//! #[derive(Clone, Serialize, Deserialize)]
//! struct MyGenome(f32);
//!
//! impl Genotype for MyGenome {
//!     fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
//!         if rng.random::<f32>() < rate {
//!             self.0 += rng.random::<f32>() - 0.5;
//!         }
//!     }
//!     fn crossover<R: Rng>(&self, other: &Self, _rng: &mut R) -> Self {
//!         MyGenome((self.0 + other.0) / 2.0)
//!     }
//! }
//!
//! // Define your fitness function
//! struct MaximizeFitness;
//! impl Evaluator<MyGenome> for MaximizeFitness {
//!     fn evaluate(&self, g: &MyGenome) -> (f32, Vec<f32>, Vec<f32>) {
//!         let fitness = -g.0.powi(2); // Minimize x^2 (maximize -x^2)
//!         (fitness, vec![fitness], vec![])
//!     }
//! }
//!
//! // Run evolution
//! let initial: Vec<MyGenome> = (0..50).map(|i| MyGenome(i as f32 - 25.0)).collect();
//! let mut ga = SimpleGA::new(initial, 0.1, 5, 42);
//!
//! for _ in 0..100 {
//!     ga.step(&MaximizeFitness);
//! }
//!
//! let best = ga.population().iter().max_by(|a, b| {
//!     a.fitness.partial_cmp(&b.fitness).unwrap()
//! }).unwrap();
//! println!("Best fitness: {}", best.fitness);
//! ```
//!
//! ## Algorithms
//!
//! | Algorithm | Use Case | Key Feature |
//! |-----------|----------|-------------|
//! | [`SimpleGA`](algorithms::simple::SimpleGA) | Single-objective optimization | Fast, simple, elitism support |
//! | [`Nsga2`](algorithms::nsga2::Nsga2) | Multi-objective optimization | Pareto front discovery |
//! | [`MapElites`](algorithms::map_elites::MapElites) | Quality-diversity | Behavioral diversity archive |
//!
//! ## Feature Flags
//!
//! - `parallel` (default): Enable parallel fitness evaluation using Rayon
//!
//! ## Serialization
//!
//! All algorithm state is serializable, enabling checkpointing and resumption:
//!
//! ```rust,ignore
//! // Save state
//! let json = serde_json::to_string(&ga)?;
//!
//! // Restore and continue
//! let mut ga: SimpleGA<MyGenome> = serde_json::from_str(&json)?;
//! ga.step(&evaluator);
//! ```

use rand::Rng;
use serde::{Deserialize, Serialize};

/// A genotype represents the genetic encoding of an individual.
///
/// Implement this trait for your custom genome type to use it with the
/// evolutionary algorithms in this crate.
///
/// # Requirements
///
/// - Must be [`Clone`], [`Serialize`], [`Deserialize`], [`Send`], and [`Sync`]
/// - Must implement mutation and crossover operators
///
/// # Example
///
/// ```rust
/// use rand::Rng;
/// use serde::{Deserialize, Serialize};
/// use symbios_genetics::Genotype;
///
/// #[derive(Clone, Serialize, Deserialize)]
/// struct BitVec(Vec<bool>);
///
/// impl Genotype for BitVec {
///     fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
///         for bit in &mut self.0 {
///             if rng.random::<f32>() < rate {
///                 *bit = !*bit;
///             }
///         }
///     }
///
///     fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
///         let point = rng.random_range(0..self.0.len());
///         let mut child = self.0[..point].to_vec();
///         child.extend_from_slice(&other.0[point..]);
///         BitVec(child)
///     }
/// }
/// ```
pub trait Genotype: Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync {
    /// Apply mutation to this genotype.
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator for stochastic mutation
    /// * `rate` - Mutation rate, typically in range `[0.0, 1.0]`
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32);

    /// Create offspring by combining this genotype with another.
    ///
    /// # Arguments
    ///
    /// * `other` - The other parent genotype
    /// * `rng` - Random number generator for stochastic crossover
    ///
    /// # Returns
    ///
    /// A new genotype combining genetic material from both parents
    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self;
}

/// A phenotype represents an evaluated individual with fitness information.
///
/// The phenotype contains both the genetic encoding ([`genotype`](Phenotype::genotype))
/// and the results of fitness evaluation.
///
/// # Fields
///
/// * `genotype` - The genetic encoding
/// * `fitness` - Single scalar fitness value (for single-objective algorithms)
/// * `objectives` - Multiple objective values (for multi-objective algorithms like NSGA-II)
/// * `descriptor` - Behavioral descriptor (for quality-diversity algorithms like MAP-Elites)
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "G: Genotype")]
pub struct Phenotype<G: Genotype> {
    /// The genetic encoding of this individual.
    pub genotype: G,
    /// Scalar fitness value. Higher is better.
    pub fitness: f32,
    /// Objective values for multi-objective optimization. Higher is better for each.
    pub objectives: Vec<f32>,
    /// Behavioral descriptor for quality-diversity algorithms.
    /// Values should be normalized to `[0.0, 1.0]` for MAP-Elites.
    pub descriptor: Vec<f32>,
}

/// Evaluates genotypes to produce fitness scores.
///
/// Implement this trait to define your optimization problem's fitness function.
///
/// # Thread Safety
///
/// Evaluators must be [`Send`] and [`Sync`] to support parallel evaluation.
/// If your evaluator has mutable state, wrap it in appropriate synchronization
/// primitives (e.g., `Arc<Mutex<T>>`).
///
/// # Example
///
/// ```rust
/// use symbios_genetics::{Evaluator, Genotype};
/// use serde::{Deserialize, Serialize};
/// use rand::Rng;
///
/// #[derive(Clone, Serialize, Deserialize)]
/// struct Point(f32, f32);
///
/// impl Genotype for Point {
///     fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
///         if rng.random::<f32>() < rate {
///             self.0 += rng.random::<f32>() - 0.5;
///             self.1 += rng.random::<f32>() - 0.5;
///         }
///     }
///     fn crossover<R: Rng>(&self, other: &Self, _rng: &mut R) -> Self {
///         Point((self.0 + other.0) / 2.0, (self.1 + other.1) / 2.0)
///     }
/// }
///
/// // Minimize distance to origin
/// struct DistanceToOrigin;
///
/// impl Evaluator<Point> for DistanceToOrigin {
///     fn evaluate(&self, g: &Point) -> (f32, Vec<f32>, Vec<f32>) {
///         let dist = (g.0.powi(2) + g.1.powi(2)).sqrt();
///         let fitness = -dist; // Negate because higher fitness is better
///         (fitness, vec![fitness], vec![])
///     }
/// }
/// ```
pub trait Evaluator<G: Genotype>: Send + Sync {
    /// Evaluate a genotype and return fitness information.
    ///
    /// # Arguments
    ///
    /// * `genotype` - The genotype to evaluate
    ///
    /// # Returns
    ///
    /// A tuple of `(fitness, objectives, descriptor)`:
    /// - `fitness`: Scalar fitness value (higher is better)
    /// - `objectives`: Vector of objective values for multi-objective optimization
    /// - `descriptor`: Behavioral descriptor for quality-diversity (values in `[0.0, 1.0]`)
    fn evaluate(&self, genotype: &G) -> (f32, Vec<f32>, Vec<f32>);
}

/// The core evolutionary algorithm trait.
///
/// All evolutionary algorithms implement this trait, providing a uniform
/// interface for running evolution steps and accessing the population.
///
/// # Example
///
/// ```rust,ignore
/// use symbios_genetics::{Evolver, Evaluator};
///
/// fn run_evolution<G, E, Ev>(mut evolver: Ev, evaluator: &E, generations: usize)
/// where
///     G: Genotype,
///     E: Evaluator<G>,
///     Ev: Evolver<G>,
/// {
///     for _ in 0..generations {
///         evolver.step(evaluator);
///     }
///     let best = evolver.population()
///         .iter()
///         .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
///     println!("Best fitness: {:?}", best.map(|p| p.fitness));
/// }
/// ```
pub trait Evolver<G: Genotype> {
    /// Perform one generation of evolution.
    ///
    /// This typically involves:
    /// 1. Evaluating the current population
    /// 2. Selecting parents
    /// 3. Creating offspring via crossover and mutation
    /// 4. Selecting survivors for the next generation
    ///
    /// # Arguments
    ///
    /// * `evaluator` - The fitness evaluator
    fn step<E: Evaluator<G>>(&mut self, evaluator: &E);

    /// Get a reference to the current population.
    ///
    /// # Returns
    ///
    /// A slice of all phenotypes in the current population
    fn population(&mut self) -> &[Phenotype<G>];
}

/// Evolutionary algorithm implementations.
///
/// This module contains three evolutionary algorithms:
///
/// - [`simple::SimpleGA`](algorithms::simple::SimpleGA) - A simple generational genetic algorithm with elitism
/// - [`nsga2::Nsga2`](algorithms::nsga2::Nsga2) - NSGA-II for multi-objective optimization
/// - [`map_elites::MapElites`](algorithms::map_elites::MapElites) - MAP-Elites for quality-diversity optimization
pub mod algorithms {
    /// Simple generational genetic algorithm.
    pub mod map_elites;
    /// NSGA-II multi-objective evolutionary algorithm.
    pub mod nsga2;
    /// Simple genetic algorithm with elitism.
    pub mod simple;
}
