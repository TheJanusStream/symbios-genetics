//! Simple generational genetic algorithm with elitism.
//!
//! A straightforward genetic algorithm for single-objective optimization.
//! Uses tournament selection, crossover, mutation, and elitism.
//!
//! # Overview
//!
//! SimpleGA is ideal for single-objective optimization problems where you want
//! to maximize a scalar fitness value. It's simple to use and efficient for
//! many common optimization tasks.
//!
//! # Features
//!
//! - **Elitism**: Top individuals survive unchanged to the next generation
//! - **Tournament Selection**: Size-3 tournaments balance exploration and exploitation
//! - **Parallel Evaluation**: Optional parallel fitness evaluation via Rayon
//! - **NaN Handling**: Gracefully handles NaN fitness values
//!
//! # Example
//!
//! ```rust
//! use rand::Rng;
//! use serde::{Deserialize, Serialize};
//! use symbios_genetics::{Evaluator, Evolver, Genotype, algorithms::simple::SimpleGA};
//!
//! #[derive(Clone, Serialize, Deserialize)]
//! struct FloatVec(Vec<f32>);
//!
//! impl Genotype for FloatVec {
//!     fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
//!         for v in &mut self.0 {
//!             if rng.random::<f32>() < rate {
//!                 *v += rng.random::<f32>() - 0.5;
//!             }
//!         }
//!     }
//!     fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
//!         let point = rng.random_range(0..self.0.len());
//!         let mut child = self.0[..point].to_vec();
//!         child.extend_from_slice(&other.0[point..]);
//!         FloatVec(child)
//!     }
//! }
//!
//! // Maximize sum of values
//! struct SumFitness;
//! impl Evaluator<FloatVec> for SumFitness {
//!     fn evaluate(&self, g: &FloatVec) -> (f32, Vec<f32>, Vec<f32>) {
//!         let sum: f32 = g.0.iter().sum();
//!         (sum, vec![sum], vec![])
//!     }
//! }
//!
//! let initial: Vec<FloatVec> = (0..50)
//!     .map(|_| FloatVec(vec![0.0; 10]))
//!     .collect();
//!
//! let mut ga = SimpleGA::new(initial, 0.1, 5, 42);
//!
//! for _ in 0..100 {
//!     ga.step(&SumFitness);
//! }
//!
//! let best = ga.population().first().unwrap();
//! println!("Best fitness: {}", best.fitness);
//! ```
//!
//! # Algorithm Details
//!
//! Each generation:
//! 1. Evaluate all individuals
//! 2. Sort by fitness (descending)
//! 3. Copy top `elitism` individuals to next generation
//! 4. Fill remaining slots via tournament selection, crossover, and mutation

use crate::{Evaluator, Evolver, Genotype, Phenotype};
use rand::prelude::{IndexedRandom, SeedableRng};
use rand_pcg::Pcg64;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Compares two f32 values, treating NaN as less than all other values.
/// This ensures NaN fitness individuals sort to the end (lowest priority).
fn cmp_f32_nan_last(a: f32, b: f32) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
    }
}

/// Simple generational genetic algorithm with elitism.
///
/// A classic genetic algorithm that maintains a fixed-size population and
/// uses tournament selection for parent selection.
///
/// # Type Parameters
///
/// * `G` - The genotype type, must implement [`Genotype`]
///
/// # Selection Mechanism
///
/// Uses tournament selection with size 3 (or smaller for tiny populations).
/// The winner of each tournament is selected as a parent.
///
/// # Elitism
///
/// The top `elitism` individuals (by fitness) are copied unchanged to the
/// next generation, ensuring the best solutions are never lost.
#[derive(Serialize, Deserialize)]
#[serde(bound = "G: Genotype")]
pub struct SimpleGA<G: Genotype> {
    population: Vec<Phenotype<G>>,
    pop_size: usize,
    mutation_rate: f32,
    elitism: usize,
    rng: Pcg64,
}

impl<G: Genotype> SimpleGA<G> {
    /// Creates a new SimpleGA instance.
    ///
    /// # Arguments
    ///
    /// * `initial_pop` - Initial population of genotypes
    /// * `mutation_rate` - Probability of mutation, typically in `[0.0, 1.0]`
    /// * `elitism` - Number of top individuals to preserve each generation
    /// * `seed` - RNG seed for deterministic execution
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let initial: Vec<MyGenome> = (0..100).map(|_| MyGenome::random()).collect();
    /// let mut ga = SimpleGA::new(initial, 0.1, 5, 42);
    /// ```
    pub fn new(initial_pop: Vec<G>, mutation_rate: f32, elitism: usize, seed: u64) -> Self {
        let pop_size = initial_pop.len();

        // Warn if elitism configuration may cause unexpected behavior
        if elitism >= pop_size && pop_size > 0 {
            eprintln!(
                "Warning: SimpleGA elitism ({}) >= pop_size ({}). \
                 Elitism will be clamped to {} to ensure evolution progresses.",
                elitism,
                pop_size,
                pop_size.saturating_sub(1)
            );
        }

        let population = initial_pop
            .into_iter()
            .map(|g| Phenotype {
                genotype: g,
                fitness: 0.0,
                objectives: vec![],
                descriptor: vec![],
            })
            .collect();

        Self {
            population,
            pop_size,
            mutation_rate,
            elitism,
            rng: Pcg64::seed_from_u64(seed),
        }
    }

    /// Returns the population size.
    pub fn pop_size(&self) -> usize {
        self.pop_size
    }

    /// Returns the current mutation rate.
    pub fn mutation_rate(&self) -> f32 {
        self.mutation_rate
    }

    /// Returns the elitism count.
    pub fn elitism(&self) -> usize {
        self.elitism
    }
}
impl<G: Genotype> Evolver<G> for SimpleGA<G> {
    fn step<E: Evaluator<G>>(&mut self, evaluator: &E) {
        if self.population.is_empty() {
            return;
        }

        #[cfg(feature = "parallel")]
        self.population.par_iter_mut().for_each(|p| {
            let (f, obj, desc) = evaluator.evaluate(&p.genotype);
            p.fitness = f;
            p.objectives = obj;
            p.descriptor = desc;
        });
        #[cfg(not(feature = "parallel"))]
        for p in &mut self.population {
            let (f, obj, desc) = evaluator.evaluate(&p.genotype);
            p.fitness = f;
            p.objectives = obj;
            p.descriptor = desc;
        }

        // Sort by fitness descending, with NaN values pushed to the end
        self.population
            .sort_by(|a, b| cmp_f32_nan_last(b.fitness, a.fitness));

        // Clamp elitism to pop_size - 1 to ensure at least one offspring is generated
        // This prevents evolution from halting when elitism >= pop_size
        let effective_elitism = self.elitism.min(self.pop_size.saturating_sub(1));
        let mut next_gen = self.population[..effective_elitism].to_vec();

        // Tournament selection with graceful handling of small populations
        let tournament_size = 3.min(self.population.len());
        while next_gen.len() < self.pop_size {
            let p_a = self
                .population
                .choose_multiple(&mut self.rng, tournament_size)
                .max_by(|a, b| cmp_f32_nan_last(a.fitness, b.fitness))
                .unwrap();
            let p_b = self
                .population
                .choose_multiple(&mut self.rng, tournament_size)
                .max_by(|a, b| cmp_f32_nan_last(a.fitness, b.fitness))
                .unwrap();
            let mut child_dna = p_a.genotype.crossover(&p_b.genotype, &mut self.rng);
            child_dna.mutate(&mut self.rng, self.mutation_rate);
            next_gen.push(Phenotype {
                genotype: child_dna,
                fitness: 0.0,
                objectives: vec![],
                descriptor: vec![],
            });
        }
        self.population = next_gen;
    }
    fn population(&mut self) -> &[Phenotype<G>] {
        &self.population
    }
}
