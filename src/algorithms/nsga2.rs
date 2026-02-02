//! NSGA-II multi-objective evolutionary algorithm.
//!
//! NSGA-II (Non-dominated Sorting Genetic Algorithm II) is a popular algorithm
//! for multi-objective optimization that finds a set of Pareto-optimal solutions.
//!
//! # Overview
//!
//! In multi-objective optimization, there is no single "best" solution because
//! objectives may conflict. Instead, NSGA-II finds the *Pareto front*: solutions
//! where no objective can be improved without worsening another.
//!
//! # Key Concepts
//!
//! - **Pareto Dominance**: Solution A *dominates* B if A is at least as good in
//!   all objectives and strictly better in at least one
//! - **Pareto Front**: The set of non-dominated solutions (rank 0)
//! - **Crowding Distance**: Measures solution diversity; higher means more isolated
//!
//! # Example
//!
//! ```rust
//! use rand::Rng;
//! use serde::{Deserialize, Serialize};
//! use symbios_genetics::{Evaluator, Evolver, Genotype, algorithms::nsga2::Nsga2};
//!
//! #[derive(Clone, Serialize, Deserialize)]
//! struct Design { weight: f32, cost: f32 }
//!
//! impl Genotype for Design {
//!     fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
//!         if rng.random::<f32>() < rate {
//!             self.weight += rng.random::<f32>() - 0.5;
//!             self.cost += rng.random::<f32>() - 0.5;
//!         }
//!     }
//!     fn crossover<R: Rng>(&self, other: &Self, _rng: &mut R) -> Self {
//!         Design {
//!             weight: (self.weight + other.weight) / 2.0,
//!             cost: (self.cost + other.cost) / 2.0,
//!         }
//!     }
//! }
//!
//! // Minimize weight, minimize cost (negate for maximization)
//! struct MultiObjective;
//! impl Evaluator<Design> for MultiObjective {
//!     fn evaluate(&self, g: &Design) -> (f32, Vec<f32>, Vec<f32>) {
//!         let obj1 = -g.weight; // Minimize weight
//!         let obj2 = -g.cost;   // Minimize cost
//!         (obj1 + obj2, vec![obj1, obj2], vec![])
//!     }
//! }
//!
//! let initial: Vec<Design> = (0..50)
//!     .map(|i| Design { weight: i as f32, cost: 50.0 - i as f32 })
//!     .collect();
//!
//! let mut nsga = Nsga2::new(initial, 0.1, 42);
//! for _ in 0..100 {
//!     nsga.step(&MultiObjective);
//! }
//!
//! // Population now approximates the Pareto front
//! ```
//!
//! # Algorithm Details
//!
//! Each generation:
//! 1. Creates offspring via binary tournament selection, crossover, and mutation
//! 2. Combines parents and offspring (2N individuals)
//! 3. Performs fast non-dominated sorting to rank solutions
//! 4. Calculates crowding distance within each front
//! 5. Selects best N individuals by rank, then by crowding distance
//!
//! # References
//!
//! Deb, K., et al. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II.
//! IEEE Transactions on Evolutionary Computation, 6(2), 182-197.

use crate::{Evaluator, Evolver, Genotype, Phenotype};
use rand::prelude::SeedableRng;
use rand_pcg::Pcg64;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Internal representation for serialization (excludes derived state).
#[derive(Serialize, Deserialize)]
#[serde(bound = "G: Genotype")]
struct Nsga2Data<G: Genotype> {
    population: Vec<Phenotype<G>>,
    ranks: Vec<usize>,
    crowding_distances: Vec<f32>,
    pop_size: usize,
    mutation_rate: f32,
    rng: Pcg64,
}

/// NSGA-II multi-objective evolutionary algorithm.
///
/// Maintains a population that evolves toward the Pareto front of a
/// multi-objective optimization problem.
///
/// # Type Parameters
///
/// * `G` - The genotype type, must implement [`Genotype`]
///
/// # Selection Mechanism
///
/// NSGA-II uses binary tournament selection based on:
/// 1. **Rank** (lower is better): Solutions on the Pareto front have rank 0
/// 2. **Crowding distance** (higher is better): Tie-breaker that favors diversity
pub struct Nsga2<G: Genotype> {
    population: Vec<Phenotype<G>>,
    /// Pareto rank for each individual (0 = non-dominated front).
    ranks: Vec<usize>,
    /// Crowding distance for diversity preservation (higher = more isolated).
    crowding_distances: Vec<f32>,
    pop_size: usize,
    mutation_rate: f32,
    rng: Pcg64,
}

impl<G: Genotype> Serialize for Nsga2<G> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Nsga2", 6)?;
        state.serialize_field("population", &self.population)?;
        state.serialize_field("ranks", &self.ranks)?;
        state.serialize_field("crowding_distances", &self.crowding_distances)?;
        state.serialize_field("pop_size", &self.pop_size)?;
        state.serialize_field("mutation_rate", &self.mutation_rate)?;
        state.serialize_field("rng", &self.rng)?;
        state.end()
    }
}

impl<'de, G: Genotype> Deserialize<'de> for Nsga2<G> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;

        let data = Nsga2Data::<G>::deserialize(deserializer)?;

        // Validate population is not empty (empty is handled gracefully but pop_size would be wrong)
        if data.pop_size == 0 {
            return Err(D::Error::custom("pop_size must be greater than 0"));
        }

        // Validate population length matches pop_size
        if data.population.len() != data.pop_size {
            return Err(D::Error::custom(format!(
                "population length ({}) does not match pop_size ({})",
                data.population.len(),
                data.pop_size
            )));
        }

        // Validate ranks length matches population length
        // This prevents index out of bounds in binary_tournament
        if data.ranks.len() != data.population.len() {
            return Err(D::Error::custom(format!(
                "ranks length ({}) does not match population length ({})",
                data.ranks.len(),
                data.population.len()
            )));
        }

        // Validate crowding_distances length matches population length
        // This prevents index out of bounds in binary_tournament
        if data.crowding_distances.len() != data.population.len() {
            return Err(D::Error::custom(format!(
                "crowding_distances length ({}) does not match population length ({})",
                data.crowding_distances.len(),
                data.population.len()
            )));
        }

        // Validate mutation_rate is not NaN or infinite
        if data.mutation_rate.is_nan() || data.mutation_rate.is_infinite() {
            return Err(D::Error::custom("mutation_rate must be a finite number"));
        }

        Ok(Self {
            population: data.population,
            ranks: data.ranks,
            crowding_distances: data.crowding_distances,
            pop_size: data.pop_size,
            mutation_rate: data.mutation_rate,
            rng: data.rng,
        })
    }
}

impl<G: Genotype> Nsga2<G> {
    /// Returns the population size.
    pub fn pop_size(&self) -> usize {
        self.pop_size
    }

    /// Returns the current mutation rate.
    pub fn mutation_rate(&self) -> f32 {
        self.mutation_rate
    }

    /// Sets the mutation rate.
    ///
    /// # Arguments
    ///
    /// * `rate` - New mutation rate, typically in `[0.0, 1.0]`
    pub fn set_mutation_rate(&mut self, rate: f32) {
        self.mutation_rate = rate;
    }
}

/// Internal wrapper for sorting individuals by rank and crowding distance.
/// Uses index into combined population to avoid cloning phenotypes during sorting.
#[derive(Clone, Copy)]
pub struct SortWrapper {
    /// Index into the combined population vector.
    pub index: usize,
    /// Pareto rank (0 = non-dominated front).
    pub rank: usize,
    /// Crowding distance for diversity.
    pub distance: f32,
}

impl<G: Genotype> Nsga2<G> {
    /// Creates a new NSGA-II instance.
    ///
    /// # Arguments
    ///
    /// * `initial_pop` - Initial population of genotypes
    /// * `mutation_rate` - Probability of mutation, typically in `[0.0, 1.0]`
    /// * `seed` - RNG seed for deterministic execution
    ///
    /// # Note
    ///
    /// The initial population is created with placeholder fitness values.
    /// Ranks and crowding distances are calculated based on the initial
    /// (unevaluated) state to ensure consistency. Call `step()` with an
    /// evaluator to compute meaningful ranks based on actual objectives.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let initial: Vec<MyGenome> = (0..100).map(|_| MyGenome::random()).collect();
    /// let mut nsga = Nsga2::new(initial, 0.1, 42);
    /// ```
    pub fn new(initial_pop: Vec<G>, mutation_rate: f32, seed: u64) -> Self {
        let pop_size = initial_pop.len();
        let population: Vec<_> = initial_pop
            .into_iter()
            .map(|g| Phenotype {
                genotype: g,
                fitness: 0.0,
                objectives: vec![],
                descriptor: vec![],
            })
            .collect();

        // Calculate initial ranks and crowding distances for consistency
        // With empty objectives, all individuals are non-dominated (rank 0)
        let (ranks, crowding_distances) = Self::calculate_ranks_and_distances(&population);

        Self {
            population,
            ranks,
            crowding_distances,
            pop_size,
            mutation_rate,
            rng: Pcg64::seed_from_u64(seed),
        }
    }

    /// Calculates ranks and crowding distances for a population.
    ///
    /// This is used both during initialization and could be used to
    /// inspect the current Pareto structure without running a full step.
    fn calculate_ranks_and_distances(population: &[Phenotype<G>]) -> (Vec<usize>, Vec<f32>) {
        if population.is_empty() {
            return (vec![], vec![]);
        }

        let fronts = Self::fast_non_dominated_sort(population);
        let mut ranks = vec![0; population.len()];
        let mut crowding_distances = vec![0.0; population.len()];

        for (rank, indices) in fronts.iter().enumerate() {
            let mut front_wrappers: Vec<_> = indices
                .iter()
                .map(|&i| SortWrapper {
                    index: i,
                    rank,
                    distance: 0.0,
                })
                .collect();

            Self::calculate_crowding_distance(&mut front_wrappers, population);

            for wrapper in front_wrappers {
                ranks[wrapper.index] = rank;
                crowding_distances[wrapper.index] = wrapper.distance;
            }
        }

        (ranks, crowding_distances)
    }

    /// Binary tournament selection.
    ///
    /// Picks 2 random individuals and returns the index of the better one.
    /// Comparison is by rank (lower is better), then crowding distance (higher is better).
    fn binary_tournament(&mut self) -> usize {
        use rand::Rng;
        let n = self.population.len();
        let i = self.rng.random_range(0..n);
        let j = self.rng.random_range(0..n);

        match self.ranks[i].cmp(&self.ranks[j]) {
            Ordering::Less => i,
            Ordering::Greater => j,
            Ordering::Equal => {
                // Use total_cmp for NaN-safe comparison (NaN is treated as greater than all values)
                // Higher crowding distance is better, so we want the one with greater distance
                match self.crowding_distances[i].total_cmp(&self.crowding_distances[j]) {
                    Ordering::Greater | Ordering::Equal => i,
                    Ordering::Less => j,
                }
            }
        }
    }

    /// Performs fast non-dominated sorting.
    ///
    /// Partitions the population into Pareto fronts where front 0 contains
    /// non-dominated solutions, front 1 contains solutions dominated only by
    /// front 0, and so on.
    ///
    /// # Arguments
    ///
    /// * `combined` - Slice of phenotypes to sort
    ///
    /// # Returns
    ///
    /// Vector of fronts, where each front is a vector of indices into `combined`.
    ///
    /// # Complexity
    ///
    /// O(MN²) where M is the number of objectives and N is the population size.
    pub fn fast_non_dominated_sort(combined: &[Phenotype<G>]) -> Vec<Vec<usize>> {
        let n = combined.len();
        let mut fronts = vec![vec![]];
        let mut domination_count = vec![0; n];
        let mut dominated_indices = vec![vec![]; n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                if Self::dominates(&combined[i], &combined[j]) {
                    dominated_indices[i].push(j);
                } else if Self::dominates(&combined[j], &combined[i]) {
                    domination_count[i] += 1;
                }
            }
            if domination_count[i] == 0 {
                fronts[0].push(i);
            }
        }

        let mut curr = 0;
        while curr < fronts.len() && !fronts[curr].is_empty() {
            let mut next_front = vec![];
            for &i in &fronts[curr] {
                for &j in &dominated_indices[i] {
                    domination_count[j] -= 1;
                    if domination_count[j] == 0 {
                        next_front.push(j);
                    }
                }
            }
            if next_front.is_empty() {
                break;
            }
            fronts.push(next_front);
            curr += 1;
        }
        fronts
    }

    /// Calculates crowding distance for a Pareto front.
    ///
    /// Crowding distance measures how isolated a solution is in objective space.
    /// Solutions at the boundaries get infinite distance; interior solutions get
    /// distance based on their neighbors.
    ///
    /// # Arguments
    ///
    /// * `front` - Mutable slice of sort wrappers (indices) in the same Pareto front
    /// * `combined` - Reference to the combined population for accessing objectives
    ///
    /// # Algorithm
    ///
    /// For each objective:
    /// 1. Sort the front by that objective
    /// 2. Assign infinite distance to boundary solutions
    /// 3. Add normalized neighbor distance to interior solutions
    pub fn calculate_crowding_distance(front: &mut [SortWrapper], combined: &[Phenotype<G>]) {
        let n = front.len();
        if n <= 2 {
            for ind in front {
                ind.distance = f32::INFINITY;
            }
            return;
        }

        let min_obj = front
            .iter()
            .map(|w| combined[w.index].objectives.len())
            .min()
            .unwrap_or(0);
        let max_obj = front
            .iter()
            .map(|w| combined[w.index].objectives.len())
            .max()
            .unwrap_or(0);

        if min_obj == 0 {
            for ind in front {
                ind.distance = f32::INFINITY;
            }
            return;
        }

        if min_obj != max_obj {
            for ind in front {
                ind.distance = f32::INFINITY;
            }
            return;
        }

        let obj_count = min_obj;

        for m in 0..obj_count {
            // Use total_cmp for NaN-safe sorting (NaN sorts to end)
            front.sort_by(|a, b| {
                combined[a.index].objectives[m].total_cmp(&combined[b.index].objectives[m])
            });
            let range =
                combined[front[n - 1].index].objectives[m] - combined[front[0].index].objectives[m];
            front[0].distance = f32::INFINITY;
            front[n - 1].distance = f32::INFINITY;
            if range > 0.0 {
                for i in 1..(n - 1) {
                    if front[i].distance != f32::INFINITY {
                        front[i].distance += (combined[front[i + 1].index].objectives[m]
                            - combined[front[i - 1].index].objectives[m])
                            / range;
                    }
                }
            }
        }
    }

    /// Tests if solution `a` Pareto-dominates solution `b`.
    ///
    /// Solution `a` dominates `b` if:
    /// - `a` is at least as good as `b` in all objectives
    /// - `a` is strictly better than `b` in at least one objective
    ///
    /// # Arguments
    ///
    /// * `a` - First phenotype
    /// * `b` - Second phenotype
    ///
    /// # Returns
    ///
    /// `true` if `a` dominates `b`, `false` otherwise.
    /// Returns `false` if objective counts differ (incomparable).
    ///
    /// # Example
    ///
    /// ```text
    /// a = [3, 2], b = [2, 1] -> a dominates b (better in both)
    /// a = [3, 1], b = [2, 2] -> neither dominates (trade-off)
    /// ```
    pub fn dominates(a: &Phenotype<G>, b: &Phenotype<G>) -> bool {
        if a.objectives.len() != b.objectives.len() {
            return false;
        }
        let mut better_in_any = false;
        for (oa, ob) in a.objectives.iter().zip(b.objectives.iter()) {
            if oa < ob {
                return false;
            }
            if oa > ob {
                better_in_any = true;
            }
        }
        better_in_any
    }
}

impl<G: Genotype> Evolver<G> for Nsga2<G> {
    fn step<E: Evaluator<G>>(&mut self, evaluator: &E) {
        if self.population.is_empty() {
            return;
        }

        // Generate offspring using binary tournament selection
        let mut offspring = vec![];
        while offspring.len() < self.pop_size {
            let idx1 = self.binary_tournament();
            let idx2 = self.binary_tournament();
            let p1 = &self.population[idx1];
            let p2 = &self.population[idx2];
            let mut child_dna = p1.genotype.crossover(&p2.genotype, &mut self.rng);
            child_dna.mutate(&mut self.rng, self.mutation_rate);
            offspring.push(Phenotype {
                genotype: child_dna,
                fitness: 0.0,
                objectives: vec![],
                descriptor: vec![],
            });
        }

        let mut combined = self.population.clone();
        combined.extend(offspring);

        #[cfg(feature = "parallel")]
        combined.par_iter_mut().for_each(|p| {
            let (fit, obj, desc) = evaluator.evaluate(&p.genotype);
            p.fitness = fit;
            p.objectives = obj;
            p.descriptor = desc;
        });
        #[cfg(not(feature = "parallel"))]
        for p in &mut combined {
            let (fit, obj, desc) = evaluator.evaluate(&p.genotype);
            p.fitness = fit;
            p.objectives = obj;
            p.descriptor = desc;
        }

        let fronts = Self::fast_non_dominated_sort(&combined);
        let mut next_gen: Vec<SortWrapper> = vec![];
        for (rank, indices) in fronts.iter().enumerate() {
            let mut current_front: Vec<_> = indices
                .iter()
                .map(|&i| SortWrapper {
                    index: i,
                    rank,
                    distance: 0.0,
                })
                .collect();
            Self::calculate_crowding_distance(&mut current_front, &combined);
            if next_gen.len() + current_front.len() <= self.pop_size {
                next_gen.extend(current_front);
            } else {
                // Use total_cmp for NaN-safe sorting (higher distance is better, so reverse order)
                current_front.sort_by(|a, b| b.distance.total_cmp(&a.distance));
                next_gen.extend(
                    current_front
                        .into_iter()
                        .take(self.pop_size - next_gen.len()),
                );
                break;
            }
        }

        // Store ranks and crowding distances for next generation's tournament selection
        self.ranks = next_gen.iter().map(|w| w.rank).collect();
        self.crowding_distances = next_gen.iter().map(|w| w.distance).collect();
        // Only clone phenotypes when building the final population
        self.population = next_gen
            .into_iter()
            .map(|w| combined[w.index].clone())
            .collect();
    }
    fn population(&mut self) -> &[Phenotype<G>] {
        &self.population
    }
}
