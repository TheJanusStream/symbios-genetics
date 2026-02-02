use crate::{Evaluator, Evolver, Genotype, Phenotype};
use rand::prelude::SeedableRng;
use rand_pcg::Pcg64; // Specific, serializable generator
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Serialize, Deserialize)]
#[serde(bound = "G: Genotype")]
pub struct Nsga2<G: Genotype> {
    population: Vec<Phenotype<G>>,
    /// Stored rank for each individual (lower is better)
    ranks: Vec<usize>,
    /// Stored crowding distance for each individual (higher is better)
    crowding_distances: Vec<f32>,
    pop_size: usize,
    mutation_rate: f32,
    rng: Pcg64,
}

impl<G: Genotype> Nsga2<G> {
    pub fn pop_size(&self) -> usize {
        self.pop_size
    }

    pub fn mutation_rate(&self) -> f32 {
        self.mutation_rate
    }

    pub fn set_mutation_rate(&mut self, rate: f32) {
        self.mutation_rate = rate;
    }
}

#[derive(Clone)]
pub struct SortWrapper<G: Genotype> {
    pub pheno: Phenotype<G>,
    pub rank: usize,
    pub distance: f32,
}

impl<G: Genotype> Nsga2<G> {
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

        // Initialize with default ranks (all equal) and infinite crowding distance
        let ranks = vec![0; population.len()];
        let crowding_distances = vec![f32::INFINITY; population.len()];

        Self {
            population,
            ranks,
            crowding_distances,
            pop_size,
            mutation_rate,
            rng: Pcg64::seed_from_u64(seed),
        }
    }

    /// Binary tournament selection: picks 2 random individuals and returns
    /// the one with better rank (lower), or better crowding distance (higher) if tied.
    fn binary_tournament(&mut self) -> usize {
        use rand::Rng;
        let n = self.population.len();
        let i = self.rng.random_range(0..n);
        let j = self.rng.random_range(0..n);

        // Compare by rank first (lower is better)
        match self.ranks[i].cmp(&self.ranks[j]) {
            Ordering::Less => i,
            Ordering::Greater => j,
            Ordering::Equal => {
                // Tie-break by crowding distance (higher is better)
                if self.crowding_distances[i] >= self.crowding_distances[j] {
                    i
                } else {
                    j
                }
            }
        }
    }

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

    pub fn calculate_crowding_distance(front: &mut [SortWrapper<G>]) {
        let n = front.len();
        if n <= 2 {
            for ind in front {
                ind.distance = f32::INFINITY;
            }
            return;
        }

        // Check for consistent objective counts
        let min_obj = front
            .iter()
            .map(|w| w.pheno.objectives.len())
            .min()
            .unwrap_or(0);
        let max_obj = front
            .iter()
            .map(|w| w.pheno.objectives.len())
            .max()
            .unwrap_or(0);

        if min_obj == 0 {
            // No objectives: all get infinite distance (equally preferred)
            for ind in front {
                ind.distance = f32::INFINITY;
            }
            return;
        }

        if min_obj != max_obj {
            // Ragged objectives: individuals are incomparable for crowding
            // Give all infinite distance so selection falls back to rank only
            for ind in front {
                ind.distance = f32::INFINITY;
            }
            return;
        }

        let obj_count = min_obj;

        for m in 0..obj_count {
            front.sort_by(|a, b| {
                a.pheno.objectives[m]
                    .partial_cmp(&b.pheno.objectives[m])
                    .unwrap_or(Ordering::Equal)
            });
            let range = front[n - 1].pheno.objectives[m] - front[0].pheno.objectives[m];
            front[0].distance = f32::INFINITY;
            front[n - 1].distance = f32::INFINITY;
            if range > 0.0 {
                for i in 1..(n - 1) {
                    if front[i].distance != f32::INFINITY {
                        front[i].distance += (front[i + 1].pheno.objectives[m]
                            - front[i - 1].pheno.objectives[m])
                            / range;
                    }
                }
            }
        }
    }

    pub fn dominates(a: &Phenotype<G>, b: &Phenotype<G>) -> bool {
        // Incomparable if objective counts differ
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
        let mut next_gen = vec![];
        for (rank, indices) in fronts.iter().enumerate() {
            let mut current_front: Vec<_> = indices
                .iter()
                .map(|&i| SortWrapper {
                    pheno: combined[i].clone(),
                    rank,
                    distance: 0.0,
                })
                .collect();
            Self::calculate_crowding_distance(&mut current_front);
            if next_gen.len() + current_front.len() <= self.pop_size {
                next_gen.extend(current_front);
            } else {
                current_front.sort_by(|a, b| {
                    b.distance
                        .partial_cmp(&a.distance)
                        .unwrap_or(Ordering::Equal)
                });
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
        self.population = next_gen.into_iter().map(|w| w.pheno).collect();
    }
    fn population(&mut self) -> &[Phenotype<G>] {
        &self.population
    }
}
