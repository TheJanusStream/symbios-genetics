use crate::{Evaluator, Evolver, Genotype, Phenotype};
use rand::prelude::{IndexedRandom, SeedableRng};
use rand_pcg::Pcg64; // Specific, serializable generator
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Compare two f32 values, treating NaN as less than all other values.
/// This ensures NaN fitness individuals sort to the end (lowest priority).
fn cmp_f32_nan_last(a: f32, b: f32) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
    }
}

#[derive(Serialize, Deserialize)]
#[serde(bound = "G: Genotype")]
pub struct SimpleGA<G: Genotype> {
    pub population: Vec<Phenotype<G>>,
    pub pop_size: usize,
    pub mutation_rate: f32,
    pub elitism: usize,
    rng: Pcg64, // Now satisfies the Serialize/Deserialize bound
}

impl<G: Genotype> SimpleGA<G> {
    pub fn new(initial_pop: Vec<G>, mutation_rate: f32, elitism: usize, seed: u64) -> Self {
        let pop_size = initial_pop.len();
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

        // Clamp elitism to population size to prevent slice overflow
        let effective_elitism = self.elitism.min(self.pop_size);
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
    fn population(&self) -> &[Phenotype<G>] {
        &self.population
    }
}
