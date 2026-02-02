use crate::{Evaluator, Evolver, Genotype, Phenotype};
use rand::prelude::{IndexedRandom, SeedableRng};
use rand_pcg::Pcg64; // Specific, serializable generator
use serde::{Deserialize, Serialize};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

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

        self.population
            .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        let mut next_gen = self.population[..self.elitism].to_vec();

        while next_gen.len() < self.pop_size {
            let p_a = self
                .population
                .choose_multiple(&mut self.rng, 3)
                .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
                .unwrap();
            let p_b = self
                .population
                .choose_multiple(&mut self.rng, 3)
                .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
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
