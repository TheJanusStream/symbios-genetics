use crate::{Evaluator, Evolver, Genotype, Phenotype};
use rand::prelude::IndexedRandom;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub struct SimpleGA<G: Genotype> {
    population: Vec<Phenotype<G>>,
    pop_size: usize,
    mutation_rate: f32,
    elitism: usize,
}

impl<G: Genotype> SimpleGA<G> {
    pub fn new(initial_pop: Vec<G>, mutation_rate: f32, elitism: usize) -> Self {
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
        }
    }
}

impl<G: Genotype> Evolver<G> for SimpleGA<G> {
    fn step<E: Evaluator<G>>(&mut self, evaluator: &E) {
        // 1. Parallel Evaluation
        #[cfg(feature = "parallel")]
        {
            self.population.par_iter_mut().for_each(|p| {
                let (f, obj, desc) = evaluator.evaluate(&p.genotype);
                p.fitness = f;
                p.objectives = obj;
                p.descriptor = desc;
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for p in &mut self.population {
                let (f, obj, desc) = evaluator.evaluate(&p.genotype);
                p.fitness = f;
                p.objectives = obj;
                p.descriptor = desc;
            }
        }

        self.population
            .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        let mut next_gen = self.population[..self.elitism].to_vec();
        let mut rng = rand::rng();

        while next_gen.len() < self.pop_size {
            let parent_a = self
                .population
                .choose_multiple(&mut rng, 3)
                .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
                .unwrap();
            let parent_b = self
                .population
                .choose_multiple(&mut rng, 3)
                .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
                .unwrap();

            let mut child_dna = parent_a.genotype.crossover(&parent_b.genotype, &mut rng);
            child_dna.mutate(&mut rng, self.mutation_rate);

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
