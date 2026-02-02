use crate::{Evaluator, Genotype, Phenotype};
use rand::prelude::IndexedRandom;
use std::collections::HashMap;

pub struct MapElites<G: Genotype> {
    pub archive: HashMap<Vec<usize>, Phenotype<G>>,
    resolution: usize,
    mutation_rate: f32,
}

impl<G: Genotype> MapElites<G> {
    pub fn new(_dims: usize, resolution: usize, mutation_rate: f32) -> Self {
        Self {
            archive: HashMap::new(),
            resolution,
            mutation_rate,
        }
    }

    fn to_index(&self, descriptor: &[f32]) -> Vec<usize> {
        descriptor
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * (self.resolution - 1) as f32) as usize)
            .collect()
    }

    pub fn step<E: Evaluator<G>>(&mut self, evaluator: &E, batch_size: usize) {
        let mut rng = rand::rng();
        let keys: Vec<_> = self.archive.keys().cloned().collect();

        for _ in 0..batch_size {
            let mut dna = if keys.is_empty() {
                return;
            } else {
                let key = keys.choose(&mut rng).unwrap();
                self.archive.get(key).unwrap().genotype.clone()
            };

            dna.mutate(&mut rng, self.mutation_rate);
            let (f, obj, desc) = evaluator.evaluate(&dna);

            let idx = self.to_index(&desc);
            let new_pheno = Phenotype {
                genotype: dna,
                fitness: f,
                objectives: obj,
                descriptor: desc,
            };

            if let Some(existing) = self.archive.get(&idx) {
                if new_pheno.fitness > existing.fitness {
                    self.archive.insert(idx, new_pheno);
                }
            } else {
                self.archive.insert(idx, new_pheno);
            }
        }
    }
}
