use crate::{Evaluator, Evolver, Genotype, Phenotype};
use rand::prelude::IndexedRandom;
use std::collections::HashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub struct MapElites<G: Genotype> {
    pub archive: HashMap<Vec<usize>, Phenotype<G>>,
    /// Cached population for trait compatibility
    population_cache: Vec<Phenotype<G>>,
    resolution: usize,
    mutation_rate: f32,
    pub batch_size: usize,
}

impl<G: Genotype> MapElites<G> {
    pub fn new(_dims: usize, resolution: usize, mutation_rate: f32) -> Self {
        Self {
            archive: HashMap::new(),
            population_cache: Vec::new(),
            resolution,
            mutation_rate,
            batch_size: 1, // Default
        }
    }

    pub fn map_to_index(&self, descriptor: &[f32]) -> Vec<usize> {
        descriptor
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * (self.resolution - 1) as f32).round() as usize)
            .collect()
    }
}

impl<G: Genotype> Evolver<G> for MapElites<G> {
    fn step<E: Evaluator<G>>(&mut self, evaluator: &E) {
        let mut rng = rand::rng();
        let keys: Vec<_> = self.archive.keys().cloned().collect();

        if keys.is_empty() {
            return;
        }

        let candidates: Vec<G> = (0..self.batch_size)
            .map(|_| {
                let key = keys.choose(&mut rng).unwrap();
                let mut dna = self.archive.get(key).unwrap().genotype.clone();
                dna.mutate(&mut rng, self.mutation_rate);
                dna
            })
            .collect();

        #[cfg(feature = "parallel")]
        let results: Vec<(G, f32, Vec<f32>, Vec<f32>)> = candidates
            .into_par_iter()
            .map(|dna| {
                let (f, obj, desc) = evaluator.evaluate(&dna);
                (dna, f, obj, desc)
            })
            .collect();

        #[cfg(not(feature = "parallel"))]
        let results: Vec<(G, f32, Vec<f32>, Vec<f32>)> = candidates
            .into_iter()
            .map(|dna| {
                let (f, obj, desc) = evaluator.evaluate(&dna);
                (dna, f, obj, desc)
            })
            .collect();

        for (dna, f, obj, desc) in results {
            let idx = self.map_to_index(&desc);
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

        // Refresh cache
        self.population_cache = self.archive.values().cloned().collect();
    }

    fn population(&self) -> &[Phenotype<G>] {
        &self.population_cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Genotype;
    use rand::Rng;
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Serialize, Deserialize)]
    struct Dummy;
    impl Genotype for Dummy {
        fn mutate<R: Rng>(&mut self, _: &mut R, _: f32) {}
        fn crossover<R: Rng>(&self, _: &Self, _: &mut R) -> Self {
            Dummy
        }
        fn distance(&self, _: &Self) -> f32 {
            0.0
        }
    }

    #[test]
    fn test_high_dim_indexing() {
        let engine = MapElites::<Dummy>::new(4, 100, 0.1);
        // Test 4D coordinate: [0.1, 0.5, 0.9, 1.0]
        let idx = engine.map_to_index(&[0.1, 0.5, 0.9, 1.0]);
        // 0.1 * 99 = 9.9 -> 10
        // 0.5 * 99 = 49.5 -> 50
        // 0.9 * 99 = 89.1 -> 89
        // 1.0 * 99 = 99
        assert_eq!(idx, vec![10, 50, 89, 99]);
    }

    #[test]
    fn test_index_out_of_bounds_protection() {
        let engine = MapElites::<Dummy>::new(1, 10, 0.1);
        // Test clamping: -5.0 and 5.0 should be forced into [0, 9]
        assert_eq!(engine.map_to_index(&[-5.0]), vec![0]);
        assert_eq!(engine.map_to_index(&[5.0]), vec![9]);
    }
}
