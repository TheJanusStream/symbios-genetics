use crate::{Evaluator, Evolver, Genotype, Phenotype};
use rand::prelude::{IndexedRandom, SeedableRng};
use rand_pcg::Pcg64; // Specific, serializable generator
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Internal representation for serialization (without cache)
#[derive(Serialize, Deserialize)]
#[serde(bound = "G: Genotype")]
struct MapElitesData<G: Genotype> {
    archive: HashMap<Vec<usize>, Phenotype<G>>,
    resolution: usize,
    mutation_rate: f32,
    batch_size: usize,
    rng: Pcg64,
}

pub struct MapElites<G: Genotype> {
    pub archive: HashMap<Vec<usize>, Phenotype<G>>,
    population_cache: Vec<Phenotype<G>>,
    pub resolution: usize,
    pub mutation_rate: f32,
    pub batch_size: usize,
    rng: Pcg64,
}

impl<G: Genotype> Serialize for MapElites<G> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("MapElites", 5)?;
        state.serialize_field("archive", &self.archive)?;
        state.serialize_field("resolution", &self.resolution)?;
        state.serialize_field("mutation_rate", &self.mutation_rate)?;
        state.serialize_field("batch_size", &self.batch_size)?;
        state.serialize_field("rng", &self.rng)?;
        state.end()
    }
}

impl<'de, G: Genotype> Deserialize<'de> for MapElites<G> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let data = MapElitesData::<G>::deserialize(deserializer)?;
        let population_cache: Vec<Phenotype<G>> = data.archive.values().cloned().collect();
        Ok(Self {
            archive: data.archive,
            population_cache,
            resolution: data.resolution,
            mutation_rate: data.mutation_rate,
            batch_size: data.batch_size,
            rng: data.rng,
        })
    }
}

impl<G: Genotype> MapElites<G> {
    pub fn new(resolution: usize, mutation_rate: f32, seed: u64) -> Self {
        Self {
            archive: HashMap::new(),
            population_cache: Vec::new(),
            resolution,
            mutation_rate,
            batch_size: 64, // Reasonable default for parallel evaluation
            rng: Pcg64::seed_from_u64(seed),
        }
    }

    pub fn seed_population<E: Evaluator<G>>(&mut self, initial: Vec<G>, evaluator: &E) {
        for dna in initial {
            let (f, obj, desc) = evaluator.evaluate(&dna);
            let idx = self.map_to_index(&desc);
            self.archive.insert(
                idx,
                Phenotype {
                    genotype: dna,
                    fitness: f,
                    objectives: obj,
                    descriptor: desc,
                },
            );
        }
        self.rebuild_cache();
    }

    fn rebuild_cache(&mut self) {
        self.population_cache = self.archive.values().cloned().collect();
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
        if self.archive.is_empty() {
            return;
        }

        // Collect genotypes directly instead of cloning Vec keys
        // This avoids O(n) heap allocations per step for large archives
        let parents: Vec<&G> = self.archive.values().map(|p| &p.genotype).collect();

        let candidates: Vec<G> = (0..self.batch_size)
            .map(|_| {
                let parent = *parents.choose(&mut self.rng).unwrap();
                let mut dna = parent.clone();
                dna.mutate(&mut self.rng, self.mutation_rate);
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
            if self
                .archive
                .get(&idx)
                .is_none_or(|e| new_pheno.fitness > e.fitness)
            {
                self.archive.insert(idx, new_pheno);
            }
        }
        self.rebuild_cache();
    }

    fn population(&self) -> &[Phenotype<G>] {
        &self.population_cache
    }
}
