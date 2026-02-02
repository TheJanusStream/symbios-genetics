use crate::{Evaluator, Evolver, Genotype, Phenotype};
use rand::prelude::SeedableRng;
use rand_pcg::Pcg64; // Specific, serializable generator
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Internal representation for serialization (without cache)
#[derive(Serialize, Deserialize)]
#[serde(bound = "G: Genotype")]
struct MapElitesData<G: Genotype> {
    archive: BTreeMap<Vec<usize>, Phenotype<G>>,
    resolution: usize,
    mutation_rate: f32,
    batch_size: usize,
    rng: Pcg64,
}

pub struct MapElites<G: Genotype> {
    archive: BTreeMap<Vec<usize>, Phenotype<G>>,
    population_cache: Vec<Phenotype<G>>,
    cache_valid: bool,
    resolution: usize,
    mutation_rate: f32,
    batch_size: usize,
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
            cache_valid: true,
            resolution: data.resolution,
            mutation_rate: data.mutation_rate,
            batch_size: data.batch_size,
            rng: data.rng,
        })
    }
}

impl<G: Genotype> MapElites<G> {
    pub fn new(resolution: usize, mutation_rate: f32, seed: u64) -> Self {
        assert!(resolution > 0, "resolution must be greater than 0");
        Self {
            archive: BTreeMap::new(),
            population_cache: Vec::new(),
            cache_valid: true,
            resolution,
            mutation_rate,
            batch_size: 64, // Reasonable default for parallel evaluation
            rng: Pcg64::seed_from_u64(seed),
        }
    }

    pub fn resolution(&self) -> usize {
        self.resolution
    }

    pub fn mutation_rate(&self) -> f32 {
        self.mutation_rate
    }

    pub fn set_mutation_rate(&mut self, rate: f32) {
        self.mutation_rate = rate;
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn set_batch_size(&mut self, size: usize) {
        assert!(size > 0, "batch_size must be greater than 0");
        self.batch_size = size;
    }

    pub fn archive_len(&self) -> usize {
        self.archive.len()
    }

    pub fn archive_get(&self, key: &[usize]) -> Option<&Phenotype<G>> {
        self.archive.get(key)
    }

    pub fn archive_keys(&self) -> impl Iterator<Item = &Vec<usize>> {
        self.archive.keys()
    }

    pub fn archive_iter(&self) -> impl Iterator<Item = (&Vec<usize>, &Phenotype<G>)> {
        self.archive.iter()
    }

    pub fn best_by_fitness(&self) -> Option<&Phenotype<G>> {
        self.archive.values().max_by(|a, b| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    pub fn seed_population<E: Evaluator<G>>(&mut self, initial: Vec<G>, evaluator: &E) {
        for dna in initial {
            let (f, obj, desc) = evaluator.evaluate(&dna);
            let idx = self.map_to_index(&desc);
            let new_pheno = Phenotype {
                genotype: dna,
                fitness: f,
                objectives: obj,
                descriptor: desc,
            };
            // Respect elitism: only insert if no existing elite or new individual is better
            if self
                .archive
                .get(&idx)
                .is_none_or(|existing| new_pheno.fitness > existing.fitness)
            {
                self.archive.insert(idx, new_pheno);
                self.cache_valid = false;
            }
        }
    }

    fn ensure_cache_valid(&mut self) {
        if !self.cache_valid {
            self.population_cache = self.archive.values().cloned().collect();
            self.cache_valid = true;
        }
    }

    pub fn map_to_index(&self, descriptor: &[f32]) -> Vec<usize> {
        descriptor
            .iter()
            .map(|&v| {
                // Use floor() for uniform bin sizes, clamp index to valid range
                let scaled = v.clamp(0.0, 1.0) * self.resolution as f32;
                (scaled.floor() as usize).min(self.resolution - 1)
            })
            .collect()
    }
}

impl<G: Genotype> Evolver<G> for MapElites<G> {
    fn step<E: Evaluator<G>>(&mut self, evaluator: &E) {
        use rand::Rng;

        if self.archive.is_empty() {
            return;
        }

        // Collect genotypes directly instead of cloning Vec keys
        let parents: Vec<&G> = self.archive.values().map(|p| &p.genotype).collect();
        let mutation_rate = self.mutation_rate;

        // Pre-select parents and generate RNG seeds serially (RNG needs mutable access)
        let selections: Vec<(usize, u64)> = (0..self.batch_size)
            .map(|_| {
                let parent_idx = self.rng.random_range(0..parents.len());
                let seed = self.rng.random::<u64>();
                (parent_idx, seed)
            })
            .collect();

        // Parallel: clone parents, mutate with per-task RNG, and evaluate
        #[cfg(feature = "parallel")]
        let results: Vec<(G, f32, Vec<f32>, Vec<f32>)> = selections
            .into_par_iter()
            .map(|(parent_idx, seed)| {
                let mut rng = Pcg64::seed_from_u64(seed);
                let mut dna = parents[parent_idx].clone();
                dna.mutate(&mut rng, mutation_rate);
                let (f, obj, desc) = evaluator.evaluate(&dna);
                (dna, f, obj, desc)
            })
            .collect();

        #[cfg(not(feature = "parallel"))]
        let results: Vec<(G, f32, Vec<f32>, Vec<f32>)> = selections
            .into_iter()
            .map(|(parent_idx, seed)| {
                let mut rng = Pcg64::seed_from_u64(seed);
                let mut dna = parents[parent_idx].clone();
                dna.mutate(&mut rng, mutation_rate);
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
                self.cache_valid = false;
            }
        }
    }

    fn population(&mut self) -> &[Phenotype<G>] {
        self.ensure_cache_valid();
        &self.population_cache
    }
}
