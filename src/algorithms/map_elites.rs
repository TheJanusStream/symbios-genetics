//! MAP-Elites quality-diversity algorithm.
//!
//! MAP-Elites (Multi-dimensional Archive of Phenotypic Elites) is a quality-diversity
//! algorithm that maintains an archive of high-performing solutions across a space
//! of behavioral descriptors.
//!
//! # Overview
//!
//! Unlike traditional evolutionary algorithms that converge to a single optimum,
//! MAP-Elites discovers a diverse collection of high-quality solutions. Each cell
//! in the behavioral descriptor space stores the best individual found for that
//! behavioral niche.
//!
//! # Key Concepts
//!
//! - **Behavioral Descriptor**: A vector of values (typically in `[0.0, 1.0]`) that
//!   characterizes *how* a solution behaves, not just *how well* it performs
//! - **Archive**: A grid-based data structure that stores elites for each behavioral niche
//! - **Resolution**: The number of bins per descriptor dimension
//!
//! # Example
//!
//! ```rust
//! use rand::Rng;
//! use serde::{Deserialize, Serialize};
//! use symbios_genetics::{Evaluator, Evolver, Genotype, algorithms::map_elites::MapElites};
//!
//! #[derive(Clone, Serialize, Deserialize)]
//! struct Point(f32, f32);
//!
//! impl Genotype for Point {
//!     fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
//!         if rng.random::<f32>() < rate {
//!             self.0 = (self.0 + rng.random::<f32>() - 0.5).clamp(0.0, 1.0);
//!             self.1 = (self.1 + rng.random::<f32>() - 0.5).clamp(0.0, 1.0);
//!         }
//!     }
//!     fn crossover<R: Rng>(&self, other: &Self, _rng: &mut R) -> Self {
//!         Point((self.0 + other.0) / 2.0, (self.1 + other.1) / 2.0)
//!     }
//! }
//!
//! struct Rastrigin;
//! impl Evaluator<Point> for Rastrigin {
//!     fn evaluate(&self, g: &Point) -> (f32, Vec<f32>, Vec<f32>) {
//!         // Rastrigin function (negated for maximization)
//!         let x = g.0 * 10.0 - 5.0;
//!         let y = g.1 * 10.0 - 5.0;
//!         let fitness = -(20.0 + x*x + y*y - 10.0*(x*2.0*std::f32::consts::PI).cos()
//!                        - 10.0*(y*2.0*std::f32::consts::PI).cos());
//!         // Use position as behavioral descriptor
//!         (fitness, vec![fitness], vec![g.0, g.1])
//!     }
//! }
//!
//! let mut me = MapElites::<Point>::new(10, 0.3, 42);
//! me.seed_population(
//!     (0..100).map(|_| Point(rand::random(), rand::random())).collect(),
//!     &Rastrigin,
//! );
//!
//! for _ in 0..100 {
//!     me.step(&Rastrigin);
//! }
//!
//! println!("Archive contains {} elites", me.archive_len());
//! ```
//!
//! # Algorithm Details
//!
//! Each step of MAP-Elites:
//! 1. Randomly selects parents from the archive
//! 2. Creates offspring via mutation
//! 3. Evaluates offspring fitness and behavioral descriptors
//! 4. Places offspring in archive cells if they improve on existing elites
//!
//! # References
//!
//! Mouret, J.-B., & Clune, J. (2015). Illuminating search spaces by mapping elites.
//! arXiv preprint arXiv:1504.04909.

use crate::{Evaluator, Evolver, Genotype, Phenotype};
use rand::prelude::SeedableRng;
use rand_pcg::Pcg64;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Internal representation for serialization (excludes transient cache).
#[derive(Serialize, Deserialize)]
#[serde(bound = "G: Genotype")]
struct MapElitesData<G: Genotype> {
    archive: BTreeMap<Vec<usize>, Phenotype<G>>,
    /// Vec of archive keys for O(1) random parent sampling.
    /// Kept in sync with archive to avoid O(N) iteration during step().
    archive_keys_vec: Vec<Vec<usize>>,
    resolution: usize,
    mutation_rate: f32,
    batch_size: usize,
    rng: Pcg64,
}

/// MAP-Elites quality-diversity algorithm.
///
/// Maintains an archive of high-performing solutions across a discretized
/// behavioral descriptor space. Each cell stores the elite (highest fitness)
/// individual discovered for that behavioral niche.
///
/// # Type Parameters
///
/// * `G` - The genotype type, must implement [`Genotype`]
///
/// # Archive Structure
///
/// The archive is a multi-dimensional grid where:
/// - Each dimension corresponds to a behavioral descriptor
/// - Each cell is indexed by `Vec<usize>` (bin indices)
/// - Resolution determines bins per dimension (e.g., resolution=10 → 10 bins)
///
/// # Determinism
///
/// MAP-Elites uses a seeded RNG ([`Pcg64`]) and deterministic iteration order
/// ([`BTreeMap`]) to ensure reproducible results across runs.
pub struct MapElites<G: Genotype> {
    archive: BTreeMap<Vec<usize>, Phenotype<G>>,
    /// Vec of archive keys for O(1) random parent sampling.
    /// Avoids O(N) iteration of BTreeMap during step().
    archive_keys_vec: Vec<Vec<usize>>,
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
        let mut state = serializer.serialize_struct("MapElites", 6)?;
        state.serialize_field("archive", &self.archive)?;
        state.serialize_field("archive_keys_vec", &self.archive_keys_vec)?;
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
        use serde::de::Error;

        let data = MapElitesData::<G>::deserialize(deserializer)?;

        // Validate resolution to prevent division/underflow panic in map_single_dimension
        if data.resolution == 0 {
            return Err(D::Error::custom("resolution must be greater than 0"));
        }

        // Validate batch_size to prevent empty offspring generation
        if data.batch_size == 0 {
            return Err(D::Error::custom("batch_size must be greater than 0"));
        }

        // Validate archive_keys_vec integrity: each key must exist in archive
        // This prevents panic from unwrap() during parent selection in step()
        for key in &data.archive_keys_vec {
            if !data.archive.contains_key(key) {
                return Err(D::Error::custom(
                    "archive_keys_vec contains key not present in archive",
                ));
            }
        }

        // Validate archive integrity: each archive key should be in archive_keys_vec
        // This ensures the key vec is complete for random sampling
        if data.archive.len() != data.archive_keys_vec.len() {
            return Err(D::Error::custom(
                "archive_keys_vec length does not match archive size",
            ));
        }

        let population_cache: Vec<Phenotype<G>> = data.archive.values().cloned().collect();
        Ok(Self {
            archive: data.archive,
            archive_keys_vec: data.archive_keys_vec,
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
    /// Creates a new MAP-Elites instance.
    ///
    /// # Arguments
    ///
    /// * `resolution` - Number of bins per descriptor dimension. Must be > 0.
    /// * `mutation_rate` - Probability of mutation, typically in `[0.0, 1.0]`
    /// * `seed` - RNG seed for deterministic execution
    ///
    /// # Panics
    ///
    /// Panics if `resolution` is 0.
    ///
    /// # Example
    ///
    /// ```rust
    /// use symbios_genetics::algorithms::map_elites::MapElites;
    /// # use serde::{Serialize, Deserialize};
    /// # use rand::Rng;
    /// # #[derive(Clone, Serialize, Deserialize)]
    /// # struct G;
    /// # impl symbios_genetics::Genotype for G {
    /// #     fn mutate<R: Rng>(&mut self, _: &mut R, _: f32) {}
    /// #     fn crossover<R: Rng>(&self, _: &Self, _: &mut R) -> Self { G }
    /// # }
    ///
    /// // 20x20 grid (400 cells) with 30% mutation rate
    /// let me = MapElites::<G>::new(20, 0.3, 42);
    /// ```
    pub fn new(resolution: usize, mutation_rate: f32, seed: u64) -> Self {
        assert!(resolution > 0, "resolution must be greater than 0");
        Self {
            archive: BTreeMap::new(),
            archive_keys_vec: Vec::new(),
            population_cache: Vec::new(),
            cache_valid: true,
            resolution,
            mutation_rate,
            batch_size: 64,
            rng: Pcg64::seed_from_u64(seed),
        }
    }

    /// Returns the resolution (bins per descriptor dimension).
    pub fn resolution(&self) -> usize {
        self.resolution
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

    /// Returns the batch size (offspring per step).
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Sets the batch size for parallel evaluation.
    ///
    /// Larger batch sizes improve parallelism but increase memory usage.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of offspring to generate per step. Must be > 0.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0.
    pub fn set_batch_size(&mut self, size: usize) {
        assert!(size > 0, "batch_size must be greater than 0");
        self.batch_size = size;
    }

    /// Returns the number of elites in the archive.
    ///
    /// This is the count of occupied cells, not the total grid size.
    pub fn archive_len(&self) -> usize {
        self.archive.len()
    }

    /// Gets an elite by its cell index.
    ///
    /// # Arguments
    ///
    /// * `key` - The cell index (bin indices for each descriptor dimension)
    ///
    /// # Returns
    ///
    /// The elite at that cell, or `None` if the cell is empty.
    pub fn archive_get(&self, key: &[usize]) -> Option<&Phenotype<G>> {
        self.archive.get(key)
    }

    /// Returns an iterator over all occupied cell indices.
    ///
    /// Iteration order is deterministic (sorted by index).
    pub fn archive_keys(&self) -> impl Iterator<Item = &Vec<usize>> {
        self.archive.keys()
    }

    /// Returns an iterator over all (index, elite) pairs.
    ///
    /// Iteration order is deterministic (sorted by index).
    pub fn archive_iter(&self) -> impl Iterator<Item = (&Vec<usize>, &Phenotype<G>)> {
        self.archive.iter()
    }

    /// Returns the elite with the highest fitness across all cells.
    ///
    /// # Returns
    ///
    /// The best elite, or `None` if the archive is empty.
    pub fn best_by_fitness(&self) -> Option<&Phenotype<G>> {
        self.archive.values().max_by(|a, b| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Seeds the archive with initial individuals.
    ///
    /// Each individual is evaluated and placed in its corresponding cell.
    /// Respects elitism: only replaces existing elites if the new individual
    /// has higher fitness.
    ///
    /// # Arguments
    ///
    /// * `initial` - Vector of genotypes to seed
    /// * `evaluator` - Fitness evaluator
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let initial: Vec<MyGenome> = (0..100)
    ///     .map(|_| MyGenome::random())
    ///     .collect();
    /// me.seed_population(initial, &evaluator);
    /// ```
    pub fn seed_population<E: Evaluator<G>>(&mut self, initial: Vec<G>, evaluator: &E) {
        for dna in initial {
            let (f, obj, desc) = evaluator.evaluate(&dna);

            // Skip individuals with NaN fitness - they cannot meaningfully compete
            if f.is_nan() {
                continue;
            }

            // Skip individuals with NaN descriptors - they cannot be properly mapped
            if desc.iter().any(|v| v.is_nan()) {
                continue;
            }

            let idx = self.map_to_index(&desc);
            let new_pheno = Phenotype {
                genotype: dna,
                fitness: f,
                objectives: obj,
                descriptor: desc,
            };
            // Replace existing elite if:
            // - Cell is empty, OR
            // - New fitness is strictly better, OR
            // - Existing elite has NaN fitness (corrupted state recovery)
            if self.archive.get(&idx).is_none_or(|existing| {
                new_pheno.fitness > existing.fitness || existing.fitness.is_nan()
            }) {
                let is_new_key = !self.archive.contains_key(&idx);
                self.archive.insert(idx.clone(), new_pheno);
                if is_new_key {
                    self.archive_keys_vec.push(idx);
                }
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

    /// Maps a behavioral descriptor to cell indices.
    ///
    /// Descriptor values are clamped to `[0.0, 1.0]` and discretized into
    /// bins based on the resolution.
    ///
    /// # Arguments
    ///
    /// * `descriptor` - Behavioral descriptor values
    ///
    /// # Returns
    ///
    /// Vector of bin indices, one per descriptor dimension.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let me = MapElites::<G>::new(10, 0.1, 42);
    /// let idx = me.map_to_index(&[0.25, 0.75]);
    /// assert_eq!(idx, vec![2, 7]); // bins 2 and 7 out of 0-9
    /// ```
    pub fn map_to_index(&self, descriptor: &[f32]) -> Vec<usize> {
        descriptor
            .iter()
            .map(|&v| self.map_single_dimension(v))
            .collect()
    }

    /// Maps a behavioral descriptor into a pre-allocated buffer.
    ///
    /// This is an allocation-free alternative to [`map_to_index`] for
    /// high-throughput use cases. The buffer must have at least as many
    /// elements as the descriptor.
    ///
    /// # Arguments
    ///
    /// * `descriptor` - Behavioral descriptor values
    /// * `buffer` - Pre-allocated buffer to write indices into
    ///
    /// # Panics
    ///
    /// Panics if `buffer.len() < descriptor.len()`.
    pub fn map_to_index_into(&self, descriptor: &[f32], buffer: &mut [usize]) {
        assert!(
            buffer.len() >= descriptor.len(),
            "buffer too small: {} < {}",
            buffer.len(),
            descriptor.len()
        );
        for (i, &v) in descriptor.iter().enumerate() {
            buffer[i] = self.map_single_dimension(v);
        }
    }

    /// Maps a single descriptor dimension to a bin index.
    #[inline]
    fn map_single_dimension(&self, v: f32) -> usize {
        let scaled = v.clamp(0.0, 1.0) * self.resolution as f32;
        (scaled.floor() as usize).min(self.resolution - 1)
    }
}

impl<G: Genotype> Evolver<G> for MapElites<G> {
    fn step<E: Evaluator<G>>(&mut self, evaluator: &E) {
        use rand::Rng;

        if self.archive.is_empty() {
            return;
        }

        let mutation_rate = self.mutation_rate;
        let num_keys = self.archive_keys_vec.len();

        // Pre-select parent keys and generate RNG seeds serially (RNG needs mutable access)
        // O(batch_size) instead of O(archive_size)
        let selections: Vec<(Vec<usize>, u64)> = (0..self.batch_size)
            .map(|_| {
                let key_idx = self.rng.random_range(0..num_keys);
                let key = self.archive_keys_vec[key_idx].clone();
                let seed = self.rng.random::<u64>();
                (key, seed)
            })
            .collect();

        // Clone parents outside parallel section to avoid holding reference across threads
        let parents: Vec<G> = selections
            .iter()
            .map(|(key, _)| self.archive.get(key).unwrap().genotype.clone())
            .collect();

        // Parallel: mutate with per-task RNG and evaluate
        #[cfg(feature = "parallel")]
        let results: Vec<(G, f32, Vec<f32>, Vec<f32>)> = parents
            .into_par_iter()
            .zip(selections.into_par_iter())
            .map(|(mut dna, (_, seed))| {
                let mut rng = Pcg64::seed_from_u64(seed);
                dna.mutate(&mut rng, mutation_rate);
                let (f, obj, desc) = evaluator.evaluate(&dna);
                (dna, f, obj, desc)
            })
            .collect();

        #[cfg(not(feature = "parallel"))]
        let results: Vec<(G, f32, Vec<f32>, Vec<f32>)> = parents
            .into_iter()
            .zip(selections.into_iter())
            .map(|(mut dna, (_, seed))| {
                let mut rng = Pcg64::seed_from_u64(seed);
                dna.mutate(&mut rng, mutation_rate);
                let (f, obj, desc) = evaluator.evaluate(&dna);
                (dna, f, obj, desc)
            })
            .collect();

        // Reuse a single buffer for index mapping to avoid per-offspring allocations
        let mut idx_buffer: Vec<usize> = Vec::new();

        for (dna, f, obj, desc) in results {
            // Skip individuals with NaN fitness - they cannot meaningfully compete
            // with existing elites and would corrupt the archive if inserted.
            // NaN comparisons always return false, so without this check:
            // - NaN <= existing.fitness is false -> would overwrite valid elites
            // - valid <= NaN is false -> would overwrite NaN (self-correcting but causes churn)
            if f.is_nan() {
                continue;
            }

            // Skip individuals with NaN descriptors - they cannot be properly mapped.
            // NaN.clamp(0.0, 1.0) returns NaN, and NaN as usize saturates to 0,
            // which would silently map all NaN-descriptor individuals to bin 0.
            if desc.iter().any(|v| v.is_nan()) {
                continue;
            }

            // Resize buffer to match descriptor dimensions, reusing capacity
            idx_buffer.resize(desc.len(), 0);
            self.map_to_index_into(&desc, &mut idx_buffer);

            // Check if offspring should enter archive before cloning the key
            // Using > instead of <= ensures NaN in existing elites gets replaced by valid fitness
            let dominated = self
                .archive
                .get(&idx_buffer)
                .is_some_and(|e| f <= e.fitness && !e.fitness.is_nan());

            if !dominated {
                let new_pheno = Phenotype {
                    genotype: dna,
                    fitness: f,
                    objectives: obj,
                    descriptor: desc,
                };
                let is_new_key = !self.archive.contains_key(&idx_buffer);
                // Only allocate key Vec when actually inserting
                let idx = idx_buffer.clone();
                self.archive.insert(idx.clone(), new_pheno);
                if is_new_key {
                    self.archive_keys_vec.push(idx);
                }
                self.cache_valid = false;
            }
        }
    }

    fn population(&mut self) -> &[Phenotype<G>] {
        self.ensure_cache_valid();
        &self.population_cache
    }
}
