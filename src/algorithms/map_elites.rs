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
//! let mut me = MapElites::<Point>::new(10, 0.3, 64, 42);
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

use crate::algorithms::archive::Archive;
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
    archive: Archive<G>,
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
        state.serialize_field("archive", self.archive.cells())?;
        state.serialize_field("archive_keys_vec", self.archive.keys_vec())?;
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

        // Reconstruct archive and validate keys_vec ↔ cells consistency.
        // This prevents an unwrap() panic during parent selection in step()
        // and rejects truncated/desynced state from malicious deserialization.
        let archive = Archive::from_raw(data.archive, data.archive_keys_vec);
        archive.validate().map_err(D::Error::custom)?;

        Ok(Self {
            archive,
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
    /// * `batch_size` - Number of offspring generated per [`step`](Self::step). Must be > 0.
    ///   A typical default is 64; larger values improve parallelism at the cost of memory.
    /// * `seed` - RNG seed for deterministic execution
    ///
    /// # Panics
    ///
    /// Panics if `resolution` or `batch_size` is 0.
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
    /// // 20x20 grid (400 cells), 30% mutation rate, 64 offspring per step
    /// let me = MapElites::<G>::new(20, 0.3, 64, 42);
    /// ```
    pub fn new(resolution: usize, mutation_rate: f32, batch_size: usize, seed: u64) -> Self {
        assert!(resolution > 0, "resolution must be greater than 0");
        assert!(batch_size > 0, "batch_size must be greater than 0");
        Self {
            archive: Archive::new(),
            resolution,
            mutation_rate,
            batch_size,
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
        self.archive.best_by_fitness()
    }

    /// Returns the fraction of cells occupied in the archive.
    ///
    /// Coverage is `archive_len / resolution^dimensions`, where dimensions is
    /// taken from the first inserted elite's descriptor length. Returns `0.0`
    /// for an empty archive.
    ///
    /// This is one of the two standard quality-diversity metrics. See also
    /// [`qd_score`](Self::qd_score).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use symbios_genetics::algorithms::map_elites::MapElites;
    /// # use serde::{Serialize, Deserialize};
    /// # use rand::Rng;
    /// # #[derive(Clone, Serialize, Deserialize)]
    /// # struct G;
    /// # impl symbios_genetics::Genotype for G {
    /// #     fn mutate<R: Rng>(&mut self, _: &mut R, _: f32) {}
    /// #     fn crossover<R: Rng>(&self, _: &Self, _: &mut R) -> Self { G }
    /// # }
    /// let me = MapElites::<G>::new(10, 0.1, 64, 42);
    /// assert_eq!(me.coverage(), 0.0);
    /// ```
    pub fn coverage(&self) -> f64 {
        let occupied = self.archive.len();
        if occupied == 0 {
            return 0.0;
        }
        let dim = self
            .archive
            .keys()
            .next()
            .expect("archive non-empty checked above")
            .len();
        let total = (self.resolution as f64).powi(dim as i32);
        if total == 0.0 {
            0.0
        } else {
            occupied as f64 / total
        }
    }

    /// Returns the QD score (sum of fitness across all occupied cells).
    ///
    /// This is the standard quality-diversity metric: a higher score reflects
    /// either better fitness, broader coverage, or both. NaN-fitness elites
    /// are skipped (they should not enter the archive but are filtered
    /// defensively).
    ///
    /// # Caveat: negative fitness
    ///
    /// QD score assumes non-negative fitness. If your fitness can be negative
    /// (e.g. `fitness = -distance`), shift it to non-negative before relying on
    /// this metric for cross-run comparison.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use symbios_genetics::algorithms::map_elites::MapElites;
    /// # use serde::{Serialize, Deserialize};
    /// # use rand::Rng;
    /// # #[derive(Clone, Serialize, Deserialize)]
    /// # struct G;
    /// # impl symbios_genetics::Genotype for G {
    /// #     fn mutate<R: Rng>(&mut self, _: &mut R, _: f32) {}
    /// #     fn crossover<R: Rng>(&self, _: &Self, _: &mut R) -> Self { G }
    /// # }
    /// let me = MapElites::<G>::new(10, 0.1, 64, 42);
    /// assert_eq!(me.qd_score(), 0.0);
    /// ```
    pub fn qd_score(&self) -> f64 {
        self.archive.qd_score()
    }

    /// Exports the archive as CSV to the given writer.
    ///
    /// One header row plus one row per occupied cell, in the deterministic
    /// iteration order of [`archive_iter`](Self::archive_iter). Columns:
    ///
    /// - `key` — bin indices joined with `;`
    /// - `descriptor` — descriptor floats joined with `;`
    /// - `fitness` — scalar fitness
    /// - `objectives` — objective floats joined with `;`
    /// - `genotype_hash` — 16-hex-char `seahash` of the bincode-serialised genotype
    ///
    /// The genotype hash is stable across runs given the same genotype bytes,
    /// suitable for joining the CSV against a separate genotype dump.
    ///
    /// Available with the `export` feature.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`](std::io::Error) if the writer fails or if a
    /// genotype cannot be serialised by `bincode`.
    #[cfg(feature = "export")]
    pub fn export_csv<W: std::io::Write>(&self, writer: W) -> std::io::Result<()> {
        self.archive.export_csv(writer)
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

            // NaN-fitness and NaN-descriptor candidates are filtered by Archive::insert_if_better.
            // We pre-check the descriptor here to avoid building a key for a candidate that will
            // be rejected anyway.
            if f.is_nan() || desc.iter().any(|v| v.is_nan()) {
                continue;
            }

            let key = self.map_to_index(&desc);
            self.archive.insert_if_better(
                key,
                Phenotype {
                    genotype: dna,
                    fitness: f,
                    objectives: obj,
                    descriptor: desc,
                },
            );
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
    /// let me = MapElites::<G>::new(10, 0.1, 64, 42);
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

        // Pre-select parent keys and generate RNG seeds serially (RNG needs mutable access).
        // O(batch_size) sampling using Archive::sample_key.
        let selections: Vec<(Vec<usize>, u64)> = (0..self.batch_size)
            .map(|_| {
                let key = self
                    .archive
                    .sample_key(&mut self.rng)
                    .expect("archive non-empty checked above")
                    .clone();
                let seed = self.rng.random::<u64>();
                (key, seed)
            })
            .collect();

        // Clone parents outside parallel section to avoid holding reference across threads
        let parents: Vec<G> = selections
            .iter()
            .map(|(key, _)| {
                self.archive
                    .get(key)
                    .expect("sampled key exists in archive")
                    .genotype
                    .clone()
            })
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

        // Reuse a single buffer for index mapping to avoid per-offspring allocations.
        let mut idx_buffer: Vec<usize> = Vec::new();

        for (dna, f, obj, desc) in results {
            // NaN-fitness and NaN-descriptor candidates are filtered by Archive::insert_if_better,
            // but pre-checking the descriptor here lets us skip the index-mapping work entirely.
            if f.is_nan() || desc.iter().any(|v| v.is_nan()) {
                continue;
            }

            idx_buffer.resize(desc.len(), 0);
            self.map_to_index_into(&desc, &mut idx_buffer);

            self.archive.insert_if_better(
                idx_buffer.clone(),
                Phenotype {
                    genotype: dna,
                    fitness: f,
                    objectives: obj,
                    descriptor: desc,
                },
            );
        }
    }

    fn population(&mut self) -> &[Phenotype<G>] {
        self.archive.population()
    }
}
