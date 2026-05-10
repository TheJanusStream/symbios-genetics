//! Speciation primitives for population-based evolutionary algorithms.
//!
//! Speciation clusters genotypes into species based on a user-supplied
//! [`CompatibilityDistance`](crate::speciation::CompatibilityDistance) metric,
//! allowing fitness sharing within niches and a dynamic threshold that targets
//! a configured species count.
//!
//! This module is genotype-agnostic: the caller supplies the distance metric.
//! The classic NEAT (Stanley & Miikkulainen, 2002) usage is to wrap a
//! topology-genome's `compatibility_distance` method in a
//! [`CompatibilityDistance`](crate::speciation::CompatibilityDistance) impl and
//! feed it to [`Speciation`](crate::speciation::Speciation).
//!
//! # Algorithm sketch
//!
//! Each generation:
//! 1. [`Speciation::assign`](crate::speciation::Speciation::assign) walks the
//!    population. Each phenotype is placed in the first existing species whose
//!    representative is within `threshold` distance, or a new species is created.
//! 2. [`Speciation::share_fitness`](crate::speciation::Speciation::share_fitness)
//!    divides each phenotype's `fitness` by its species size — Stanley's
//!    *explicit fitness sharing*, which prevents any one species from
//!    dominating the population.
//! 3. [`Speciation::adjust_threshold`](crate::speciation::Speciation::adjust_threshold)
//!    nudges `threshold` up or down by `threshold_step` to drive the species
//!    count toward `target_count`.
//!
//! # Example
//!
//! ```rust
//! use rand::Rng;
//! use serde::{Deserialize, Serialize};
//! use symbios_genetics::{
//!     speciation::{CompatibilityDistance, Speciation},
//!     Genotype, Phenotype,
//! };
//!
//! #[derive(Clone, Serialize, Deserialize)]
//! struct Vec3([f32; 3]);
//!
//! impl Genotype for Vec3 {
//!     fn mutate<R: Rng>(&mut self, rng: &mut R, _rate: f32) {
//!         for v in &mut self.0 { *v += rng.random::<f32>() - 0.5; }
//!     }
//!     fn crossover<R: Rng>(&self, other: &Self, _rng: &mut R) -> Self {
//!         Vec3([
//!             (self.0[0] + other.0[0]) * 0.5,
//!             (self.0[1] + other.0[1]) * 0.5,
//!             (self.0[2] + other.0[2]) * 0.5,
//!         ])
//!     }
//! }
//!
//! struct Euclidean;
//! impl CompatibilityDistance<Vec3> for Euclidean {
//!     fn distance(&self, a: &Vec3, b: &Vec3) -> f32 {
//!         a.0.iter().zip(&b.0).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
//!     }
//! }
//!
//! let population: Vec<Phenotype<Vec3>> = (0..40)
//!     .map(|i| Phenotype {
//!         genotype: Vec3([i as f32 * 0.1, 0.0, 0.0]),
//!         fitness: 1.0,
//!         objectives: vec![],
//!         descriptor: vec![],
//!     })
//!     .collect();
//!
//! let mut spec = Speciation::new(Euclidean, 0.5, 5);
//! let mut pop = population;
//! spec.assign(&pop);
//! spec.share_fitness(&mut pop);
//! spec.adjust_threshold();
//! assert!(!spec.species().is_empty());
//! ```

use std::marker::PhantomData;

use crate::{Genotype, Phenotype};

/// Distance metric used to decide whether two genotypes share a species.
///
/// Implementors return a non-negative scalar where `0.0` indicates identical
/// genotypes. The metric does not need to satisfy the triangle inequality —
/// speciation only relies on threshold comparisons.
pub trait CompatibilityDistance<G: Genotype>: Send + Sync {
    /// Compute the compatibility distance between two genotypes.
    fn distance(&self, a: &G, b: &G) -> f32;
}

/// A cluster of genotypes whose representative has been measured within the
/// current speciation threshold.
#[derive(Clone, Debug)]
pub struct Species<G: Genotype> {
    /// Stable identifier assigned at species creation.
    pub id: u64,
    /// Representative genotype used for distance comparisons in [`Speciation::assign`].
    pub representative: G,
    /// Indices into the population slice supplied to [`Speciation::assign`].
    pub member_indices: Vec<usize>,
}

impl<G: Genotype> Species<G> {
    /// Number of members currently assigned to this species.
    #[must_use]
    pub fn size(&self) -> usize {
        self.member_indices.len()
    }
}

/// Speciation state: a list of [`Species`] plus a dynamically-tuned threshold.
///
/// The threshold controls how aggressive speciation is. Lower thresholds
/// produce more, smaller species; higher thresholds produce fewer, larger
/// species. [`adjust_threshold`](Self::adjust_threshold) moves it toward
/// whichever direction brings the current count closer to `target_count`.
pub struct Speciation<G: Genotype, D: CompatibilityDistance<G>> {
    species: Vec<Species<G>>,
    threshold: f32,
    target_count: usize,
    threshold_step: f32,
    min_threshold: f32,
    distance: D,
    next_species_id: u64,
    _marker: PhantomData<fn() -> G>,
}

impl<G: Genotype, D: CompatibilityDistance<G>> Speciation<G, D> {
    /// Default threshold step used when [`new`](Self::new) is called.
    pub const DEFAULT_THRESHOLD_STEP: f32 = 0.3;
    /// Default minimum threshold floor used when [`new`](Self::new) is called.
    pub const DEFAULT_MIN_THRESHOLD: f32 = 0.1;

    /// Create a new speciation manager.
    ///
    /// # Arguments
    ///
    /// * `distance` - Compatibility distance metric.
    /// * `initial_threshold` - Starting compatibility threshold.
    /// * `target_count` - Desired number of species. The threshold is adjusted
    ///   each call to [`adjust_threshold`](Self::adjust_threshold) toward this target.
    #[must_use]
    pub fn new(distance: D, initial_threshold: f32, target_count: usize) -> Self {
        Self {
            species: Vec::new(),
            threshold: initial_threshold,
            target_count,
            threshold_step: Self::DEFAULT_THRESHOLD_STEP,
            min_threshold: Self::DEFAULT_MIN_THRESHOLD,
            distance,
            next_species_id: 0,
            _marker: PhantomData,
        }
    }

    /// Override the per-generation threshold-adjustment step (default 0.3).
    #[must_use]
    pub fn with_threshold_step(mut self, step: f32) -> Self {
        self.threshold_step = step;
        self
    }

    /// Override the floor for [`adjust_threshold`](Self::adjust_threshold) (default 0.1).
    ///
    /// The threshold will not be reduced below this value.
    #[must_use]
    pub fn with_min_threshold(mut self, min: f32) -> Self {
        self.min_threshold = min;
        self
    }

    /// Assign each phenotype in `population` to a species.
    ///
    /// Existing species are retained across generations to keep IDs stable;
    /// their representative is updated to a current member, and species with
    /// no surviving members are dropped.
    pub fn assign(&mut self, population: &[Phenotype<G>]) {
        // Preserve old representatives but clear membership; we will refill below.
        for s in &mut self.species {
            s.member_indices.clear();
        }

        for (idx, phen) in population.iter().enumerate() {
            let mut placed = false;
            for s in &mut self.species {
                if self.distance.distance(&phen.genotype, &s.representative) < self.threshold {
                    s.member_indices.push(idx);
                    placed = true;
                    break;
                }
            }
            if !placed {
                self.species.push(Species {
                    id: self.next_species_id,
                    representative: phen.genotype.clone(),
                    member_indices: vec![idx],
                });
                self.next_species_id += 1;
            }
        }

        // Drop empty species. Update representatives to a current member so
        // species drift with their cluster across generations.
        self.species.retain(|s| !s.member_indices.is_empty());
        for s in &mut self.species {
            // Use the first member as the new representative — cheap and stable.
            // SAFETY: retain() above ensures member_indices is non-empty.
            let rep_idx = s.member_indices[0];
            s.representative = population[rep_idx].genotype.clone();
        }
    }

    /// Apply Stanley & Miikkulainen explicit fitness sharing.
    ///
    /// Each phenotype's `fitness` is divided by the size of its species,
    /// preventing dominant species from monopolizing reproduction. Must be
    /// called after [`assign`](Self::assign).
    ///
    /// Phenotypes not assigned to any species (only possible if the population
    /// passed here differs from the one passed to [`assign`](Self::assign))
    /// are left unmodified.
    pub fn share_fitness(&self, population: &mut [Phenotype<G>]) {
        for s in &self.species {
            let divisor = s.member_indices.len() as f32;
            if divisor == 0.0 {
                continue;
            }
            for &idx in &s.member_indices {
                if let Some(phen) = population.get_mut(idx) {
                    phen.fitness /= divisor;
                }
            }
        }
    }

    /// Move the compatibility threshold toward whichever direction brings the
    /// current species count closer to `target_count`.
    ///
    /// Call once per generation after [`assign`](Self::assign). The threshold
    /// is clamped to a floor of `min_threshold` (default
    /// [`DEFAULT_MIN_THRESHOLD`](Self::DEFAULT_MIN_THRESHOLD)).
    ///
    /// If the current count equals `target_count` the threshold is held
    /// steady, preventing oscillation around discrete count plateaus.
    pub fn adjust_threshold(&mut self) {
        let count = self.species.len();
        if count > self.target_count {
            self.threshold += self.threshold_step;
        } else if count < self.target_count {
            self.threshold = (self.threshold - self.threshold_step).max(self.min_threshold);
        }
    }

    /// Current species list.
    #[must_use]
    pub fn species(&self) -> &[Species<G>] {
        &self.species
    }

    /// Current compatibility threshold.
    #[must_use]
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Configured target number of species.
    #[must_use]
    pub fn target_count(&self) -> usize {
        self.target_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Serialize, Deserialize, Debug)]
    struct Scalar(f32);

    impl Genotype for Scalar {
        fn mutate<R: Rng>(&mut self, rng: &mut R, _rate: f32) {
            self.0 += rng.random::<f32>() - 0.5;
        }
        fn crossover<R: Rng>(&self, other: &Self, _rng: &mut R) -> Self {
            Scalar((self.0 + other.0) * 0.5)
        }
    }

    struct Abs;
    impl CompatibilityDistance<Scalar> for Abs {
        fn distance(&self, a: &Scalar, b: &Scalar) -> f32 {
            (a.0 - b.0).abs()
        }
    }

    fn make_pop(values: &[f32]) -> Vec<Phenotype<Scalar>> {
        values
            .iter()
            .map(|&v| Phenotype {
                genotype: Scalar(v),
                fitness: 1.0,
                objectives: vec![],
                descriptor: vec![],
            })
            .collect()
    }

    #[test]
    fn assign_clusters_close_genotypes() {
        let pop = make_pop(&[0.0, 0.05, 0.1, 5.0, 5.1]);
        let mut spec = Speciation::new(Abs, 0.5, 2);
        spec.assign(&pop);
        assert_eq!(spec.species().len(), 2);
    }

    #[test]
    fn fitness_sharing_divides_by_species_size() {
        let mut pop = make_pop(&[0.0, 0.1, 0.2, 5.0]);
        for p in &mut pop {
            p.fitness = 4.0;
        }
        let mut spec = Speciation::new(Abs, 0.5, 2);
        spec.assign(&pop);
        spec.share_fitness(&mut pop);

        // First three are in one species (size 3), last is alone (size 1).
        assert!((pop[0].fitness - 4.0 / 3.0).abs() < 1e-5);
        assert!((pop[1].fitness - 4.0 / 3.0).abs() < 1e-5);
        assert!((pop[2].fitness - 4.0 / 3.0).abs() < 1e-5);
        assert!((pop[3].fitness - 4.0).abs() < 1e-5);
    }

    #[test]
    fn threshold_increases_when_too_many_species() {
        let pop = make_pop(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let mut spec = Speciation::new(Abs, 0.5, 1).with_threshold_step(0.5);
        spec.assign(&pop);
        let t0 = spec.threshold();
        spec.adjust_threshold();
        assert!(spec.threshold() > t0);
    }

    #[test]
    fn threshold_decreases_when_too_few_species() {
        let pop = make_pop(&[0.0, 0.05, 0.1]);
        let mut spec = Speciation::new(Abs, 5.0, 5).with_threshold_step(0.5);
        spec.assign(&pop);
        let t0 = spec.threshold();
        spec.adjust_threshold();
        assert!(spec.threshold() < t0);
    }

    #[test]
    fn threshold_floors_at_min() {
        let pop = make_pop(&[0.0]);
        let mut spec = Speciation::new(Abs, 0.2, 99)
            .with_threshold_step(1.0)
            .with_min_threshold(0.1);
        spec.assign(&pop);
        spec.adjust_threshold();
        assert!((spec.threshold() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn species_ids_are_stable_across_generations() {
        let mut pop = make_pop(&[0.0, 0.05, 5.0, 5.05]);
        let mut spec = Speciation::new(Abs, 0.5, 2);
        spec.assign(&pop);
        let ids_gen0: Vec<u64> = spec.species().iter().map(|s| s.id).collect();

        // Mutate slightly and reassign; species IDs should persist.
        for p in &mut pop {
            p.genotype.0 += 0.01;
        }
        spec.assign(&pop);
        let ids_gen1: Vec<u64> = spec.species().iter().map(|s| s.id).collect();
        assert_eq!(ids_gen0, ids_gen1);
    }

    #[test]
    fn empty_species_dropped() {
        let pop = make_pop(&[0.0, 5.0]);
        let mut spec = Speciation::new(Abs, 0.5, 2);
        spec.assign(&pop);
        assert_eq!(spec.species().len(), 2);

        // Now reassign with all members in one cluster.
        let pop2 = make_pop(&[0.0, 0.05]);
        spec.assign(&pop2);
        assert_eq!(spec.species().len(), 1);
    }

    #[test]
    fn species_count_converges_toward_target() {
        // Wide range of values; tune threshold over many generations.
        let pop = make_pop(&[
            0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5,
        ]);
        let target = 4;
        let mut spec = Speciation::new(Abs, 0.5, target).with_threshold_step(0.1);
        for _ in 0..100 {
            spec.assign(&pop);
            spec.adjust_threshold();
        }
        // Should be at or near target after convergence.
        let count = spec.species().len();
        assert!(
            count.abs_diff(target) <= 1,
            "expected count near {target}, got {count}"
        );
    }
}
