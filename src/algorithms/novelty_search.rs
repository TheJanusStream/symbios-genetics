//! Novelty search (Lehman & Stanley, 2011).
//!
//! Novelty search ignores the fitness landscape and instead drives selection
//! by *behavioural novelty* — how different an individual's descriptor is
//! from the descriptors it has previously seen. This breaks out of deceptive
//! local optima where pure fitness search gets stuck.
//!
//! # Algorithm
//!
//! Each generation:
//!
//! 1. Evaluate population. Each phenotype gets `(fitness, objectives, descriptor)`.
//! 2. Compute *novelty* for each individual as the mean distance to its `k`
//!    nearest neighbours in the union of (current population descriptors,
//!    behavioural archive).
//! 3. Compute a selection score: `alpha * novelty + (1 - alpha) * fitness`.
//!    `alpha = 1.0` is pure novelty search; `alpha = 0.0` is fitness-only;
//!    intermediate values blend the two ("novelty + fitness" hybrid).
//! 4. Optionally add each descriptor to the behaviour archive according to
//!    the chosen [`ArchivePolicy`](crate::algorithms::novelty_search::ArchivePolicy).
//! 5. Tournament-select on the score, crossover, mutate, with elitism.
//!
//! # Knobs
//!
//! | Knob | Typical | Effect |
//! |------|---------|--------|
//! | `k` | 15 | kNN window for novelty. Smaller = more local; larger = smoother. |
//! | `alpha` | 1.0 | Novelty/fitness blend. 1.0 = pure novelty (deceptive problems). 0.5 = hybrid. |
//! | `policy` | `Probabilistic(0.05)` | When to add descriptors to the behaviour archive. |
//! | `tournament_size` | 3 | Selection pressure. Higher = greedier. |
//! | `elitism` | 1-5 | Number of top-scoring individuals copied unchanged. |
//!
//! # References
//!
//! Lehman, J., & Stanley, K. O. (2011). Abandoning Objectives: Evolution
//! Through the Search for Novelty Alone. Evolutionary Computation, 19(2).

use crate::{Evaluator, Evolver, Genotype, Phenotype};
use rand::Rng;
use rand::prelude::SeedableRng;
use rand_pcg::Pcg64;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Distance function over flat-vector behaviour descriptors.
///
/// Implementations must be deterministic and symmetric. Used by
/// [`NoveltySearch`] for kNN novelty computation.
pub trait BehaviourDistance: Send + Sync {
    /// Distance between two descriptors. Both slices have the same length
    /// (the descriptor dimension produced by the user's evaluator).
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;
}

/// Default Euclidean (L2) distance for `Vec<f32>` descriptors.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct EuclideanDistance;

impl BehaviourDistance for EuclideanDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

/// When (and how often) to add an individual's descriptor to the behaviour
/// archive that drives novelty computation.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum ArchivePolicy {
    /// Every offspring's descriptor is added to the archive. The archive
    /// grows unboundedly; novelty cost grows with archive size. Best for
    /// short runs or small populations.
    AlwaysAdd,
    /// Each offspring's descriptor is added with probability `p` (where
    /// `0.0 < p <= 1.0`). The archive grows on average `p * batch_size`
    /// entries per generation.
    Probabilistic(f32),
}

/// Bundles the novelty-search-specific knobs separately from GA-shared knobs
/// (`mutation_rate`, `elitism`, etc.) for a cleaner constructor signature.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct NoveltyConfig {
    /// kNN window for novelty computation. Must be > 0. Typical: 15.
    pub k: usize,
    /// Novelty/fitness blend. `1.0` = pure novelty, `0.0` = fitness-only.
    /// Must be in `[0.0, 1.0]`. Typical: `1.0` for deceptive problems.
    pub alpha: f32,
    /// How often to add descriptors to the behaviour archive.
    pub policy: ArchivePolicy,
}

impl NoveltyConfig {
    /// Convenience constructor.
    ///
    /// # Panics
    ///
    /// Panics if `k == 0`, `alpha` is outside `[0, 1]`, or `policy` is
    /// `Probabilistic(p)` with `p` outside `(0, 1]`.
    pub fn new(k: usize, alpha: f32, policy: ArchivePolicy) -> Self {
        let cfg = Self { k, alpha, policy };
        cfg.validate().expect("invalid NoveltyConfig");
        cfg
    }

    fn validate(&self) -> Result<(), String> {
        if self.k == 0 {
            return Err("k must be greater than 0".into());
        }
        if !(0.0..=1.0).contains(&self.alpha) {
            return Err(format!("alpha must be in [0.0, 1.0], got {}", self.alpha));
        }
        if let ArchivePolicy::Probabilistic(p) = self.policy
            && (!(0.0..=1.0).contains(&p) || p <= 0.0)
        {
            return Err(format!(
                "probabilistic archive policy requires 0.0 < p <= 1.0, got {p}"
            ));
        }
        Ok(())
    }
}

fn cmp_f32_nan_last(a: f32, b: f32) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
    }
}

/// Internal representation for serialization with validating deserializer.
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "G: Genotype, D: BehaviourDistance + Serialize",
    deserialize = "G: Genotype, D: BehaviourDistance + Deserialize<'de>"
))]
struct NoveltySearchData<G: Genotype, D: BehaviourDistance> {
    population: Vec<Phenotype<G>>,
    pop_size: usize,
    novelty: Vec<f32>,
    behaviour_archive: Vec<Vec<f32>>,
    mutation_rate: f32,
    elitism: usize,
    k: usize,
    alpha: f32,
    policy: ArchivePolicy,
    distance: D,
    rng: Pcg64,
}

/// Novelty search evolver.
///
/// See module docs for algorithm and knob descriptions.
pub struct NoveltySearch<G: Genotype, D: BehaviourDistance = EuclideanDistance> {
    population: Vec<Phenotype<G>>,
    pop_size: usize,
    /// Aligned with `population`: `novelty[i]` is the most recently computed
    /// novelty score for `population[i]`. Empty before the first `step()`.
    novelty: Vec<f32>,
    /// Behavioural archive: descriptors only.
    behaviour_archive: Vec<Vec<f32>>,
    mutation_rate: f32,
    elitism: usize,
    k: usize,
    alpha: f32,
    policy: ArchivePolicy,
    distance: D,
    rng: Pcg64,
}

impl<G: Genotype, D: BehaviourDistance + Serialize> Serialize for NoveltySearch<G, D> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("NoveltySearch", 11)?;
        state.serialize_field("population", &self.population)?;
        state.serialize_field("pop_size", &self.pop_size)?;
        state.serialize_field("novelty", &self.novelty)?;
        state.serialize_field("behaviour_archive", &self.behaviour_archive)?;
        state.serialize_field("mutation_rate", &self.mutation_rate)?;
        state.serialize_field("elitism", &self.elitism)?;
        state.serialize_field("k", &self.k)?;
        state.serialize_field("alpha", &self.alpha)?;
        state.serialize_field("policy", &self.policy)?;
        state.serialize_field("distance", &self.distance)?;
        state.serialize_field("rng", &self.rng)?;
        state.end()
    }
}

impl<'de, G: Genotype, D: BehaviourDistance + Deserialize<'de>> Deserialize<'de>
    for NoveltySearch<G, D>
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        use serde::de::Error;

        let data = NoveltySearchData::<G, D>::deserialize(deserializer)?;

        // pop_size must match population length. The step() loop runs until
        // next_gen.len() reaches pop_size — a desynced pop_size is an OOM
        // vector (e.g. pop_size = usize::MAX with a tiny population).
        if data.pop_size != data.population.len() {
            return Err(De::Error::custom(format!(
                "pop_size ({}) does not match population length ({})",
                data.pop_size,
                data.population.len()
            )));
        }

        // Re-run the same hyper-parameter validation as NoveltyConfig::new,
        // closing the silent-corruption window where k=0 produced 0/0=NaN
        // and an out-of-range alpha skewed selection.
        let cfg = NoveltyConfig {
            k: data.k,
            alpha: data.alpha,
            policy: data.policy,
        };
        cfg.validate().map_err(De::Error::custom)?;

        if data.mutation_rate.is_nan() || data.mutation_rate.is_infinite() {
            return Err(De::Error::custom("mutation_rate must be a finite number"));
        }

        // novelty is either empty (pre-first-step) or aligned with population.
        if !data.novelty.is_empty() && data.novelty.len() != data.population.len() {
            return Err(De::Error::custom(format!(
                "novelty length ({}) does not match population length ({})",
                data.novelty.len(),
                data.population.len()
            )));
        }

        Ok(Self {
            population: data.population,
            pop_size: data.pop_size,
            novelty: data.novelty,
            behaviour_archive: data.behaviour_archive,
            mutation_rate: data.mutation_rate,
            elitism: data.elitism,
            k: data.k,
            alpha: data.alpha,
            policy: data.policy,
            distance: data.distance,
            rng: data.rng,
        })
    }
}

impl<G: Genotype> NoveltySearch<G, EuclideanDistance> {
    /// Creates a novelty search instance with the default Euclidean distance.
    ///
    /// See [`with_distance`](Self::with_distance) to use a custom distance.
    ///
    /// # Arguments
    ///
    /// * `initial_pop` - Initial population. Must be non-empty.
    /// * `mutation_rate` - Probability of mutation, in `[0.0, 1.0]`.
    /// * `elitism` - Number of top-scoring individuals copied unchanged each generation.
    /// * `config` - Novelty-search-specific knobs (`k`, `alpha`, `policy`).
    /// * `seed` - RNG seed for deterministic execution.
    ///
    /// # Panics
    ///
    /// Panics if `initial_pop` is empty or if `config` is invalid.
    pub fn new(
        initial_pop: Vec<G>,
        mutation_rate: f32,
        elitism: usize,
        config: NoveltyConfig,
        seed: u64,
    ) -> Self {
        Self::with_distance(
            initial_pop,
            mutation_rate,
            elitism,
            config,
            EuclideanDistance,
            seed,
        )
    }
}

impl<G: Genotype, D: BehaviourDistance> NoveltySearch<G, D> {
    /// Creates a novelty search instance with a caller-supplied distance.
    ///
    /// See [`new`](Self::new) for argument documentation.
    pub fn with_distance(
        initial_pop: Vec<G>,
        mutation_rate: f32,
        elitism: usize,
        config: NoveltyConfig,
        distance: D,
        seed: u64,
    ) -> Self {
        assert!(
            !initial_pop.is_empty(),
            "initial population must be non-empty"
        );
        config.validate().expect("invalid NoveltyConfig");

        let pop_size = initial_pop.len();
        if elitism >= pop_size {
            eprintln!(
                "Warning: NoveltySearch elitism ({}) >= pop_size ({}). \
                 Elitism will be clamped to {} to ensure evolution progresses.",
                elitism,
                pop_size,
                pop_size.saturating_sub(1)
            );
        }

        let population: Vec<Phenotype<G>> = initial_pop
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
            novelty: Vec::new(),
            behaviour_archive: Vec::new(),
            mutation_rate,
            elitism,
            k: config.k,
            alpha: config.alpha,
            policy: config.policy,
            distance,
            rng: Pcg64::seed_from_u64(seed),
        }
    }

    /// Returns the population size.
    pub fn pop_size(&self) -> usize {
        self.pop_size
    }

    /// Returns the kNN window size.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Returns the novelty/fitness blend factor.
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Sets the novelty/fitness blend factor.
    ///
    /// # Panics
    ///
    /// Panics if `alpha` is not in `[0.0, 1.0]`.
    pub fn set_alpha(&mut self, alpha: f32) {
        assert!(
            (0.0..=1.0).contains(&alpha),
            "alpha must be in [0.0, 1.0], got {alpha}"
        );
        self.alpha = alpha;
    }

    /// Returns the current archive policy.
    pub fn policy(&self) -> ArchivePolicy {
        self.policy
    }

    /// Returns the number of descriptors currently in the behavioural archive.
    pub fn archive_len(&self) -> usize {
        self.behaviour_archive.len()
    }

    /// Returns the descriptors currently in the behavioural archive.
    pub fn archive(&self) -> &[Vec<f32>] {
        &self.behaviour_archive
    }

    /// Returns the novelty scores used for selection in the most recent
    /// [`step`](Evolver::step), one per population slot. Empty before the
    /// first step.
    ///
    /// These values align with the population state *that drove the last
    /// selection*, not with the current population (which is the offspring
    /// produced by that selection). They are useful for monitoring how
    /// novelty evolved over the run; they are *not* a per-individual
    /// score for the current population.
    pub fn novelty(&self) -> &[f32] {
        &self.novelty
    }

    /// Computes the score (alpha-blended novelty + fitness) for an individual.
    fn score(&self, i: usize) -> f32 {
        let nov = self.novelty.get(i).copied().unwrap_or(0.0);
        let fit = self.population[i].fitness;
        let fit = if fit.is_nan() { 0.0 } else { fit };
        self.alpha * nov + (1.0 - self.alpha) * fit
    }

    /// Picks the index of the tournament winner among `tournament_size` random
    /// individuals from the population, ranked by `scores`.
    fn tournament_pick(&mut self, tournament_size: usize, scores: &[f32]) -> usize {
        let n = self.population.len();
        let mut best_idx = self.rng.random_range(0..n);
        let mut best_score = scores[best_idx];
        for _ in 1..tournament_size {
            let candidate = self.rng.random_range(0..n);
            if cmp_f32_nan_last(scores[candidate], best_score) == Ordering::Greater {
                best_idx = candidate;
                best_score = scores[candidate];
            }
        }
        best_idx
    }
}

/// Computes mean distance to k nearest neighbours among `others`.
/// Returns 0.0 if `others` is empty (after filtering for matching dimension).
fn mean_knn_distance<'a, D, I>(point: &[f32], others: I, k: usize, distance: &D) -> f32
where
    D: BehaviourDistance,
    I: Iterator<Item = &'a [f32]>,
{
    let mut distances: Vec<f32> = others
        .filter(|o: &&[f32]| o.len() == point.len())
        .map(|o| distance.distance(point, o))
        .filter(|d: &f32| !d.is_nan())
        .collect();
    if distances.is_empty() {
        return 0.0;
    }
    distances.sort_by(|a: &f32, b: &f32| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let take = k.min(distances.len());
    distances.iter().take(take).sum::<f32>() / take as f32
}

impl<G: Genotype, D: BehaviourDistance> Evolver<G> for NoveltySearch<G, D> {
    fn step<E: Evaluator<G>>(&mut self, evaluator: &E) {
        if self.population.is_empty() {
            return;
        }

        // 1. Evaluate population (parallel where possible).
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

        // 2. Compute novelty: mean distance to k nearest in (population ∪ archive).
        // For each individual, neighbours are everyone *else* in the population plus
        // every archive descriptor. NaN-descriptor individuals get novelty 0 so they
        // don't game selection.
        let pop_descriptors: Vec<&[f32]> = self
            .population
            .iter()
            .map(|p| p.descriptor.as_slice())
            .collect();

        let n = self.population.len();
        self.novelty.clear();
        self.novelty.reserve(n);
        for i in 0..n {
            let me = pop_descriptors[i];
            if me.is_empty() || me.iter().any(|v| v.is_nan()) {
                self.novelty.push(0.0);
                continue;
            }
            let neighbours = pop_descriptors
                .iter()
                .enumerate()
                .filter_map(|(j, d)| if j == i { None } else { Some(*d) })
                .chain(self.behaviour_archive.iter().map(|v| v.as_slice()));
            let nov = mean_knn_distance(me, neighbours, self.k, &self.distance);
            self.novelty.push(nov);
        }

        // 3. Update behaviour archive per policy. Skip NaN-descriptor and empty entries.
        let add_to_archive: Box<dyn Fn(&mut Pcg64) -> bool> = match self.policy {
            ArchivePolicy::AlwaysAdd => Box::new(|_| true),
            ArchivePolicy::Probabilistic(p) => Box::new(move |rng| rng.random::<f32>() < p),
        };
        for p in &self.population {
            if p.descriptor.is_empty() || p.descriptor.iter().any(|v| v.is_nan()) {
                continue;
            }
            if add_to_archive(&mut self.rng) {
                self.behaviour_archive.push(p.descriptor.clone());
            }
        }

        // 4. Selection by score = alpha*novelty + (1-alpha)*fitness, with elitism.
        // Sort indices by score descending; tournament from the score-ordered list.
        let mut indices: Vec<usize> = (0..n).collect();
        let scores: Vec<f32> = (0..n).map(|i| self.score(i)).collect();
        indices.sort_by(|&a, &b| cmp_f32_nan_last(scores[b], scores[a]));

        let effective_elitism = self.elitism.min(self.pop_size.saturating_sub(1));
        let mut next_gen: Vec<Phenotype<G>> = indices
            .iter()
            .take(effective_elitism)
            .map(|&i| self.population[i].clone())
            .collect();

        let tournament_size = 3.min(n);
        while next_gen.len() < self.pop_size {
            let p_a = self.tournament_pick(tournament_size, &scores);
            let p_b = self.tournament_pick(tournament_size, &scores);
            let mut child_dna = self.population[p_a]
                .genotype
                .crossover(&self.population[p_b].genotype, &mut self.rng);
            child_dna.mutate(&mut self.rng, self.mutation_rate);
            next_gen.push(Phenotype {
                genotype: child_dna,
                fitness: 0.0,
                objectives: vec![],
                descriptor: vec![],
            });
        }
        self.population = next_gen;
        // Note: `self.novelty` now holds the values used for *this* step's selection,
        // which aligned with the population *before* it was replaced. The next step()
        // overwrites these. Until then, `novelty()` returns the most recent selection
        // novelty — useful for monitoring but not aligned with the new population's
        // genotypes.
    }

    fn population(&mut self) -> &[Phenotype<G>] {
        &self.population
    }
}
