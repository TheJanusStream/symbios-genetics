//! Tests for novelty search.

use rand::Rng;
use serde::{Deserialize, Serialize};
use symbios_genetics::algorithms::novelty_search::{
    ArchivePolicy, BehaviourDistance, EuclideanDistance, NoveltyConfig, NoveltySearch,
};
use symbios_genetics::{Evaluator, Evolver, Genotype};

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
struct OneD(f32);

impl Genotype for OneD {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        if rng.random::<f32>() < rate {
            self.0 = (self.0 + (rng.random::<f32>() - 0.5) * 0.2).clamp(-10.0, 10.0);
        }
    }
    fn crossover<R: Rng>(&self, other: &Self, _rng: &mut R) -> Self {
        OneD((self.0 + other.0) / 2.0)
    }
}

/// Descriptor = position; fitness = 0 (so novelty drives selection alone).
struct PositionAsDescriptor;
impl Evaluator<OneD> for PositionAsDescriptor {
    fn evaluate(&self, g: &OneD) -> (f32, Vec<f32>, Vec<f32>) {
        (0.0, vec![], vec![g.0])
    }
}

#[test]
fn euclidean_distance_is_correct() {
    let d = EuclideanDistance;
    assert!((d.distance(&[0.0, 0.0], &[3.0, 4.0]) - 5.0).abs() < 1e-6);
    assert_eq!(d.distance(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0);
}

#[test]
fn novelty_config_rejects_invalid_values() {
    use std::panic::catch_unwind;
    assert!(catch_unwind(|| NoveltyConfig::new(0, 1.0, ArchivePolicy::AlwaysAdd)).is_err());
    assert!(catch_unwind(|| NoveltyConfig::new(15, -0.1, ArchivePolicy::AlwaysAdd)).is_err());
    assert!(catch_unwind(|| NoveltyConfig::new(15, 1.1, ArchivePolicy::AlwaysAdd)).is_err());
    assert!(
        catch_unwind(|| NoveltyConfig::new(15, 1.0, ArchivePolicy::Probabilistic(0.0))).is_err()
    );
    assert!(
        catch_unwind(|| NoveltyConfig::new(15, 1.0, ArchivePolicy::Probabilistic(1.5))).is_err()
    );
}

#[test]
fn step_populates_novelty_aligned_with_population() {
    let initial: Vec<OneD> = (0..20).map(|i| OneD(i as f32 - 10.0)).collect();
    let cfg = NoveltyConfig::new(3, 1.0, ArchivePolicy::AlwaysAdd);
    let mut ns = NoveltySearch::new(initial, 0.0, 1, cfg, 42);
    ns.step(&PositionAsDescriptor);
    let nov_len = ns.novelty().len();
    let pop_len = ns.population().len();
    assert_eq!(
        pop_len, nov_len,
        "novelty length must match population length"
    );
    // Novelty was non-zero before step() rebuilt the population (we computed it then cleared).
    // After a fresh step, we'll see novelty values for the *new* population once we step again.
    ns.step(&PositionAsDescriptor);
    assert!(ns.novelty().iter().any(|&v| v > 0.0));
}

#[test]
fn always_add_archive_grows_each_step() {
    let initial: Vec<OneD> = (0..10).map(|i| OneD(i as f32)).collect();
    let cfg = NoveltyConfig::new(3, 1.0, ArchivePolicy::AlwaysAdd);
    let mut ns = NoveltySearch::new(initial, 0.0, 1, cfg, 42);
    let mut prev = 0;
    for _ in 0..5 {
        ns.step(&PositionAsDescriptor);
        let now = ns.archive_len();
        assert!(
            now > prev,
            "archive should grow under AlwaysAdd ({prev} -> {now})"
        );
        prev = now;
    }
}

#[test]
fn probabilistic_archive_grows_more_slowly_than_always_add() {
    let initial_a: Vec<OneD> = (0..30).map(|i| OneD(i as f32)).collect();
    let initial_b = initial_a.clone();
    let mut always = NoveltySearch::new(
        initial_a,
        0.0,
        1,
        NoveltyConfig::new(3, 1.0, ArchivePolicy::AlwaysAdd),
        42,
    );
    let mut prob = NoveltySearch::new(
        initial_b,
        0.0,
        1,
        NoveltyConfig::new(3, 1.0, ArchivePolicy::Probabilistic(0.1)),
        42,
    );
    for _ in 0..10 {
        always.step(&PositionAsDescriptor);
        prob.step(&PositionAsDescriptor);
    }
    assert!(
        prob.archive_len() < always.archive_len(),
        "probabilistic ({}) should grow slower than always-add ({})",
        prob.archive_len(),
        always.archive_len()
    );
}

#[test]
fn deterministic_under_fixed_seed() {
    fn run() -> (usize, Vec<f32>) {
        let initial: Vec<OneD> = (0..20).map(|i| OneD(i as f32 / 2.0 - 5.0)).collect();
        let cfg = NoveltyConfig::new(5, 1.0, ArchivePolicy::AlwaysAdd);
        let mut ns = NoveltySearch::new(initial, 0.2, 2, cfg, 42);
        for _ in 0..10 {
            ns.step(&PositionAsDescriptor);
        }
        // Use genotypes (always present) plus archive size as the determinism signature.
        let archive_len = ns.archive_len();
        let positions: Vec<f32> = ns.population().iter().map(|p| p.genotype.0).collect();
        (archive_len, positions)
    }
    assert_eq!(run(), run());
}

/// Deceptive 1D problem: fitness has a sharp local maximum at x = 0.0,
/// surrounded by a low-fitness moat. The global optimum is at x = 5.0.
/// Pure fitness selection gets stuck at x = 0; novelty selection escapes.
#[derive(Clone, Serialize, Deserialize, Debug)]
struct DeceptiveDNA(f32);

impl Genotype for DeceptiveDNA {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        if rng.random::<f32>() < rate {
            self.0 = (self.0 + (rng.random::<f32>() - 0.5) * 0.5).clamp(-10.0, 10.0);
        }
    }
    fn crossover<R: Rng>(&self, other: &Self, _rng: &mut R) -> Self {
        DeceptiveDNA((self.0 + other.0) / 2.0)
    }
}

struct DeceptiveTrap;
impl Evaluator<DeceptiveDNA> for DeceptiveTrap {
    fn evaluate(&self, g: &DeceptiveDNA) -> (f32, Vec<f32>, Vec<f32>) {
        // Sharp local max at 0 (height 5), broad global max at 5 (height 1),
        // moat between them with negative fitness around |x| ~ 1..4.
        let local = 5.0 * (-100.0 * g.0 * g.0).exp();
        let global = (-((g.0 - 5.0).powi(2))).exp();
        let fitness = local + global - 0.5;
        (fitness, vec![fitness], vec![g.0])
    }
}

#[test]
fn novelty_explores_further_than_pure_fitness() {
    // All initial individuals are clustered around the deceptive trap at x=0.
    let initial: Vec<DeceptiveDNA> = (0..30).map(|i| DeceptiveDNA(i as f32 * 0.01)).collect();

    // Pure fitness search: alpha = 0.0
    let mut fitness_only = NoveltySearch::new(
        initial.clone(),
        0.3,
        2,
        NoveltyConfig::new(5, 0.0, ArchivePolicy::AlwaysAdd),
        42,
    );
    for _ in 0..50 {
        fitness_only.step(&DeceptiveTrap);
    }
    let max_pos_fitness: f32 = fitness_only
        .population()
        .iter()
        .map(|p| p.genotype.0.abs())
        .fold(0.0, f32::max);

    // Pure novelty search: alpha = 1.0
    let mut novelty_only = NoveltySearch::new(
        initial,
        0.3,
        2,
        NoveltyConfig::new(5, 1.0, ArchivePolicy::AlwaysAdd),
        42,
    );
    for _ in 0..50 {
        novelty_only.step(&DeceptiveTrap);
    }
    let max_pos_novelty: f32 = novelty_only
        .population()
        .iter()
        .map(|p| p.genotype.0.abs())
        .fold(0.0, f32::max);

    // Novelty should reach further from the deceptive trap.
    assert!(
        max_pos_novelty > max_pos_fitness,
        "novelty search should explore further than fitness-only on the deceptive trap.\n\
         fitness-only max |x|: {max_pos_fitness}\n\
         novelty-only max |x|: {max_pos_novelty}"
    );
}

#[test]
fn custom_distance_via_with_distance() {
    /// Manhattan (L1) distance — a different, valid behaviour distance.
    #[derive(Clone, Copy, Default, Serialize, Deserialize)]
    struct ManhattanDistance;
    impl BehaviourDistance for ManhattanDistance {
        fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
        }
    }

    let initial: Vec<OneD> = (0..10).map(|i| OneD(i as f32)).collect();
    let cfg = NoveltyConfig::new(3, 1.0, ArchivePolicy::AlwaysAdd);
    let mut ns = NoveltySearch::with_distance(initial, 0.1, 1, cfg, ManhattanDistance, 42);
    ns.step(&PositionAsDescriptor);
    assert_eq!(ns.novelty().len(), ns.population().len());
}

#[test]
fn serde_roundtrip_preserves_state() {
    let initial: Vec<OneD> = (0..20).map(|i| OneD(i as f32 / 2.0)).collect();
    let cfg = NoveltyConfig::new(3, 0.5, ArchivePolicy::Probabilistic(0.5));
    let mut ns = NoveltySearch::new(initial, 0.1, 2, cfg, 42);
    for _ in 0..5 {
        ns.step(&PositionAsDescriptor);
    }
    let archive_len = ns.archive_len();
    let pop_len = ns.population().len();

    let bytes = bincode::serialize(&ns).expect("serialize");
    let mut restored: NoveltySearch<OneD> = bincode::deserialize(&bytes).expect("deserialize");
    assert_eq!(restored.archive_len(), archive_len);
    assert_eq!(restored.population().len(), pop_len);

    // Continued evolution after restore must work.
    restored.step(&PositionAsDescriptor);
}
