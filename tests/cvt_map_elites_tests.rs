//! Tests for CvtMapElites.

use rand::Rng;
use serde::{Deserialize, Serialize};
use symbios_genetics::algorithms::cvt_map_elites::CvtMapElites;
use symbios_genetics::{Evaluator, Evolver, Genotype};

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
struct TwoD(f32, f32);

impl Genotype for TwoD {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        if rng.random::<f32>() < rate {
            self.0 = (self.0 + rng.random::<f32>() - 0.5).clamp(0.0, 1.0);
        }
        if rng.random::<f32>() < rate {
            self.1 = (self.1 + rng.random::<f32>() - 0.5).clamp(0.0, 1.0);
        }
    }
    fn crossover<R: Rng>(&self, other: &Self, _rng: &mut R) -> Self {
        TwoD((self.0 + other.0) / 2.0, (self.1 + other.1) / 2.0)
    }
}

/// Descriptor = position; fitness rewards proximity to (0.5, 0.5).
struct CenterFit;
impl Evaluator<TwoD> for CenterFit {
    fn evaluate(&self, g: &TwoD) -> (f32, Vec<f32>, Vec<f32>) {
        let dx = g.0 - 0.5;
        let dy = g.1 - 0.5;
        let f = -(dx * dx + dy * dy).sqrt();
        (f, vec![f], vec![g.0, g.1])
    }
}

#[test]
fn with_lloyd_is_deterministic() {
    let a = CvtMapElites::<TwoD>::with_lloyd(64, 2, 5_000, 10, 0.2, 16, 42);
    let b = CvtMapElites::<TwoD>::with_lloyd(64, 2, 5_000, 10, 0.2, 16, 42);
    assert_eq!(a.centroids(), b.centroids());
}

#[test]
fn different_seeds_produce_different_centroids() {
    let a = CvtMapElites::<TwoD>::with_lloyd(64, 2, 5_000, 10, 0.2, 16, 1);
    let b = CvtMapElites::<TwoD>::with_lloyd(64, 2, 5_000, 10, 0.2, 16, 2);
    assert_ne!(a.centroids(), b.centroids());
}

#[test]
fn assign_to_centroid_returns_nearest() {
    let centroids = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    let me = CvtMapElites::<TwoD>::new(centroids, 0.1, 16, 42);
    assert_eq!(me.assign_to_centroid(&[0.1, 0.1]), 0);
    assert_eq!(me.assign_to_centroid(&[0.9, 0.1]), 1);
    assert_eq!(me.assign_to_centroid(&[0.1, 0.9]), 2);
    assert_eq!(me.assign_to_centroid(&[0.9, 0.9]), 3);
}

#[test]
fn coverage_and_qd_score_track_archive_state() {
    let centroids = vec![
        vec![0.1, 0.1],
        vec![0.9, 0.1],
        vec![0.5, 0.9],
        vec![0.5, 0.5],
    ];
    let mut me = CvtMapElites::<TwoD>::new(centroids, 0.1, 16, 42);
    assert_eq!(me.coverage(), 0.0);
    assert_eq!(me.qd_score(), 0.0);

    me.seed_population(
        vec![TwoD(0.1, 0.1), TwoD(0.9, 0.1), TwoD(0.5, 0.9)],
        &CenterFit,
    );
    assert_eq!(me.archive_len(), 3);
    assert!((me.coverage() - 0.75).abs() < 1e-9);
    // Three distinct corners; sum of three negative distances to (0.5, 0.5).
    assert!(me.qd_score() < 0.0);
}

#[test]
fn step_is_deterministic_under_fixed_seed() {
    fn run() -> (usize, Vec<f32>) {
        let mut me = CvtMapElites::<TwoD>::with_lloyd(32, 2, 2_000, 8, 0.3, 16, 42);
        let initial: Vec<TwoD> = (0..40)
            .map(|i| {
                let t = i as f32 / 40.0;
                TwoD(t, 1.0 - t)
            })
            .collect();
        me.seed_population(initial, &CenterFit);
        for _ in 0..20 {
            me.step(&CenterFit);
        }
        let archive_len = me.archive_len();
        let fitnesses: Vec<f32> = me.archive_iter().map(|(_, p)| p.fitness).collect();
        (archive_len, fitnesses)
    }
    assert_eq!(run(), run());
}

#[test]
fn coverage_difference_grid_vs_cvt_in_high_dim() {
    // High descriptor dimension where a grid would explode (10^6 cells)
    // but CVT stays bounded at num_centroids.
    let mut cvt = CvtMapElites::<TwoD>::with_lloyd(
        128,   // num_centroids
        6,     // descriptor_dim — would be 10^6 = 1M grid cells at res=10
        3_000, // num_samples
        8,     // lloyd_iters
        0.2, 16, 42,
    );

    // 6D descriptor evaluator: position + 4 zero-padding dims.
    struct SixDFit;
    impl Evaluator<TwoD> for SixDFit {
        fn evaluate(&self, g: &TwoD) -> (f32, Vec<f32>, Vec<f32>) {
            let f = -((g.0 - 0.5).powi(2) + (g.1 - 0.5).powi(2)).sqrt();
            (f, vec![f], vec![g.0, g.1, 0.5, 0.5, 0.5, 0.5])
        }
    }

    cvt.seed_population(
        (0..50)
            .map(|i| TwoD(i as f32 / 50.0, (49 - i) as f32 / 50.0))
            .collect(),
        &SixDFit,
    );
    for _ in 0..10 {
        cvt.step(&SixDFit);
    }

    // CVT bounds archive size to num_centroids regardless of dim.
    assert!(cvt.archive_len() <= cvt.num_centroids());
    // And we have non-trivial coverage from the seeded diversity.
    assert!(cvt.archive_len() > 0);
}

#[test]
#[should_panic(expected = "centroids must be non-empty")]
fn new_rejects_empty_centroids() {
    let _ = CvtMapElites::<TwoD>::new(vec![], 0.1, 16, 42);
}

#[test]
#[should_panic(expected = "all centroids must have the same dimension")]
fn new_rejects_mixed_dimension_centroids() {
    let _ = CvtMapElites::<TwoD>::new(vec![vec![0.0, 0.0], vec![1.0]], 0.1, 16, 42);
}

#[test]
#[should_panic(expected = "batch_size must be greater than 0")]
fn new_rejects_zero_batch_size() {
    let _ = CvtMapElites::<TwoD>::new(vec![vec![0.5, 0.5]], 0.1, 0, 42);
}

#[test]
fn serde_roundtrip_preserves_archive() {
    let mut me = CvtMapElites::<TwoD>::with_lloyd(16, 2, 1_000, 5, 0.2, 8, 42);
    me.seed_population(
        vec![TwoD(0.1, 0.9), TwoD(0.5, 0.5), TwoD(0.9, 0.1)],
        &CenterFit,
    );
    let before_len = me.archive_len();

    let bytes = bincode::serialize(&me).expect("serialize failed");
    let restored: CvtMapElites<TwoD> = bincode::deserialize(&bytes).expect("deserialize failed");
    assert_eq!(restored.archive_len(), before_len);
    assert_eq!(restored.num_centroids(), me.num_centroids());
    assert_eq!(restored.centroids(), me.centroids());
}
