//! Tests for MapElites coverage, qd_score, and CSV export.

use rand::Rng;
use serde::{Deserialize, Serialize};
use symbios_genetics::algorithms::map_elites::MapElites;
use symbios_genetics::{Evaluator, Genotype};

#[derive(Clone, Serialize, Deserialize, Debug)]
struct OneD(f32);

impl Genotype for OneD {
    fn mutate<R: Rng>(&mut self, _rng: &mut R, _rate: f32) {}
    fn crossover<R: Rng>(&self, other: &Self, _rng: &mut R) -> Self {
        OneD((self.0 + other.0) / 2.0)
    }
}

/// Evaluator placing each individual in a 1D descriptor cell equal to `g.0`,
/// with fitness equal to `g.0` so each cell's elite has a known fitness.
struct DescIsValue;
impl Evaluator<OneD> for DescIsValue {
    fn evaluate(&self, g: &OneD) -> (f32, Vec<f32>, Vec<f32>) {
        (g.0, vec![g.0], vec![g.0.clamp(0.0, 1.0)])
    }
}

#[test]
fn coverage_is_zero_on_empty_archive() {
    let me = MapElites::<OneD>::new(10, 0.1, 64, 42);
    assert_eq!(me.coverage(), 0.0);
}

#[test]
fn coverage_reflects_distinct_cells_filled() {
    let mut me = MapElites::<OneD>::new(10, 0.1, 64, 42);
    // Descriptors 0.05, 0.15, 0.25, 0.35, 0.45 land in distinct bins 0..=4.
    let seeds = vec![OneD(0.05), OneD(0.15), OneD(0.25), OneD(0.35), OneD(0.45)];
    me.seed_population(seeds, &DescIsValue);
    assert_eq!(me.archive_len(), 5);
    // 1D, resolution=10 → total cells = 10, 5 occupied → coverage = 0.5
    assert!((me.coverage() - 0.5).abs() < 1e-9);
}

#[test]
fn qd_score_is_zero_on_empty_archive() {
    let me = MapElites::<OneD>::new(10, 0.1, 64, 42);
    assert_eq!(me.qd_score(), 0.0);
}

#[test]
fn qd_score_sums_fitness_across_cells() {
    let mut me = MapElites::<OneD>::new(10, 0.1, 64, 42);
    // Distinct cells, fitness == descriptor value.
    let seeds = vec![OneD(0.05), OneD(0.15), OneD(0.25)];
    me.seed_population(seeds, &DescIsValue);
    let expected = 0.05_f64 + 0.15_f64 + 0.25_f64;
    assert!(
        (me.qd_score() - expected).abs() < 1e-5,
        "qd_score = {}, expected {}",
        me.qd_score(),
        expected
    );
}

#[cfg(feature = "export")]
#[test]
fn export_csv_writes_header_and_one_row_per_cell() {
    let mut me = MapElites::<OneD>::new(10, 0.1, 64, 42);
    me.seed_population(vec![OneD(0.05), OneD(0.15), OneD(0.95)], &DescIsValue);

    let mut out = Vec::new();
    me.export_csv(&mut out).expect("export_csv failed");
    let text = String::from_utf8(out).expect("non-utf8 output");

    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[0], "key,descriptor,fitness,objectives,genotype_hash");
    assert_eq!(lines.len(), 1 + 3, "header + 3 elites");

    // Each data row has 5 comma-separated columns.
    for row in &lines[1..] {
        assert_eq!(row.matches(',').count(), 4, "row malformed: {row}");
    }
}

#[cfg(feature = "export")]
#[test]
fn export_csv_is_deterministic() {
    fn run() -> String {
        let mut me = MapElites::<OneD>::new(10, 0.1, 64, 42);
        me.seed_population(vec![OneD(0.05), OneD(0.55), OneD(0.85)], &DescIsValue);
        let mut out = Vec::new();
        me.export_csv(&mut out).unwrap();
        String::from_utf8(out).unwrap()
    }
    assert_eq!(run(), run());
}

#[cfg(feature = "export")]
#[test]
fn export_csv_hashes_distinct_genotypes_distinctly() {
    let mut me = MapElites::<OneD>::new(10, 0.1, 64, 42);
    me.seed_population(vec![OneD(0.05), OneD(0.55)], &DescIsValue);

    let mut out = Vec::new();
    me.export_csv(&mut out).unwrap();
    let text = String::from_utf8(out).unwrap();

    let hashes: Vec<&str> = text
        .lines()
        .skip(1)
        .map(|row| row.rsplit(',').next().unwrap())
        .collect();

    assert_eq!(hashes.len(), 2);
    assert_ne!(
        hashes[0], hashes[1],
        "different genotypes must hash differently"
    );
    // 16 hex chars = 64-bit seahash output.
    for h in &hashes {
        assert_eq!(h.len(), 16, "hash {h} not 16 hex chars");
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()), "non-hex: {h}");
    }
}
