//! Tests for issues identified in the adversarial code review.
//! Each test documents a specific issue and verifies the fix.

use rand::Rng;
use serde::{Deserialize, Serialize};
use symbios_genetics::{
    Evaluator, Evolver, Genotype, algorithms::map_elites::MapElites, algorithms::simple::SimpleGA,
};

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
struct TestDNA(f32);

impl Genotype for TestDNA {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        if rng.random::<f32>() < rate {
            self.0 += rng.random::<f32>() - 0.5;
        }
    }
    fn crossover<R: Rng>(&self, other: &Self, _rng: &mut R) -> Self {
        TestDNA((self.0 + other.0) / 2.0)
    }
}

struct TestEval;
impl Evaluator<TestDNA> for TestEval {
    fn evaluate(&self, genotype: &TestDNA) -> (f32, Vec<f32>, Vec<f32>) {
        (
            genotype.0,
            vec![genotype.0, -genotype.0],
            vec![genotype.0.clamp(0.0, 1.0)],
        )
    }
}

// ============================================================================
// Issue #25: Non-Deterministic Execution in MapElites (HashMap iteration order)
// ============================================================================

/// Test that MapElites produces deterministic results across runs with the same seed.
/// The fix is to use BTreeMap instead of HashMap for deterministic iteration order.
#[test]
fn test_map_elites_deterministic_execution() {
    fn run_evolution(seed: u64) -> Vec<f32> {
        let mut engine = MapElites::<TestDNA>::new(10, 0.5, seed);
        engine.set_batch_size(8);

        // Seed with diverse individuals
        let initial: Vec<TestDNA> = (0..20).map(|i| TestDNA(i as f32 / 20.0)).collect();
        engine.seed_population(initial, &TestEval);

        // Run several steps
        for _ in 0..10 {
            engine.step(&TestEval);
        }

        // Collect results in a deterministic order (sorted by archive key)
        let mut results: Vec<(Vec<usize>, f32)> = engine
            .archive_iter()
            .map(|(k, p)| (k.clone(), p.fitness))
            .collect();
        results.sort_by(|a, b| a.0.cmp(&b.0));
        results.iter().map(|(_, f)| *f).collect()
    }

    // Run the same evolution twice with the same seed
    let results1 = run_evolution(42);
    let results2 = run_evolution(42);

    // Results should be identical
    assert_eq!(
        results1, results2,
        "MapElites should produce identical results with the same seed.\n\
         Run 1: {:?}\nRun 2: {:?}\n\
         This failure indicates non-deterministic iteration order (HashMap bug).",
        results1, results2
    );
}

/// Test that archive iteration order is consistent (deterministic).
#[test]
fn test_map_elites_archive_iteration_order_deterministic() {
    let mut engine = MapElites::<TestDNA>::new(5, 0.1, 42);

    // Seed with individuals that map to different bins
    let initial: Vec<TestDNA> = vec![
        TestDNA(0.1),
        TestDNA(0.3),
        TestDNA(0.5),
        TestDNA(0.7),
        TestDNA(0.9),
    ];
    engine.seed_population(initial, &TestEval);

    // Collect keys in iteration order multiple times
    let keys1: Vec<Vec<usize>> = engine.archive_keys().cloned().collect();
    let keys2: Vec<Vec<usize>> = engine.archive_keys().cloned().collect();
    let keys3: Vec<Vec<usize>> = engine.archive_keys().cloned().collect();

    // With BTreeMap, iteration order should be consistent and sorted
    assert_eq!(
        keys1, keys2,
        "Archive key iteration order should be consistent"
    );
    assert_eq!(
        keys2, keys3,
        "Archive key iteration order should be consistent"
    );

    // Verify keys are sorted (BTreeMap property)
    let mut sorted_keys = keys1.clone();
    sorted_keys.sort();
    assert_eq!(
        keys1, sorted_keys,
        "Archive keys should iterate in sorted order (BTreeMap)"
    );
}

// ============================================================================
// Issue #26: Panic via Integer Underflow in MapElites (resolution=0)
// ============================================================================

#[test]
#[should_panic(expected = "resolution must be greater than 0")]
fn test_map_elites_rejects_zero_resolution() {
    // resolution=0 would cause underflow in map_to_index: (resolution - 1)
    let _engine = MapElites::<TestDNA>::new(0, 0.1, 42);
}

#[test]
fn test_map_elites_resolution_one_works() {
    // resolution=1 is valid (single bin)
    let mut engine = MapElites::<TestDNA>::new(1, 0.1, 42);
    engine.seed_population(vec![TestDNA(0.5)], &TestEval);

    // All descriptors should map to bin 0
    let idx = engine.map_to_index(&[0.0]);
    assert_eq!(
        idx,
        vec![0],
        "All values should map to bin 0 with resolution=1"
    );

    let idx = engine.map_to_index(&[1.0]);
    assert_eq!(
        idx,
        vec![0],
        "All values should map to bin 0 with resolution=1"
    );

    let idx = engine.map_to_index(&[0.5]);
    assert_eq!(
        idx,
        vec![0],
        "All values should map to bin 0 with resolution=1"
    );
}

// ============================================================================
// Issue #27: seed_population Violates Elitism Invariant
// ============================================================================

#[test]
fn test_map_elites_seed_population_respects_elitism() {
    let mut engine = MapElites::<TestDNA>::new(10, 0.1, 42);

    struct ControlledEval;
    impl Evaluator<TestDNA> for ControlledEval {
        fn evaluate(&self, g: &TestDNA) -> (f32, Vec<f32>, Vec<f32>) {
            // Fitness equals the DNA value, descriptor always maps to bin 5
            (g.0, vec![g.0], vec![0.5])
        }
    }

    // Seed with a high-fitness individual (fitness = 100.0)
    engine.seed_population(vec![TestDNA(100.0)], &ControlledEval);
    let initial_fitness = engine.archive_get(&vec![5]).unwrap().fitness;
    assert_eq!(
        initial_fitness, 100.0,
        "Initial elite should have fitness 100.0"
    );

    // Try to seed with a lower-fitness individual in the same bin (fitness = 10.0)
    engine.seed_population(vec![TestDNA(10.0)], &ControlledEval);

    // The high-fitness individual should still be there (elitism preserved)
    let final_fitness = engine.archive_get(&vec![5]).unwrap().fitness;
    assert_eq!(
        final_fitness, 100.0,
        "seed_population should preserve better elites. \
         Expected fitness 100.0, got {}. \
         The lower-fitness seed (10.0) should not overwrite the elite.",
        final_fitness
    );
}

#[test]
fn test_map_elites_seed_population_replaces_worse() {
    let mut engine = MapElites::<TestDNA>::new(10, 0.1, 42);

    struct ControlledEval;
    impl Evaluator<TestDNA> for ControlledEval {
        fn evaluate(&self, g: &TestDNA) -> (f32, Vec<f32>, Vec<f32>) {
            (g.0, vec![g.0], vec![0.5])
        }
    }

    // Seed with a low-fitness individual (fitness = 10.0)
    engine.seed_population(vec![TestDNA(10.0)], &ControlledEval);

    // Seed with a higher-fitness individual (fitness = 100.0)
    engine.seed_population(vec![TestDNA(100.0)], &ControlledEval);

    // The higher-fitness individual should replace the lower one
    let final_fitness = engine.archive_get(&vec![5]).unwrap().fitness;
    assert_eq!(
        final_fitness, 100.0,
        "seed_population should replace worse individuals with better ones"
    );
}

// ============================================================================
// Issue #28: Broken Encapsulation in SimpleGA (public fields)
// ============================================================================

// Note: After the fix, fields should be private with getter methods.
// The test below verifies the public API works correctly.

#[test]
fn test_simple_ga_encapsulation_via_public_api() {
    let initial: Vec<TestDNA> = (0..10).map(|i| TestDNA(i as f32)).collect();
    let mut ga = SimpleGA::new(initial, 0.1, 2, 42);

    // Access via public API getter methods (not direct field access)
    let pop_size = ga.pop_size();
    assert_eq!(pop_size, 10, "pop_size() should return the population size");

    let mutation_rate = ga.mutation_rate();
    assert!((mutation_rate - 0.1).abs() < f32::EPSILON);

    let elitism = ga.elitism();
    assert_eq!(elitism, 2);

    // Verify population() returns correct data
    let pop = ga.population();
    assert_eq!(pop.len(), 10);
}

#[test]
fn test_simple_ga_population_size_invariant_maintained() {
    let initial: Vec<TestDNA> = (0..10).map(|i| TestDNA(i as f32)).collect();
    let mut ga = SimpleGA::new(initial, 0.1, 2, 42);

    // Run several steps
    for _ in 0..5 {
        ga.step(&TestEval);
    }

    // Population size should remain constant
    assert_eq!(
        ga.population().len(),
        10,
        "Population size invariant should be maintained by encapsulation"
    );
}

// ============================================================================
// Issue #29: MapElites Stall with Zero Batch Size
// ============================================================================

#[test]
#[should_panic(expected = "batch_size must be greater than 0")]
fn test_map_elites_rejects_zero_batch_size() {
    let mut engine = MapElites::<TestDNA>::new(10, 0.1, 42);
    engine.set_batch_size(0);
}

#[test]
fn test_map_elites_batch_size_one_works() {
    let mut engine = MapElites::<TestDNA>::new(10, 0.1, 42);
    engine.set_batch_size(1);

    engine.seed_population(vec![TestDNA(0.5)], &TestEval);
    engine.step(&TestEval);

    // Should complete without stalling
    assert!(engine.archive_len() >= 1);
}
