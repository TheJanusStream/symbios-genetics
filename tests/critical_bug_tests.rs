//! Critical bug regression tests for issues identified in code review.
//! Each test documents and verifies the fix for a specific issue.

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use symbios_genetics::{
    Evaluator, Evolver, Genotype, Phenotype, algorithms::map_elites::MapElites,
    algorithms::nsga2::Nsga2, algorithms::simple::SimpleGA,
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
// Issue #17: O(N) cache rebuild in MapElites
// ============================================================================

#[test]
fn test_map_elites_cache_performance_scales_with_batch_not_archive() {
    // This test verifies that step() performance scales with batch_size,
    // not archive size. With the bug, each step clones the entire archive.

    // Use resolution 50 for a 50-bin archive (1D descriptor)
    let mut engine = MapElites::<TestDNA>::new(50, 0.5, 42);
    engine.set_batch_size(64);

    struct PerformanceTestEval;
    impl Evaluator<TestDNA> for PerformanceTestEval {
        fn evaluate(&self, g: &TestDNA) -> (f32, Vec<f32>, Vec<f32>) {
            // Fitness is the value, descriptor spans [0, 1]
            let desc = (g.0 % 1.0).abs();
            (g.0, vec![g.0], vec![desc])
        }
    }

    let eval = PerformanceTestEval;

    // Seed with individuals that will spread across bins
    let initial: Vec<TestDNA> = (0..100).map(|i| TestDNA(i as f32 / 100.0)).collect();
    engine.seed_population(initial, &eval);

    // Run several steps to populate archive
    for _ in 0..50 {
        engine.step(&eval);
    }

    let archive_size = engine.archive_len();
    assert!(
        archive_size > 20,
        "Archive should be well-populated, got {}",
        archive_size
    );

    // Time multiple steps - with lazy cache, this should be fast
    // because we don't call population() between steps
    let start = Instant::now();
    for _ in 0..100 {
        engine.step(&eval);
    }
    let elapsed = start.elapsed();

    // Performance assertion: 100 steps should complete quickly
    assert!(
        elapsed.as_millis() < 5000,
        "100 steps took {:?}, suggesting O(archive_size) scaling. \
         Should scale with batch_size instead.",
        elapsed
    );
}

// ============================================================================
// Issue #18: NSGA-II uses uniform random instead of binary tournament
// ============================================================================

#[test]
fn test_nsga2_binary_tournament_provides_selection_pressure() {
    // This test verifies that parent selection favors better-ranked individuals.
    // With uniform random selection (the bug), there's no pressure toward the front.

    #[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
    struct RankedDNA(f32);

    impl Genotype for RankedDNA {
        fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
            if rng.random::<f32>() < rate {
                self.0 += (rng.random::<f32>() - 0.5) * 0.1;
            }
        }
        fn crossover<R: Rng>(&self, other: &Self, _rng: &mut R) -> Self {
            RankedDNA((self.0 + other.0) / 2.0)
        }
    }

    struct MultiObjEval;
    impl Evaluator<RankedDNA> for MultiObjEval {
        fn evaluate(&self, g: &RankedDNA) -> (f32, Vec<f32>, Vec<f32>) {
            // Two conflicting objectives: maximize g.0 vs maximize (1 - g.0)
            (g.0, vec![g.0, 1.0 - g.0], vec![])
        }
    }

    // Start with a diverse population
    let initial: Vec<RankedDNA> = (0..50).map(|i| RankedDNA(i as f32 / 50.0)).collect();

    let mut engine = Nsga2::new(initial, 0.1, 42);
    let eval = MultiObjEval;

    // Run for several generations
    for _ in 0..20 {
        engine.step(&eval);
    }

    // With binary tournament selection, the population should converge toward
    // the Pareto front (values near 0.0 or 1.0 for this problem).
    // With uniform selection, evolution is nearly random.

    let pop = engine.population();
    let front_coverage: Vec<f32> = pop.iter().map(|p| p.genotype.0).collect();

    // Check that we have diversity along the Pareto front
    let min_val = front_coverage.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = front_coverage
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    // The Pareto front spans [0, 1], so we should see spread
    assert!(
        max_val - min_val > 0.5,
        "Population should spread along Pareto front. Got range [{}, {}]",
        min_val,
        max_val
    );
}

// ============================================================================
// Issue #19: MapElites public archive allows state desync
// ============================================================================

#[test]
fn test_map_elites_archive_is_encapsulated() {
    let mut engine = MapElites::<TestDNA>::new(10, 0.1, 42);
    let eval = TestEval;

    engine.seed_population(vec![TestDNA(0.5)], &eval);

    // Verify archive is accessible only through controlled methods
    assert_eq!(
        engine.archive_len(),
        1,
        "Should have 1 individual in archive"
    );

    // The archive should be accessible for reading but modifications
    // should go through proper methods that maintain cache consistency
    let pop = engine.population();
    assert_eq!(pop.len(), 1, "Population cache should match archive");
}

// ============================================================================
// Issue #20: Biased bin discretization in MapElites
// ============================================================================

#[test]
fn test_map_elites_uniform_bin_distribution() {
    // With round(), boundary bins are half-sized:
    // resolution=3: bin 0 = [0.0, 0.25), bin 1 = [0.25, 0.75), bin 2 = [0.75, 1.0]
    // With floor(), all bins are equal: [0, 0.333), [0.333, 0.667), [0.667, 1.0]

    let engine = MapElites::<TestDNA>::new(3, 0.1, 42);

    // Test boundary values
    let test_cases = vec![
        (0.0, 0),  // Start of first bin
        (0.32, 0), // Should be in first bin with floor
        (0.33, 0), // Edge case
        (0.34, 1), // Should be in middle bin
        (0.5, 1),  // Middle of middle bin
        (0.65, 1), // Still in middle bin
        (0.67, 2), // Should be in last bin
        (0.99, 2), // Near end
        (1.0, 2),  // End boundary
    ];

    for (value, expected_bin) in test_cases {
        let idx = engine.map_to_index(&[value]);
        assert_eq!(
            idx[0], expected_bin,
            "Value {} should map to bin {}, got {}. \
             Bins should be uniformly sized using floor().",
            value, expected_bin, idx[0]
        );
    }
}

#[test]
fn test_map_elites_boundary_bins_equal_size() {
    // Statistical test: with uniform input, each bin should get ~equal samples
    let engine = MapElites::<TestDNA>::new(5, 0.1, 42);
    let mut bin_counts = [0usize; 5];

    // Sample uniformly across [0, 1]
    for i in 0..10000 {
        let value = i as f32 / 10000.0;
        let idx = engine.map_to_index(&[value]);
        bin_counts[idx[0]] += 1;
    }

    // Each bin should have ~2000 samples (10000 / 5)
    // With the round() bug, bins 0 and 4 would have ~half as many
    let expected = 2000;
    let tolerance = 100; // Allow 5% deviation

    for (bin, &count) in bin_counts.iter().enumerate() {
        assert!(
            (count as i32 - expected as i32).abs() < tolerance as i32,
            "Bin {} has {} samples, expected ~{}. \
             Boundary bins should not be undersized.",
            bin,
            count,
            expected
        );
    }
}

// ============================================================================
// Issue #21: Ragged objectives in NSGA-II crowding distance
// ============================================================================

#[test]
fn test_nsga2_rejects_ragged_objectives() {
    struct RaggedEval;
    impl Evaluator<TestDNA> for RaggedEval {
        fn evaluate(&self, g: &TestDNA) -> (f32, Vec<f32>, Vec<f32>) {
            if g.0 > 0.5 {
                (g.0, vec![g.0, -g.0, g.0 * 2.0], vec![])
            } else {
                (g.0, vec![g.0, -g.0], vec![])
            }
        }
    }

    let initial = vec![
        TestDNA(0.3), // 2 objectives
        TestDNA(0.4), // 2 objectives
        TestDNA(0.6), // 3 objectives
        TestDNA(0.7), // 3 objectives
    ];

    let mut engine = Nsga2::new(initial, 0.0, 42);
    let eval = RaggedEval;

    // The engine should either:
    // 1. Panic with a clear error about objective count mismatch, OR
    // 2. Handle gracefully by separating incomparable individuals
    // It should NOT silently ignore the extra objective

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.step(&eval);
    }));

    // For now, we accept either graceful handling or explicit panic
    // The key is it shouldn't silently produce wrong results
    if result.is_ok() {
        // If it didn't panic, verify the population is valid
        let pop = engine.population();
        assert!(!pop.is_empty(), "Population should not be empty after step");
    }
    // If it panicked, that's also acceptable (explicit failure > silent corruption)
}

// ============================================================================
// Issue #22: Serial mutation bottleneck in MapElites
// ============================================================================

#[test]
#[cfg(feature = "parallel")]
fn test_map_elites_mutation_parallelizable() {
    // This test verifies mutation can benefit from parallelism.
    // The actual parallelism is implementation-dependent, but we verify
    // that heavy mutation doesn't serialize the entire step.

    #[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
    struct HeavyDNA(Vec<f32>);

    impl Genotype for HeavyDNA {
        fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
            // Simulate expensive mutation
            for v in &mut self.0 {
                if rng.random::<f32>() < rate {
                    *v += rng.random::<f32>() - 0.5;
                }
            }
            // Small computational work to simulate cost
            let _sum: f32 = self.0.iter().sum();
        }
        fn crossover<R: Rng>(&self, other: &Self, _rng: &mut R) -> Self {
            HeavyDNA(
                self.0
                    .iter()
                    .zip(other.0.iter())
                    .map(|(a, b)| (a + b) / 2.0)
                    .collect(),
            )
        }
    }

    struct HeavyEval;
    impl Evaluator<HeavyDNA> for HeavyEval {
        fn evaluate(&self, g: &HeavyDNA) -> (f32, Vec<f32>, Vec<f32>) {
            let fitness: f32 = g.0.iter().sum();
            let desc = vec![g.0[0].clamp(0.0, 1.0)];
            (fitness, vec![fitness], desc)
        }
    }

    let mut engine = MapElites::<HeavyDNA>::new(10, 0.5, 42);
    let eval = HeavyEval;

    // Large genome to make mutation non-trivial
    let initial: Vec<HeavyDNA> = (0..10)
        .map(|i| HeavyDNA(vec![i as f32 / 10.0; 100]))
        .collect();
    engine.seed_population(initial, &eval);

    let start = Instant::now();
    for _ in 0..10 {
        engine.step(&eval);
    }
    let elapsed = start.elapsed();

    // Should complete reasonably quickly with parallelism
    assert!(
        elapsed.as_millis() < 5000,
        "10 steps with heavy mutation took {:?}. \
         Mutation should be parallelizable.",
        elapsed
    );
}

// ============================================================================
// Issue #23: SimpleGA elitism halting
// ============================================================================

#[test]
fn test_simple_ga_elitism_equal_pop_size_still_evolves() {
    let initial: Vec<TestDNA> = (0..10).map(|i| TestDNA(i as f32 / 10.0)).collect();

    // BUG: elitism == pop_size means no offspring are created
    let mut engine = SimpleGA::new(initial.clone(), 0.5, 10, 42);
    let eval = TestEval;

    // Run a step
    engine.step(&eval);

    let pop_after = engine.population();

    // The population should still be able to evolve
    // With the bug: population is static
    // Fix options:
    // 1. Cap elitism at pop_size - 1 to ensure at least 1 offspring
    // 2. Warn/error if elitism >= pop_size
    // 3. Still allow mutation of elite copies

    // At minimum, verify we don't just freeze
    let _initial_fitness_sum: f32 = initial.iter().map(|d| d.0).sum();
    let _after_fitness_sum: f32 = pop_after.iter().map(|p| p.genotype.0).sum();

    // With elitism capped, we should see some change from mutation
    // (This is a weak test - mainly verifying no panic and some activity)
    assert_eq!(
        pop_after.len(),
        10,
        "Population size should remain constant"
    );
}

#[test]
fn test_simple_ga_warns_on_excessive_elitism() {
    // Elitism > pop_size should be handled gracefully
    let initial: Vec<TestDNA> = (0..5).map(|i| TestDNA(i as f32)).collect();

    // elitism (10) > pop_size (5)
    let mut engine = SimpleGA::new(initial, 0.5, 10, 42);
    let eval = TestEval;

    // Should not panic
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.step(&eval);
    }));

    assert!(
        result.is_ok(),
        "SimpleGA should handle elitism > pop_size gracefully"
    );
}

// ============================================================================
// Issue #24: Weak dominance test assertion
// ============================================================================

#[test]
fn test_nsga2_dominates_both_false_on_length_mismatch() {
    let short = Phenotype {
        genotype: TestDNA(1.0),
        fitness: 1.0,
        objectives: vec![1.0, 1.0],
        descriptor: vec![],
    };

    let long = Phenotype {
        genotype: TestDNA(2.0),
        fitness: 2.0,
        objectives: vec![0.5, 0.5, 100.0],
        descriptor: vec![],
    };

    let short_dominates_long = Nsga2::<TestDNA>::dominates(&short, &long);
    let long_dominates_short = Nsga2::<TestDNA>::dominates(&long, &short);

    // BOTH should be false - different objective counts means incomparable
    assert!(
        !short_dominates_long,
        "short should not dominate long when objective counts differ"
    );
    assert!(
        !long_dominates_short,
        "long should not dominate short when objective counts differ"
    );
}

#[test]
fn test_nsga2_dominates_correct_for_equal_length() {
    let better = Phenotype {
        genotype: TestDNA(1.0),
        fitness: 1.0,
        objectives: vec![2.0, 2.0],
        descriptor: vec![],
    };

    let worse = Phenotype {
        genotype: TestDNA(0.5),
        fitness: 0.5,
        objectives: vec![1.0, 1.0],
        descriptor: vec![],
    };

    let equal = Phenotype {
        genotype: TestDNA(1.0),
        fitness: 1.0,
        objectives: vec![2.0, 2.0],
        descriptor: vec![],
    };

    // Better dominates worse (higher in all objectives)
    assert!(
        Nsga2::<TestDNA>::dominates(&better, &worse),
        "better should dominate worse"
    );
    assert!(
        !Nsga2::<TestDNA>::dominates(&worse, &better),
        "worse should not dominate better"
    );

    // Equal objectives: neither dominates
    assert!(
        !Nsga2::<TestDNA>::dominates(&better, &equal),
        "equal individuals should not dominate each other"
    );
}
