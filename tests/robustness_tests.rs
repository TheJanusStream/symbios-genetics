use rand::Rng;
use serde::{Deserialize, Serialize};
use symbios_genetics::{
    Evaluator, Evolver, Genotype, algorithms::map_elites::MapElites, algorithms::nsga2::Nsga2,
    algorithms::simple::SimpleGA,
};

// --- Mock Infrastructure for Edge Case Testing ---

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
struct TestDNA(f32);

impl Genotype for TestDNA {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        if rng.random::<f32>() < rate {
            self.0 += 1.0;
        }
    }
    fn crossover<R: Rng>(&self, other: &Self, _rng: &mut R) -> Self {
        TestDNA((self.0 + other.0) / 2.0)
    }
}

// ============================================================================
// Issue #1: NaN Fitness Panic Tests
// ============================================================================

/// Evaluator that returns NaN fitness (simulates divide-by-zero, log(-1), etc.)
struct NaNEvaluator;
impl Evaluator<TestDNA> for NaNEvaluator {
    fn evaluate(&self, _genotype: &TestDNA) -> (f32, Vec<f32>, Vec<f32>) {
        (f32::NAN, vec![f32::NAN, f32::NAN], vec![0.5])
    }
}

/// Evaluator that returns NaN for some individuals (more realistic scenario)
struct PartialNaNEvaluator;
impl Evaluator<TestDNA> for PartialNaNEvaluator {
    fn evaluate(&self, genotype: &TestDNA) -> (f32, Vec<f32>, Vec<f32>) {
        // Every third individual gets NaN fitness
        if genotype.0 as i32 % 3 == 0 {
            (f32::NAN, vec![f32::NAN], vec![0.5])
        } else {
            (genotype.0, vec![genotype.0], vec![genotype.0 / 10.0])
        }
    }
}

#[test]
fn test_simple_ga_handles_nan_fitness_without_panic() {
    let initial_pop: Vec<TestDNA> = (0..10).map(|i| TestDNA(i as f32)).collect();
    let mut ga = SimpleGA::new(initial_pop, 0.1, 2, 42);
    let eval = NaNEvaluator;

    // This should NOT panic - currently it does due to partial_cmp().unwrap()
    ga.step(&eval);

    // Verify the algorithm continues to function
    assert!(!ga.population().is_empty());
}

#[test]
fn test_simple_ga_handles_partial_nan_fitness() {
    let initial_pop: Vec<TestDNA> = (0..10).map(|i| TestDNA(i as f32)).collect();
    let mut ga = SimpleGA::new(initial_pop, 0.1, 2, 42);
    let eval = PartialNaNEvaluator;

    // Should handle mixed NaN/valid fitness values
    ga.step(&eval);
    assert!(!ga.population().is_empty());
}

#[test]
fn test_nsga2_handles_nan_objectives_without_panic() {
    let initial_pop: Vec<TestDNA> = (0..10).map(|i| TestDNA(i as f32)).collect();
    let mut nsga2 = Nsga2::new(initial_pop, 0.1, 42);
    let eval = NaNEvaluator;

    // Should not panic even with all NaN objectives
    nsga2.step(&eval);
    assert!(!nsga2.population().is_empty());
}

// ============================================================================
// Issue #2: Configuration-Induced Panic Tests
// ============================================================================

#[test]
fn test_simple_ga_elitism_exceeds_population() {
    // elitism (15) > pop_size (10) - this should be handled gracefully
    let initial_pop: Vec<TestDNA> = (0..10).map(|i| TestDNA(i as f32)).collect();

    // This currently panics at: self.population[..self.elitism].to_vec()
    let mut ga = SimpleGA::new(initial_pop, 0.1, 15, 42);

    struct SimpleEval;
    impl Evaluator<TestDNA> for SimpleEval {
        fn evaluate(&self, g: &TestDNA) -> (f32, Vec<f32>, Vec<f32>) {
            (g.0, vec![], vec![])
        }
    }

    ga.step(&SimpleEval);
    assert!(!ga.population().is_empty());
}

#[test]
fn test_simple_ga_population_smaller_than_tournament_size() {
    // Population of 2, but tournament selection uses 3
    let initial_pop = vec![TestDNA(1.0), TestDNA(2.0)];
    let mut ga = SimpleGA::new(initial_pop, 0.1, 1, 42);

    struct SimpleEval;
    impl Evaluator<TestDNA> for SimpleEval {
        fn evaluate(&self, g: &TestDNA) -> (f32, Vec<f32>, Vec<f32>) {
            (g.0, vec![], vec![])
        }
    }

    // Should handle gracefully, not panic
    ga.step(&SimpleEval);
    assert_eq!(ga.population().len(), 2);
}

#[test]
fn test_simple_ga_single_individual_population() {
    let initial_pop = vec![TestDNA(1.0)];
    let mut ga = SimpleGA::new(initial_pop, 0.1, 1, 42);

    struct SimpleEval;
    impl Evaluator<TestDNA> for SimpleEval {
        fn evaluate(&self, g: &TestDNA) -> (f32, Vec<f32>, Vec<f32>) {
            (g.0, vec![], vec![])
        }
    }

    ga.step(&SimpleEval);
    assert_eq!(ga.population().len(), 1);
}

#[test]
fn test_simple_ga_empty_population() {
    let initial_pop: Vec<TestDNA> = vec![];
    let mut ga = SimpleGA::new(initial_pop, 0.1, 0, 42);

    struct SimpleEval;
    impl Evaluator<TestDNA> for SimpleEval {
        fn evaluate(&self, g: &TestDNA) -> (f32, Vec<f32>, Vec<f32>) {
            (g.0, vec![], vec![])
        }
    }

    // Should handle empty population without panic
    ga.step(&SimpleEval);
    assert!(ga.population().is_empty());
}

// ============================================================================
// Issue #5: NSGA-II Crowding Distance Division by Zero
// ============================================================================

/// Evaluator that returns identical objectives for all individuals
struct UniformObjectivesEvaluator;
impl Evaluator<TestDNA> for UniformObjectivesEvaluator {
    fn evaluate(&self, _genotype: &TestDNA) -> (f32, Vec<f32>, Vec<f32>) {
        // All individuals have the same objectives -> range = 0 -> div by zero
        (1.0, vec![5.0, 5.0], vec![0.5])
    }
}

#[test]
fn test_nsga2_crowding_distance_zero_range() {
    let initial_pop: Vec<TestDNA> = (0..10).map(|i| TestDNA(i as f32)).collect();
    let mut nsga2 = Nsga2::new(initial_pop, 0.0, 42);
    let eval = UniformObjectivesEvaluator;

    // This should not produce Infinity or NaN that breaks sorting
    nsga2.step(&eval);

    // Verify no NaN or Infinity in distances would corrupt selection
    let pop = nsga2.population();
    assert!(!pop.is_empty());
    for p in pop {
        assert!(!p.fitness.is_nan(), "Fitness should not be NaN");
    }
}

#[test]
fn test_nsga2_crowding_distance_single_objective_all_same() {
    let initial_pop: Vec<TestDNA> = (0..5).map(|_| TestDNA(1.0)).collect();
    let mut nsga2 = Nsga2::new(initial_pop, 0.0, 42);

    struct ConstantEval;
    impl Evaluator<TestDNA> for ConstantEval {
        fn evaluate(&self, _g: &TestDNA) -> (f32, Vec<f32>, Vec<f32>) {
            (1.0, vec![1.0], vec![0.5])
        }
    }

    nsga2.step(&ConstantEval);
    assert!(!nsga2.population().is_empty());
}

// ============================================================================
// Issue #3 & #4: Map-Elites Performance and Batch Size Tests
// ============================================================================

#[test]
fn test_map_elites_default_batch_size_is_reasonable() {
    let me = MapElites::<TestDNA>::new(10, 0.1, 42);

    // batch_size of 1 defeats the purpose of parallel evaluation
    // It should default to something reasonable (e.g., 32, 64, or num_cpus)
    assert!(
        me.batch_size > 1,
        "Default batch_size should be > 1 for effective parallelism, got {}",
        me.batch_size
    );
}

#[test]
fn test_map_elites_large_archive_performance() {
    use std::time::Instant;

    let mut me = MapElites::<TestDNA>::new(100, 0.1, 42);
    me.batch_size = 32;

    struct FastEval;
    impl Evaluator<TestDNA> for FastEval {
        fn evaluate(&self, g: &TestDNA) -> (f32, Vec<f32>, Vec<f32>) {
            (g.0, vec![], vec![(g.0 / 1000.0).clamp(0.0, 1.0)])
        }
    }

    // Seed with many individuals to create a large archive
    let initial: Vec<TestDNA> = (0..1000).map(|i| TestDNA(i as f32)).collect();
    me.seed_population(initial, &FastEval);

    // Measure time for 100 steps
    let start = Instant::now();
    for _ in 0..100 {
        me.step(&FastEval);
    }
    let duration = start.elapsed();

    // Performance assertion: 100 steps should complete in reasonable time
    // This will help catch O(n^2) or worse allocation patterns
    println!("Map-Elites 100 steps took {:?}", duration);

    // Very generous limit - mainly catches catastrophic performance issues
    assert!(
        duration.as_secs() < 10,
        "Map-Elites performance degraded: 100 steps took {:?}",
        duration
    );
}

// ============================================================================
// Issue #7: Flaky Timing Test - Relative Comparison Version
// ============================================================================

#[cfg(feature = "parallel")]
mod parallel_tests {
    use super::*;
    use std::time::{Duration, Instant};
    use symbios_genetics::algorithms::simple::SimpleGA;

    #[derive(Clone, Serialize, Deserialize)]
    struct SlowDNA(Vec<f32>);

    impl Genotype for SlowDNA {
        fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
            for v in &mut self.0 {
                if rng.random_bool(rate as f64) {
                    *v += 1.0;
                }
            }
        }
        fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
            SlowDNA(
                self.0
                    .iter()
                    .zip(&other.0)
                    .map(|(&a, &b)| if rng.random_bool(0.5) { a } else { b })
                    .collect(),
            )
        }
    }

    struct SlowEvaluator {
        delay_micros: u64,
    }

    impl Evaluator<SlowDNA> for SlowEvaluator {
        fn evaluate(&self, genotype: &SlowDNA) -> (f32, Vec<f32>, Vec<f32>) {
            // Use spin-wait instead of sleep for more consistent timing
            let start = Instant::now();
            while start.elapsed() < Duration::from_micros(self.delay_micros) {
                std::hint::spin_loop();
            }
            let sum: f32 = genotype.0.iter().sum();
            (sum, vec![sum], vec![])
        }
    }

    #[test]
    fn test_parallel_provides_speedup_relative_to_sequential() {
        let pop_size = 100;
        let delay_micros = 500; // 500 microseconds per eval

        // Sequential baseline (single-threaded simulation)
        let sequential_estimate = Duration::from_micros(delay_micros * pop_size as u64);

        // Parallel run
        let initial_pop: Vec<SlowDNA> = (0..pop_size).map(|_| SlowDNA(vec![0.0; 10])).collect();
        let mut ga = SimpleGA::new(initial_pop, 0.1, 10, 42);
        let eval = SlowEvaluator { delay_micros };

        let start = Instant::now();
        ga.step(&eval);
        let parallel_duration = start.elapsed();

        // We expect at least 2x speedup on any multi-core machine
        // But we use a conservative ratio that should pass even on slow CI
        let speedup_ratio =
            sequential_estimate.as_micros() as f64 / parallel_duration.as_micros() as f64;

        println!(
            "Sequential estimate: {:?}, Parallel actual: {:?}, Speedup: {:.2}x",
            sequential_estimate, parallel_duration, speedup_ratio
        );

        // Require at least 1.5x speedup (very conservative for CI environments)
        assert!(
            speedup_ratio > 1.5,
            "Parallel execution should provide speedup. Got {:.2}x (expected > 1.5x)",
            speedup_ratio
        );
    }
}
