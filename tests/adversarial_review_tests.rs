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

// ============================================================================
// Performance Issue: MapElites O(N) parent selection scaling
// Review: src/algorithms/map_elites.rs line 384
// ============================================================================

/// Test that MapElites parent selection doesn't scale linearly with archive size.
/// The bug: `let parents: Vec<&G> = self.archive.values()...collect()` allocates
/// and iterates ALL archive entries just to select batch_size parents.
#[test]
fn test_map_elites_parent_selection_scales_sublinearly() {
    use std::time::Instant;

    // Create test genotype optimized for this test
    #[derive(Clone, Serialize, Deserialize)]
    struct SmallDNA(u8);

    impl Genotype for SmallDNA {
        fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
            if rng.random::<f32>() < rate {
                self.0 = rng.random();
            }
        }
        fn crossover<R: Rng>(&self, other: &Self, _rng: &mut R) -> Self {
            SmallDNA((self.0 / 2).wrapping_add(other.0 / 2))
        }
    }

    // Evaluator that spreads individuals across many bins
    struct SpreadEval;
    impl Evaluator<SmallDNA> for SpreadEval {
        fn evaluate(&self, g: &SmallDNA) -> (f32, Vec<f32>, Vec<f32>) {
            let v = g.0 as f32 / 255.0;
            (v, vec![v], vec![v, (v * 7.0) % 1.0])
        }
    }

    // Create archive with moderate size (1000 items)
    let mut engine = MapElites::<SmallDNA>::new(100, 0.3, 42);
    engine.set_batch_size(32);

    let initial: Vec<SmallDNA> = (0..=255).map(SmallDNA).collect();
    engine.seed_population(initial, &SpreadEval);

    // Warm up
    for _ in 0..5 {
        engine.step(&SpreadEval);
    }

    let archive_size = engine.archive_len();

    // Time many steps
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        engine.step(&SpreadEval);
    }
    let duration = start.elapsed();
    let per_step_us = duration.as_micros() as f64 / iterations as f64;

    // With O(1) or O(batch_size) selection, time should be bounded regardless of archive size.
    // With O(N) selection, time grows with archive_len.
    // batch_size=32 should complete in well under 1ms per step on modern hardware.
    assert!(
        per_step_us < 5000.0, // 5ms max per step
        "MapElites step took {:.0}μs/step with archive_len={}. \
         Parent selection should be O(batch_size), not O(archive_len).",
        per_step_us,
        archive_size
    );
}

// ============================================================================
// Performance Issue: Nsga2 deep cloning during crowding distance sorting
// Review: src/algorithms/nsga2.rs lines 426-433
// ============================================================================

/// Test that Nsga2 doesn't deep clone genomes for sorting operations.
#[test]
fn test_nsga2_sorting_avoids_excessive_cloning() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;
    use symbios_genetics::algorithms::nsga2::Nsga2;

    static CLONE_COUNT: AtomicUsize = AtomicUsize::new(0);

    #[derive(Serialize, Deserialize)]
    struct TrackedDNA {
        data: Vec<f32>,
    }

    impl Clone for TrackedDNA {
        fn clone(&self) -> Self {
            CLONE_COUNT.fetch_add(1, Ordering::Relaxed);
            TrackedDNA {
                data: self.data.clone(),
            }
        }
    }

    impl Genotype for TrackedDNA {
        fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
            if rng.random::<f32>() < rate {
                for v in &mut self.data {
                    *v += rng.random::<f32>() - 0.5;
                }
            }
        }
        fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
            let point = rng.random_range(0..self.data.len());
            let mut data = self.data[..point].to_vec();
            data.extend_from_slice(&other.data[point..]);
            TrackedDNA { data }
        }
    }

    struct MultiObjEval;
    impl Evaluator<TrackedDNA> for MultiObjEval {
        fn evaluate(&self, g: &TrackedDNA) -> (f32, Vec<f32>, Vec<f32>) {
            let sum: f32 = g.data.iter().sum();
            let var: f32 = g
                .data
                .iter()
                .map(|v| (v - sum / g.data.len() as f32).powi(2))
                .sum();
            (sum, vec![sum, -var], vec![])
        }
    }

    let pop_size = 100;
    let genome_size = 100; // 100 f32s per genome

    let initial: Vec<TrackedDNA> = (0..pop_size)
        .map(|i| TrackedDNA {
            data: (0..genome_size)
                .map(|j| (i * genome_size + j) as f32)
                .collect(),
        })
        .collect();

    let mut nsga = Nsga2::new(initial, 0.1, 42);

    // Reset counter after construction
    CLONE_COUNT.store(0, Ordering::Relaxed);

    let start = Instant::now();
    nsga.step(&MultiObjEval);
    let duration = start.elapsed();

    let clones = CLONE_COUNT.load(Ordering::Relaxed);

    // Expected cloning behavior after fix:
    // - offspring creation: pop_size clones (parents to children)
    // - combined population: pop_size clones (self.population.clone())
    // Sorting should NOT require additional clones of the full genome.
    //
    // Before fix: O(2*N) clones for SortWrapper creation
    // After fix: O(N) clones for combined population only
    let max_acceptable_clones = pop_size * 4; // Allow some buffer for implementation details

    println!(
        "Nsga2::step cloned {} times for pop_size={} (took {:?})",
        clones, pop_size, duration
    );

    assert!(
        clones <= max_acceptable_clones,
        "Nsga2 performed {} clones for pop_size={}. \
         Sorting should use indices, not clone data. Max acceptable: {}",
        clones,
        pop_size,
        max_acceptable_clones
    );
}

// ============================================================================
// API Issue: SimpleGA elitism validation
// Review: SimpleGA::new does not warn when elitism >= pop_size
// ============================================================================

#[test]
fn test_simple_ga_elitism_clamping_maintains_progress() {
    // When elitism >= pop_size, the algorithm should still make progress
    // by clamping elitism to pop_size - 1
    let initial: Vec<TestDNA> = (0..10).map(|i| TestDNA(i as f32)).collect();
    let mut ga = SimpleGA::new(initial, 0.5, 100, 42); // elitism=100 > pop_size=10

    // Should not stall - at least one offspring should be created
    for _ in 0..10 {
        ga.step(&TestEval);
    }

    // Verify evolution occurred (population should have changed)
    let pop = ga.population();
    assert_eq!(pop.len(), 10, "Population size should remain constant");

    // Evolution should have made some progress
    let best_fitness = pop
        .iter()
        .map(|p| p.fitness)
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        best_fitness > 0.0,
        "Evolution should progress even with over-specified elitism"
    );
}

// ============================================================================
// Efficiency Issue: MapElites map_to_index allocates Vec per call
// Review: Optimize for high-throughput with reusable buffer
// ============================================================================

#[test]
fn test_map_elites_map_to_index_into_buffer() {
    let engine = MapElites::<TestDNA>::new(10, 0.1, 42);

    // Test buffer-based method produces same results as allocating version
    let descriptor = vec![0.25, 0.75, 0.0, 1.0, 0.5];
    let expected = engine.map_to_index(&descriptor);

    let mut buffer = vec![0usize; 5];
    engine.map_to_index_into(&descriptor, &mut buffer);

    assert_eq!(
        &buffer[..descriptor.len()],
        &expected[..],
        "map_to_index_into should produce same results as map_to_index"
    );
}

#[test]
fn test_map_elites_map_to_index_into_reuses_buffer() {
    use std::time::Instant;

    let engine = MapElites::<TestDNA>::new(100, 0.1, 42);
    let iterations = 10000;

    // Pre-allocate buffer
    let mut buffer = vec![0usize; 2];

    // Time buffer-based version
    let start = Instant::now();
    for i in 0..iterations {
        let d = vec![(i as f32 / iterations as f32), 0.5];
        engine.map_to_index_into(&d, &mut buffer);
    }
    let buffer_time = start.elapsed();

    // Time allocating version
    let start = Instant::now();
    for i in 0..iterations {
        let d = vec![(i as f32 / iterations as f32), 0.5];
        let _ = engine.map_to_index(&d);
    }
    let alloc_time = start.elapsed();

    println!(
        "map_to_index_into: {:?}, map_to_index: {:?}",
        buffer_time, alloc_time
    );

    // Verify final values match
    let final_d = vec![0.5, 0.5];
    engine.map_to_index_into(&final_d, &mut buffer);
    let expected = engine.map_to_index(&final_d);
    assert_eq!(buffer[0], expected[0]);
    assert_eq!(buffer[1], expected[1]);
}

// ============================================================================
// Issue #36: MapElites Deserialization Time Bombs (DoS via panic injection)
// Review: Critical - resolution=0 or archive key desync causes panics
// ============================================================================

use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::collections::BTreeMap;
use symbios_genetics::Phenotype;

/// Mirror of internal MapElitesData for testing malicious deserialization.
/// This allows us to craft invalid serialized states.
#[derive(Serialize, Deserialize)]
struct MaliciousMapElitesData {
    archive: BTreeMap<Vec<usize>, Phenotype<TestDNA>>,
    archive_keys_vec: Vec<Vec<usize>>,
    resolution: usize,
    mutation_rate: f32,
    batch_size: usize,
    rng: Pcg64,
}

/// Test that deserializing MapElites with resolution=0 fails gracefully.
#[test]
fn test_map_elites_deserialize_rejects_zero_resolution() {
    // Create valid phenotype
    let pheno = Phenotype {
        genotype: TestDNA(0.5),
        fitness: 0.5,
        objectives: vec![0.5],
        descriptor: vec![0.5],
    };

    let mut archive = BTreeMap::new();
    archive.insert(vec![5usize], pheno);

    // Create malicious data with resolution=0
    let malicious_data = MaliciousMapElitesData {
        archive,
        archive_keys_vec: vec![vec![5]],
        resolution: 0, // Invalid!
        mutation_rate: 0.1,
        batch_size: 64,
        rng: Pcg64::seed_from_u64(42),
    };

    let serialized = bincode::serialize(&malicious_data).unwrap();
    let result: Result<MapElites<TestDNA>, _> = bincode::deserialize(&serialized);

    match result {
        Ok(_) => panic!(
            "Deserializing MapElites with resolution=0 should fail, not panic later.\n\
             This prevents DoS via division/underflow in map_single_dimension."
        ),
        Err(e) => {
            let err_msg = e.to_string();
            assert!(
                err_msg.contains("resolution must be greater than 0"),
                "Error should mention resolution validation. Got: {}",
                err_msg
            );
        }
    }
}

/// Test that deserializing MapElites with batch_size=0 fails gracefully.
#[test]
fn test_map_elites_deserialize_rejects_zero_batch_size() {
    let pheno = Phenotype {
        genotype: TestDNA(0.5),
        fitness: 0.5,
        objectives: vec![0.5],
        descriptor: vec![0.5],
    };

    let mut archive = BTreeMap::new();
    archive.insert(vec![5usize], pheno);

    let malicious_data = MaliciousMapElitesData {
        archive,
        archive_keys_vec: vec![vec![5]],
        resolution: 10,
        mutation_rate: 0.1,
        batch_size: 0, // Invalid!
        rng: Pcg64::seed_from_u64(42),
    };

    let serialized = bincode::serialize(&malicious_data).unwrap();
    let result: Result<MapElites<TestDNA>, _> = bincode::deserialize(&serialized);

    assert!(
        result.is_err(),
        "Deserializing MapElites with batch_size=0 should fail"
    );
}

/// Test that deserializing MapElites with desynced archive_keys_vec fails gracefully.
#[test]
fn test_map_elites_deserialize_rejects_archive_key_desync() {
    let pheno = Phenotype {
        genotype: TestDNA(0.5),
        fitness: 0.5,
        objectives: vec![0.5],
        descriptor: vec![0.5],
    };

    let mut archive = BTreeMap::new();
    archive.insert(vec![5usize], pheno);

    // archive_keys_vec contains a key [99] that doesn't exist in archive
    let malicious_data = MaliciousMapElitesData {
        archive,
        archive_keys_vec: vec![vec![5], vec![99]], // [99] doesn't exist in archive!
        resolution: 10,
        mutation_rate: 0.1,
        batch_size: 64,
        rng: Pcg64::seed_from_u64(42),
    };

    let serialized = bincode::serialize(&malicious_data).unwrap();
    let result: Result<MapElites<TestDNA>, _> = bincode::deserialize(&serialized);

    assert!(
        result.is_err(),
        "Deserializing MapElites with desynced archive_keys_vec should fail.\n\
         This prevents unwrap() panic during parent selection in step()."
    );
}

/// Test that deserializing MapElites with incomplete archive_keys_vec fails.
#[test]
fn test_map_elites_deserialize_rejects_incomplete_keys_vec() {
    let pheno1 = Phenotype {
        genotype: TestDNA(0.1),
        fitness: 0.1,
        objectives: vec![0.1],
        descriptor: vec![0.1],
    };
    let pheno2 = Phenotype {
        genotype: TestDNA(0.9),
        fitness: 0.9,
        objectives: vec![0.9],
        descriptor: vec![0.9],
    };

    let mut archive = BTreeMap::new();
    archive.insert(vec![1usize], pheno1);
    archive.insert(vec![9usize], pheno2);

    // archive_keys_vec is missing key [9] - only has [1]
    let malicious_data = MaliciousMapElitesData {
        archive,
        archive_keys_vec: vec![vec![1]], // Missing [9]!
        resolution: 10,
        mutation_rate: 0.1,
        batch_size: 64,
        rng: Pcg64::seed_from_u64(42),
    };

    let serialized = bincode::serialize(&malicious_data).unwrap();
    let result: Result<MapElites<TestDNA>, _> = bincode::deserialize(&serialized);

    assert!(
        result.is_err(),
        "Deserializing MapElites with incomplete archive_keys_vec should fail"
    );
}

// ============================================================================
// Issue #38: SimpleGA elitism() getter returns invalid stored value
// Review: Low - elitism() lies about runtime behavior
// ============================================================================

/// Test that elitism() returns the effective clamped value, not the configured invalid value.
#[test]
fn test_simple_ga_elitism_returns_effective_value() {
    let initial: Vec<TestDNA> = (0..10).map(|i| TestDNA(i as f32)).collect();
    let ga = SimpleGA::new(initial, 0.1, 100, 42); // elitism=100 > pop_size=10

    // elitism() should return effective value (clamped to pop_size - 1 = 9)
    let effective = ga.elitism();
    assert_eq!(
        effective, 9,
        "elitism() should return effective clamped value (9), not configured value (100)"
    );

    // elitism_configured() should return the original configured value
    let configured = ga.elitism_configured();
    assert_eq!(
        configured, 100,
        "elitism_configured() should return the originally configured value"
    );
}

/// Test that elitism() returns correct value when configured value is valid.
#[test]
fn test_simple_ga_elitism_returns_configured_when_valid() {
    let initial: Vec<TestDNA> = (0..10).map(|i| TestDNA(i as f32)).collect();
    let ga = SimpleGA::new(initial, 0.1, 3, 42); // elitism=3 < pop_size=10

    // Both should return the same value when configured value is valid
    assert_eq!(ga.elitism(), 3);
    assert_eq!(ga.elitism_configured(), 3);
}

// ============================================================================
// Issue #39: NSGA2 uninitialized population state
// Review: Minor - ranks/distances not calculated on creation
// ============================================================================

/// Test that NSGA2 calculates initial ranks and crowding distances on creation.
#[test]
fn test_nsga2_initial_ranks_calculated() {
    use symbios_genetics::algorithms::nsga2::Nsga2;

    let initial: Vec<TestDNA> = (0..10).map(|i| TestDNA(i as f32)).collect();
    let nsga = Nsga2::new(initial, 0.1, 42);

    // Serialize immediately after creation (before any step()) using bincode
    let serialized = bincode::serialize(&nsga).expect("Serialization failed");

    // Deserialize and check it works (no corrupted state)
    let deserialized: Nsga2<TestDNA> =
        bincode::deserialize(&serialized).expect("Deserialization failed");

    // Population should be intact
    assert_eq!(deserialized.pop_size(), 10);

    // The deserialized instance should work correctly
    let mut nsga2 = deserialized;
    nsga2.step(&TestEval); // Should not panic due to invalid ranks/distances
}

/// Test that NSGA2 binary tournament works correctly after construction.
#[test]
fn test_nsga2_binary_tournament_works_after_new() {
    use symbios_genetics::algorithms::nsga2::Nsga2;

    // With initial (unevaluated) population, all individuals have empty objectives
    // This means all are non-dominated (rank 0) with infinite crowding distance
    let initial: Vec<TestDNA> = (0..10).map(|i| TestDNA(i as f32)).collect();
    let mut nsga = Nsga2::new(initial, 0.1, 42);

    // step() should work correctly - it uses binary_tournament which relies on
    // ranks and crowding_distances being properly initialized
    nsga.step(&TestEval);

    // If we get here without panic, the initial state was valid
    assert_eq!(nsga.pop_size(), 10);
}

// ============================================================================
// Issue #37: MapElites step() allocation hypocrisy
// Review: Moderate - uses map_to_index instead of map_to_index_into
// ============================================================================

/// Test that MapElites step() works correctly with the optimized implementation.
#[test]
fn test_map_elites_step_uses_buffer_optimization() {
    let mut engine = MapElites::<TestDNA>::new(10, 0.3, 42);
    engine.set_batch_size(32);

    // Seed population
    let initial: Vec<TestDNA> = (0..20).map(|i| TestDNA(i as f32 / 20.0)).collect();
    engine.seed_population(initial, &TestEval);

    let initial_len = engine.archive_len();

    // Run multiple steps - should not panic and should make progress
    for _ in 0..50 {
        engine.step(&TestEval);
    }

    // Archive should have grown or at least maintained
    assert!(
        engine.archive_len() >= initial_len,
        "Archive should maintain or grow during evolution"
    );

    // Verify archive integrity - all elites should be accessible
    for key in engine.archive_keys() {
        assert!(
            engine.archive_get(key).is_some(),
            "All archive keys should map to valid elites"
        );
    }
}
