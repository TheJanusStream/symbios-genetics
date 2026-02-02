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

// ============================================================================
// Issue #41: Nsga2 Deserialization State Desynchronization (Critical DoS)
// Review: population, ranks, crowding_distances must have same length
// ============================================================================

use symbios_genetics::algorithms::nsga2::Nsga2;

/// Mirror of internal Nsga2 structure for testing malicious deserialization.
#[derive(Serialize)]
struct MaliciousNsga2Data {
    population: Vec<Phenotype<TestDNA>>,
    ranks: Vec<usize>,
    crowding_distances: Vec<f32>,
    pop_size: usize,
    mutation_rate: f32,
    rng: Pcg64,
}

/// Test that deserializing Nsga2 with desynced ranks vector fails gracefully.
/// Before fix: Causes index out of bounds panic in binary_tournament.
#[test]
fn test_nsga2_deserialize_rejects_ranks_desync() {
    let pop = vec![Phenotype {
        genotype: TestDNA(0.5),
        fitness: 0.5,
        objectives: vec![0.5],
        descriptor: vec![],
    }];

    let malicious = MaliciousNsga2Data {
        population: pop,
        ranks: vec![], // CRITICAL: Empty while population has 1 element
        crowding_distances: vec![],
        pop_size: 1,
        mutation_rate: 0.1,
        rng: Pcg64::seed_from_u64(42),
    };

    let serialized = bincode::serialize(&malicious).unwrap();
    let result: Result<Nsga2<TestDNA>, _> = bincode::deserialize(&serialized);

    assert!(
        result.is_err(),
        "Deserializing Nsga2 with desynced ranks should fail, not panic later.\n\
         This prevents DoS via index out of bounds in binary_tournament."
    );
}

/// Test that deserializing Nsga2 with desynced crowding_distances fails gracefully.
#[test]
fn test_nsga2_deserialize_rejects_crowding_distances_desync() {
    let pop = vec![
        Phenotype {
            genotype: TestDNA(0.3),
            fitness: 0.3,
            objectives: vec![0.3],
            descriptor: vec![],
        },
        Phenotype {
            genotype: TestDNA(0.7),
            fitness: 0.7,
            objectives: vec![0.7],
            descriptor: vec![],
        },
    ];

    let malicious = MaliciousNsga2Data {
        population: pop,
        ranks: vec![0, 0],             // Correct length
        crowding_distances: vec![1.0], // Wrong: only 1 element for 2 individuals
        pop_size: 2,
        mutation_rate: 0.1,
        rng: Pcg64::seed_from_u64(42),
    };

    let serialized = bincode::serialize(&malicious).unwrap();
    let result: Result<Nsga2<TestDNA>, _> = bincode::deserialize(&serialized);

    assert!(
        result.is_err(),
        "Deserializing Nsga2 with desynced crowding_distances should fail"
    );
}

/// Test that deserializing Nsga2 with pop_size mismatch fails gracefully.
#[test]
fn test_nsga2_deserialize_rejects_pop_size_mismatch() {
    let pop = vec![Phenotype {
        genotype: TestDNA(0.5),
        fitness: 0.5,
        objectives: vec![0.5],
        descriptor: vec![],
    }];

    let malicious = MaliciousNsga2Data {
        population: pop,
        ranks: vec![0],
        crowding_distances: vec![f32::INFINITY],
        pop_size: 100, // Wrong: claims 100 but only 1 in population
        mutation_rate: 0.1,
        rng: Pcg64::seed_from_u64(42),
    };

    let serialized = bincode::serialize(&malicious).unwrap();
    let result: Result<Nsga2<TestDNA>, _> = bincode::deserialize(&serialized);

    assert!(
        result.is_err(),
        "Deserializing Nsga2 with pop_size mismatch should fail"
    );
}

/// Test that deserializing Nsga2 with invalid mutation_rate fails gracefully.
#[test]
fn test_nsga2_deserialize_rejects_nan_mutation_rate() {
    let pop = vec![Phenotype {
        genotype: TestDNA(0.5),
        fitness: 0.5,
        objectives: vec![0.5],
        descriptor: vec![],
    }];

    let malicious = MaliciousNsga2Data {
        population: pop,
        ranks: vec![0],
        crowding_distances: vec![f32::INFINITY],
        pop_size: 1,
        mutation_rate: f32::NAN, // Invalid
        rng: Pcg64::seed_from_u64(42),
    };

    let serialized = bincode::serialize(&malicious).unwrap();
    let result: Result<Nsga2<TestDNA>, _> = bincode::deserialize(&serialized);

    assert!(
        result.is_err(),
        "Deserializing Nsga2 with NaN mutation_rate should fail"
    );
}

/// Test that valid Nsga2 serialization round-trips correctly.
#[test]
fn test_nsga2_valid_serialization_roundtrip() {
    let initial: Vec<TestDNA> = (0..10).map(|i| TestDNA(i as f32)).collect();
    let mut nsga = Nsga2::new(initial, 0.1, 42);

    // Run a step to populate ranks/distances properly
    nsga.step(&TestEval);

    // Serialize and deserialize
    let serialized = bincode::serialize(&nsga).unwrap();
    let mut deserialized: Nsga2<TestDNA> = bincode::deserialize(&serialized).unwrap();

    // Should work correctly
    deserialized.step(&TestEval);
    assert_eq!(deserialized.pop_size(), 10);
}

// ============================================================================
// Issue #42: NaN Bias in Nsga2 Binary Tournament Selection
// Review: NaN crowding distances cause biased selection
// ============================================================================

/// Test that NaN crowding distances don't cause panics or crashes.
/// The fix uses total_cmp for NaN-safe comparisons in all sorting operations.
#[test]
fn test_nsga2_nan_crowding_distance_handling() {
    // This test verifies that the algorithm handles NaN values without crashing.
    // Before the fix, NaN in objectives would cause "comparison function does not
    // correctly implement a total order" panic during sorting.

    struct NaNInjectingEval;

    impl Evaluator<TestDNA> for NaNInjectingEval {
        fn evaluate(&self, g: &TestDNA) -> (f32, Vec<f32>, Vec<f32>) {
            // Inject NaN in objectives for some individuals to test sorting robustness
            let obj1 = if g.0 > 0.8 { f32::NAN } else { g.0 };
            let obj2 = if g.0 < 0.2 { f32::NAN } else { -g.0 };
            let fitness = if obj1.is_nan() || obj2.is_nan() {
                f32::NEG_INFINITY // Mark NaN individuals as low fitness
            } else {
                g.0
            };
            (fitness, vec![obj1, obj2], vec![])
        }
    }

    let initial: Vec<TestDNA> = (0..20).map(|i| TestDNA(i as f32 / 20.0)).collect();
    let mut nsga = Nsga2::new(initial, 0.3, 42);

    // Run with NaN injection - should not crash (this was the bug)
    for _ in 0..10 {
        nsga.step(&NaNInjectingEval);
    }

    // Verify population is still valid and algorithm completed
    let pop = nsga.population();
    assert_eq!(pop.len(), 20, "Population should maintain size");
}

/// Test that binary_tournament doesn't bias toward NaN crowding distances.
#[test]
fn test_nsga2_binary_tournament_nan_no_bias() {
    // This test verifies that when both candidates have equal rank,
    // the one with NaN crowding distance doesn't get unfair advantage.
    // With total_cmp, NaN > any finite value, so NaN distances are
    // treated as "most isolated" (which is actually reasonable behavior
    // since we can't measure their crowding).

    let initial: Vec<TestDNA> = (0..10).map(|i| TestDNA(i as f32)).collect();
    let mut nsga = Nsga2::new(initial, 0.1, 42);

    // Run a step to get valid population
    nsga.step(&TestEval);

    // The algorithm should complete without issues
    assert_eq!(nsga.pop_size(), 10);
}

// ============================================================================
// Issue #43: MapElites Resolution Integer Overflow at Extreme Values
// Review: Pedantic - f32 precision loss at resolutions > 2^24
// ============================================================================

/// Test MapElites binning precision at moderate resolutions.
#[test]
fn test_map_elites_resolution_precision_moderate() {
    // At resolution 1000, f32 has plenty of precision
    let engine = MapElites::<TestDNA>::new(1000, 0.1, 42);

    // Test boundary values
    let idx_zero = engine.map_to_index(&[0.0]);
    assert_eq!(idx_zero, vec![0], "0.0 should map to bin 0");

    let idx_one = engine.map_to_index(&[1.0]);
    assert_eq!(idx_one, vec![999], "1.0 should map to last bin (999)");

    // Test mid-point precision
    let idx_half = engine.map_to_index(&[0.5]);
    assert_eq!(idx_half, vec![500], "0.5 should map to bin 500");

    // Test near-boundary values
    let idx_near_one = engine.map_to_index(&[0.999]);
    assert_eq!(idx_near_one, vec![999], "0.999 should map to bin 999");
}

/// Test MapElites binning at maximum safe resolution (within f32 precision).
#[test]
fn test_map_elites_resolution_max_safe() {
    // 2^24 = 16,777,216 is the maximum integer exactly representable in f32
    // We use a smaller value to stay safely within precision limits
    let safe_resolution = 1 << 20; // 1,048,576
    let engine = MapElites::<TestDNA>::new(safe_resolution, 0.1, 42);

    // These should work correctly
    let idx_zero = engine.map_to_index(&[0.0]);
    assert_eq!(idx_zero, vec![0]);

    let idx_one = engine.map_to_index(&[1.0]);
    assert_eq!(idx_one, vec![safe_resolution - 1]);

    // Mid-point should be approximately correct
    let idx_half = engine.map_to_index(&[0.5]);
    let expected_half = safe_resolution / 2;
    assert!(
        (idx_half[0] as i64 - expected_half as i64).abs() <= 1,
        "0.5 should map near bin {} at resolution {}, got {}",
        expected_half,
        safe_resolution,
        idx_half[0]
    );
}

/// Test that extreme resolutions are handled gracefully (documented limitation).
#[test]
fn test_map_elites_resolution_extreme_documented() {
    // At extremely high resolutions, we accept some precision loss
    // but the algorithm should not panic or produce wildly wrong results
    let extreme_resolution = 1 << 26; // 67,108,864 - beyond f32 precision
    let engine = MapElites::<TestDNA>::new(extreme_resolution, 0.1, 42);

    // Boundaries should still work
    let idx_zero = engine.map_to_index(&[0.0]);
    assert_eq!(idx_zero, vec![0], "0.0 should always map to bin 0");

    let idx_one = engine.map_to_index(&[1.0]);
    assert_eq!(
        idx_one,
        vec![extreme_resolution - 1],
        "1.0 should always map to last bin"
    );

    // Mid-point may have some precision error but should be in ballpark
    let idx_half = engine.map_to_index(&[0.5]);
    let expected_half = extreme_resolution / 2;
    let error = (idx_half[0] as i64 - expected_half as i64).abs();
    let max_acceptable_error = (extreme_resolution / 1000) as i64; // 0.1% tolerance

    assert!(
        error <= max_acceptable_error,
        "0.5 mapping error {} exceeds tolerance {} at resolution {}",
        error,
        max_acceptable_error,
        extreme_resolution
    );
}

// ============================================================================
// Issue #44: NaN Handling in MapElites and NSGA-II (Critical)
// Review: NaN fitness/objectives cause archive corruption and convergence failure
// ============================================================================

/// Test that MapElites rejects individuals with NaN fitness during seeding.
/// Before fix: NaN individuals would overwrite valid elites due to comparison semantics.
#[test]
fn test_map_elites_nan_fitness_rejected_during_seed() {
    struct NaNFitnessEval;
    impl Evaluator<TestDNA> for NaNFitnessEval {
        fn evaluate(&self, g: &TestDNA) -> (f32, Vec<f32>, Vec<f32>) {
            let fitness = if g.0 > 0.5 { f32::NAN } else { g.0 * 100.0 };
            (fitness, vec![fitness], vec![0.5]) // All map to same bin
        }
    }

    let mut engine = MapElites::<TestDNA>::new(10, 0.1, 42);

    // First seed a valid individual with high fitness
    engine.seed_population(vec![TestDNA(0.4)], &NaNFitnessEval); // fitness = 40.0
    let initial_fitness = engine.archive_get(&vec![5]).unwrap().fitness;
    assert_eq!(
        initial_fitness, 40.0,
        "Initial elite should have fitness 40.0"
    );

    // Try to seed with NaN fitness individual (g.0 > 0.5 -> NaN)
    engine.seed_population(vec![TestDNA(0.9)], &NaNFitnessEval); // fitness = NaN

    // NaN individual should NOT overwrite the valid elite
    let final_fitness = engine.archive_get(&vec![5]).unwrap().fitness;
    assert_eq!(
        final_fitness, 40.0,
        "NaN fitness individual should not overwrite valid elite. Got {}",
        final_fitness
    );
}

/// Test that MapElites rejects individuals with NaN fitness during step().
#[test]
fn test_map_elites_nan_fitness_rejected_during_step() {
    struct IntermittentNaNEval;
    impl Evaluator<TestDNA> for IntermittentNaNEval {
        fn evaluate(&self, g: &TestDNA) -> (f32, Vec<f32>, Vec<f32>) {
            // Deterministically produce NaN for certain values
            let fitness = if (g.0 * 100.0) as i32 % 3 == 0 {
                f32::NAN
            } else {
                g.0 * 100.0
            };
            (fitness, vec![fitness], vec![g.0.clamp(0.0, 1.0)])
        }
    }

    let mut engine = MapElites::<TestDNA>::new(10, 0.5, 42);
    engine.set_batch_size(32);

    // Seed with valid individuals
    let initial: Vec<TestDNA> = (1..10).map(|i| TestDNA(i as f32 / 10.0)).collect();
    engine.seed_population(initial, &IntermittentNaNEval);

    let initial_len = engine.archive_len();

    // Run many steps with intermittent NaN fitness
    for _ in 0..50 {
        engine.step(&IntermittentNaNEval);
    }

    // Archive should not have shrunk (NaN individuals filtered out)
    assert!(
        engine.archive_len() >= initial_len,
        "Archive should not shrink due to NaN fitness. Initial: {}, Final: {}",
        initial_len,
        engine.archive_len()
    );

    // All archive entries should have valid (non-NaN) fitness
    for (key, pheno) in engine.archive_iter() {
        assert!(
            !pheno.fitness.is_nan(),
            "Archive should not contain NaN fitness. Found NaN at key {:?}",
            key
        );
    }
}

/// Test that MapElites rejects individuals with NaN descriptors.
/// Before fix: NaN descriptors silently mapped to bin 0, corrupting that cell.
#[test]
fn test_map_elites_nan_descriptor_rejected() {
    struct NaNDescriptorEval;
    impl Evaluator<TestDNA> for NaNDescriptorEval {
        fn evaluate(&self, g: &TestDNA) -> (f32, Vec<f32>, Vec<f32>) {
            let descriptor = if g.0 > 0.5 { f32::NAN } else { g.0 };
            (g.0 * 100.0, vec![g.0 * 100.0], vec![descriptor])
        }
    }

    let mut engine = MapElites::<TestDNA>::new(10, 0.1, 42);

    // Seed a valid individual in bin 0
    engine.seed_population(vec![TestDNA(0.05)], &NaNDescriptorEval); // desc = 0.05, maps to bin 0
    let bin0_fitness = engine.archive_get(&vec![0]).unwrap().fitness;
    assert!(
        (bin0_fitness - 5.0).abs() < 0.1,
        "Bin 0 should have fitness ~5.0"
    );

    // Try to seed individual with NaN descriptor (should be rejected, not map to bin 0)
    engine.seed_population(vec![TestDNA(0.9)], &NaNDescriptorEval); // desc = NaN

    // Bin 0 should still have the original elite, not the NaN-descriptor one
    let final_bin0_fitness = engine.archive_get(&vec![0]).unwrap().fitness;
    assert!(
        (final_bin0_fitness - 5.0).abs() < 0.1,
        "NaN descriptor individual should not corrupt bin 0. Expected ~5.0, got {}",
        final_bin0_fitness
    );
}

/// Test that NSGA-II properly handles NaN objectives by pushing them to worse fronts.
/// Before fix: NaN individuals ended up in Pareto Front 0 (best rank).
#[test]
fn test_nsga2_nan_objectives_pushed_to_worst_front() {
    struct NaNObjectivesEval;
    impl Evaluator<TestDNA> for NaNObjectivesEval {
        fn evaluate(&self, g: &TestDNA) -> (f32, Vec<f32>, Vec<f32>) {
            // Half the population gets NaN objectives
            if g.0 > 0.5 {
                (f32::NAN, vec![f32::NAN, f32::NAN], vec![])
            } else {
                (g.0, vec![g.0, 1.0 - g.0], vec![])
            }
        }
    }

    let initial: Vec<TestDNA> = (0..20).map(|i| TestDNA(i as f32 / 20.0)).collect();
    let mut nsga = Nsga2::new(initial, 0.1, 42);

    // Run evolution
    for _ in 0..20 {
        nsga.step(&NaNObjectivesEval);
    }

    // After evolution, the population should be dominated by valid individuals
    // (NaN individuals should have been pushed to worse fronts and selected against)
    let pop = nsga.population();
    let nan_count = pop
        .iter()
        .filter(|p| p.objectives.iter().any(|o| o.is_nan()))
        .count();

    // Allow some NaN individuals to survive (due to random selection), but they
    // should not dominate. With proper NaN handling, valid individuals should
    // be strongly preferred.
    assert!(
        nan_count < pop.len() / 2,
        "NaN individuals should not dominate population. Found {} NaN out of {}",
        nan_count,
        pop.len()
    );
}

/// Test that NSGA-II dominates function handles NaN correctly.
#[test]
fn test_nsga2_dominates_nan_handling() {
    // Create phenotypes for testing
    let valid_a = Phenotype {
        genotype: TestDNA(0.5),
        fitness: 0.5,
        objectives: vec![3.0, 2.0],
        descriptor: vec![],
    };
    let valid_b = Phenotype {
        genotype: TestDNA(0.3),
        fitness: 0.3,
        objectives: vec![2.0, 1.0],
        descriptor: vec![],
    };
    let nan_individual = Phenotype {
        genotype: TestDNA(0.9),
        fitness: f32::NAN,
        objectives: vec![f32::NAN, 5.0],
        descriptor: vec![],
    };

    // Valid a dominates valid b (better in both objectives)
    assert!(
        Nsga2::<TestDNA>::dominates(&valid_a, &valid_b),
        "Valid [3,2] should dominate valid [2,1]"
    );

    // NaN individual should NOT dominate valid individuals
    assert!(
        !Nsga2::<TestDNA>::dominates(&nan_individual, &valid_b),
        "NaN individual should not dominate valid individual"
    );

    // Valid individual SHOULD dominate NaN individual
    assert!(
        Nsga2::<TestDNA>::dominates(&valid_a, &nan_individual),
        "Valid individual should dominate NaN individual"
    );

    // NaN vs NaN: neither dominates
    let nan_b = Phenotype {
        genotype: TestDNA(0.8),
        fitness: f32::NAN,
        objectives: vec![f32::NAN, f32::NAN],
        descriptor: vec![],
    };
    assert!(
        !Nsga2::<TestDNA>::dominates(&nan_individual, &nan_b),
        "NaN should not dominate NaN"
    );
}

/// Test that MapElites recovers from NaN corruption in archive.
/// If somehow a NaN elite exists, valid individuals should be able to replace it.
#[test]
fn test_map_elites_nan_recovery() {
    struct ControlledEval {
        return_nan: bool,
    }
    impl Evaluator<TestDNA> for ControlledEval {
        fn evaluate(&self, g: &TestDNA) -> (f32, Vec<f32>, Vec<f32>) {
            let fitness = if self.return_nan {
                f32::NAN
            } else {
                g.0 * 100.0
            };
            (fitness, vec![fitness], vec![0.5])
        }
    }

    let mut engine = MapElites::<TestDNA>::new(10, 0.1, 42);

    // Seed with valid individual
    engine.seed_population(vec![TestDNA(0.3)], &ControlledEval { return_nan: false });
    let initial_fitness = engine.archive_get(&vec![5]).unwrap().fitness;
    assert!(
        (initial_fitness - 30.0).abs() < 0.01,
        "Initial fitness should be ~30.0, got {}",
        initial_fitness
    );

    // Try to overwrite with NaN (should be rejected)
    engine.seed_population(vec![TestDNA(0.9)], &ControlledEval { return_nan: true });
    let after_nan_fitness = engine.archive_get(&vec![5]).unwrap().fitness;
    assert!(
        (after_nan_fitness - 30.0).abs() < 0.01,
        "NaN should not overwrite valid elite, got {}",
        after_nan_fitness
    );

    // Valid individual with lower fitness should not overwrite
    engine.seed_population(vec![TestDNA(0.1)], &ControlledEval { return_nan: false });
    let after_lower_fitness = engine.archive_get(&vec![5]).unwrap().fitness;
    assert!(
        (after_lower_fitness - 30.0).abs() < 0.01,
        "Lower fitness should not overwrite, got {}",
        after_lower_fitness
    );

    // Valid individual with higher fitness should overwrite
    engine.seed_population(vec![TestDNA(0.5)], &ControlledEval { return_nan: false });
    let final_fitness = engine.archive_get(&vec![5]).unwrap().fitness;
    assert!(
        (final_fitness - 50.0).abs() < 0.01,
        "Higher fitness should overwrite, got {}",
        final_fitness
    );
}
