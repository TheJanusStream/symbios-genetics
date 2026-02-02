use rand::Rng;
use serde::{Deserialize, Serialize};
use symbios_genetics::{
    Evaluator, Evolver, Genotype, algorithms::map_elites::MapElites, algorithms::nsga2::Nsga2,
};

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
struct SimpleDNA(f32);

impl Genotype for SimpleDNA {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        if rng.random::<f32>() < rate {
            self.0 += rng.random::<f32>() - 0.5;
        }
    }
    fn crossover<R: Rng>(&self, other: &Self, _rng: &mut R) -> Self {
        SimpleDNA((self.0 + other.0) / 2.0)
    }
}

struct SimpleEval;
impl Evaluator<SimpleDNA> for SimpleEval {
    fn evaluate(&self, genotype: &SimpleDNA) -> (f32, Vec<f32>, Vec<f32>) {
        (
            genotype.0,
            vec![genotype.0, -genotype.0],
            vec![genotype.0.clamp(0.0, 1.0)],
        )
    }
}

// ============================================================================
// Issue #10: MapElites population_cache deserialization bug
// ============================================================================

#[test]
fn test_map_elites_population_visible_after_deserialization() {
    let mut engine = MapElites::<SimpleDNA>::new(10, 0.1, 42);
    let eval = SimpleEval;

    // Seed with some individuals
    engine.seed_population(vec![SimpleDNA(0.5), SimpleDNA(0.3)], &eval);

    // Run a step to populate the cache
    engine.step(&eval);

    let pop_before = engine.population().len();
    assert!(
        pop_before > 0,
        "Population should have individuals before serialization"
    );

    // Serialize and deserialize (using bincode since JSON doesn't support Vec keys)
    let serialized = bincode::serialize(&engine).expect("Serialization failed");
    let mut deserialized: MapElites<SimpleDNA> =
        bincode::deserialize(&serialized).expect("Deserialization failed");

    // BUG: population_cache is #[serde(skip)], so it's empty after deserialization
    let pop_after = deserialized.population().len();
    assert_eq!(
        pop_after, pop_before,
        "Population should be visible immediately after deserialization without calling step(). \
         Got {} individuals, expected {}. This is the deserialization cache bug.",
        pop_after, pop_before
    );
}

// ============================================================================
// Issue #11: MapElites seed_population doesn't update cache
// ============================================================================

#[test]
fn test_map_elites_population_visible_after_seeding() {
    let mut engine = MapElites::<SimpleDNA>::new(10, 0.1, 42);
    let eval = SimpleEval;

    // Seed the population
    engine.seed_population(vec![SimpleDNA(0.5), SimpleDNA(0.3), SimpleDNA(0.7)], &eval);

    // BUG: seed_population doesn't update population_cache
    let pop = engine.population();
    assert!(
        !pop.is_empty(),
        "Population should be visible immediately after seeding without calling step(). \
         This is the seed_population cache update bug."
    );

    // Verify we can see all the seeded individuals (up to archive collisions)
    assert!(
        engine.archive_len() > 0,
        "Archive should contain seeded individuals"
    );
}

// ============================================================================
// Issue #13: NSGA-II ragged objectives panic
// ============================================================================

struct RaggedEval;
impl Evaluator<SimpleDNA> for RaggedEval {
    fn evaluate(&self, genotype: &SimpleDNA) -> (f32, Vec<f32>, Vec<f32>) {
        // Return different objective counts based on genotype value
        // This simulates conditional evaluation (e.g., invalid genomes get fewer objectives)
        if genotype.0 > 0.5 {
            (
                genotype.0,
                vec![genotype.0, -genotype.0, genotype.0 * 2.0],
                vec![],
            )
        } else {
            (genotype.0, vec![genotype.0, -genotype.0], vec![])
        }
    }
}

#[test]
fn test_nsga2_handles_ragged_objectives_gracefully() {
    // Create population that will produce ragged objectives
    let initial = vec![
        SimpleDNA(0.3), // Will get 2 objectives
        SimpleDNA(0.4), // Will get 2 objectives
        SimpleDNA(0.6), // Will get 3 objectives
        SimpleDNA(0.7), // Will get 3 objectives
    ];

    let mut engine = Nsga2::new(initial, 0.0, 42);
    let eval = RaggedEval;

    // BUG: This panics with index out of bounds in calculate_crowding_distance
    // because it assumes all individuals have the same objective count
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.step(&eval);
    }));

    assert!(
        result.is_ok(),
        "NSGA-II should handle ragged objectives gracefully instead of panicking. \
         Either validate and reject, or handle consistently."
    );
}

// ============================================================================
// Issue #14: NSGA-II public field encapsulation allows invariant violation
// ============================================================================

#[test]
fn test_nsga2_fields_are_encapsulated() {
    let initial = vec![SimpleDNA(0.5), SimpleDNA(0.6)];
    let mut engine = Nsga2::new(initial, 0.1, 42);

    // Fields should now be private and only accessible via getters
    // This test verifies the API is encapsulated
    assert_eq!(engine.pop_size(), 2);
    assert!((engine.mutation_rate() - 0.1).abs() < f32::EPSILON);

    // Verify population is accessible via trait method
    // (population field is private, but population() method exists)
    let pop = engine.population();
    assert_eq!(pop.len(), 2);
}

#[test]
fn test_nsga2_empty_initial_population_handled() {
    // Creating with empty population should be handled gracefully
    let initial: Vec<SimpleDNA> = vec![];
    let mut engine = Nsga2::new(initial, 0.1, 42);
    let eval = SimpleEval;

    // step() with empty population should not panic
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.step(&eval);
    }));

    assert!(
        result.is_ok(),
        "NSGA-II step() should handle empty population gracefully."
    );
}

// ============================================================================
// Issue #15: NSGA-II dominates function zip truncation
// ============================================================================

#[test]
fn test_nsga2_dominates_detects_length_mismatch() {
    use symbios_genetics::Phenotype;

    let short = Phenotype {
        genotype: SimpleDNA(1.0),
        fitness: 1.0,
        objectives: vec![1.0, 1.0],
        descriptor: vec![],
    };

    let long = Phenotype {
        genotype: SimpleDNA(2.0),
        fitness: 2.0,
        objectives: vec![0.5, 0.5, 100.0], // Third objective is much better
        descriptor: vec![],
    };

    // BUG: zip() silently truncates, so the comparison only considers first 2 objectives
    // This makes `short` appear to dominate `long` when considering only [1.0, 1.0] vs [0.5, 0.5]
    // But the third objective (100.0) should make this comparison invalid or different

    // The dominates function should either:
    // 1. Panic/error on mismatched lengths (strict validation)
    // 2. Return false for both (neither dominates if we can't compare all objectives)

    let short_dominates_long = Nsga2::<SimpleDNA>::dominates(&short, &long);
    let long_dominates_short = Nsga2::<SimpleDNA>::dominates(&long, &short);

    // With the bug: short_dominates_long = true (zip truncates, 1.0 > 0.5 in both)
    // Correct behavior: BOTH should be false because we can't properly compare

    // Strict assertion: neither should dominate when objective counts differ
    assert!(
        !short_dominates_long,
        "short should not dominate long when objective counts differ"
    );
    assert!(
        !long_dominates_short,
        "long should not dominate short when objective counts differ"
    );
}
