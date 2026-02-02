use rand::Rng;
use serde::{Deserialize, Serialize};
use symbios_genetics::{
    Evaluator, Evolver, Genotype, Phenotype, algorithms::map_elites::MapElites,
    algorithms::nsga2::Nsga2,
};

// --- Mock Infrastructure ---

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
struct MockDNA(f32);

impl Genotype for MockDNA {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        // Corrected: only mutate if roll < rate
        if rng.random::<f32>() < rate {
            self.0 += 1.0;
        }
    }
    fn crossover<R: Rng>(&self, other: &Self, _rng: &mut R) -> Self {
        MockDNA((self.0 + other.0) / 2.0)
    }
    fn distance(&self, other: &Self) -> f32 {
        (self.0 - other.0).abs()
    }
}

struct MockEval;
impl Evaluator<MockDNA> for MockEval {
    fn evaluate(&self, genotype: &MockDNA) -> (f32, Vec<f32>, Vec<f32>) {
        // Maximizing Obj0 (Value) and Obj1 (-Value) ensures a perfectly linear Pareto Front
        (
            genotype.0,
            vec![genotype.0, -genotype.0],
            vec![genotype.0 / 10.0],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_elites_index_mapping() {
        let engine = MapElites::<MockDNA>::new(10, 0.1, 42);
        assert_eq!(engine.map_to_index(&[0.0]), vec![0]);
        assert_eq!(engine.map_to_index(&[1.0]), vec![9]);
    }

    #[test]
    fn test_nsga2_step_diversity() {
        // Use a mutation rate of 0.0 to ensure 1.0 and 3.0 remain extreme boundaries
        let initial_pop = vec![MockDNA(1.0), MockDNA(2.0), MockDNA(3.0)];
        let mut engine = Nsga2::new(initial_pop, 0.0, 42);
        let eval = MockEval;

        engine.step(&eval);

        let pop = engine.population();
        let values: Vec<f32> = pop.iter().map(|p| p.genotype.0).collect();

        // With 0.0 mutation, 1.0 and 3.0 are guaranteed to be Pareto boundaries
        // and should be preserved by Crowding Distance logic.
        assert!(
            values.contains(&1.0),
            "Population missing 1.0 boundary: {:?}",
            values
        );
        assert!(
            values.contains(&3.0),
            "Population missing 3.0 boundary: {:?}",
            values
        );
    }

    #[test]
    fn test_nsga2_dominance_logic() {
        let p_strong = Phenotype {
            genotype: MockDNA(10.0),
            fitness: 10.0,
            objectives: vec![10.0, 10.0],
            descriptor: vec![],
        };
        let p_weak = Phenotype {
            genotype: MockDNA(1.0),
            fitness: 1.0,
            objectives: vec![1.0, 1.0],
            descriptor: vec![],
        };
        assert!(Nsga2::dominates(&p_strong, &p_weak));
        assert!(!Nsga2::dominates(&p_weak, &p_strong));
    }
}
