use rand::Rng;
use serde::{Deserialize, Serialize};

/// The 'DNA' of an individual.
/// Defined by how it changes, not what it does.
pub trait Genotype: Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32);
    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self;

    /// Distance metric for speciation or diversity tracking.
    fn distance(&self, other: &Self) -> f32;
}

/// The 'Body' expressed from DNA.
/// Holds the results of evaluation.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "G: Genotype")]
pub struct Phenotype<G: Genotype> {
    pub genotype: G,
    pub fitness: f32,
    /// Multi-objective values (used by NSGA-II)
    pub objectives: Vec<f32>,
    /// N-dimensional position in the morphospace (used by MAP-Elites)
    pub descriptor: Vec<f32>,
}

/// A trait for systems that can turn DNA into a Body and evaluate it.
pub trait Evaluator<G: Genotype>: Send + Sync {
    /// Express DNA and return (fitness, objectives, descriptor).
    /// This is where the 'Physics' or 'Logic' happens.
    fn evaluate(&self, genotype: &G) -> (f32, Vec<f32>, Vec<f32>);
}

/// The master engine trait.
pub trait Evolver<G: Genotype> {
    fn step<E: Evaluator<G>>(&mut self, evaluator: &E);
    fn population(&self) -> &[Phenotype<G>];
}

pub mod algorithms {
    pub mod map_elites;
    pub mod nsga2;
    pub mod simple;
}
