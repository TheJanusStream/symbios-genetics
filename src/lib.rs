use rand::Rng;
use serde::{Deserialize, Serialize};

pub trait Genotype: Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32);
    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self;
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "G: Genotype")]
pub struct Phenotype<G: Genotype> {
    pub genotype: G,
    pub fitness: f32,
    pub objectives: Vec<f32>,
    pub descriptor: Vec<f32>,
}

pub trait Evaluator<G: Genotype>: Send + Sync {
    fn evaluate(&self, genotype: &G) -> (f32, Vec<f32>, Vec<f32>);
}

pub trait Evolver<G: Genotype> {
    fn step<E: Evaluator<G>>(&mut self, evaluator: &E);
    fn population(&mut self) -> &[Phenotype<G>];
}

pub mod algorithms {
    pub mod map_elites;
    pub mod nsga2;
    pub mod simple;
}
