use rand::Rng;
use serde::{Deserialize, Serialize};
use symbios_genetics::{Evaluator, Evolver, Genotype, algorithms::simple::SimpleGA};

const TARGET: &str = "Sovereign Symbiosis";

#[derive(Clone, Serialize, Deserialize, Debug)]
struct StringDNA(Vec<u8>);

impl Genotype for StringDNA {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        for byte in &mut self.0 {
            if rng.random::<f32>() < rate {
                *byte = rng.random_range(32..126);
            }
        }
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        let split = rng.random_range(0..self.0.len());
        let mut child = self.0[..split].to_vec();
        child.extend_from_slice(&other.0[split..]);
        StringDNA(child)
    }

    fn distance(&self, other: &Self) -> f32 {
        self.0.iter().zip(&other.0).filter(|(a, b)| a != b).count() as f32
    }
}

struct StringEvaluator;
impl Evaluator<StringDNA> for StringEvaluator {
    fn evaluate(&self, genotype: &StringDNA) -> (f32, Vec<f32>, Vec<f32>) {
        let fitness = genotype
            .0
            .iter()
            .zip(TARGET.as_bytes())
            .filter(|(a, b)| a == b)
            .count() as f32;
        (fitness, vec![fitness], vec![])
    }
}

fn main() {
    let mut rng = rand::rng();
    let initial_pop: Vec<StringDNA> = (0..100)
        .map(|_| {
            StringDNA(
                (0..TARGET.len())
                    .map(|_| rng.random_range(32..126))
                    .collect(),
            )
        })
        .collect();

    let mut ga = SimpleGA::new(initial_pop, 0.05, 5, 42);
    let eval = StringEvaluator;

    for generation in 0..500 {
        ga.step(&eval);
        let best = &ga.population()[0];
        let current_str = String::from_utf8_lossy(&best.genotype.0);

        if generation % 50 == 0 {
            println!(
                "Gen {}: [{}] (Fitness: {})",
                generation, current_str, best.fitness
            );
        }

        if current_str == TARGET {
            println!("🎯 Target reached at Gen {}!", generation);
            break;
        }
    }
}
