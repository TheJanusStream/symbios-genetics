use rand::Rng;
use serde::{Deserialize, Serialize};
use symbios_genetics::{Evaluator, Evolver, Genotype, algorithms::simple::SimpleGA};

const BIT_COUNT: usize = 128;

#[derive(Clone, Serialize, Deserialize, Debug)]
struct BitDNA(Vec<bool>);

impl Genotype for BitDNA {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        for bit in &mut self.0 {
            if rng.random::<f32>() < rate {
                *bit = !*bit; // Flip the bit
            }
        }
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        let split = rng.random_range(0..self.0.len());
        let mut child = self.0[..split].to_vec();
        child.extend_from_slice(&other.0[split..]);
        BitDNA(child)
    }

    fn distance(&self, other: &Self) -> f32 {
        // Hamming distance
        self.0.iter().zip(&other.0).filter(|(a, b)| a != b).count() as f32
    }
}

struct BitEvaluator {
    target: Vec<bool>,
}

impl Evaluator<BitDNA> for BitEvaluator {
    fn evaluate(&self, genotype: &BitDNA) -> (f32, Vec<f32>, Vec<f32>) {
        let matches = genotype
            .0
            .iter()
            .zip(&self.target)
            .filter(|(a, b)| a == b)
            .count() as f32;

        let fitness = matches / BIT_COUNT as f32; // Normalized 0.0 to 1.0
        (fitness, vec![fitness], vec![fitness])
    }
}

fn main() {
    let mut rng = rand::rng();

    // Generate a random target pattern to match
    let target_pattern: Vec<bool> = (0..BIT_COUNT).map(|_| rng.random_bool(0.5)).collect();

    let initial_pop: Vec<BitDNA> = (0..100)
        .map(|_| BitDNA((0..BIT_COUNT).map(|_| rng.random_bool(0.5)).collect()))
        .collect();

    let mut ga = SimpleGA::new(initial_pop, 0.02, 2);
    let eval = BitEvaluator {
        target: target_pattern,
    };

    println!("Evolving 128-bit pattern matching...");

    for generation in 0..1000 {
        ga.step(&eval);
        let best = &ga.population()[0];

        if generation % 100 == 0 {
            println!(
                "Generation {}: Best Accuracy: {:.2}%",
                generation,
                best.fitness * 100.0
            );
        }

        if best.fitness >= 1.0 {
            println!("🎯 Perfect match found at Generation {}!", generation);
            break;
        }
    }
}
