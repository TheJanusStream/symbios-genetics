use rand::Rng;
use serde::{Deserialize, Serialize};
use symbios_genetics::{Evaluator, Evolver, Genotype, algorithms::nsga2::Nsga2};

#[derive(Clone, Serialize, Deserialize, Debug)]
struct RectDNA {
    w: f32,
    h: f32,
}

impl Genotype for RectDNA {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        if rng.random::<f32>() < rate {
            self.w = (self.w + rng.random_range(-1.0..1.0)).clamp(0.1, 10.0);
            self.h = (self.h + rng.random_range(-1.0..1.0)).clamp(0.1, 10.0);
        }
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        RectDNA {
            w: if rng.random_bool(0.5) {
                self.w
            } else {
                other.w
            },
            h: if rng.random_bool(0.5) {
                self.h
            } else {
                other.h
            },
        }
    }

    fn distance(&self, other: &Self) -> f32 {
        ((self.w - other.w).powi(2) + (self.h - other.h).powi(2)).sqrt()
    }
}

struct RectEvaluator;
impl Evaluator<RectDNA> for RectEvaluator {
    fn evaluate(&self, genotype: &RectDNA) -> (f32, Vec<f32>, Vec<f32>) {
        let area = genotype.w * genotype.h;
        let perimeter = 2.0 * (genotype.w + genotype.h);
        (area, vec![area, -perimeter], vec![area, perimeter])
    }
}

fn main() {
    let mut rng = rand::rng();
    let initial_pop: Vec<RectDNA> = (0..50)
        .map(|_| RectDNA {
            w: rng.random_range(1.0..5.0),
            h: rng.random_range(1.0..5.0),
        })
        .collect();

    let mut engine = Nsga2::new(initial_pop, 0.1, 42);
    let eval = RectEvaluator;

    println!("Running Rectangle Trade-off (Max Area vs Min Perimeter)...");

    for _ in 0..100 {
        engine.step(&eval);
    }

    println!("\nPareto Front Results (Samples):");
    println!(
        "{:<10} | {:<10} | {:<10} | {:<10}",
        "Width", "Height", "Area", "Perimeter"
    );
    println!("-----------------------------------------------------------");

    for p in engine.population().iter().take(10) {
        println!(
            "{:<10.2} | {:<10.2} | {:<10.2} | {:<10.2}",
            p.genotype.w, p.genotype.h, p.objectives[0], -p.objectives[1]
        );
    }
}
