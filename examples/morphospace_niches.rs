use rand::Rng;
use serde::{Deserialize, Serialize};
use symbios_genetics::{
    Evaluator, Evolver, Genotype, Phenotype, algorithms::map_elites::MapElites,
};

#[derive(Clone, Serialize, Deserialize, Debug)]
struct ShapeDNA {
    h: f32, // Height [0.1, 10.0]
    w: f32, // Width  [0.1, 10.0]
}

impl Genotype for ShapeDNA {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        if rng.random::<f32>() < rate {
            // Evolutionary drift
            self.h = (self.h + rng.random_range(-0.5..0.5)).clamp(0.1, 10.0);
            self.w = (self.w + rng.random_range(-0.5..0.5)).clamp(0.1, 10.0);
        }
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        ShapeDNA {
            h: if rng.random_bool(0.5) {
                self.h
            } else {
                other.h
            },
            w: if rng.random_bool(0.5) {
                self.w
            } else {
                other.w
            },
        }
    }
}

struct ShapeEvaluator;
impl Evaluator<ShapeDNA> for ShapeEvaluator {
    fn evaluate(&self, genotype: &ShapeDNA) -> (f32, Vec<f32>, Vec<f32>) {
        let area = genotype.h * genotype.w;

        // Descriptor: Normalized [h, w] to map into the archive grid
        // We divide by 10.0 because our range is 0.1 to 10.0
        let descriptor = vec![genotype.h / 10.0, genotype.w / 10.0];

        (area, vec![area], descriptor)
    }
}

fn main() {
    let mut rng = rand::rng();
    let mutation_rate = 0.2;
    let resolution = 10; // 10x10 grid = 100 niches

    let mut engine = MapElites::new(resolution, mutation_rate, 42);
    engine.batch_size = 50;
    let eval = ShapeEvaluator;

    // 1. Initial Seeding: Drop a few random shapes into the void
    for _ in 0..5 {
        let dna = ShapeDNA {
            h: rng.random_range(1.0..5.0),
            w: rng.random_range(1.0..5.0),
        };
        let (f, obj, desc) = eval.evaluate(&dna);
        let idx = desc
            .iter()
            .map(|&v| (v * (resolution - 1) as f32) as usize)
            .collect::<Vec<_>>();
        engine.archive.insert(
            idx,
            Phenotype {
                genotype: dna,
                fitness: f,
                objectives: obj,
                descriptor: desc,
            },
        );
    }

    println!("Mapping 10x10 Morphospace (100 niches)...");

    // 2. Evolutionary Steps
    for generation in 0..100 {
        // Use parallel batching (size 50)
        engine.step(&eval);

        if generation % 20 == 0 {
            println!(
                "Generation {}: Archive Coverage: {}/{}",
                generation,
                engine.archive.len(),
                resolution * resolution
            );
        }
    }

    // 3. Visualize the "Map" (ASCII Representation)
    println!("\nArchive Map (H=Height, W=Width):");
    println!("(Numbers = Fitness/Area in that niche)");
    println!("      W 0    1    2    3    4    5    6    7    8    9");
    for row in 0..resolution {
        print!("H {} |", row);
        for col in 0..resolution {
            let key = vec![row, col];
            if let Some(elite) = engine.archive.get(&key) {
                print!("{:>4.0} ", elite.fitness);
            } else {
                print!("  .  ");
            }
        }
        println!();
    }

    println!("\nFinal Stats:");
    println!("Total Elites Found: {}", engine.archive.len());
    let best_overall = engine
        .archive
        .values()
        .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
        .unwrap();
    println!(
        "Overall Area King: H:{:.2} W:{:.2} (Area: {:.2})",
        best_overall.genotype.h, best_overall.genotype.w, best_overall.fitness
    );
}
