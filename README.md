# symbios-genetics

**A sovereign, battle-hardened evolutionary computation engine for Rust.**

`symbios-genetics` is a trait-based library designed for **Morphogenetic Engineering**, **Artificial Life**, and **Creative AI**. Unlike general-purpose genetic libraries, it prioritizes **correctness**, **reproducibility**, and **serialization** above all else.

## Key Features

*   **Trait-Based Architecture:** Strictly decouples the *Genotype* (Mutation/Crossover logic) from the *Phenotype* (Performance/Metrics).
*   **Parallel-First:** Built-in `rayon` support allows for $O(N)$ scaling of fitness evaluation.
*   **Zero-Copy Design:** Uses efficient data structures (e.g., `BTreeMap` for sparse MAP-Elites archives) to minimize allocation overhead during evolution steps.

## Algorithms

The library implements three distinct evolutionary strategies covering the spectrum of optimization needs:

| Algorithm | Type | Best Use Case |
|-----------|------|---------------|
| **SimpleGA** | Single-Objective | Converging on a specific optimal solution (e.g., maximizing speed). Features Elitism and Tournament Selection. |
| **NSGA-II** | Multi-Objective | Finding the *Pareto Front* of trade-offs between conflicting goals (e.g., maximize strength AND minimize weight). |
| **MAP-Elites** | Quality-Diversity | Illuminating the search space. Finds the best solution for every possible niche (e.g., "fastest robot for every possible height"). |

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
symbios-genetics = "0.1.0"
serde = { version = "1.0", features = ["derive"] }
rand = "0.9"
```

### Defining a Genome
Implement the `Genotype` trait for your data structure.

```rust
use rand::Rng;
use serde::{Deserialize, Serialize};
use symbios_genetics::Genotype;

#[derive(Clone, Serialize, Deserialize, Debug)]
struct MyDNA {
    value: f32,
}

impl Genotype for MyDNA {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        if rng.random::<f32>() < rate {
            self.value += rng.random_range(-0.1..0.1);
        }
    }

    fn crossover<R: Rng>(&self, other: &Self, _rng: &mut R) -> Self {
        MyDNA {
            value: (self.value + other.value) / 2.0,
        }
    }
}
```

### Defining an Evaluator

Implement `Evaluator` to bridge your genome to the engine. Returns a tuple of `(Fitness, Objectives, Descriptor)`.

```rust
use symbios_genetics::Evaluator;

struct MyEvaluator;

impl Evaluator<MyDNA> for MyEvaluator {
    fn evaluate(&self, dna: &MyDNA) -> (f32, Vec<f32>, Vec<f32>) {
        // 1. Fitness (Scalar): Used by SimpleGA
        let fitness = -(dna.value - 42.0).abs(); 
        
        // 2. Objectives (Vector): Used by NSGA-II
        let objectives = vec![fitness, -dna.value]; 
        
        // 3. Descriptor (Vector): Used by MAP-Elites (Normalized 0.0-1.0)
        let descriptor = vec![dna.value.clamp(0.0, 100.0) / 100.0];
        
        (fitness, objectives, descriptor)
    }
}
```

### Running Evolution

```rust
use symbios_genetics::{
    algorithms::simple::SimpleGA,
    Evolver,
};

fn main() {
    // 1. Initialize
    let initial_pop = (0..100).map(|_| MyDNA { value: 0.0 }).collect();
    
    // 2. Configure Engine (Pop, Mutation Rate, Elitism, Seed)
    let mut engine = SimpleGA::new(initial_pop, 0.1, 5, 12345);
    let evaluator = MyEvaluator;

    // 3. Evolve
    for _ in 0..100 {
        engine.step(&evaluator);
    }

    // 4. Inspect
    let best = &engine.population()[0];
    println!("Best DNA: {:?} (Fitness: {})", best.genotype, best.fitness);
}
```

## Architecture

### The `Evolver` Trait
All algorithms implement the `Evolver<G>` trait. This allows you to write simulation harnesses (e.g., a Bevy plugin) that are agnostic to the specific evolutionary strategy. You can hot-swap `SimpleGA` for `MapElites` without rewriting your game loop.

### Parallelism
To enable parallel evaluation, ensure the `parallel` feature is enabled (default) and your `Evaluator` implements `Send + Sync`. The engine automatically dispatches evaluation tasks via `rayon::par_iter`.

```toml
symbios-genetics = { version = "0.1.0", features = ["parallel"] }
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
