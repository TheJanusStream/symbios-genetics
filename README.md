# symbios-genetics

**A sovereign, high-performance evolutionary computation engine for Rust.**

`symbios-genetics` is a trait-based library designed for **Morphogenetic Engineering**, **Artificial Life**, and **Creative AI**. It prioritizes correctness, reproducibility, and massive parallelism over ease-of-use for trivial tasks. It is built to power the **Symbios Ecosystem**, enabling the evolution of complex morphologies, neural controllers, and procedural artifacts.

> **Status:** v0.1.0 (Hardened MVP). Ready for research and production experimentation.

## Key Features

*   **Trait-Based Architecture:** Strictly decouples the *Genotype* (DNA/Mutation logic) from the *Phenotype* (Performance/Metrics).
*   **Parallel-First:** Built-in `rayon` support for `O(N)` scaling of population evaluation.
*   **Sovereign Persistence:** Full `serde` support for all engine states, including the Random Number Generator (`Pcg64`). You can pause, save to disk, and resume an evolution bit-perfectly on a different machine.
*   **Adversarially Hardened:** Rigorously tested against NaN fitness, empty populations, and edge-case configurations to ensure panic-free operation in long-running simulations.
*   **Deterministic:** Uses `rand_pcg` for portable, seedable determinism.

## Algorithms

The library implements three distinct evolutionary strategies covering the spectrum of optimization needs:

1.  **SimpleGA:** Standard Genetic Algorithm with Elitism and Tournament Selection. Best for single-objective convergence.
2.  **NSGA-II:** Non-dominated Sorting Genetic Algorithm II. Best for Multi-Objective Optimization (finding the Pareto Front).
3.  **MAP-Elites:** Multi-dimensional Archive of Phenotypic Elites. Best for **Quality-Diversity** (QD) and search space illumination.

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
symbios-genetics = "0.1.0"
serde = { version = "1.0", features = ["derive"] }
rand = "0.9"
```

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
        // Simple average crossover
        MyDNA {
            value: (self.value + other.value) / 2.0,
        }
    }
}
```

```rust
use symbios_genetics::Evaluator;

struct MyEvaluator;

impl Evaluator<MyDNA> for MyEvaluator {
    fn evaluate(&self, dna: &MyDNA) -> (f32, Vec<f32>, Vec<f32>) {
        // 1. Fitness (Single scalar for SimpleGA/MAP-Elites)
        let fitness = -(dna.value - 42.0).abs(); // Target 42.0
        
        // 2. Objectives (Vector for NSGA-II)
        let objectives = vec![fitness]; 
        
        // 3. Descriptor (Vector for MAP-Elites niches)
        let descriptor = vec![dna.value.clamp(0.0, 100.0) / 100.0];
        
        (fitness, objectives, descriptor)
    }
}
```

```rust
use symbios_genetics::{
    algorithms::simple::SimpleGA,
    Evolver,
};

fn main() {
    // Initialize population
    let initial_pop = (0..100).map(|_| MyDNA { value: 0.0 }).collect();
    
    // Create engine (Mutation Rate: 0.1, Elitism: 5, Seed: 12345)
    let mut engine = SimpleGA::new(initial_pop, 0.1, 5, 12345);
    let evaluator = MyEvaluator;

    for _ in 0..100 {
        engine.step(&evaluator);
    }

    let best = &engine.population()[0];
    println!("Best DNA: {:?} (Fitness: {})", best.genotype, best.fitness);
}
```

## Architectural Concepts
### The Evolver Trait

All algorithms implement the Evolver<G> trait. This allows you to write simulation harnesses (e.g., a Bevy plugin) that are agnostic to the specific evolutionary strategy being used. You can hot-swap SimpleGA for NSGA-II without rewriting your loop.

### The Phenotype Wrapper

The engine wraps your Genotype in a Phenotype<G> struct. This stores the metadata (fitness, objectives, descriptors) alongside the DNA, preventing re-evaluation of unchanged individuals and simplifying serialization.

### Parallelism

To enable parallel evaluation, enable the parallel feature in Cargo.toml:

```toml
symbios-genetics = { version = "0.1.0", features = ["parallel"] }
```

The engine automatically uses rayon::par_iter during the step() function. Ensure your Evaluator is Send + Sync.

### Stability Guarantees

This crate adheres to a Zero-Panic Policy for runtime operations.

NaN Safety: Fitness values that evaluate to NaN are strictly handled (sorted to the bottom).

Empty Populations: Functions gracefully handle empty input vectors.

Serialization: All internal state is serializable, ensuring "Save/Load" is always possible.