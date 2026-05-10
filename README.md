# symbios-genetics

**A battle-hardened evolutionary computation engine for Rust.**

`symbios-genetics` is a trait-based library designed for **Morphogenetic Engineering**, **Artificial Life**, and **Creative AI**. Unlike general-purpose genetic libraries, it prioritizes **correctness**, **reproducibility**, and **serialization** above all else.

## Algorithms

The library implements three distinct evolutionary strategies covering the spectrum of optimization needs:

| Algorithm          | Type                   | Best Use Case                                                                                                                                            |
|--------------------|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| **SimpleGA**       | Single-Objective       | Converging on a specific optimal solution (e.g., maximizing speed). Features Elitism and Tournament Selection.                                           |
| **NSGA-II**        | Multi-Objective        | Finding the *Pareto Front* of trade-offs between conflicting goals (e.g., maximize strength AND minimize weight).                                        |
| **MAP-Elites**     | Quality-Diversity      | Illuminating the search space. Finds the best solution for every possible niche (e.g., "fastest robot for every possible height").                       |
| **CVT-MAP-Elites** | Quality-Diversity      | MAP-Elites for high-dimensional or non-uniform descriptor spaces. Decouples archive size from descriptor dimensionality via a Voronoi tessellation.      |
| **Novelty Search** | Open-ended exploration | Replaces fitness with kNN behavioural distance. Escapes deceptive local optima where pure fitness search gets stuck. Configurable novelty/fitness blend. |

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
symbios-genetics = "0.2"
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

## Quality-Diversity Metrics & Export

`MapElites` exposes the standard QD metrics and a CSV export of the archive:

```rust,ignore
use std::fs::File;

let coverage = engine.coverage();   // fraction of cells occupied (0.0..=1.0)
let qd_score = engine.qd_score();   // sum of fitness across cells

// CSV export requires the `export` feature.
let mut out = File::create("archive.csv")?;
engine.export_csv(&mut out)?;
```

Enable export with:

```toml
symbios-genetics = { version = "0.2", features = ["export"] }
```

The CSV has columns `key,descriptor,fitness,objectives,genotype_hash` — one row per occupied cell, in deterministic iteration order. The genotype hash is a 16-hex-char `seahash` of the bincode-serialised genotype, suitable for joining against a separate genotype dump.

> **Note**: `qd_score` is the raw sum of fitness. If your fitness can be negative (e.g. `-distance`), shift it to non-negative before relying on the score for cross-run comparison.

## Pre-made Scorers

The [`scorers`](https://docs.rs/symbios-genetics/latest/symbios_genetics/scorers/) module ships composable building blocks for locomotion / robotics fitness:

- **Concrete scorers** over a [`Trajectory`] struct: `Displacement`, `UpAlignment`, `Height`, `EnergyEfficiency` — and `Const` for literals.
- **Combinators**: `Multiply`, `Sum`, `Penalize`, `Normalize` for assembling multi-term objectives.
- **`CompositeEvaluator`** to bridge a scorer composition into the `Evaluator<G>` trait used by every algorithm. The simulator function runs once per genotype; the result is fanned out to fitness, `objectives` (NSGA-II), and `descriptor` (MAP-Elites) signals.

```rust,ignore
use symbios_genetics::scorers::{
    CompositeEvaluator, Const, Displacement, EnergyEfficiency, Height,
    Multiply, Sum, Trajectory, UpAlignment,
};

// locomotion = displacement * up_alignment * (height + 0.5)
let locomotion = Multiply(
    Multiply(Displacement, UpAlignment),
    Sum(Height, Const(0.5_f32)),
);

let evaluator = CompositeEvaluator::<Robot, Trajectory, _>::new(
    |robot| run_physics_sim(robot),  // user-supplied: Robot -> Trajectory
    Box::new(locomotion),
)
.with_objectives(vec![Box::new(Displacement), Box::new(EnergyEfficiency)])
.with_descriptors(vec![Box::new(UpAlignment), Box::new(Height)]);
```

Behavioural diversity is intentionally not provided as a scorer — it's a population-level signal. Use [`NoveltySearch`](https://docs.rs/symbios-genetics/latest/symbios_genetics/algorithms/novelty_search/) for that.

## Speciation

The [`speciation`](https://docs.rs/symbios-genetics/latest/symbios_genetics/speciation/) module is a standalone NEAT-style speciation primitive that operates on a `&mut [Phenotype<G>]`. You supply a `CompatibilityDistance<G>` metric; `Speciation` clusters the population by that distance, applies Stanley & Miikkulainen *explicit fitness sharing* (each phenotype's fitness is divided by its species size), and adapts the compatibility threshold each generation toward a configured target species count.

It is genotype-agnostic and not coupled to a specific [`Evolver`] — drive it from your own selection loop, or against a population you maintain alongside an engine.

```rust,ignore
use symbios_genetics::{Phenotype, speciation::{CompatibilityDistance, Speciation}};

struct Euclidean;
impl CompatibilityDistance<MyDNA> for Euclidean {
    fn distance(&self, a: &MyDNA, b: &MyDNA) -> f32 { /* ... */ 0.0 }
}

let mut spec = Speciation::new(Euclidean, /*initial_threshold*/ 0.5, /*target_count*/ 8);
let mut pop: Vec<Phenotype<MyDNA>> = /* evaluated population you own */ vec![];

// Each generation, after evaluation:
spec.assign(&pop);
spec.share_fitness(&mut pop);
spec.adjust_threshold();
```

Species IDs are stable across generations as long as a representative survives, making it straightforward to track lineage. See the module docs for the algorithm sketch and tuning knobs (`with_threshold_step`, `with_min_threshold`).

## Architecture

### The `Evolver` Trait

All algorithms implement the `Evolver<G>` trait. This allows you to write simulation harnesses (e.g., a Bevy plugin) that are agnostic to the specific evolutionary strategy. You can hot-swap `SimpleGA` for `MapElites` without rewriting your game loop.

### Parallelism

To enable parallel evaluation, ensure the `parallel` feature is enabled (default) and your `Evaluator` implements `Send + Sync`. The engine automatically dispatches evaluation tasks via `rayon::par_iter`.

```toml
symbios-genetics = { version = "0.2", features = ["parallel"] }
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
