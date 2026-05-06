//! Grid MAP-Elites vs CVT-MAP-Elites on the same problem.
//!
//! Both algorithms maximise rectangle area (`h * w`) over a 2D behaviour
//! space `(h, w) ∈ [0, 1]²`, with fitness gated to the unit disk centred
//! at (0.5, 0.5). With a uniform 2D descriptor and matched archive budget
//! (144 cells = 12×12 grid ≈ 144 CVT centroids), grid and CVT achieve
//! comparable coverage and QD score — confirming CVT loses nothing when
//! the descriptor distribution is uniform.
//!
//! The real win for CVT shows up in two cases this example *doesn't*
//! exercise:
//!
//! 1. **High descriptor dimensionality.** A 6-D grid at resolution 10
//!    needs 10⁶ cells; CVT keeps the archive at whatever `num_centroids`
//!    you pick, decoupled from dim. See the
//!    `coverage_difference_grid_vs_cvt_in_high_dim` test.
//! 2. **Non-uniform descriptor distributions.** With caller-supplied
//!    centroids (via [`CvtMapElites::new`]), centroids can be precomputed
//!    against the actual descriptor distribution, concentrating archive
//!    capacity where the action is.
//!
//! Run with: `cargo run --example grid_vs_cvt --release`

use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use serde::{Deserialize, Serialize};
use symbios_genetics::algorithms::cvt_map_elites::CvtMapElites;
use symbios_genetics::algorithms::map_elites::MapElites;
use symbios_genetics::{Evaluator, Evolver, Genotype};

#[derive(Clone, Serialize, Deserialize, Debug)]
struct Rect {
    h: f32,
    w: f32,
}

impl Genotype for Rect {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        if rng.random::<f32>() < rate {
            self.h = (self.h + rng.random_range(-0.1..0.1)).clamp(0.0, 1.0);
        }
        if rng.random::<f32>() < rate {
            self.w = (self.w + rng.random_range(-0.1..0.1)).clamp(0.0, 1.0);
        }
    }
    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        Rect {
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

/// Fitness = area, but only inside the unit disk centred at (0.5, 0.5).
/// Outside the disk, fitness is 0 — those cells stay empty in practice
/// because any in-disk competitor with positive area dominates.
struct DiskRect;
impl Evaluator<Rect> for DiskRect {
    fn evaluate(&self, g: &Rect) -> (f32, Vec<f32>, Vec<f32>) {
        let dx = g.h - 0.5;
        let dy = g.w - 0.5;
        let in_disk = dx * dx + dy * dy <= 0.25;
        let fitness = if in_disk { g.h * g.w } else { 0.0 };
        (fitness, vec![fitness], vec![g.h, g.w])
    }
}

fn random_rects<R: Rng>(n: usize, rng: &mut R) -> Vec<Rect> {
    (0..n)
        .map(|_| Rect {
            h: rng.random::<f32>(),
            w: rng.random::<f32>(),
        })
        .collect()
}

fn main() {
    const MUTATION_RATE: f32 = 0.3;
    const BATCH_SIZE: usize = 64;
    const SEED: u64 = 42;
    const STEPS: usize = 200;
    const SEED_POP: usize = 200;

    let eval = DiskRect;

    // Grid MAP-Elites: 12x12 = 144 cells.
    let mut grid_seed_rng = Pcg64::seed_from_u64(SEED);
    let mut grid = MapElites::<Rect>::new(12, MUTATION_RATE, BATCH_SIZE, SEED);
    grid.seed_population(random_rects(SEED_POP, &mut grid_seed_rng), &eval);
    for _ in 0..STEPS {
        grid.step(&eval);
    }

    // CVT-MAP-Elites: 144 centroids via Lloyd's algorithm.
    let mut cvt_seed_rng = Pcg64::seed_from_u64(SEED);
    let mut cvt = CvtMapElites::<Rect>::with_lloyd(
        144,    // num_centroids
        2,      // descriptor_dim
        10_000, // num_samples for Lloyd's
        25,     // lloyd iterations
        MUTATION_RATE,
        BATCH_SIZE,
        SEED,
    );
    cvt.seed_population(random_rects(SEED_POP, &mut cvt_seed_rng), &eval);
    for _ in 0..STEPS {
        cvt.step(&eval);
    }

    println!("Comparison after {STEPS} steps (rectangle area inside unit disk):");
    println!();
    println!("                          Grid (12x12)    CVT (144 centroids)");
    println!(
        "  Total cells / centroids       {:>4}                 {:>4}",
        12 * 12,
        cvt.num_centroids()
    );
    println!(
        "  Occupied cells                {:>4}                 {:>4}",
        grid.archive_len(),
        cvt.archive_len()
    );
    println!(
        "  Coverage                       {:>5.1}%               {:>5.1}%",
        grid.coverage() * 100.0,
        cvt.coverage() * 100.0
    );
    println!(
        "  QD score (sum of fitness)      {:>6.3}              {:>6.3}",
        grid.qd_score(),
        cvt.qd_score()
    );
    println!(
        "  Best fitness                   {:>6.3}              {:>6.3}",
        grid.best_by_fitness().map(|p| p.fitness).unwrap_or(0.0),
        cvt.best_by_fitness().map(|p| p.fitness).unwrap_or(0.0),
    );
    println!();
    println!(
        "With a uniform 2D descriptor and matched 144-cell budget, grid and CVT \
         track each other closely. CVT's advantage shows up in higher dimensions \
         (where grid is exponential in dim) and with caller-supplied centroids \
         that match a non-uniform descriptor distribution."
    );
}
