use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use rand::Rng;
use rand::prelude::SeedableRng;
use rand_pcg::Pcg64;
use serde::{Deserialize, Serialize};
use symbios_genetics::{
    Evaluator, Evolver, Genotype, Phenotype,
    algorithms::{
        map_elites::MapElites,
        nsga2::Nsga2,
        simple::SimpleGA,
    },
};

// =============================================================================
// Common test genotypes and evaluators
// =============================================================================

/// Bit vector genotype for single-objective benchmarks
#[derive(Clone, Serialize, Deserialize)]
struct BitVec(Vec<bool>);

impl Genotype for BitVec {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        for bit in &mut self.0 {
            if rng.random::<f32>() < rate {
                *bit = !*bit;
            }
        }
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        let point = rng.random_range(0..self.0.len());
        let mut child = self.0[..point].to_vec();
        child.extend_from_slice(&other.0[point..]);
        BitVec(child)
    }
}

/// Float vector genotype for multi-objective benchmarks
#[derive(Clone, Serialize, Deserialize)]
struct FloatVec(Vec<f32>);

impl Genotype for FloatVec {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        for val in &mut self.0 {
            if rng.random::<f32>() < rate {
                *val = (*val + rng.random::<f32>() - 0.5).clamp(0.0, 1.0);
            }
        }
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        let point = rng.random_range(0..self.0.len());
        let mut child = self.0[..point].to_vec();
        child.extend_from_slice(&other.0[point..]);
        FloatVec(child)
    }
}

/// OneMax evaluator: maximizes the number of 1s in a bit vector
struct OneMaxEval;

impl Evaluator<BitVec> for OneMaxEval {
    fn evaluate(&self, g: &BitVec) -> (f32, Vec<f32>, Vec<f32>) {
        let fitness = g.0.iter().filter(|&&b| b).count() as f32;
        (fitness, vec![fitness], vec![])
    }
}

/// ZDT1-like evaluator for multi-objective benchmarks
struct Zdt1Eval;

impl Evaluator<FloatVec> for Zdt1Eval {
    fn evaluate(&self, g: &FloatVec) -> (f32, Vec<f32>, Vec<f32>) {
        let n = g.0.len() as f32;
        let f1 = g.0[0];
        let g_val = 1.0 + 9.0 * g.0[1..].iter().sum::<f32>() / (n - 1.0).max(1.0);
        let f2 = g_val * (1.0 - (f1 / g_val).sqrt());
        // Negate for maximization
        (-f1, vec![-f1, -f2], vec![])
    }
}

/// Rastrigin-like evaluator for MAP-Elites with 2D descriptor
struct RastriginEval;

impl Evaluator<FloatVec> for RastriginEval {
    fn evaluate(&self, g: &FloatVec) -> (f32, Vec<f32>, Vec<f32>) {
        let x = g.0.get(0).copied().unwrap_or(0.5) * 10.0 - 5.0;
        let y = g.0.get(1).copied().unwrap_or(0.5) * 10.0 - 5.0;
        let fitness = -(20.0 + x * x + y * y
            - 10.0 * (x * 2.0 * std::f32::consts::PI).cos()
            - 10.0 * (y * 2.0 * std::f32::consts::PI).cos());
        // Use first two dimensions as behavioral descriptor
        let desc = vec![
            g.0.get(0).copied().unwrap_or(0.5),
            g.0.get(1).copied().unwrap_or(0.5),
        ];
        (fitness, vec![fitness], desc)
    }
}

// =============================================================================
// Helper functions
// =============================================================================

fn create_bitvec_population(size: usize, genome_len: usize, seed: u64) -> Vec<BitVec> {
    let mut rng = Pcg64::seed_from_u64(seed);
    (0..size)
        .map(|_| BitVec((0..genome_len).map(|_| rng.random_bool(0.5)).collect()))
        .collect()
}

fn create_floatvec_population(size: usize, genome_len: usize, seed: u64) -> Vec<FloatVec> {
    let mut rng = Pcg64::seed_from_u64(seed);
    (0..size)
        .map(|_| FloatVec((0..genome_len).map(|_| rng.random::<f32>()).collect()))
        .collect()
}

fn create_phenotypes_for_sorting(size: usize, num_objectives: usize, seed: u64) -> Vec<Phenotype<FloatVec>> {
    let mut rng = Pcg64::seed_from_u64(seed);
    (0..size)
        .map(|_| {
            let objectives: Vec<f32> = (0..num_objectives).map(|_| rng.random::<f32>()).collect();
            Phenotype {
                genotype: FloatVec(vec![0.0; 2]),
                fitness: objectives[0],
                objectives,
                descriptor: vec![],
            }
        })
        .collect()
}

// =============================================================================
// SimpleGA Benchmarks
// =============================================================================

fn bench_simple_ga_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("SimpleGA/step");

    for pop_size in [50, 100, 200, 500].iter() {
        group.throughput(Throughput::Elements(*pop_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(pop_size),
            pop_size,
            |b, &size| {
                let population = create_bitvec_population(size, 128, 42);
                let eval = OneMaxEval;
                b.iter_batched(
                    || SimpleGA::new(population.clone(), 0.02, 2, 42),
                    |mut ga| {
                        ga.step(&eval);
                        black_box(ga)
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_simple_ga_genome_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("SimpleGA/genome_size");

    for genome_len in [64, 128, 256, 512].iter() {
        group.throughput(Throughput::Elements(*genome_len as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(genome_len),
            genome_len,
            |b, &len| {
                let population = create_bitvec_population(100, len, 42);
                let eval = OneMaxEval;
                b.iter_batched(
                    || SimpleGA::new(population.clone(), 0.02, 2, 42),
                    |mut ga| {
                        ga.step(&eval);
                        black_box(ga)
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

// =============================================================================
// NSGA-II Benchmarks
// =============================================================================

fn bench_nsga2_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("NSGA2/step");

    for pop_size in [50, 100, 200].iter() {
        group.throughput(Throughput::Elements(*pop_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(pop_size),
            pop_size,
            |b, &size| {
                let population = create_floatvec_population(size, 10, 42);
                let eval = Zdt1Eval;
                b.iter_batched(
                    || Nsga2::new(population.clone(), 0.1, 42),
                    |mut nsga| {
                        nsga.step(&eval);
                        black_box(nsga)
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_nsga2_non_dominated_sort(c: &mut Criterion) {
    let mut group = c.benchmark_group("NSGA2/non_dominated_sort");

    for pop_size in [50, 100, 200, 500].iter() {
        group.throughput(Throughput::Elements(*pop_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(pop_size),
            pop_size,
            |b, &size| {
                let phenotypes = create_phenotypes_for_sorting(size, 2, 42);
                b.iter(|| {
                    black_box(Nsga2::<FloatVec>::fast_non_dominated_sort(&phenotypes))
                });
            },
        );
    }
    group.finish();
}

fn bench_nsga2_crowding_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("NSGA2/crowding_distance");

    for front_size in [20, 50, 100, 200].iter() {
        group.throughput(Throughput::Elements(*front_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(front_size),
            front_size,
            |b, &size| {
                let phenotypes = create_phenotypes_for_sorting(size, 3, 42);
                let mut front: Vec<_> = (0..size)
                    .map(|i| symbios_genetics::algorithms::nsga2::SortWrapper {
                        index: i,
                        rank: 0,
                        distance: 0.0,
                    })
                    .collect();
                b.iter(|| {
                    // Reset distances
                    for w in &mut front {
                        w.distance = 0.0;
                    }
                    Nsga2::<FloatVec>::calculate_crowding_distance(&mut front, &phenotypes);
                    black_box(front.len())
                });
            },
        );
    }
    group.finish();
}

fn bench_nsga2_objectives_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("NSGA2/num_objectives");

    for num_obj in [2, 3, 5, 8].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_obj),
            num_obj,
            |b, &num| {
                let phenotypes = create_phenotypes_for_sorting(100, num, 42);
                b.iter(|| {
                    black_box(Nsga2::<FloatVec>::fast_non_dominated_sort(&phenotypes))
                });
            },
        );
    }
    group.finish();
}

// =============================================================================
// MAP-Elites Benchmarks
// =============================================================================

fn bench_map_elites_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("MapElites/step");

    for resolution in [10, 20, 50].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(resolution),
            resolution,
            |b, &res| {
                let population = create_floatvec_population(200, 4, 42);
                let eval = RastriginEval;
                b.iter_batched(
                    || {
                        let mut me = MapElites::<FloatVec>::new(res, 0.3, 42);
                        me.seed_population(population.clone(), &eval);
                        me
                    },
                    |mut me| {
                        me.step(&eval);
                        black_box(me)
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_map_elites_seed(c: &mut Criterion) {
    let mut group = c.benchmark_group("MapElites/seed_population");

    for seed_size in [100, 500, 1000].iter() {
        group.throughput(Throughput::Elements(*seed_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(seed_size),
            seed_size,
            |b, &size| {
                let population = create_floatvec_population(size, 4, 42);
                let eval = RastriginEval;
                b.iter_batched(
                    || MapElites::<FloatVec>::new(20, 0.3, 42),
                    |mut me| {
                        me.seed_population(population.clone(), &eval);
                        black_box(me)
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_map_elites_map_to_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("MapElites/map_to_index");

    for dims in [2, 4, 8, 16].iter() {
        group.throughput(Throughput::Elements(*dims as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(dims),
            dims,
            |b, &d| {
                let me = MapElites::<FloatVec>::new(100, 0.3, 42);
                let mut rng = Pcg64::seed_from_u64(42);
                let descriptor: Vec<f32> = (0..d).map(|_| rng.random::<f32>()).collect();
                b.iter(|| {
                    black_box(me.map_to_index(&descriptor))
                });
            },
        );
    }
    group.finish();
}

fn bench_map_elites_batch_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("MapElites/batch_size");

    for batch in [32, 64, 128, 256].iter() {
        group.throughput(Throughput::Elements(*batch as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch),
            batch,
            |b, &batch_size| {
                let population = create_floatvec_population(200, 4, 42);
                let eval = RastriginEval;
                b.iter_batched(
                    || {
                        let mut me = MapElites::<FloatVec>::new(20, 0.3, 42);
                        me.set_batch_size(batch_size);
                        me.seed_population(population.clone(), &eval);
                        me
                    },
                    |mut me| {
                        me.step(&eval);
                        black_box(me)
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

// =============================================================================
// Criterion setup
// =============================================================================

criterion_group!(
    simple_ga_benches,
    bench_simple_ga_step,
    bench_simple_ga_genome_scaling,
);

criterion_group!(
    nsga2_benches,
    bench_nsga2_step,
    bench_nsga2_non_dominated_sort,
    bench_nsga2_crowding_distance,
    bench_nsga2_objectives_scaling,
);

criterion_group!(
    map_elites_benches,
    bench_map_elites_step,
    bench_map_elites_seed,
    bench_map_elites_map_to_index,
    bench_map_elites_batch_scaling,
);

criterion_main!(simple_ga_benches, nsga2_benches, map_elites_benches);
