use rand::Rng;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use symbios_genetics::{Evaluator, Evolver, Genotype, algorithms::simple::SimpleGA};

#[derive(Clone, Serialize, Deserialize)]
struct HeavyDNA(Vec<f32>);
impl Genotype for HeavyDNA {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        for v in &mut self.0 {
            if rng.random_bool(rate as f64) {
                *v += 1.0;
            }
        }
    }
    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        HeavyDNA(
            self.0
                .iter()
                .zip(&other.0)
                .map(|(&a, &b)| if rng.random_bool(0.5) { a } else { b })
                .collect(),
        )
    }
}

struct HeavyEvaluator;
impl Evaluator<HeavyDNA> for HeavyEvaluator {
    fn evaluate(&self, genotype: &HeavyDNA) -> (f32, Vec<f32>, Vec<f32>) {
        // Simulate heavy physics work (1ms)
        std::thread::sleep(Duration::from_millis(1));
        let sum: f32 = genotype.0.iter().sum();
        (sum, vec![sum], vec![])
    }
}

#[test]
fn stress_test_parallel_throughput() {
    let pop_size = 1000;
    let initial_pop: Vec<HeavyDNA> = (0..pop_size).map(|_| HeavyDNA(vec![0.0; 10])).collect();
    let mut ga = SimpleGA::new(initial_pop, 0.1, 10, 42);
    let eval = HeavyEvaluator;

    println!("Starting Heavy Stress Test (1000 individuals @ 1ms sleep)...");
    let start = Instant::now();
    ga.step(&eval);
    let duration = start.elapsed();

    println!("Throughput: {} evals in {:?}", pop_size, duration);

    #[cfg(feature = "parallel")]
    {
        // Use relative timing: compare against theoretical sequential time
        // Sequential would be at least 1000 * 1ms = 1000ms
        let sequential_estimate_ms = pop_size as u64;
        let actual_ms = duration.as_millis() as u64;
        let speedup = sequential_estimate_ms as f64 / actual_ms.max(1) as f64;

        println!(
            "Sequential estimate: {}ms, Actual: {}ms, Speedup: {:.2}x",
            sequential_estimate_ms, actual_ms, speedup
        );

        // Require at least 1.5x speedup (conservative for CI environments)
        assert!(
            speedup > 1.5,
            "Parallel execution should provide speedup. Got {:.2}x (expected > 1.5x)",
            speedup
        );
    }
}
