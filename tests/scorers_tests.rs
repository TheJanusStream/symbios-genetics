//! Tests for the scorers module: concrete scorers, combinators, and the
//! CompositeEvaluator bridge into Evaluator<G>.

use rand::Rng;
use serde::{Deserialize, Serialize};
use symbios_genetics::scorers::{
    CompositeEvaluator, Const, Displacement, EnergyEfficiency, Height, Multiply, Normalize,
    NumLike, Penalize, Scorer, Sum, Trajectory, UpAlignment,
};
use symbios_genetics::{Evaluator, Genotype};

#[derive(Clone, Serialize, Deserialize, Debug)]
struct Robot {
    forward: f32,
    lift: f32,
    energy: f32,
}

impl Genotype for Robot {
    fn mutate<R: Rng>(&mut self, _rng: &mut R, _rate: f32) {}
    fn crossover<R: Rng>(&self, _other: &Self, _rng: &mut R) -> Self {
        self.clone()
    }
}

/// Sim turning a Robot into a Trajectory. Forward → end.x, lift → end.y +
/// max_height, energy → energy_used. Up-vector points straight up so
/// UpAlignment = 1.0.
fn run_sim(robot: &Robot) -> Trajectory {
    Trajectory {
        start: [0.0, 0.0, 0.0],
        end: [robot.forward, robot.lift, 0.0],
        up: [0.0, 1.0, 0.0],
        max_height: robot.lift,
        energy_used: robot.energy,
        final_descriptor: vec![robot.forward / 10.0, robot.lift / 10.0],
    }
}

#[test]
fn displacement_is_euclidean() {
    let t = Trajectory {
        start: [0.0, 0.0, 0.0],
        end: [3.0, 4.0, 0.0],
        ..Default::default()
    };
    assert!((Displacement.score(&t) - 5.0).abs() < 1e-6);
}

#[test]
fn up_alignment_clamps_to_unit_interval() {
    let mut t = Trajectory {
        up: [0.0, 0.7, 0.0],
        ..Default::default()
    };
    assert!((UpAlignment.score(&t) - 0.7).abs() < 1e-6);
    t.up = [0.0, -0.5, 0.0];
    assert_eq!(UpAlignment.score(&t), 0.0);
    t.up = [0.0, 1.5, 0.0]; // unphysical but defensive
    assert_eq!(UpAlignment.score(&t), 1.0);
}

#[test]
fn height_returns_max_height() {
    let t = Trajectory {
        max_height: 7.5,
        ..Default::default()
    };
    assert_eq!(Height.score(&t), 7.5);
}

#[test]
fn energy_efficiency_is_displacement_per_energy() {
    let t = Trajectory {
        start: [0.0, 0.0, 0.0],
        end: [10.0, 0.0, 0.0],
        energy_used: 5.0,
        ..Default::default()
    };
    assert_eq!(EnergyEfficiency.score(&t), 2.0);

    // Zero energy returns 0 instead of dividing.
    let t_zero = Trajectory {
        start: [0.0, 0.0, 0.0],
        end: [10.0, 0.0, 0.0],
        energy_used: 0.0,
        ..Default::default()
    };
    assert_eq!(EnergyEfficiency.score(&t_zero), 0.0);
}

/// The headline composition from the issue:
///   locomotion = displacement * up_alignment * (height + 0.5)
#[test]
fn locomotion_composition_matches_issue_formula() {
    let locomotion = Multiply(
        Multiply(Displacement, UpAlignment),
        Sum(Height, Const(0.5_f32)),
    );

    // displacement = 10, up_alignment = 1, height = 1.5 → 10 * 1 * (1.5 + 0.5) = 20
    let t = Trajectory {
        start: [0.0, 0.0, 0.0],
        end: [10.0, 0.0, 0.0],
        up: [0.0, 1.0, 0.0],
        max_height: 1.5,
        energy_used: 4.0,
        final_descriptor: vec![],
    };
    assert!((locomotion.score(&t) - 20.0).abs() < 1e-5);
}

#[test]
fn penalize_subtracts_penalty_term() {
    let scorer = Penalize {
        base: Displacement,
        penalty: Multiply(Const(0.1_f32), Height),
    };
    // displacement = 5, height = 10 → 5 - 0.1*10 = 4
    let t = Trajectory {
        start: [0.0, 0.0, 0.0],
        end: [5.0, 0.0, 0.0],
        max_height: 10.0,
        ..Default::default()
    };
    assert!((scorer.score(&t) - 4.0).abs() < 1e-5);
}

#[test]
fn normalize_scales_and_clamps() {
    let scorer = Normalize {
        inner: Displacement,
        max: 10.0_f32,
    };
    let t = Trajectory {
        start: [0.0, 0.0, 0.0],
        end: [5.0, 0.0, 0.0],
        ..Default::default()
    };
    assert!((scorer.score(&t) - 0.5).abs() < 1e-5);

    // Above max → clamped to 1.
    let t_far = Trajectory {
        start: [0.0, 0.0, 0.0],
        end: [50.0, 0.0, 0.0],
        ..Default::default()
    };
    assert_eq!(scorer.score(&t_far), 1.0);
}

#[test]
#[should_panic(expected = "Normalize::max must be > 0")]
fn normalize_panics_on_zero_max() {
    let scorer = Normalize {
        inner: Displacement,
        max: 0.0_f32,
    };
    let t = Trajectory::default();
    let _ = scorer.score(&t);
}

#[test]
fn composite_evaluator_runs_sim_once_and_distributes_results() {
    // Multi-objective: scalar fitness = locomotion;
    // objectives for NSGA-II: [Displacement, EnergyEfficiency];
    // descriptor for MAP-Elites: [Height-normalized, UpAlignment]
    let fitness = Box::new(Multiply(Displacement, UpAlignment));
    let objectives: Vec<Box<dyn Scorer<Trajectory, f32> + Send + Sync>> =
        vec![Box::new(Displacement), Box::new(EnergyEfficiency)];
    let descriptors: Vec<Box<dyn Scorer<Trajectory, f32> + Send + Sync>> = vec![
        Box::new(Normalize {
            inner: Height,
            max: 10.0_f32,
        }),
        Box::new(UpAlignment),
    ];

    let eval = CompositeEvaluator::<Robot, Trajectory, _>::new(run_sim, fitness)
        .with_objectives(objectives)
        .with_descriptors(descriptors);

    let robot = Robot {
        forward: 8.0,
        lift: 2.5,
        energy: 4.0,
    };
    let (fit, obj, desc) = eval.evaluate(&robot);

    // Sim: end = (8, 2.5, 0), up = (0, 1, 0), height = 2.5, energy = 4
    // displacement = sqrt(64 + 6.25) = sqrt(70.25) ≈ 8.381
    let displacement = (64.0_f32 + 6.25).sqrt();
    assert!((fit - displacement * 1.0).abs() < 1e-4);
    assert_eq!(obj.len(), 2);
    assert!((obj[0] - displacement).abs() < 1e-4);
    assert!((obj[1] - displacement / 4.0).abs() < 1e-4);
    assert_eq!(desc.len(), 2);
    assert!((desc[0] - 0.25).abs() < 1e-5); // 2.5 / 10 = 0.25
    assert_eq!(desc[1], 1.0);
}

#[test]
fn numlike_works_with_f64_and_f32() {
    let scorer_f32 = Sum(Const(1.0_f32), Const(2.0_f32));
    let scorer_f64 = Sum(Const(1.0_f64), Const(2.0_f64));
    let s = Trajectory::default();
    assert!((Scorer::<Trajectory, f32>::score(&scorer_f32, &s) - 3.0).abs() < 1e-6);
    assert!((Scorer::<Trajectory, f64>::score(&scorer_f64, &s) - 3.0).abs() < 1e-9);
}

#[test]
fn numlike_clamp_to_handles_bounds() {
    assert_eq!((-0.5_f32).clamp_to(0.0, 1.0), 0.0);
    assert_eq!((1.5_f32).clamp_to(0.0, 1.0), 1.0);
    assert_eq!((0.5_f32).clamp_to(0.0, 1.0), 0.5);
}
