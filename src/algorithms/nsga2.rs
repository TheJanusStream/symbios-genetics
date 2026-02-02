use crate::{Evaluator, Evolver, Genotype, Phenotype};
use rand::prelude::IndexedRandom;

pub struct Nsga2<G: Genotype> {
    population: Vec<Phenotype<G>>,
    pop_size: usize,
    mutation_rate: f32,
}

#[derive(Clone)]
struct SortWrapper<G: Genotype> {
    pheno: Phenotype<G>,
    _rank: usize,
    distance: f32,
}

impl<G: Genotype> Nsga2<G> {
    pub fn new(initial_pop: Vec<G>, mutation_rate: f32) -> Self {
        let pop_size = initial_pop.len();
        let population = initial_pop
            .into_iter()
            .map(|g| Phenotype {
                genotype: g,
                fitness: 0.0,
                objectives: vec![],
                descriptor: vec![],
            })
            .collect();

        Self {
            population,
            pop_size,
            mutation_rate,
        }
    }

    fn fast_non_dominated_sort(combined: &[Phenotype<G>]) -> Vec<Vec<usize>> {
        let n = combined.len();
        let mut fronts = vec![vec![]];
        let mut domination_count = vec![0; n];
        let mut dominated_indices = vec![vec![]; n];

        for i in 0..n {
            for j in 0..n {
                if Self::dominates(&combined[i], &combined[j]) {
                    dominated_indices[i].push(j);
                } else if Self::dominates(&combined[j], &combined[i]) {
                    domination_count[i] += 1;
                }
            }
            if domination_count[i] == 0 {
                fronts[0].push(i);
            }
        }

        let mut curr = 0;
        while !fronts[curr].is_empty() {
            let mut next_front = vec![];
            for &i in &fronts[curr] {
                for &j in &dominated_indices[i] {
                    domination_count[j] -= 1;
                    if domination_count[j] == 0 {
                        next_front.push(j);
                    }
                }
            }
            curr += 1;
            fronts.push(next_front);
        }
        fronts
    }

    fn calculate_crowding_distance(front: &mut [SortWrapper<G>]) {
        let n = front.len();
        if n == 0 {
            return;
        }
        if n <= 2 {
            for ind in front {
                ind.distance = f32::INFINITY;
            }
            return;
        }

        let obj_count = front[0].pheno.objectives.len();
        for m in 0..obj_count {
            front.sort_by(|a, b| {
                a.pheno.objectives[m]
                    .partial_cmp(&b.pheno.objectives[m])
                    .unwrap()
            });

            let min_obj = front[0].pheno.objectives[m];
            let max_obj = front[n - 1].pheno.objectives[m];
            let range = max_obj - min_obj;

            front[0].distance = f32::INFINITY;
            front[n - 1].distance = f32::INFINITY;

            if range > 0.0 {
                for i in 1..(n - 1) {
                    if front[i].distance != f32::INFINITY {
                        front[i].distance += (front[i + 1].pheno.objectives[m]
                            - front[i - 1].pheno.objectives[m])
                            / range;
                    }
                }
            }
        }
    }

    fn dominates(a: &Phenotype<G>, b: &Phenotype<G>) -> bool {
        let mut better_in_any = false;
        for (oa, ob) in a.objectives.iter().zip(b.objectives.iter()) {
            if oa < ob {
                return false;
            }
            if oa > ob {
                better_in_any = true;
            }
        }
        better_in_any
    }
}

impl<G: Genotype> Evolver<G> for Nsga2<G> {
    fn step<E: Evaluator<G>>(&mut self, evaluator: &E) {
        let mut rng = rand::rng();

        let mut offspring = vec![];
        while offspring.len() < self.pop_size {
            let p1 = self.population.choose(&mut rng).unwrap();
            let p2 = self.population.choose(&mut rng).unwrap();
            let mut child_dna = p1.genotype.crossover(&p2.genotype, &mut rng);
            child_dna.mutate(&mut rng, self.mutation_rate);
            offspring.push(Phenotype {
                genotype: child_dna,
                fitness: 0.0,
                objectives: vec![],
                descriptor: vec![],
            });
        }

        let mut combined = self.population.clone();
        combined.extend(offspring);
        for p in &mut combined {
            let (fit, obj, desc) = evaluator.evaluate(&p.genotype);
            p.fitness = fit;
            p.objectives = obj;
            p.descriptor = desc;
        }

        let fronts = Self::fast_non_dominated_sort(&combined);

        let mut next_gen_wrappers = vec![];
        for (rank, indices) in fronts.iter().enumerate() {
            if indices.is_empty() {
                continue;
            }

            let mut current_front: Vec<SortWrapper<G>> = indices
                .iter()
                .map(|&i| SortWrapper {
                    pheno: combined[i].clone(),
                    _rank: rank,
                    distance: 0.0,
                })
                .collect();

            Self::calculate_crowding_distance(&mut current_front);

            if next_gen_wrappers.len() + current_front.len() <= self.pop_size {
                next_gen_wrappers.extend(current_front);
            } else {
                current_front.sort_by(|a, b| b.distance.partial_cmp(&a.distance).unwrap());
                let needed = self.pop_size - next_gen_wrappers.len();
                next_gen_wrappers.extend(current_front.into_iter().take(needed));
                break;
            }
        }

        self.population = next_gen_wrappers.into_iter().map(|w| w.pheno).collect();
    }

    fn population(&self) -> &[Phenotype<G>] {
        &self.population
    }
}
