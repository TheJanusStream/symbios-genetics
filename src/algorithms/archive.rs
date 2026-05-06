//! Internal cell-keyed archive shared by MAP-Elites variants.
//!
//! The archive stores at most one elite per cell key, using strict-better
//! fitness for replacement. Cells are keyed by `Vec<usize>` regardless of
//! how the key is computed (grid bin indices for [`super::map_elites`],
//! centroid IDs for the CVT variant).

use crate::{Genotype, Phenotype};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Cell-keyed elite archive.
///
/// `cells` holds one phenotype per occupied key; `keys_vec` mirrors the
/// occupied keys for O(1) random parent sampling. The two collections are
/// kept in sync by [`Archive::insert_if_better`].
#[derive(Serialize, Deserialize)]
#[serde(bound = "G: Genotype")]
pub(crate) struct Archive<G: Genotype> {
    cells: BTreeMap<Vec<usize>, Phenotype<G>>,
    keys_vec: Vec<Vec<usize>>,

    #[serde(skip)]
    population_cache: Vec<Phenotype<G>>,
    #[serde(skip, default = "default_cache_valid_after_load")]
    cache_valid: bool,
}

fn default_cache_valid_after_load() -> bool {
    false
}

impl<G: Genotype> Default for Archive<G> {
    fn default() -> Self {
        Self::new()
    }
}

impl<G: Genotype> Archive<G> {
    pub(crate) fn new() -> Self {
        Self {
            cells: BTreeMap::new(),
            keys_vec: Vec::new(),
            population_cache: Vec::new(),
            cache_valid: true,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.cells.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    pub(crate) fn get(&self, key: &[usize]) -> Option<&Phenotype<G>> {
        self.cells.get(key)
    }

    pub(crate) fn keys(&self) -> impl Iterator<Item = &Vec<usize>> {
        self.cells.keys()
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&Vec<usize>, &Phenotype<G>)> {
        self.cells.iter()
    }

    /// O(1) random parent key. Returns `None` if the archive is empty.
    pub(crate) fn sample_key<R: Rng>(&self, rng: &mut R) -> Option<&Vec<usize>> {
        if self.keys_vec.is_empty() {
            None
        } else {
            let idx = rng.random_range(0..self.keys_vec.len());
            Some(&self.keys_vec[idx])
        }
    }

    /// Insert `candidate` at `key`, replacing the existing elite when:
    ///
    /// - the cell is empty, OR
    /// - `candidate.fitness > existing.fitness`, OR
    /// - the existing elite has NaN fitness (corrupted-state recovery).
    ///
    /// Candidates with NaN fitness or NaN descriptor entries are silently
    /// rejected — they cannot meaningfully compete and would corrupt the
    /// archive if inserted.
    pub(crate) fn insert_if_better(&mut self, key: Vec<usize>, candidate: Phenotype<G>) {
        if candidate.fitness.is_nan() || candidate.descriptor.iter().any(|v| v.is_nan()) {
            return;
        }

        let should_insert = self.cells.get(&key).is_none_or(|existing| {
            candidate.fitness > existing.fitness || existing.fitness.is_nan()
        });
        if !should_insert {
            return;
        }

        let is_new_key = !self.cells.contains_key(&key);
        if is_new_key {
            self.keys_vec.push(key.clone());
        }
        self.cells.insert(key, candidate);
        self.cache_valid = false;
    }

    pub(crate) fn best_by_fitness(&self) -> Option<&Phenotype<G>> {
        self.cells.values().max_by(|a, b| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Sum of fitness across occupied cells, skipping NaN entries.
    pub(crate) fn qd_score(&self) -> f64 {
        self.cells
            .values()
            .filter_map(|p| {
                if p.fitness.is_nan() {
                    None
                } else {
                    Some(p.fitness as f64)
                }
            })
            .sum()
    }

    /// Returns the cached population view, rebuilding it from cells if stale.
    pub(crate) fn population(&mut self) -> &[Phenotype<G>] {
        if !self.cache_valid {
            self.population_cache = self.cells.values().cloned().collect();
            self.cache_valid = true;
        }
        &self.population_cache
    }

    /// Validate that `keys_vec` is in sync with `cells`. Called after
    /// deserialization to reject malicious or corrupted state.
    pub(crate) fn validate(&self) -> Result<(), &'static str> {
        for key in &self.keys_vec {
            if !self.cells.contains_key(key) {
                return Err("archive_keys_vec contains key not present in archive");
            }
        }
        if self.cells.len() != self.keys_vec.len() {
            return Err("archive_keys_vec length does not match archive size");
        }
        Ok(())
    }

    /// Reconstruct from already-validated raw fields. Used by container
    /// types whose Deserialize impl validates upstream.
    pub(crate) fn from_raw(
        cells: BTreeMap<Vec<usize>, Phenotype<G>>,
        keys_vec: Vec<Vec<usize>>,
    ) -> Self {
        let population_cache: Vec<Phenotype<G>> = cells.values().cloned().collect();
        Self {
            cells,
            keys_vec,
            population_cache,
            cache_valid: true,
        }
    }

    /// Borrow the raw cells map for serialization.
    pub(crate) fn cells(&self) -> &BTreeMap<Vec<usize>, Phenotype<G>> {
        &self.cells
    }

    /// Borrow the raw keys vec for serialization.
    pub(crate) fn keys_vec(&self) -> &Vec<Vec<usize>> {
        &self.keys_vec
    }
}

#[cfg(feature = "export")]
impl<G: Genotype> Archive<G> {
    /// Write a CSV row per occupied cell. See [`super::map_elites::MapElites::export_csv`].
    pub(crate) fn export_csv<W: std::io::Write>(&self, mut writer: W) -> std::io::Result<()> {
        use std::fmt::Write as _;

        writeln!(writer, "key,descriptor,fitness,objectives,genotype_hash")?;

        let mut buf = String::new();
        for (key, pheno) in &self.cells {
            buf.clear();

            for (i, b) in key.iter().enumerate() {
                if i > 0 {
                    buf.push(';');
                }
                write!(buf, "{}", b).expect("write to String is infallible");
            }
            buf.push(',');

            for (i, v) in pheno.descriptor.iter().enumerate() {
                if i > 0 {
                    buf.push(';');
                }
                write!(buf, "{}", v).expect("write to String is infallible");
            }
            buf.push(',');

            write!(buf, "{}", pheno.fitness).expect("write to String is infallible");
            buf.push(',');

            for (i, v) in pheno.objectives.iter().enumerate() {
                if i > 0 {
                    buf.push(';');
                }
                write!(buf, "{}", v).expect("write to String is infallible");
            }
            buf.push(',');

            let bytes = bincode::serialize(&pheno.genotype).map_err(std::io::Error::other)?;
            let hash = seahash::hash(&bytes);
            write!(buf, "{:016x}", hash).expect("write to String is infallible");

            writeln!(writer, "{}", buf)?;
        }
        Ok(())
    }
}
