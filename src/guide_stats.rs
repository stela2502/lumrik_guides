//guide_stats.rs

// src/guide_stants.rs

use std::collections::{BTreeMap, HashMap};
use std::fmt::{Display, Formatter, Result as FmtResult};

use crate::{
    GuideDataset,
    GuideFeatureIndex,
};
use crate::caller::{
    GuideCall,
    GuideCalls,
};

use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;


pub trait MultiGuideGapStatsTable {
    fn write_table(
        &self,
        out: &PathBuf,
    ) -> Result<()>;

    fn print_table(&self);
}

#[derive(Debug, Clone)]
pub struct MultiGuideGapStats {
    /// Exact number of called guides.
    ///
    /// None represents the combined statistics across all multi-guide cells.
    n_called_guides: Option<usize>,

    n_cells: usize,

    mean: f64,

    /// Sample standard deviation.
    ///
    /// None if only one cell is present in the group.
    std_dev: Option<f64>,
}

impl MultiGuideGapStats {
    pub fn new(
        n_called_guides: Option<usize>,
        values: &[f64],
    ) -> Self {
        assert!(
            !values.is_empty(),
            "cannot calculate statistics from an empty value set"
        );

        let n_cells = values.len();

        let mean =
            values.iter().sum::<f64>()
                / n_cells as f64;

        let std_dev = if n_cells > 1 {
            let variance = values
                .iter()
                .map(|value| {
                    let diff = *value - mean;
                    diff * diff
                })
                .sum::<f64>()
                / (n_cells - 1) as f64;

            Some(variance.sqrt())
        } else {
            None
        };

        Self {
            n_called_guides,
            n_cells,
            mean,
            std_dev,
        }
    }

    pub fn n_called_guides(&self) -> Option<usize> {
        self.n_called_guides
    }

    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }

    pub fn std_dev(&self) -> Option<f64> {
        self.std_dev
    }

    pub fn group_name(&self) -> String {
        self.n_called_guides
            .map(|n| n.to_string())
            .unwrap_or_else(|| "ALL".to_string())
    }

    /// Collect posterior-gap statistics for all cells with >= 2 called guides.
    ///
    /// The returned vector contains:
    ///
    ///  one entry for every exact guide multiplicit
    ///  followed by one combined ALL entry.
    pub fn collect(
        index: &GuideFeatureIndex,
        data: &GuideDataset,
        calls: &GuideCalls,
    ) -> Vec<Self> {
        let call_lookup: HashMap<(u64, u32), &GuideCall> = calls
            .flat
            .iter()
            .map(|call| {
                (
                    (call.cell_id, call.guide_id),
                    call,
                )
            })
            .collect();

        let mut by_multiplicity:
            BTreeMap<usize, Vec<f64>> =
            BTreeMap::new();

        let mut all_multi = Vec::new();

        for &cell_id in &data.cell_ids {
            let n_called =
                calls.called_for_cell(cell_id).count();

            if n_called < 2 {
                continue;
            }

            let gap = posterior_gap_for_cell(
                cell_id,
                index,
                &call_lookup,
            );

            by_multiplicity
                .entry(n_called)
                .or_default()
                .push(gap);

            all_multi.push(gap);
        }

        let mut result = Vec::new();

        for (n_called_guides, values)
            in by_multiplicity
        {
            result.push(Self::new(
                Some(n_called_guides),
                &values,
            ));
        }

        if !all_multi.is_empty() {
            result.push(Self::new(
                None,
                &all_multi,
            ));
        }

        result
    }

    /// Header matching Display output.
    pub fn header() -> &'static str {
        "called_guides\tcells\tmean_posterior_gap\tstd_posterior_gap"
    }
}

impl MultiGuideGapStatsTable for [MultiGuideGapStats] {
    fn write_table(
        &self,
        out: &PathBuf,
    ) -> Result<()> {
        let path =
            out.join("multi_guide_posterior_gap_stats.tsv");

        let mut writer = BufWriter::new(
            File::create(&path)
                .with_context(|| {
                    format!("creating {}", path.display())
                })?,
        );

        writeln!(
            writer,
            "{}",
            MultiGuideGapStats::header()
        )?;

        for stat in self {
            writeln!(writer, "{stat}")?;
        }

        writer
            .flush()
            .with_context(|| {
                format!("flushing {}", path.display())
            })?;

        Ok(())
    }

    fn print_table(&self) {
        println!("{}", MultiGuideGapStats::header());

        for stat in self {
            println!("{stat}");
        }
    }
}

impl Display for MultiGuideGapStats {
    fn fmt(
        &self,
        f: &mut Formatter<'_>,
    ) -> FmtResult {
        match self.std_dev {
            Some(std_dev) => {
                write!(
                    f,
                    "{}\t{}\t{:.8}\t{:.8}",
                    self.group_name(),
                    self.n_cells,
                    self.mean,
                    std_dev,
                )
            }

            None => {
                write!(
                    f,
                    "{}\t{}\t{:.8}\tNA",
                    self.group_name(),
                    self.n_cells,
                    self.mean,
                )
            }
        }
    }
}


fn posterior_gap_for_cell(
    cell_id: u64,
    index: &GuideFeatureIndex,
    call_lookup: &HashMap<(u64, u32), &GuideCall>,
) -> f64 {
    let mut best = 0.0_f64;
    let mut second = 0.0_f64;

    for guide_id in 0..index.guides().len() {
        let posterior = call_lookup
            .get(&(cell_id, guide_id as u32))
            .map(|call| call.posterior_real)
            .unwrap_or(0.0);

        if posterior > best {
            second = best;
            best = posterior;
        } else if posterior > second {
            second = posterior;
        }
    }

    best - second
}
