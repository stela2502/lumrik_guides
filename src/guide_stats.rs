// src/guide_stants.rs

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;


/// Cell-level information needed for posterior-gap statistics.
///
/// The posterior gap itself is calculated upstream when the guides
/// are ranked for the cell. This module only summarizes those values.
#[derive(Debug, Clone)]
pub struct CellGuideGap {
    pub n_called_guides: usize,
    pub posterior_gap: f64,
}


#[derive(Debug, Clone)]
pub struct MultiGuideGapStats {
    n_called_guides: Option<usize>,
    n_cells: usize,

    mean: f64,

    min: f64,
    q1: f64,
    median: f64,
    q3: f64,
    p90: f64,
    p95: f64,
    max: f64,
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

        let mut sorted = values.to_vec();

        sorted.sort_unstable_by(|a, b| {
            a.total_cmp(b)
        });

        let n_cells = sorted.len();

        let mean =
            sorted.iter().sum::<f64>()
                / n_cells as f64;

        Self {
            n_called_guides,
            n_cells,

            mean,

            min: sorted[0],
            q1: quantile(&sorted, 0.25),
            median: quantile(&sorted, 0.50),
            q3: quantile(&sorted, 0.75),
            p90: quantile(&sorted, 0.90),
            p95: quantile(&sorted, 0.95),
            max: *sorted
                .last()
                .expect("sorted values unexpectedly empty"),
        }
    }


    /// Build statistics from posterior gaps that have already been
    /// calculated at the cell level.
    ///
    /// Only cells with >= 2 called guides contribute.
    pub fn collect(
        gaps: &[(usize, f64)],
    ) -> Vec<Self> {
        let mut by_multiplicity:
            BTreeMap<usize, Vec<f64>> =
            BTreeMap::new();

        let mut all_multi = Vec::new();

        for gap in gaps {
            if gap.0 < 2 {
                continue;
            }

            by_multiplicity
                .entry(gap.0)
                .or_default()
                .push(gap.1);

            all_multi.push(
                gap.1
            );
        }

        let mut result =
            Vec::with_capacity(
                by_multiplicity.len() + 1
            );

        for (n_called_guides, values)
            in by_multiplicity
        {
            result.push(
                Self::new(
                    Some(n_called_guides),
                    &values,
                )
            );
        }

        if !all_multi.is_empty() {
            result.push(
                Self::new(
                    None,
                    &all_multi,
                )
            );
        }

        result
    }


    pub fn n_called_guides(
        &self,
    ) -> Option<usize> {
        self.n_called_guides
    }


    pub fn n_cells(
        &self,
    ) -> usize {
        self.n_cells
    }


    pub fn mean(
        &self,
    ) -> f64 {
        self.mean
    }


    pub fn min(
        &self,
    ) -> f64 {
        self.min
    }


    pub fn q1(
        &self,
    ) -> f64 {
        self.q1
    }


    pub fn median(
        &self,
    ) -> f64 {
        self.median
    }


    pub fn q3(
        &self,
    ) -> f64 {
        self.q3
    }


    pub fn p90(
        &self,
    ) -> f64 {
        self.p90
    }


    pub fn p95(
        &self,
    ) -> f64 {
        self.p95
    }


    pub fn max(
        &self,
    ) -> f64 {
        self.max
    }


    pub fn group_name(
        &self,
    ) -> String {
        self.n_called_guides
            .map(|n| n.to_string())
            .unwrap_or_else(|| "ALL".to_string())
    }


    pub fn header() -> &'static str {
        concat!(
            "called_guides",
            "\tcells",
            "\tmean",
            "\tmin",
            "\tq1",
            "\tmedian",
            "\tq3",
            "\tp90",
            "\tp95",
            "\tmax"
        )
    }
}


impl Display for MultiGuideGapStats {
    fn fmt(
        &self,
        f: &mut Formatter<'_>,
    ) -> FmtResult {
        write!(
            f,
            concat!(
                "{}\t{}",
                "\t{:.8}",
                "\t{:.8}",
                "\t{:.8}",
                "\t{:.8}",
                "\t{:.8}",
                "\t{:.8}",
                "\t{:.8}",
                "\t{:.8}"
            ),
            self.group_name(),
            self.n_cells,
            self.mean,
            self.min,
            self.q1,
            self.median,
            self.q3,
            self.p90,
            self.p95,
            self.max,
        )
    }
}


pub trait MultiGuideGapStatsTable {
    fn print_table(
        &self,
    );

    fn write_table(
        &self,
        out: &PathBuf,
    ) -> Result<()>;
}


impl MultiGuideGapStatsTable for [MultiGuideGapStats] {
    fn print_table(
        &self,
    ) {
        println!(
            "Multi-guide posterior-gap statistics"
        );

        println!(
            "{}",
            MultiGuideGapStats::header()
        );

        for stat in self {
            println!("{stat}");
        }
    }


    fn write_table(
        &self,
        out: &PathBuf,
    ) -> Result<()> {
        let path = out.join(
            "multi_guide_posterior_gap_stats.tsv"
        );

        let mut writer = BufWriter::new(
            File::create(&path)
                .with_context(|| {
                    format!(
                        "creating {}",
                        path.display()
                    )
                })?,
        );

        writeln!(
            writer,
            "{}",
            MultiGuideGapStats::header()
        )?;

        for stat in self {
            writeln!(
                writer,
                "{stat}"
            )?;
        }

        writer
            .flush()
            .with_context(|| {
                format!(
                    "flushing {}",
                    path.display()
                )
            })?;

        Ok(())
    }
}


fn quantile(
    values: &[f64],
    q: f64,
) -> f64 {
    assert!(
        !values.is_empty(),
        "cannot calculate quantile of empty data"
    );

    assert!(
        (0.0..=1.0).contains(&q),
        "quantile must be between 0 and 1"
    );

    if values.len() == 1 {
        return values[0];
    }

    let position =
        q * (values.len() - 1) as f64;

    let lower =
        position.floor() as usize;

    let upper =
        position.ceil() as usize;

    if lower == upper {
        return values[lower];
    }

    let fraction =
        position - lower as f64;

    values[lower]
        + fraction
            * (
                values[upper]
                    - values[lower]
            )
}