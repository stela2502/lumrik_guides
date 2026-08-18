use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::caller::{GuideCall, GuideCalls};
use crate::dataset::GuideDataset;
use crate::tenx::GuideFeatureIndex;

#[derive(Debug, Clone)]
pub struct GuideEvidence {
    pub guide_id: u32,
    pub guide_name: String,
    pub umi_count: u32,
    pub posterior: f64,
    pub log_odds: f64,
    pub q_value: f64,
    pub called: bool,
}

impl GuideEvidence {
    fn missing(
        guide_id: u32,
        guide_name: String,
    ) -> Self {
        Self {
            guide_id,
            guide_name,
            umi_count: 0,
            posterior: 0.0,
            log_odds: f64::NEG_INFINITY,
            q_value: 1.0,
            called: false,
        }
    }

    fn from_call(
        guide_name: String,
        call: &GuideCall,
    ) -> Self {
        Self {
            guide_id: call.guide_id,
            guide_name,
            umi_count: call.count,
            posterior: call.posterior.probability,
            log_odds: call.posterior.log_odds,
            q_value: call.q_value,
            called: call.called,
        }
    }
}


#[derive(Debug, Clone)]
pub struct CellGuideAssignment {
    pub cell_id: u64,
    pub barcode: String,

    pub n_called_guides: usize,
    pub assignment: &'static str,
    pub called_guides: Vec<String>,

    pub best_guide: String,
    pub best_log_odds: f64,

    pub second_guide: String,
    pub second_log_odds: f64,

    pub log_odds_gap: f64,

    pub guides: Vec<GuideEvidence>,
}

impl CellGuideAssignment {
    pub fn is_multi(&self) -> bool {
        self.n_called_guides >= 2
    }
}


#[derive(Debug, Clone)]
pub struct CellGuideAssignments {
    pub rows: Vec<CellGuideAssignment>,
    guide_names: Vec<String>,
}

impl CellGuideAssignments {
    pub fn new(
        index: &GuideFeatureIndex,
        data: &GuideDataset,
        calls: &GuideCalls,
    ) -> Self {
        let call_lookup: HashMap<(u64, u32), &GuideCall> = calls
            .flat
            .iter()
            .map(|call| ((call.cell_id, call.guide_id), call))
            .collect();

        let guide_names: Vec<String> = index
            .guides()
            .iter()
            .map(|guide| guide.name.clone())
            .collect();

        let rows = data
            .cell_ids
            .iter()
            .copied()
            .map(|cell_id| {
                Self::build_row(
                    cell_id,
                    &guide_names,
                    data,
                    &call_lookup,
                )
            })
            .collect();

        Self {
            rows,
            guide_names,
        }
    }

    fn build_row(
        cell_id: u64,
        guide_names: &[String],
        data: &GuideDataset,
        call_lookup: &HashMap<(u64, u32), &GuideCall>,
    ) -> CellGuideAssignment {
        let barcode = data
            .barcode_by_id
            .get(&cell_id)
            .cloned()
            .unwrap_or_else(|| "UNKNOWN".to_string());

        let guides: Vec<GuideEvidence> = guide_names
            .iter()
            .enumerate()
            .map(|(guide_id, guide_name)| {
                match call_lookup.get(&(cell_id, guide_id as u32)) {
                    Some(call) => {
                        GuideEvidence::from_call(
                            guide_name.clone(),
                            call,
                        )
                    }
                    None => {
                        GuideEvidence::missing(
                            guide_id as u32,
                            guide_name.clone(),
                        )
                    }
                }
            })
            .collect();

        let called_guides: Vec<String> = guides
            .iter()
            .filter(|guide| guide.called)
            .map(|guide| guide.guide_name.clone())
            .collect();

        let n_called_guides = called_guides.len();

        let assignment = match n_called_guides {
            0 => "none",
            1 => "single",
            _ => "multi",
        };

        let mut ranked: Vec<&GuideEvidence> = guides
            .iter()
            .filter(|guide| guide.umi_count > 0)
            .collect();

        ranked.sort_unstable_by(|a, b| {
            b.log_odds.total_cmp(&a.log_odds)
        });

        let best = ranked.first().copied();
        let second = ranked.get(1).copied();

        let best_guide = best
            .map(|guide| guide.guide_name.clone())
            .unwrap_or_default();

        let best_log_odds = best
            .map(|guide| guide.log_odds)
            .unwrap_or(f64::NEG_INFINITY);

        let second_guide = second
            .map(|guide| guide.guide_name.clone())
            .unwrap_or_default();

        let second_log_odds = second
            .map(|guide| guide.log_odds)
            .unwrap_or(f64::NEG_INFINITY);

        let log_odds_gap = match (best, second) {
            (Some(best), Some(second)) => {
                best.log_odds - second.log_odds
            }
            _ => f64::INFINITY,
        };

        CellGuideAssignment {
            cell_id,
            barcode,
            n_called_guides,
            assignment,
            called_guides,
            best_guide,
            best_log_odds,
            second_guide,
            second_log_odds,
            log_odds_gap,
            guides,
        }
    }

    pub fn multi_guide_gaps(
        &self,
    ) -> impl Iterator<Item = (usize, f64)> + '_ {
        self.rows
            .iter()
            .filter(|row| row.is_multi())
            .map(|row| {
                (
                    row.n_called_guides,
                    row.log_odds_gap,
                )
            })
    }

    fn write_tsv<W: Write>(
        &self,
        writer: &mut W,
    ) -> Result<()> {
        write!(
            writer,
            concat!(
                "barcode",
                "\tn_called_guides",
                "\tassignment",
                "\tcalled_guides",
                "\tbest_guide",
                "\tbest_log_odds",
                "\tsecond_guide",
                "\tsecond_log_odds",
                "\tlog_odds_gap"
            )
        )?;

        for guide_name in &self.guide_names {
            write!(
                writer,
                "\t{}_umi\t{}_posterior\t{}_log_odds\t{}_qvalue\t{}_called",
                guide_name,
                guide_name,
                guide_name,
                guide_name,
                guide_name,
            )?;
        }

        writeln!(writer)?;

        for row in &self.rows {
            write!(
                writer,
                "{}\t{}\t{}\t{}\t{}\t{:.8}\t{}\t{:.8}\t{:.8}",
                row.barcode,
                row.n_called_guides,
                row.assignment,
                row.called_guides.join(";"),
                row.best_guide,
                row.best_log_odds,
                row.second_guide,
                row.second_log_odds,
                row.log_odds_gap,
            )?;

            for guide in &row.guides {
                write!(
                    writer,
                    "\t{}\t{:.8}\t{:.8}\t{:.8e}\t{}",
                    guide.umi_count,
                    guide.posterior,
                    guide.log_odds,
                    guide.q_value,
                    guide.called,
                )?;
            }

            writeln!(writer)?;
        }

        Ok(())
    }

    pub fn print_table(
        &self,
    ) -> Result<()> {
        let stdout = std::io::stdout();
        let mut writer = stdout.lock();
        self.write_tsv(&mut writer)
    }

    pub fn write_table(
        &self,
        out: &PathBuf,
    ) -> Result<()> {
        let path = out.join("cell_guide_assignments.tsv");
        let file = File::create(&path)
            .with_context(|| format!("creating {}", path.display()))?;
        let mut writer = BufWriter::new(file);

        self.write_tsv(&mut writer)?;
        writer
            .flush()
            .with_context(|| format!("flushing {}", path.display()))?;

        Ok(())
    }
}
