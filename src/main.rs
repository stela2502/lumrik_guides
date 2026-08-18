use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::collections::HashMap;

use anyhow::{Context, Result};
use clap::Parser;
use lumrik_guides::background::{AmbientModel};
use lumrik_guides::caller::{GuideCalls, GuideCall};
use lumrik_guides::model::{fit_mixture};
use lumrik_guides::tenx::TenxGuideInput;
use lumrik_guides::cli::GuideModelCli;
use lumrik_guides::{
    GuideDataset,
    GuideFeatureIndex,
    MultiGuideGapStats,
    MultiGuideGapStatsTable,
};

use mapping_info::MappingInfo;

#[derive(Debug, Parser)]
#[command(
    name = "lumrik-guides",
    about = "Ambient-aware multi-guide caller for 10x CRISPR feature-barcode matrices"
)]
struct Cli {
    /// 10x raw_feature_bc_matrix directory.
    #[arg(long)]
    raw: PathBuf,

    /// 10x filtered_feature_bc_matrix directory.
    #[arg(long)]
    filtered: PathBuf,

    /// Output directory.
    #[arg(long)]
    out: PathBuf,

    /// 10x feature type containing the guide counts.
    #[arg(long, default_value = "CRISPR Guide Capture")]
    feature_type: String,

    /// Number of worker threads.
    /// If omitted, Rayon uses the available CPU count.
    #[arg(long)]
    pub threads: Option<usize>,

    #[command(flatten)]
    model: GuideModelCli,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    fs::create_dir_all(&cli.out)
        .with_context(|| format!("creating {}", cli.out.display()))?;

    /*
     * Use the same thread count for Rayon-based model fitting.
     */
    if let Some(threads) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads.max(1))
            .build_global()
            .context("failed to initialize Rayon thread pool")?;
    }

    /*
     * Build the 10x input.
     */
    let mut input = TenxGuideInput::new(
        cli.raw,
        cli.filtered,
    );

    input.feature_type = cli.feature_type;
    input.threads = cli
        .threads
        .unwrap_or_else(rayon::current_num_threads);
        
    let (raw_index, filtered_index) =
        input.indexes()?;

    eprintln!(
        "Found {} guide features; loading raw non-cell droplets...",
        raw_index.guides().len()
    );

    /*
     * ------------------------------------------------------------
     * Ambient/background model
     * ------------------------------------------------------------
     */
    let background_data =
        input.load_background(&raw_index)?;

    let ambient = AmbientModel::fit(
        &background_data,
        &cli.model.background_config(),
    )?;

    write_ambient(
        &cli.out,
        &raw_index,
        &ambient,
    )?;

    eprintln!(
        "Ambient model fitted from {} droplets / {} guide UMIs; loading filtered cells...",
        ambient.background_droplets,
        ambient.total_umis
    );

    /*
     * ------------------------------------------------------------
     * Filtered biological cells
     * ------------------------------------------------------------
     */
    let filtered =
        input.load_filtered(&filtered_index)?;

    /*
     * ------------------------------------------------------------
     * Ambient + genuine-guide mixture model
     * ------------------------------------------------------------
     */
    let fit_cfg =
        cli.model.fit_config();

    let fitted =
        fit_mixture(
            &filtered,
            ambient,
            &fit_cfg,
        )?;

    if fitted.biological_converged {
        eprintln!(
            "Guide assignments converged after {} iterations; calling guide/cell observations...",
            fitted.iterations
        );
    } else if fitted.mathematical_converged {
        eprintln!(
            "Model parameters converged after {} iterations; calling guide/cell observations...",
            fitted.iterations
        );
    } else {
        eprintln!(
            "WARNING: model stopped after {} iterations without convergence; calling guide/cell observations...",
            fitted.iterations
        );
    }

    /*
     * ------------------------------------------------------------
     * Final calls
     * ------------------------------------------------------------
     */
    let calls = GuideCalls::from_model(
        &fitted,
        &cli.model.call_config(),
    );

    /*
     * ------------------------------------------------------------
     * Summary statistics
     * ------------------------------------------------------------
     */
    let stats =
        collect_call_stats(
            &filtered,
            &calls,
        );

    println!(
        "{}",
        stats.report_to_string()
    );

    stats.report_to_csv(
        cli.out
            .join("guide_call_stats.tsv")
            .to_str()
            .context(
                "output path is not valid UTF-8"
            )?,
    );

    /*
     * ------------------------------------------------------------
     * Detailed output
     * ------------------------------------------------------------
     */
    write_calls(
        &cli.out,
        &filtered_index,
        &filtered,
        &calls,
    )?;

    write_guide_models(
        &cli.out,
        &filtered_index,
        &fitted,
    )?;

    write_cell_guide_assignments(
        &cli.out,
        &filtered_index,
        &filtered,
        &calls,
    )?;

    let multi_gap_stats = MultiGuideGapStats::collect(
        &filtered_index,
        &filtered,
        &calls,
    );

    let n_called =
        calls
            .flat
            .iter()
            .filter(|call| call.called)
            .count();

    eprintln!(
        "Done: {} observed cell-guide pairs, {} called genuine.",
        calls.flat.len(),
        n_called
    );

    eprintln!();
    eprintln!("Multi-guide posterior-gap statistics");
    multi_gap_stats.print_table();

    multi_gap_stats.write_table(
        &cli.out,
    )?;

    Ok(())
}

fn write_ambient(
    out: &PathBuf,
    index: &lumrik_guides::GuideFeatureIndex,
    ambient: &AmbientModel,
) -> Result<()> {
    let mut w = BufWriter::new(File::create(out.join("ambient_guides.tsv"))?);
    writeln!(w, "guide_id\tguide_name\tambient_umis\tp_g")?;
    for (gid, feature) in index.guides().iter().enumerate() {
        writeln!(
            w,
            "{}\t{}\t{}\t{:.12}",
            feature.id,
            feature.name,
            ambient.guide_umis[gid],
            ambient.guide_probability[gid]
        )?;
    }
    Ok(())
}

fn write_guide_models(
    out: &PathBuf,
    index: &lumrik_guides::GuideFeatureIndex,
    model: &lumrik_guides::FittedModel,
) -> Result<()> {
    let mut w = BufWriter::new(File::create(out.join("guide_models.tsv"))?);
    writeln!(w, "guide_id\tguide_name\tprior_real\ttrue_mean\ttheta")?;
    for (gid, g) in model.guides.iter().enumerate() {
        let feature = &index.guides()[gid];
        writeln!(
            w,
            "{}\t{}\t{:.8}\t{:.8}\t{:.8}",
            feature.id, feature.name, g.prior_real, g.mean, g.theta
        )?;
    }
    Ok(())
}

fn write_calls(
    out: &PathBuf,
    index: &lumrik_guides::GuideFeatureIndex,
    data: &lumrik_guides::GuideDataset,
    calls: &GuideCalls,
) -> Result<()> {
    let mut w = BufWriter::new(File::create(out.join("guide_calls.tsv"))?);
    writeln!(
        w,
        "barcode\tguide_id\tguide_name\tumi_count\tlambda_c\tp_g\texpected_ambient\tposterior_real\tambient_p\tq_value\tcalled"
    )?;

    for call in &calls.flat {
        let feature = &index.guides()[call.guide_id as usize];
        let barcode = data
            .barcode_by_id
            .get(&call.cell_id)
            .map(String::as_str)
            .unwrap_or("UNKNOWN");

        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{:.8}\t{:.12}\t{:.8}\t{:.8}\t{:.4e}\t{:.4e}\t{}",
            barcode,
            feature.id,
            feature.name,
            call.count,
            call.lambda_cell,
            call.ambient_probability,
            call.expected_ambient,
            call.posterior_real,
            call.ambient_p_value,
            call.q_value,
            call.called
        )?;
    }
    Ok(())
}


fn collect_call_stats(
    data: &lumrik_guides::GuideDataset,
    calls: &GuideCalls,
) -> MappingInfo {
    let mut stats = MappingInfo::new(None, 0.0, 0);

    for &cell_id in &data.cell_ids {
        stats.report("cells_total");

        let n_called = calls.called_for_cell(cell_id).count();

        match n_called {
            0 => stats.report("cells_no_guide"),
            1 => stats.report("cells_single_guide"),
            2 => {
                stats.report("cells_multi_guide");
                stats.report("cells_2_guides");
            }
            3 => {
                stats.report("cells_multi_guide");
                stats.report("cells_3_guides");
            }
            _ => {
                stats.report("cells_multi_guide");
                stats.report("cells_4plus_guides");
            }
        }

        for _ in 0..n_called {
            stats.report("called_guides_total");
        }
    }

    stats
}


#[derive(Debug)]
struct RankedGuideEvidence<'a> {
    best_guide: &'a str,
    best_posterior: f64,
    second_guide: &'a str,
    second_posterior: f64,
    posterior_gap: f64,
}

fn rank_guides_for_cell<'a>(
    cell_id: u64,
    index: &'a GuideFeatureIndex,
    call_lookup: &HashMap<(u64, u32), &'a GuideCall>,
) -> RankedGuideEvidence<'a> {
    let mut ranked: Vec<(&str, f64)> = index
        .guides()
        .iter()
        .enumerate()
        .map(|(guide_id, guide)| {
            let posterior = call_lookup
                .get(&(cell_id, guide_id as u32))
                .map(|call| call.posterior_real)
                .unwrap_or(0.0);

            (guide.name.as_str(), posterior)
        })
        .collect();

    ranked.sort_unstable_by(|a, b| {
        b.1.total_cmp(&a.1)
    });

    let (best_guide, best_posterior) = ranked
        .first()
        .copied()
        .unwrap_or(("", 0.0));

    let (second_guide, second_posterior) = ranked
        .get(1)
        .copied()
        .unwrap_or(("", 0.0));

    RankedGuideEvidence {
        best_guide,
        best_posterior,
        second_guide,
        second_posterior,
        posterior_gap: best_posterior - second_posterior,
    }
}

fn write_cell_guide_assignments(
    out: &PathBuf,
    index: &GuideFeatureIndex,
    data: &GuideDataset,
    calls: &GuideCalls,
) -> Result<()> {
    let path = out.join("cell_guide_assignments.tsv");

    let mut writer = BufWriter::new(
        File::create(&path)
            .with_context(|| format!("creating {}", path.display()))?,
    );

    /*
     * calls.flat contains only observed/non-zero cell-guide pairs.
     *
     * Missing pairs are interpreted as:
     *
     *   UMI       = 0
     *   posterior = 0
     *   q-value   = 1
     *   called    = false
     */
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

    /*
     * Cell-level annotation columns.
     */
    write!(
        writer,
        concat!(
            "barcode",
            "\tn_called_guides",
            "\tassignment",
            "\tcalled_guides",
            "\tbest_guide",
            "\tbest_posterior",
            "\tsecond_guide",
            "\tsecond_posterior",
            "\tposterior_gap"
        )
    )?;

    /*
     * Per-guide evidence columns.
     */
    for guide in index.guides() {
        write!(
            writer,
            "\t{}_umi\t{}_posterior\t{}_qvalue\t{}_called",
            guide.name,
            guide.name,
            guide.name,
            guide.name,
        )?;
    }

    writeln!(writer)?;

    /*
     * data.cell_ids follows the original filtered barcodes.tsv.gz order.
     */
    for &cell_id in &data.cell_ids {
        let barcode = data
            .barcode_by_id
            .get(&cell_id)
            .map(String::as_str)
            .unwrap_or("UNKNOWN");

        /*
         * Collect all guides passing the actual final calling rule.
         */
        let mut called_guides = Vec::new();

        for (guide_id, guide) in index.guides().iter().enumerate() {
            if let Some(call) =
                call_lookup.get(&(cell_id, guide_id as u32))
            {
                if call.called {
                    called_guides.push(guide.name.as_str());
                }
            }
        }

        let n_called = called_guides.len();

        let assignment = match n_called {
            0 => "none",
            1 => "single",
            _ => "multi",
        };

        /*
         * Rank ALL guides by posterior, not only called guides.
         *
         * This is important for identifying cells where the second-best
         * guide is close to the best guide even if it narrowly failed the
         * final call threshold.
         */
        let ranked = rank_guides_for_cell(
            cell_id,
            index,
            &call_lookup,
        );

        write!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{:.8}\t{}\t{:.8}\t{:.8}",
            barcode,
            n_called,
            assignment,
            called_guides.join(";"),
            ranked.best_guide,
            ranked.best_posterior,
            ranked.second_guide,
            ranked.second_posterior,
            ranked.posterior_gap,
        )?;

        /*
         * Detailed evidence for every guide.
         */
        for guide_id in 0..index.guides().len() {
            if let Some(call) =
                call_lookup.get(&(cell_id, guide_id as u32))
            {
                write!(
                    writer,
                    "\t{}\t{:.8}\t{:.8e}\t{}",
                    call.count,
                    call.posterior_real,
                    call.q_value,
                    call.called,
                )?;
            } else {
                write!(
                    writer,
                    "\t0\t0.00000000\t1.00000000e0\tfalse"
                )?;
            }
        }

        writeln!(writer)?;
    }

    writer.flush()?;

    Ok(())
}
