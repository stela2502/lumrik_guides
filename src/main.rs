use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use lumrik_guides::background::AmbientModel;
use lumrik_guides::caller::GuideCalls;
use lumrik_guides::cli::GuideModelCli;
use lumrik_guides::model::fit_mixture;
use lumrik_guides::tenx::TenxGuideInput;
use lumrik_guides::{
    CellGuideAssignments,
    MultiGuideGapStats,
    MultiGuideGapStatsTable,
};

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
    ///
    /// If omitted, Rayon uses the available CPU count.
    #[arg(long)]
    threads: Option<usize>,

    #[command(flatten)]
    model: GuideModelCli,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    fs::create_dir_all(&cli.out)
        .with_context(|| format!("creating {}", cli.out.display()))?;

    /*
     * ------------------------------------------------------------
     * Threads
     * ------------------------------------------------------------
     */
    if let Some(threads) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads.max(1))
            .build_global()
            .context("failed to initialize Rayon thread pool")?;
    }

    let threads = cli
        .threads
        .unwrap_or_else(rayon::current_num_threads);

    /*
     * ------------------------------------------------------------
     * Input
     * ------------------------------------------------------------
     */
    let mut input = TenxGuideInput::new(
        cli.raw,
        cli.filtered,
    );

    input.feature_type = cli.feature_type;
    input.threads = threads;

    let (raw_index, filtered_index) =
        input.indexes()?;

    eprintln!(
        "Found {} guide features.",
        raw_index.guides().len()
    );

    /*
     * ------------------------------------------------------------
     * Ambient model
     * ------------------------------------------------------------
     */
    eprintln!(
        "Loading raw-only droplets..."
    );

    let background_data =
        input.load_background(&raw_index)?;

    let ambient = AmbientModel::fit(
        &background_data,
        &cli.model.background_config(),
    )?;

    eprintln!(
        "Ambient model fitted from {} raw-only droplets / {} guide UMIs.",
        ambient.background_droplets,
        ambient.total_umis,
    );

    ambient.write_table(
        &cli.out,
        &raw_index,
    )?;

    /*
     * ------------------------------------------------------------
     * Filtered cells
     * ------------------------------------------------------------
     */
    eprintln!(
        "Loading filtered guide counts..."
    );

    let filtered =
        input.load_filtered(&filtered_index)?;

    eprintln!(
        "Loaded {} filtered cells.",
        filtered.n_cells()
    );

    /*
     * ------------------------------------------------------------
     * Mixture model
     * ------------------------------------------------------------
     */
    eprintln!(
        "Fitting ambient + true-guide mixture model..."
    );

    let fitted = fit_mixture(
        &filtered,
        ambient,
        &cli.model.fit_config(),
    )?;

    if fitted.mathematical_converged {
        eprintln!(
            "Model converged after {} iterations.",
            fitted.iterations
        );
    } else {
        eprintln!(
            "WARNING: model reached {} iterations without mathematical convergence.",
            fitted.iterations
        );
    }

    fitted.write_table(
        &cli.out,
        &filtered_index,
    )?;

    /*
     * ------------------------------------------------------------
     * Calls
     * ------------------------------------------------------------
     */
    let calls = GuideCalls::from_model(
        &fitted,
        &cli.model.call_config(),
    );

    calls.write_table(
        &cli.out,
        &filtered_index,
        &filtered,
    )?;

    /*
     * ------------------------------------------------------------
     * Cell-level annotation
     * ------------------------------------------------------------
     */
    let assignments = CellGuideAssignments::new(
        &filtered_index,
        &filtered,
        &calls,
    );

    assignments.write_table(
        &cli.out,
    )?;

    /*
     * ------------------------------------------------------------
     * Multi-guide QC
     * ------------------------------------------------------------
     */
    let multi_gap_stats =
        MultiGuideGapStats::collect(
            &assignments,
        );

    println!();

    multi_gap_stats.print_assignment_summary(
        &assignments,
    );

    multi_gap_stats.write_table(
        &cli.out,
    )?;

    /*
     * ------------------------------------------------------------
     * Final summary
     * ------------------------------------------------------------
     */
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

    Ok(())
}



fn percent(
    value: usize,
    total: usize,
) -> f64 {
    if total == 0 {
        0.0
    } else {
        value as f64
            / total as f64
            * 100.0
    }
}