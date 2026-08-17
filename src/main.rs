use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use lumrik_guides::background::{AmbientModel, BackgroundConfig};
use lumrik_guides::caller::{CallConfig, GuideCalls};
use lumrik_guides::model::{fit_mixture, FitConfig};
use lumrik_guides::tenx::TenxGuideInput;

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

    #[arg(long)]
    out: PathBuf,

    #[arg(long, default_value = "CRISPR Guide Capture")]
    feature_type: String,

    #[arg(long, default_value_t = 1)]
    threads: usize,

    #[arg(long, default_value_t = 0.5)]
    ambient_alpha: f64,

    #[arg(long, default_value_t = 0.95)]
    posterior: f64,

    #[arg(long, default_value_t = 0.05)]
    fdr: f64,

    #[arg(long, default_value_t = 100)]
    max_iterations: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    fs::create_dir_all(&cli.out)
        .with_context(|| format!("creating {}", cli.out.display()))?;

    let mut input = TenxGuideInput::new(cli.raw, cli.filtered);
    input.feature_type = cli.feature_type;
    input.threads = cli.threads;

    let (raw_index, filtered_index) = input.indexes()?;

    eprintln!(
        "Found {} guide features; loading raw non-cell droplets...",
        raw_index.guides().len()
    );

    let background_data = input.load_background(&raw_index)?;
    let ambient = AmbientModel::fit(
        &background_data,
        &BackgroundConfig {
            alpha: cli.ambient_alpha,
        },
    )?;

    write_ambient(&cli.out, &raw_index, &ambient)?;

    // Deliberate phase boundary: only now do we load the actual called cells.
    eprintln!(
        "Ambient model fitted from {} droplets / {} guide UMIs; loading filtered cells...",
        ambient.background_droplets,
        ambient.total_umis
    );

    let filtered = input.load_filtered(&filtered_index)?;

    let fit_cfg = FitConfig {
        max_iterations: cli.max_iterations,
        ..FitConfig::default()
    };
    let fitted = fit_mixture(&filtered, ambient, &fit_cfg)?;

    eprintln!(
        "Mixture model fitted in {} iterations; calling guide/cell observations...",
        fitted.iterations
    );

    let calls = GuideCalls::from_model(
        &fitted,
        &CallConfig {
            minimum_posterior: cli.posterior,
            maximum_fdr: cli.fdr,
        },
    );

    let stats = collect_call_stats(&filtered, &calls);

    println!("{}", stats.report_to_string());

    stats.report_to_csv(
        cli.out
            .join("guide_call_stats.tsv")
            .to_str()
            .context("output path is not valid UTF-8")?,
    );

    write_calls(&cli.out, &filtered_index, &filtered, &calls)?;
    write_guide_models(&cli.out, &filtered_index, &fitted)?;

    let n_called = calls.flat.iter().filter(|x| x.called).count();
    eprintln!(
        "Done: {} observed cell-guide pairs, {} called genuine.",
        calls.flat.len(),
        n_called
    );

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