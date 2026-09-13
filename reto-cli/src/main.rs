#![forbid(unsafe_code)]
//! Thin CLI front-end for Reto-Split.
//!
//! Handles CLI argument parsing, input path discovery/verification, and passes
//! verified file lists to `reto-core::run_batch`.

use clap::Parser;
use reto_core::{
    clear_retinaface_model_cache, clear_superpoint_model_cache, is_supported_image, run_batch,
    BatchProcessingRequest,
};
use std::path::{Path, PathBuf};
use tracing_subscriber::EnvFilter;

/// Command-line arguments for reto-cli.
#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "3D film camera image splitting and alignment tool",
    long_about = None
)]
pub struct Cli {
    /// Path to the input image file or directory for batch processing
    #[arg(short, long, value_name = "PATH")]
    pub input: PathBuf,

    /// Output directory for results and debug visual artifacts
    #[arg(short, long, value_name = "DIR")]
    pub output: PathBuf,

    /// Enable debug mode
    #[arg(long)]
    pub debug: bool,

    /// Inter-frame delay for Wiggle GIF in milliseconds (default: 100ms)
    #[arg(long, default_value_t = reto_core::DEFAULT_FRAME_DELAY_MS, value_name = "MS")]
    pub gif_delay: u32,

    /// Disable Floyd-Steinberg dithering during GIF color quantization
    #[arg(long)]
    pub no_dither: bool,

    /// Verbose logging level (-v for debug, -vv for trace)
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,
}

/// Discovers and validates image files from an input path using [`reto_core::is_supported_image`].
///
/// # Arguments
/// * `input_path` - Path to an individual image file or a directory containing scan files.
#[must_use]
pub fn collect_verified_images(input_path: &Path) -> Vec<PathBuf> {
    if input_path.is_file() {
        if is_supported_image(input_path) {
            vec![input_path.to_path_buf()]
        } else {
            Vec::new()
        }
    } else if input_path.is_dir() {
        let mut files = Vec::new();
        if let Ok(entries) = std::fs::read_dir(input_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() && is_supported_image(&path) {
                    files.push(path);
                }
            }
        }
        files.sort();
        files
    } else {
        Vec::new()
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let filter_directive = match cli.verbose {
        0 => {
            if cli.debug {
                "debug,ort=info"
            } else {
                "info,ort=warn"
            }
        }
        1 => "debug,ort=info",
        _ => "trace,ort=info",
    };

    tracing_subscriber::fmt()
        .with_timer(tracing_subscriber::fmt::time::UtcTime::rfc_3339())
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
        .with_target(true)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(filter_directive)),
        )
        .init();

    // Front-end discovery & input validation
    let files = collect_verified_images(&cli.input);
    if files.is_empty() {
        tracing::warn!(input = ?cli.input, "No valid image files found matching supported extensions");
        return Ok(());
    }

    tracing::info!(
        discovered_count = files.len(),
        input = ?cli.input,
        "Verified input files for processing"
    );

    let gif_config = reto_core::WiggleGifConfig::new(cli.gif_delay).with_dither(!cli.no_dither);
    let request = BatchProcessingRequest::new(files, cli.output.clone(), cli.debug)
        .with_gif_config(gif_config);
    let summary = run_batch(&request)?;

    println!(
        "[OK] Processed {} images ({} succeeded, {} failed). Output directory: {}",
        summary.total_input,
        summary.successful_count,
        summary.failed_count,
        cli.output.display()
    );

    if summary.failed_count > 0 {
        tracing::warn!(
            succeeded = summary.successful_count,
            failed = summary.failed_count,
            "Completed with some failures"
        );
    }

    // Release compiled SuperPoint and RetinaFace models from memory at program exit
    clear_superpoint_model_cache();
    clear_retinaface_model_cache();

    Ok(())
}
