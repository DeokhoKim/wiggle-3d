#![forbid(unsafe_code)]
//! Thin CLI front-end for Reto-Split.
//!
//! Handles CLI argument parsing, input path discovery/verification, and passes
//! verified file lists to `reto-core::run_batch`.

use clap::Parser;
use reto_core::{is_supported_image, run_batch, BatchProcessingRequest};
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

    /// Verbose output logging level (-v for debug, -vv for trace)
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
                "debug"
            } else {
                "info"
            }
        }
        1 => "debug",
        _ => "trace",
    };

    tracing_subscriber::fmt()
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

    let request = BatchProcessingRequest::new(files, cli.output.clone(), cli.debug);
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

    Ok(())
}
