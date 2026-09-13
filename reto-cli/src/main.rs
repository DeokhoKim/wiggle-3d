#![forbid(unsafe_code)]
//! CLI application for Reto-Split.

use std::path::{Path, PathBuf};
use clap::Parser;
use tracing_subscriber::EnvFilter;


/// Supported image file extensions.
pub const SUPPORTED_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"];

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

    /// Output directory for results and debug artifacts
    #[arg(short, long, value_name = "DIR")]
    pub output: PathBuf,

    /// Enable debug mode
    #[arg(long)]
    pub debug: bool,

    /// Verbose output logging level (-v for debug, -vv for trace)
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,
}

/// Discovers image files from a file path or directory based on supported extensions.
pub fn discover_image_files(path: &Path) -> Vec<PathBuf> {
    if path.is_file() {
        if is_supported_image(path) {
            vec![path.to_path_buf()]
        } else {
            Vec::new()
        }
    } else if path.is_dir() {
        let mut files = Vec::new();
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let entry_path = entry.path();
                if entry_path.is_file() && is_supported_image(&entry_path) {
                    files.push(entry_path);
                }
            }
        }
        files.sort();
        files
    } else {
        Vec::new()
    }
}

/// Checks if the file path has a supported image extension.
pub fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            let ext_lower = ext.to_ascii_lowercase();
            SUPPORTED_EXTENSIONS.contains(&ext_lower.as_str())
        })
        .unwrap_or(false)
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let filter_directive = match cli.verbose {
        0 => if cli.debug { "debug" } else { "info" },
        1 => "debug",
        _ => "trace",
    };

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(filter_directive)),
        )
        .init();

    let image_files = discover_image_files(&cli.input);

    tracing::info!(
        input = ?cli.input,
        output = ?cli.output,
        debug = cli.debug,
        verbose = cli.verbose,
        discovered_count = image_files.len(),
        files = ?image_files,
        "Discovered image files for processing"
    );

    Ok(())
}



