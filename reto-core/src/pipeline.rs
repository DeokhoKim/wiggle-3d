//! Core pipeline orchestrator for processing verified image batches.
//!
//! Exposes supported format metadata and batch processing operations while leaving
//! front-end file discovery and path verification to callers (CLI, TUI).

use crate::detector::{PillarStatsDetector, RoiDetectionConfig, RoiDetector};
use crate::error::Result;
use crate::geom::FrameRoiSet;
use crate::luma::{Bt709LumaConverter, ScaledGrayscaleStrip, PROJECTION_MAX_DIMENSION};
use crate::visualizer::RoiVisualizer;
use image::{DynamicImage, Rgba};
use rayon::prelude::*;
use std::path::{Path, PathBuf};

/// Default supported input image file extensions for Reto-Split batch scanning.
pub const SUPPORTED_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"];

/// Checks if the given path has a supported image extension according to core format support.
///
/// # Arguments
/// * `path` - File path to inspect.
///
/// # Examples
/// ```
/// use reto_core::is_supported_image;
/// use std::path::Path;
///
/// assert!(is_supported_image(Path::new("scan.jpg")));
/// assert!(!is_supported_image(Path::new("notes.txt")));
/// ```
#[must_use]
pub fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| {
            let ext_lower = ext.to_ascii_lowercase();
            SUPPORTED_EXTENSIONS.contains(&ext_lower.as_str())
        })
}

/// Independent processing context for a single film strip image.
///
/// Encapsulates per-image source metadata, decoded buffers, and intermediate
/// analysis state across processing pipeline stages to ensure thread safety
/// and enable parallel execution without data races.
///
/// # Examples
/// ```
/// use reto_core::ImageItemContext;
/// use std::path::PathBuf;
///
/// let item = ImageItemContext::new(PathBuf::from("scan.jpg"), PathBuf::from("output"));
/// assert_eq!(item.file_stem(), "scan");
/// ```
#[derive(Debug, Clone)]
pub struct ImageItemContext {
    /// Path to the source scan image file.
    source_path: PathBuf,
    /// Destination directory for intermediate artifacts and results.
    output_dir: PathBuf,
    /// Loaded source image buffer.
    image: Option<DynamicImage>,
    /// Intermediate detection results (frame regions of interest).
    rois: Option<FrameRoiSet>,
    /// Scaled grayscale / luma strip buffer for reusable vision analysis.
    luma_strip: Option<ScaledGrayscaleStrip>,
}

impl ImageItemContext {
    /// Initializes a new image item context from a source path and destination directory.
    ///
    /// # Arguments
    /// * `source_path` - Path to the input image file.
    /// * `output_dir` - Destination directory for output artifacts.
    #[must_use]
    pub const fn new(source_path: PathBuf, output_dir: PathBuf) -> Self {
        Self {
            source_path,
            output_dir,
            image: None,
            rois: None,
            luma_strip: None,
        }
    }

    /// Returns the source file path.
    #[must_use]
    pub fn source_path(&self) -> &Path {
        &self.source_path
    }

    /// Returns the output directory path.
    #[must_use]
    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }

    /// Returns the file stem of the source image for naming outputs.
    #[must_use]
    pub fn file_stem(&self) -> &str {
        self.source_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("strip")
    }

    /// Sets the decoded image buffer.
    pub fn set_image(&mut self, img: DynamicImage) {
        self.image = Some(img);
    }

    /// Reference to the decoded image if loaded.
    #[must_use]
    pub const fn image(&self) -> Option<&DynamicImage> {
        self.image.as_ref()
    }

    /// Takes the decoded image buffer, leaving `None` in its place.
    pub const fn take_image(&mut self) -> Option<DynamicImage> {
        self.image.take()
    }

    /// Sets the detected frame `RoIs`.
    pub fn set_rois(&mut self, rois: FrameRoiSet) {
        self.rois = Some(rois);
    }

    /// Reference to detected `RoIs` if present.
    #[must_use]
    pub const fn rois(&self) -> Option<&FrameRoiSet> {
        self.rois.as_ref()
    }

    /// Takes the detected frame `RoIs`, leaving `None` in its place.
    pub const fn take_rois(&mut self) -> Option<FrameRoiSet> {
        self.rois.take()
    }

    /// Sets the scaled grayscale / luma strip.
    pub fn set_luma_strip(&mut self, strip: ScaledGrayscaleStrip) {
        self.luma_strip = Some(strip);
    }

    /// Reference to the scaled grayscale / luma strip if generated.
    #[must_use]
    pub const fn luma_strip(&self) -> Option<&ScaledGrayscaleStrip> {
        self.luma_strip.as_ref()
    }

    /// Takes the scaled grayscale / luma strip, leaving `None` in its place.
    pub const fn take_luma_strip(&mut self) -> Option<ScaledGrayscaleStrip> {
        self.luma_strip.take()
    }
}

/// A validated batch request containing verified file paths ready for processing.
///
/// # Examples
/// ```
/// use reto_core::BatchProcessingRequest;
/// use std::path::PathBuf;
///
/// let req = BatchProcessingRequest::new(vec![PathBuf::from("img.png")], PathBuf::from("dist"), true);
/// assert!(req.debug);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchProcessingRequest {
    /// Verified image file paths.
    pub files: Vec<PathBuf>,
    /// Destination directory for output results and visual debug artifacts.
    pub output_dir: PathBuf,
    /// Whether debug visualization mode is enabled.
    pub debug: bool,
}

impl BatchProcessingRequest {
    /// Creates a new `BatchProcessingRequest`.
    ///
    /// # Arguments
    /// * `files` - List of verified image file paths to process.
    /// * `output_dir` - Destination directory for output artifacts.
    /// * `debug` - Flag to enable debug visualization output.
    #[must_use]
    pub const fn new(files: Vec<PathBuf>, output_dir: PathBuf, debug: bool) -> Self {
        Self {
            files,
            output_dir,
            debug,
        }
    }

    /// Creates independent per-image processing contexts from this batch request.
    #[must_use]
    pub fn create_item_contexts(&self) -> Vec<ImageItemContext> {
        self.files
            .iter()
            .map(|path| ImageItemContext::new(path.clone(), self.output_dir.clone()))
            .collect()
    }
}

/// Outcome status of processing an individual image item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemProcessingOutcome {
    /// Item was processed successfully.
    Success,
    /// Item processing encountered an error.
    Failure,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ItemProcessingCounts {
    successful_count: usize,
    failed_count: usize,
}

impl ItemProcessingCounts {
    #[must_use]
    const fn from_outcome(outcome: ItemProcessingOutcome) -> Self {
        match outcome {
            ItemProcessingOutcome::Success => Self {
                successful_count: 1,
                failed_count: 0,
            },
            ItemProcessingOutcome::Failure => Self {
                successful_count: 0,
                failed_count: 1,
            },
        }
    }

    #[must_use]
    const fn combine(self, other: Self) -> Self {
        Self {
            successful_count: self.successful_count + other.successful_count,
            failed_count: self.failed_count + other.failed_count,
        }
    }
}

/// Execution summary of batch processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ProcessSummary {
    /// Number of verified input image files received.
    pub total_input: usize,
    /// Number of images successfully processed.
    pub successful_count: usize,
    /// Number of images that encountered errors.
    pub failed_count: usize,
}

/// Processes a single image item context through detection, visualization, and crop export stages.
///
/// # Arguments
/// * `item` - The image item context to process.
/// * `config` - Detection parameters.
///
/// # Errors
/// Returns [`Error`] if image decoding, `RoI` detection, or file saving fails.
pub fn process_item(
    mut item: ImageItemContext,
    config: &RoiDetectionConfig,
) -> Result<ImageItemContext> {
    let source_path = item.source_path().to_path_buf();
    let file_stem = item.file_stem().to_string();
    let output_dir = item.output_dir().to_path_buf();

    tracing::info!(file = ?source_path, "Loading input image");
    let dynamic_img = match item.image.take() {
        Some(img) => img,
        None => image::open(&source_path)?,
    };

    let mut composite_tap = crate::detector::CompositeDiagnosticTap::new();
    let luma_tap = crate::detector::SaveLumaDiagnosticTap::new(
        output_dir.join(format!("{file_stem}_luma.png")),
    );
    composite_tap.add(&luma_tap);

    // Precompute luma strip and register in ImageItemContext container for reuse
    let luma_strip = ScaledGrayscaleStrip::from_image(
        &dynamic_img,
        &Bt709LumaConverter::new(),
        PROJECTION_MAX_DIMENSION,
    )?;
    item.set_luma_strip(luma_strip);

    let detector = PillarStatsDetector::new();
    let rois = detector.detect(&dynamic_img, config, Some(&composite_tap))?;

    tracing::info!(
        file = ?source_path,
        orientation = ?rois.orientation,
        frame_count = rois.len(),
        "Detected frame RoIs"
    );

    for frame in &rois.frames {
        tracing::info!(
            frame_index = frame.index,
            bounds = ?frame.bounds,
            "Frame RoI detected"
        );
    }

    // Render debug visual overlay with drawn RoI bounding boxes
    let border_color = Rgba([0, 255, 128, 255]); // High-contrast emerald green
    let overlay = RoiVisualizer::render_overlay(&dynamic_img, &rois, border_color, 4);

    let overlay_path = output_dir.join(format!("{file_stem}_roi_overlay.png"));
    overlay.save(&overlay_path)?;
    tracing::info!(overlay_path = ?overlay_path, "Saved RoI visual overlay");

    // Frame extraction intentionally disabled until explicitly requested:
    // let crops = RoiVisualizer::extract_frame_images(&dynamic_img, &rois)?;
    // for (i, crop) in crops.iter().enumerate() {
    //     let frame_path = output_dir.join(format!("{file_stem}_frame_{i}.png"));
    //     crop.save(&frame_path)?;
    // }

    item.set_image(dynamic_img);
    item.set_rois(rois);

    Ok(item)
}

/// Processes a single film strip image: detects `RoIs`, generates visual overlay, and exports sub-frame crops.
///
/// # Arguments
/// * `file_path` - Path to the input film strip scan image.
/// * `output_dir` - Destination directory where overlay and cropped frame files are written.
/// * `config` - Detection parameters.
///
/// # Errors
/// Returns [`Error`] if image decoding, `RoI` detection, or file saving fails.
pub fn process_single_image(
    file_path: &Path,
    output_dir: &Path,
    config: &RoiDetectionConfig,
) -> Result<FrameRoiSet> {
    let item = ImageItemContext::new(file_path.to_path_buf(), output_dir.to_path_buf());
    let mut processed = process_item(item, config)?;
    processed.take_rois().ok_or_else(|| {
        crate::error::Error::Unknown("Missing RoI results in processed context".to_string())
    })
}

/// Drives processing for a verified batch of image files.
///
/// Executes per-item processing concurrently across available CPU threads using `rayon`.
/// Assumes file list verification has already been conducted by the front-end.
///
/// # Arguments
/// * `request` - Batch processing specifications including files, destination, and debug flags.
///
/// # Errors
/// Returns [`Error::Io`] if output directory creation fails.
pub fn run_batch(request: &BatchProcessingRequest) -> Result<ProcessSummary> {
    if request.files.is_empty() {
        tracing::warn!("Batch request contains no files to process");
        return Ok(ProcessSummary::default());
    }

    std::fs::create_dir_all(&request.output_dir)?;

    let config = RoiDetectionConfig::default();
    let total_input = request.files.len();

    tracing::info!(
        batch_size = total_input,
        expected_frames = config.expected_frames,
        output_dir = ?request.output_dir,
        "Starting parallel batch RoI processing"
    );

    let items = request.create_item_contexts();
    let counts = items
        .into_par_iter()
        .map(|item| {
            let path = item.source_path().to_path_buf();
            let outcome = match process_item(item, &config) {
                Ok(_) => ItemProcessingOutcome::Success,
                Err(err) => {
                    tracing::error!(file = ?path, error = ?err, "Failed processing image");
                    ItemProcessingOutcome::Failure
                }
            };
            ItemProcessingCounts::from_outcome(outcome)
        })
        .reduce(ItemProcessingCounts::default, ItemProcessingCounts::combine);

    let summary = ProcessSummary {
        total_input,
        successful_count: counts.successful_count,
        failed_count: counts.failed_count,
    };

    tracing::info!(
        total = summary.total_input,
        succeeded = summary.successful_count,
        failed = summary.failed_count,
        "Batch processing finished"
    );

    Ok(summary)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::redundant_clone)]
mod tests {
    use super::*;
    use image::{Rgba, RgbaImage};

    #[test]
    fn test_is_supported_image() {
        assert!(is_supported_image(Path::new("test.jpg")));
        assert!(is_supported_image(Path::new("test.PNG")));
        assert!(is_supported_image(Path::new("test.webp")));
        assert!(!is_supported_image(Path::new("test.txt")));
        assert!(!is_supported_image(Path::new("test")));
    }

    #[test]
    fn test_run_batch_on_generated_image() {
        let temp_dir = std::env::temp_dir().join("reto_core_test_batch");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        let input_path = temp_dir.join("sample_strip.png");
        let output_path = temp_dir.join("output");

        // Create a 300x100 synthetic strip
        let img = RgbaImage::from_pixel(300, 100, Rgba([200, 200, 200, 255]));
        img.save(&input_path).unwrap();

        let request = BatchProcessingRequest::new(vec![input_path], output_path.clone(), false);
        let summary = run_batch(&request).expect("Batch should run");

        assert_eq!(summary.total_input, 1);
        assert_eq!(summary.successful_count, 1);
        assert_eq!(summary.failed_count, 0);

        // Verify generated artifacts (overlay and intermediate luma are saved)
        assert!(output_path.join("sample_strip_roi_overlay.png").exists());
        assert!(output_path.join("sample_strip_luma.png").exists());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
