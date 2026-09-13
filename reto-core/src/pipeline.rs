//! Core pipeline orchestrator for processing verified image batches.
//!
//! Exposes supported format metadata and batch processing operations while leaving
//! front-end file discovery and path verification to callers (CLI, TUI).

use crate::detector::{PillarStatsDetector, RoiDetectionConfig, RoiDetector};
use crate::error::Result;
use crate::face::RetinaFaceDetector;
use crate::feature::AlignmentDiagnosticTap;
use crate::geom::FrameRoiSet;
use crate::luma::{Bt709LumaConverter, ScaledLumaImage, PROJECTION_MAX_DIMENSION};
use crate::visualizer::RoiVisualizer;
use image::{DynamicImage, Rgba};
use rayon::prelude::*;
use std::path::{Path, PathBuf};

/// Default supported input image file extensions for Reto-Split batch scanning.
pub const SUPPORTED_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"];

/// Checks if the given path has a supported image extension according to core format support.
///
/// # Arguments
/// * `path` - The file path to test.
///
/// # Examples
/// ```
/// use reto_core::is_supported_image;
/// use std::path::Path;
///
/// assert!(is_supported_image(Path::new("scan.JPG")));
/// assert!(!is_supported_image(Path::new("scan.txt")));
/// ```
#[must_use]
pub fn is_supported_image<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref()
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| {
            SUPPORTED_EXTENSIONS
                .iter()
                .any(|&supported| ext.eq_ignore_ascii_case(supported))
        })
}

/// Independent processing context for a single film scan image.
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
    /// Scaled single-channel luma image buffer for reusable vision analysis.
    luma_image: Option<ScaledLumaImage>,
    /// Extracted feature keypoints for each detected sub-frame.
    features: Option<Vec<crate::feature::FeatureFrame>>,
    /// Target execution device for model inference.
    device: crate::feature::BackendDevice,
    /// Configuration for Wiggle GIF generation.
    gif_config: crate::gif::WiggleGifConfig,
    /// Whether debug visualization mode is enabled.
    debug: bool,
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
            luma_image: None,
            features: None,
            device: crate::feature::BackendDevice::Auto,
            gif_config: crate::gif::WiggleGifConfig {
                delay_ms: crate::gif::DEFAULT_FRAME_DELAY_MS,
                sample_factor: crate::gif::DEFAULT_NEUQUANT_SAMPLE_FAC,
                dither: true,
            },
            debug: false,
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

    /// Sets the scaled luma image buffer.
    pub fn set_luma_image(&mut self, luma: ScaledLumaImage) {
        self.luma_image = Some(luma);
    }

    /// Backward-compatible alias for [`ImageItemContext::set_luma_image`].
    pub fn set_luma_strip(&mut self, luma: ScaledLumaImage) {
        self.set_luma_image(luma);
    }

    /// Reference to the scaled luma image if generated.
    #[must_use]
    pub const fn luma_image(&self) -> Option<&ScaledLumaImage> {
        self.luma_image.as_ref()
    }

    /// Backward-compatible alias for [`ImageItemContext::luma_image`].
    #[must_use]
    pub const fn luma_strip(&self) -> Option<&ScaledLumaImage> {
        self.luma_image()
    }

    /// Takes the scaled luma image buffer, leaving `None` in its place.
    pub const fn take_luma_image(&mut self) -> Option<ScaledLumaImage> {
        self.luma_image.take()
    }

    /// Backward-compatible alias for [`ImageItemContext::take_luma_image`].
    pub const fn take_luma_strip(&mut self) -> Option<ScaledLumaImage> {
        self.take_luma_image()
    }

    /// Sets the extracted feature frames.
    pub fn set_features(&mut self, features: Vec<crate::feature::FeatureFrame>) {
        self.features = Some(features);
    }

    /// Reference to extracted feature frames if present.
    #[must_use]
    pub fn features(&self) -> Option<&[crate::feature::FeatureFrame]> {
        self.features.as_deref()
    }

    /// Returns the target execution device for model inference.
    #[must_use]
    pub const fn device(&self) -> crate::feature::BackendDevice {
        self.device
    }

    /// Sets the target execution device for model inference.
    pub const fn set_device(&mut self, device: crate::feature::BackendDevice) {
        self.device = device;
    }

    /// Returns the Wiggle GIF generation configuration.
    #[must_use]
    pub const fn gif_config(&self) -> crate::gif::WiggleGifConfig {
        self.gif_config
    }

    /// Sets the Wiggle GIF generation configuration.
    pub const fn set_gif_config(&mut self, gif_config: crate::gif::WiggleGifConfig) {
        self.gif_config = gif_config;
    }

    /// Returns whether debug visualization mode is enabled.
    #[must_use]
    pub const fn debug(&self) -> bool {
        self.debug
    }

    /// Sets whether debug visualization mode is enabled.
    pub const fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    /// Builder method to set debug visualization mode.
    #[must_use]
    pub const fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }
}

/// Progress event notification emitted during batch processing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgressEvent<'a> {
    /// Processing started for an image item.
    ItemStarted {
        /// File stem of the image item being processed.
        file_stem: &'a str,
        /// Current 1-based index in the batch.
        index: usize,
        /// Total number of items in the batch.
        total: usize,
    },
    /// Processing completed for an image item.
    ItemCompleted {
        /// File stem of the image item processed.
        file_stem: &'a str,
        /// Current 1-based index in the batch.
        index: usize,
        /// Total number of items in the batch.
        total: usize,
        /// Whether processing succeeded without error.
        success: bool,
    },
}

/// Thread-safe observer trait for receiving pipeline execution progress.
pub trait ProgressObserver: std::fmt::Debug + Send + Sync {
    /// Handles an incoming pipeline progress event.
    fn on_progress(&self, event: ProgressEvent<'_>);
}

/// A validated batch request containing verified file paths ready for processing.
///
/// # Examples
/// ```
/// use reto_core::{BackendDevice, BatchProcessingRequest};
/// use std::path::PathBuf;
///
/// let req = BatchProcessingRequest::new(vec![PathBuf::from("img.png")], PathBuf::from("dist"), true);
/// assert!(req.debug);
/// assert_eq!(req.device, BackendDevice::Auto);
/// ```
#[derive(Debug, Clone)]
pub struct BatchProcessingRequest {
    /// Verified image file paths.
    pub files: Vec<PathBuf>,
    /// Destination directory for output results and visual debug artifacts.
    pub output_dir: PathBuf,
    /// Whether debug visualization mode is enabled.
    pub debug: bool,
    /// Compute device / execution provider for model inference.
    pub device: crate::feature::BackendDevice,
    /// Configuration for Wiggle GIF generation.
    pub gif_config: crate::gif::WiggleGifConfig,
    /// Optional observer for receiving progress notifications.
    pub progress_observer: Option<std::sync::Arc<dyn ProgressObserver>>,
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
            device: crate::feature::BackendDevice::Auto,
            gif_config: crate::gif::WiggleGifConfig {
                delay_ms: crate::gif::DEFAULT_FRAME_DELAY_MS,
                sample_factor: crate::gif::DEFAULT_NEUQUANT_SAMPLE_FAC,
                dither: true,
            },
            progress_observer: None,
        }
    }

    /// Sets the compute device for inference.
    #[must_use]
    pub const fn with_device(mut self, device: crate::feature::BackendDevice) -> Self {
        self.device = device;
        self
    }

    /// Sets the Wiggle GIF generation configuration.
    #[must_use]
    pub const fn with_gif_config(mut self, gif_config: crate::gif::WiggleGifConfig) -> Self {
        self.gif_config = gif_config;
        self
    }

    /// Attaches a progress observer to receive batch processing execution events.
    #[must_use]
    pub fn with_progress_observer(
        mut self,
        observer: std::sync::Arc<dyn ProgressObserver>,
    ) -> Self {
        self.progress_observer = Some(observer);
        self
    }

    /// Creates independent per-image processing contexts from this batch request.
    #[must_use]
    pub fn create_item_contexts(&self) -> Vec<ImageItemContext> {
        self.files
            .iter()
            .map(|path| {
                let mut ctx = ImageItemContext::new(path.clone(), self.output_dir.clone());
                ctx.set_device(self.device);
                ctx.set_gif_config(self.gif_config);
                ctx.set_debug(self.debug);
                ctx
            })
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
#[tracing::instrument(level = "debug", skip_all)]
fn load_image(item: &mut ImageItemContext) -> Result<DynamicImage> {
    match item.image.take() {
        Some(img) => Ok(img),
        None => Ok(image::open(item.source_path())?),
    }
}

#[tracing::instrument(level = "debug", skip_all)]
fn prepare_luma(dynamic_img: &DynamicImage) -> Result<ScaledLumaImage> {
    ScaledLumaImage::from_image(
        dynamic_img,
        &Bt709LumaConverter::new(),
        PROJECTION_MAX_DIMENSION,
    )
    .map_err(Into::into)
}

#[tracing::instrument(level = "debug", skip(luma, config, tap))]
fn detect_rois(
    luma: &ScaledLumaImage,
    config: &RoiDetectionConfig,
    tap: Option<&dyn crate::detector::RoiDiagnosticTap>,
) -> Result<FrameRoiSet> {
    let detector = PillarStatsDetector::new();
    let rois = detector.detect_luma(luma, config, tap)?;
    tracing::debug!(
        orientation = ?rois.orientation,
        frame_count = rois.len(),
        "Detected frame RoIs"
    );
    let bounds: Vec<_> = rois.frames.iter().map(|f| &f.bounds).collect();
    tracing::debug!(
        bounds = ?bounds,
        "Frame RoI bounds configured"
    );
    Ok(rois)
}

#[tracing::instrument(level = "debug", skip(dynamic_img, rois))]
fn render_roi_overlay(
    dynamic_img: &DynamicImage,
    rois: &FrameRoiSet,
    output_dir: &Path,
    file_stem: &str,
) -> (PathBuf, DynamicImage) {
    let border_color = Rgba([0, 255, 128, 255]); // High-contrast emerald green
    let roi_overlay = RoiVisualizer::render_overlay(dynamic_img, rois, border_color, 4);
    let overlay_path = output_dir.join(format!("{file_stem}_roi_overlay.png"));
    (overlay_path, DynamicImage::ImageRgba8(roi_overlay))
}

#[allow(clippy::type_complexity)]
#[tracing::instrument(level = "debug", skip_all)]
fn extract_features(
    file_stem: &str,
    luma: &ScaledLumaImage,
    rois: &FrameRoiSet,
    overlay_path: Option<PathBuf>,
    roi_overlay: Option<DynamicImage>,
    device: crate::feature::BackendDevice,
    frame_faces: &[crate::visualizer::FrameFaceRecord],
) -> Result<(
    Vec<crate::feature::FeatureFrame>,
    Vec<crate::feature::FeatureTriplet>,
    Vec<(crate::feature::FramePair, Vec<crate::feature::FeatureMatch>)>,
)> {
    use crate::feature::{
        FeatureMatcher, PointDetector, SuperPointConfig, SuperPointDescriptorMatcher,
        SuperPointDetector, TripletConsistencyConfig,
    };

    let config = SuperPointConfig {
        device,
        ..Default::default()
    };
    let point_detector = SuperPointDetector::new(config);
    let features = point_detector.detect_luma_all(luma, &rois.frames, None)?;

    let counts: Vec<usize> = features
        .iter()
        .map(crate::feature::FeatureFrame::len)
        .collect();
    tracing::info!(
        file = %file_stem,
        keypoints = ?counts,
        "Detected frame keypoints"
    );

    let mut extracted_triplets = Vec::new();
    let mut extracted_pairs = Vec::new();

    // If at least 2 frames exist, match across frames
    if features.len() >= 2 {
        let matcher = SuperPointDescriptorMatcher::default();
        let consistency_config = TripletConsistencyConfig::with_orientation(luma.orientation);

        let tap = match (overlay_path, roi_overlay) {
            (Some(path), Some(overlay)) => {
                let t = crate::visualizer::SaveMatchesDiagnosticTap::new(
                    Some(path),
                    None,
                    overlay,
                )
                .with_rois(rois.clone());
                t.add_faces(frame_faces);
                for (idx, frame) in features.iter().enumerate() {
                    t.on_features_extracted(idx, frame);
                }
                Some(t)
            }
            _ => None,
        };

        if features.len() >= 3 {
            match matcher.extract_consistent_triplets(
                &features,
                &consistency_config,
                tap.as_ref().map(|t| t as &dyn AlignmentDiagnosticTap),
            ) {
                Ok(triplets) => {
                    tracing::info!(
                        file = %file_stem,
                        triplet_count = triplets.len(),
                        "Extracted depth-consistent feature triplets across 3 views"
                    );
                    extracted_triplets = triplets;
                }
                Err(e) => {
                    tracing::warn!(file = %file_stem, error = %e, "Failed to extract feature triplets; falling back to pairwise matching");
                }
            }
        } else {
            let pair_matches = matcher.match_pair_bidirectional(&features[0], &features[1])?;
            if let Some(ref t) = tap {
                t.on_matches_found(pair_matches.pair, &pair_matches.matches);
            }
            extracted_pairs.push((pair_matches.pair, pair_matches.matches));
        }

        if let Some(t) = tap {
            t.finish()?;
        }
    } else if let (Some(path), Some(overlay)) = (overlay_path, roi_overlay) {
        // Fallback: draw keypoints on overlay if fewer than 2 frames and debug overlay is requested
        let feat_tap = crate::visualizer::SaveFeaturesDiagnosticTap::new(
            path,
            overlay,
            Rgba([255, 64, 128, 255]),
        )
        .with_rois(rois.clone());
        for (idx, frame) in features.iter().enumerate() {
            feat_tap.on_features_extracted(idx, frame);
        }
        feat_tap.finish()?;
    }

    Ok((features, extracted_triplets, extracted_pairs))
}

/// Processes a single image item context through detection, visualization, and feature extraction.
///
/// # Arguments
/// * `item` - The image item context to process.
/// * `config` - Detection parameters.
///
/// # Errors
/// Returns [`Error`] if image loading, `RoI` detection, or feature extraction fails.
#[allow(
    clippy::cast_precision_loss,
    clippy::option_if_let_else,
    clippy::too_many_lines
)]
#[tracing::instrument(level = "debug", skip_all)]
pub fn process_item(
    mut item: ImageItemContext,
    config: &RoiDetectionConfig,
) -> Result<ImageItemContext> {
    let source_path = item.source_path().to_path_buf();
    let file_stem = item.file_stem().to_string();
    let output_dir = item.output_dir().to_path_buf();

    let dynamic_img = load_image(&mut item)?;

    let luma_image = prepare_luma(&dynamic_img)?;
    let rois = detect_rois(&luma_image, config, None)?;
    item.set_luma_image(luma_image);

    // Extract sub-frame crops for face detection and Wiggle GIF assembly
    let sub_frame_crops = RoiVisualizer::extract_frame_images(&dynamic_img, &rois)?;

    // Detect faces across each extracted sub-frame crop (batch of individual frames)
    let mut frame_faces = Vec::new();
    if !sub_frame_crops.is_empty() {
        match RetinaFaceDetector::default_engine() {
            Ok(face_detector) => match face_detector.detect_faces_for_rois(&sub_frame_crops) {
                Ok(records) => {
                    let total_faces: usize = records.iter().map(|(_, list, _)| list.len()).sum();
                    if total_faces > 0 {
                        tracing::debug!(
                            file = %file_stem,
                            faces = total_faces,
                            "Detected faces across sub-frames"
                        );
                    }
                    frame_faces = records;
                }
                Err(e) => {
                    tracing::warn!(file = %file_stem, error = %e, "Failed to run face detection on sub-frames");
                }
            },
            Err(e) => {
                tracing::warn!(file = %file_stem, error = %e, "Failed to initialize RetinaFace detector");
            }
        }
    }

    let dominant_face_bbox = frame_faces
        .iter()
        .find(|(f_idx, _, dom_idx)| *f_idx == 1 && dom_idx.is_some())
        .and_then(|(_, detections, dom_idx)| {
            dom_idx.and_then(|idx| detections.get(idx).map(|f| f.bbox))
        })
        .or_else(|| {
            frame_faces.iter().find_map(|(_, detections, dom_idx)| {
                dom_idx.and_then(|idx| detections.get(idx).map(|f| f.bbox))
            })
        });

    let (overlay_path, roi_overlay) = if item.debug() {
        let (path, overlay) = render_roi_overlay(&dynamic_img, &rois, &output_dir, &file_stem);
        (Some(path), Some(overlay))
    } else {
        (None, None)
    };

    let mut shifts = [(0.0, 0.0); 3];
    if let Some(luma) = item.luma_image() {
        let (features, triplets, pairs) = extract_features(
            &file_stem,
            luma,
            &rois,
            overlay_path,
            roi_overlay,
            item.device(),
            &frame_faces,
        )?;
        if !triplets.is_empty() {
            shifts = crate::gif::WiggleAligner::compute_depth_surface_shifts_from_triplets_with_face_priority(
                &features,
                &triplets,
                dominant_face_bbox,
                crate::gif::DEFAULT_DISPARITY_BIN_SIZE_PX,
                crate::gif::DEFAULT_CLUSTER_TOLERANCE_PX,
            );
        } else if !pairs.is_empty() {
            shifts = crate::gif::WiggleAligner::compute_depth_surface_shifts_from_pairs_with_face_priority(
                &features,
                &pairs,
                dominant_face_bbox,
                crate::gif::DEFAULT_DISPARITY_BIN_SIZE_PX,
                crate::gif::DEFAULT_CLUSTER_TOLERANCE_PX,
            );
        }
        item.set_features(features);
    }

    if !sub_frame_crops.is_empty() {
        let (scale_x, scale_y) = if let Some(luma) = item.luma_image() {
            (
                dynamic_img.width() as f32 / luma.width as f32,
                dynamic_img.height() as f32 / luma.height as f32,
            )
        } else {
            (1.0, 1.0)
        };
        let scaled_shifts: Vec<(f32, f32)> = shifts
            .iter()
            .map(|&(dx, dy)| (dx * scale_x, dy * scale_y))
            .collect();
        let aligned_frames =
            crate::gif::WiggleAligner::align_and_crop(&sub_frame_crops, &scaled_shifts)?;
        let gif_path = output_dir.join(format!("{file_stem}_wiggle.gif"));
        let mut gif_file = std::fs::File::create(&gif_path)?;
        crate::gif::WiggleGifBuilder::build_wiggle_gif(
            &aligned_frames,
            &item.gif_config(),
            &mut gif_file,
        )?;
        tracing::info!(gif_path = ?gif_path, "Saved Wiggle 3D GIF");
    }

    item.set_image(dynamic_img);
    item.set_rois(rois);

    tracing::info!(file = ?source_path, "Processed image item successfully");
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
#[tracing::instrument(level = "debug", skip_all)]
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
    let total = items.len();

    let counts = items
        .into_par_iter()
        .enumerate()
        .map(|(idx, item)| {
            let file_stem = item.file_stem().to_string();
            if let Some(ref observer) = request.progress_observer {
                observer.on_progress(ProgressEvent::ItemStarted {
                    file_stem: &file_stem,
                    index: idx + 1,
                    total,
                });
            }

            let outcome = match process_item(item, &config) {
                Ok(_) => {
                    if let Some(ref observer) = request.progress_observer {
                        observer.on_progress(ProgressEvent::ItemCompleted {
                            file_stem: &file_stem,
                            index: idx + 1,
                            total,
                            success: true,
                        });
                    }
                    ItemProcessingOutcome::Success
                }
                Err(err) => {
                    tracing::error!(file = %file_stem, error = ?err, "Failed processing image");
                    if let Some(ref observer) = request.progress_observer {
                        observer.on_progress(ProgressEvent::ItemCompleted {
                            file_stem: &file_stem,
                            index: idx + 1,
                            total,
                            success: false,
                        });
                    }
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

        // Non-debug run: verify wiggle GIF is saved, but roi_overlay is NOT saved
        let request_non_debug =
            BatchProcessingRequest::new(vec![input_path.clone()], output_path.clone(), false);
        let summary = run_batch(&request_non_debug).expect("Non-debug batch should run");
        assert_eq!(summary.total_input, 1);
        assert_eq!(summary.successful_count, 1);
        assert!(!output_path.join("sample_strip_roi_overlay.png").exists());
        assert!(output_path.join("sample_strip_wiggle.gif").exists());

        // Debug run: verify roi_overlay is saved when debug is enabled
        let request_debug = BatchProcessingRequest::new(vec![input_path], output_path.clone(), true);
        let summary_debug = run_batch(&request_debug).expect("Debug batch should run");
        assert_eq!(summary_debug.successful_count, 1);
        assert!(output_path.join("sample_strip_roi_overlay.png").exists());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[derive(Debug, Default)]
    struct MockProgressObserver {
        started: std::sync::Mutex<Vec<(String, usize, usize)>>,
        completed: std::sync::Mutex<Vec<(String, usize, usize, bool)>>,
    }

    impl ProgressObserver for MockProgressObserver {
        fn on_progress(&self, event: ProgressEvent<'_>) {
            match event {
                ProgressEvent::ItemStarted {
                    file_stem,
                    index,
                    total,
                } => {
                    self.started
                        .lock()
                        .unwrap()
                        .push((file_stem.to_string(), index, total));
                }
                ProgressEvent::ItemCompleted {
                    file_stem,
                    index,
                    total,
                    success,
                } => {
                    self.completed
                        .lock()
                        .unwrap()
                        .push((file_stem.to_string(), index, total, success));
                }
            }
        }
    }

    #[test]
    fn test_run_batch_with_progress_observer() {
        let temp_dir = std::env::temp_dir().join("reto_core_test_progress_observer");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        let input_path = temp_dir.join("obs_strip.png");
        let output_path = temp_dir.join("output");

        let img = RgbaImage::from_pixel(300, 100, Rgba([200, 200, 200, 255]));
        img.save(&input_path).unwrap();

        let observer = std::sync::Arc::new(MockProgressObserver::default());
        let request = BatchProcessingRequest::new(vec![input_path], output_path, false)
            .with_progress_observer(observer.clone());

        let summary = run_batch(&request).expect("Batch with observer should succeed");
        assert_eq!(summary.total_input, 1);
        assert_eq!(summary.successful_count, 1);

        let started = observer.started.lock().unwrap().clone();
        assert_eq!(started.len(), 1);
        assert_eq!(started[0], ("obs_strip".to_string(), 1, 1));

        let completed = observer.completed.lock().unwrap().clone();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0], ("obs_strip".to_string(), 1, 1, true));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
