//! Visual inspection overlay rendering, keypoint overlays, and diagnostic tap plugins.

use crate::error::RoiError;
use crate::face::{FaceDetection, FaceDiagnosticTap};
use crate::feature::{
    AlignmentDiagnosticTap, FeatureFrame, FeatureMatch, FeatureTriplet, FramePair,
};
use crate::geom::{FrameRoiSet, PixelRect};
use image::{DynamicImage, Rgba, RgbaImage};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

/// Per-frame face detection record: `(frame_index, detected_faces, dominant_face_index)`.
pub type FrameFaceRecord = (usize, Vec<FaceDetection>, Option<usize>);

/// Pluggable diagnostic visualizer tap that accumulates detected faces and renders an annotated overlay.
#[derive(Debug)]
pub struct SaveFacesDiagnosticTap {
    output_path: PathBuf,
    image: DynamicImage,
    rois: FrameRoiSet,
    collected: Mutex<Vec<FrameFaceRecord>>,
}

impl SaveFacesDiagnosticTap {
    /// Creates a new `SaveFacesDiagnosticTap`.
    #[must_use]
    pub const fn new(output_path: PathBuf, image: DynamicImage, rois: FrameRoiSet) -> Self {
        Self {
            output_path,
            image,
            rois,
            collected: Mutex::new(Vec::new()),
        }
    }

    /// Renders and writes the accumulated face overlay image to disk.
    ///
    /// # Errors
    /// Returns [`crate::error::Error`] if mutex is poisoned, image encoding fails, or file cannot be written.
    pub fn finish(&self) -> crate::error::Result<()> {
        let overlay = {
            let collected = self.collected.lock().map_err(|_| {
                crate::error::Error::Unknown("Mutex poisoned in SaveFacesDiagnosticTap".to_string())
            })?;
            RoiVisualizer::render_faces_overlay(&self.image, &self.rois, &collected)
        };
        overlay.save(&self.output_path)?;
        tracing::info!(output_path = ?self.output_path, "Saved face detection visual overlay");
        Ok(())
    }
}

impl FaceDiagnosticTap for SaveFacesDiagnosticTap {
    fn on_faces_detected(
        &self,
        frame_idx: usize,
        _image: &image::RgbImage,
        detections: &[FaceDetection],
        dominant_idx: Option<usize>,
    ) {
        if let Ok(mut lock) = self.collected.lock() {
            lock.push((frame_idx, detections.to_vec(), dominant_idx));
        }
    }
}

/// Pluggable diagnostic visualizer tap that accumulates detected keypoints and renders an annotated overlay.
#[derive(Debug)]
pub struct SaveFeaturesDiagnosticTap {
    output_path: PathBuf,
    marker_color: Rgba<u8>,
    image: DynamicImage,
    rois: Option<FrameRoiSet>,
    collected: Mutex<Vec<FeatureFrame>>,
    faces: Mutex<Vec<FrameFaceRecord>>,
}

impl SaveFeaturesDiagnosticTap {
    /// Creates a new `SaveFeaturesDiagnosticTap` with output path, base image, and marker color.
    #[must_use]
    pub const fn new(output_path: PathBuf, image: DynamicImage, marker_color: Rgba<u8>) -> Self {
        Self {
            output_path,
            marker_color,
            image,
            rois: None,
            collected: Mutex::new(Vec::new()),
            faces: Mutex::new(Vec::new()),
        }
    }

    /// Associates Frame RoIs for global scan strip coordinate transformations.
    #[must_use]
    pub fn with_rois(mut self, rois: FrameRoiSet) -> Self {
        self.rois = Some(rois);
        self
    }

    /// Renders and writes the accumulated feature overlay image to disk.
    ///
    /// # Errors
    /// Returns [`crate::error::Error`] if mutex is poisoned, image encoding fails, or file cannot be written.
    pub fn finish(&self) -> crate::error::Result<()> {
        let mut overlay = {
            let frames = self.collected.lock().map_err(|_| {
                crate::error::Error::Unknown(
                    "Mutex poisoned in SaveFeaturesDiagnosticTap".to_string(),
                )
            })?;
            RoiVisualizer::render_features_overlay(&self.image, &frames, self.marker_color)
        };

        if let (Some(rois), Ok(faces)) = (&self.rois, self.faces.lock()) {
            if !faces.is_empty() {
                RoiVisualizer::draw_faces_onto_image(&mut overlay, rois, &faces);
            }
        }

        overlay.save(&self.output_path)?;
        tracing::info!(output_path = ?self.output_path, "Saved feature keypoint visual overlay");
        Ok(())
    }
}

impl AlignmentDiagnosticTap for SaveFeaturesDiagnosticTap {
    fn on_features_extracted(&self, _frame_idx: usize, features: &FeatureFrame) {
        if let Ok(mut lock) = self.collected.lock() {
            lock.push(features.clone());
        }
    }
}

impl FaceDiagnosticTap for SaveFeaturesDiagnosticTap {
    fn on_faces_detected(
        &self,
        frame_idx: usize,
        _image: &image::RgbImage,
        detections: &[FaceDetection],
        dominant_idx: Option<usize>,
    ) {
        if let Ok(mut lock) = self.faces.lock() {
            lock.push((frame_idx, detections.to_vec(), dominant_idx));
        }
    }
}

/// Pluggable diagnostic visualizer tap that accumulates pairwise matches and depth-consistent triplets
/// to render inspection overlays directly onto the full scan strip.
#[derive(Debug)]
pub struct SaveMatchesDiagnosticTap {
    matches_output_path: Option<PathBuf>,
    triplets_output_path: Option<PathBuf>,
    image: DynamicImage,
    rois: Option<FrameRoiSet>,
    frames: Mutex<Vec<FeatureFrame>>,
    pairwise_matches: Mutex<Vec<(FramePair, Vec<FeatureMatch>)>>,
    triplets: Mutex<Vec<FeatureTriplet>>,
    faces: Mutex<Vec<FrameFaceRecord>>,
}

impl SaveMatchesDiagnosticTap {
    /// Creates a new `SaveMatchesDiagnosticTap`.
    #[must_use]
    pub const fn new(
        matches_output_path: Option<PathBuf>,
        triplets_output_path: Option<PathBuf>,
        image: DynamicImage,
    ) -> Self {
        Self {
            matches_output_path,
            triplets_output_path,
            image,
            rois: None,
            frames: Mutex::new(Vec::new()),
            pairwise_matches: Mutex::new(Vec::new()),
            triplets: Mutex::new(Vec::new()),
            faces: Mutex::new(Vec::new()),
        }
    }

    /// Associates Frame RoIs for global scan strip coordinate transformations.
    #[must_use]
    pub fn with_rois(mut self, rois: FrameRoiSet) -> Self {
        self.rois = Some(rois);
        self
    }

    /// Sets pre-computed face detection records onto the diagnostic tap.
    pub fn add_faces(&self, frame_faces: &[FrameFaceRecord]) {
        if let Ok(mut lock) = self.faces.lock() {
            lock.extend_from_slice(frame_faces);
        }
    }

    /// Renders and writes the accumulated pairwise match, triplet, and face overlay images to disk.
    ///
    /// # Errors
    /// Returns [`crate::error::Error`] if mutex is poisoned or image encoding fails.
    #[allow(clippy::significant_drop_tightening)]
    pub fn finish(&self) -> crate::error::Result<()> {
        let frames = self.frames.lock().map_err(|_| {
            crate::error::Error::Unknown(
                "Mutex poisoned in SaveMatchesDiagnosticTap (frames)".to_string(),
            )
        })?;

        let triplets = self.triplets.lock().map_err(|_| {
            crate::error::Error::Unknown(
                "Mutex poisoned in SaveMatchesDiagnosticTap (triplets)".to_string(),
            )
        })?;

        let faces = self.faces.lock().map_err(|_| {
            crate::error::Error::Unknown(
                "Mutex poisoned in SaveMatchesDiagnosticTap (faces)".to_string(),
            )
        })?;

        if let Some(ref path) = self.matches_output_path {
            let mut overlay = if triplets.is_empty() {
                let pairwise = self.pairwise_matches.lock().map_err(|_| {
                    crate::error::Error::Unknown(
                        "Mutex poisoned in SaveMatchesDiagnosticTap (matches)".to_string(),
                    )
                })?;
                RoiVisualizer::render_matches_overlay(&self.image, &frames, &pairwise)
            } else {
                RoiVisualizer::render_triplets_overlay(&self.image, &frames, &triplets)
            };

            if let Some(ref rois) = self.rois {
                if !faces.is_empty() {
                    RoiVisualizer::draw_faces_onto_image(&mut overlay, rois, &faces);
                }
            }

            overlay.save(path)?;
            tracing::info!(output_path = ?path, "Saved feature and face correspondence visual overlay");
        }

        if let Some(ref path) = self.triplets_output_path {
            let mut overlay =
                RoiVisualizer::render_triplets_overlay(&self.image, &frames, &triplets);
            if let Some(ref rois) = self.rois {
                if !faces.is_empty() {
                    RoiVisualizer::draw_faces_onto_image(&mut overlay, rois, &faces);
                }
            }
            overlay.save(path)?;
            tracing::info!(output_path = ?path, "Saved depth-consistent triplet visual overlay");
        }

        drop(faces);
        drop(triplets);
        drop(frames);
        Ok(())
    }
}

impl AlignmentDiagnosticTap for SaveMatchesDiagnosticTap {
    fn on_features_extracted(&self, _frame_idx: usize, features: &FeatureFrame) {
        if let Ok(mut lock) = self.frames.lock() {
            lock.push(features.clone());
        }
    }

    fn on_matches_found(&self, pair: FramePair, matches: &[FeatureMatch]) {
        if let Ok(mut lock) = self.pairwise_matches.lock() {
            lock.push((pair, matches.to_vec()));
        }
    }

    fn on_triplets_verified(&self, triplets: &[FeatureTriplet]) {
        if let Ok(mut lock) = self.triplets.lock() {
            *lock = triplets.to_vec();
        }
    }
}

impl FaceDiagnosticTap for SaveMatchesDiagnosticTap {
    fn on_faces_detected(
        &self,
        frame_idx: usize,
        _image: &image::RgbImage,
        detections: &[FaceDetection],
        dominant_idx: Option<usize>,
    ) {
        if let Ok(mut lock) = self.faces.lock() {
            lock.push((frame_idx, detections.to_vec(), dominant_idx));
        }
    }
}

/// Pluggable visual inspection and debug rendering engine.
pub struct RoiVisualizer;

impl RoiVisualizer {
    /// Renders an annotated overlay image with highlighted frame borders, labels, and indices.
    ///
    /// # Cost
    /// Allocates an `RgbaImage` only when called (i.e. strictly when the user requests debug visualization).
    ///
    /// # Arguments
    /// * `image` - Source image to annotate.
    /// * `rois` - Frame bounding boxes to outline on the overlay.
    /// * `border_color` - RGBA color for the bounding box borders.
    /// * `border_thickness` - Border stroke thickness in pixels.
    ///
    /// # Examples
    /// ```
    /// use reto_core::{EvenSplitDetector, RoiDetectionConfig, RoiDetector, RoiVisualizer};
    /// use image::{DynamicImage, Rgba, RgbaImage};
    ///
    /// let img = DynamicImage::ImageRgba8(RgbaImage::from_pixel(300, 100, Rgba([255, 255, 255, 255])));
    /// let rois = EvenSplitDetector::new().detect(&img, &RoiDetectionConfig::default(), None).unwrap();
    /// let overlay = RoiVisualizer::render_overlay(&img, &rois, Rgba([0, 255, 0, 255]), 2);
    /// assert_eq!(overlay.dimensions(), (300, 100));
    /// ```
    #[must_use]
    #[tracing::instrument(skip(image, rois), level = "debug")]
    pub fn render_overlay(
        image: &DynamicImage,
        rois: &FrameRoiSet,
        border_color: Rgba<u8>,
        border_thickness: u32,
    ) -> RgbaImage {
        let mut overlay = image.to_rgba8();
        let (width, height) = overlay.dimensions();
        let size = crate::geom::Size2D::new(width, height);

        // Draw bounding boxes for each detected frame
        for frame in &rois.frames {
            let px = frame.bounds.to_pixel_rect(size);
            draw_bounding_box(&mut overlay, &px, border_color, border_thickness);
        }

        overlay
    }

    /// Extracts each detected frame as a standalone owned `DynamicImage` for export.
    ///
    /// # Arguments
    /// * `image` - Source image to crop from.
    /// * `rois` - Set of frames to extract.
    ///
    /// # Errors
    /// Returns [`RoiError`] if any frame index is invalid or slicing fails.
    pub fn extract_frame_images(
        image: &DynamicImage,
        rois: &FrameRoiSet,
    ) -> Result<Vec<DynamicImage>, RoiError> {
        let mut crops = Vec::with_capacity(rois.len());
        for idx in 0..rois.len() {
            let sub = rois.sub_image(image, idx)?;
            crops.push(sub.to_image().into());
        }
        Ok(crops)
    }

    /// Renders an annotated overlay with detected keypoints mapped to the full scan strip.
    ///
    /// # Arguments
    /// * `image` - Source full scan strip image.
    /// * `frames` - Slice of feature frames containing local keypoints and ROI bounds.
    /// * `marker_color` - Color to draw keypoints with.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn render_features_overlay(
        image: &DynamicImage,
        frames: &[FeatureFrame],
        marker_color: Rgba<u8>,
    ) -> RgbaImage {
        let mut overlay = image.to_rgba8();
        let (width, height) = overlay.dimensions();

        for frame in frames {
            let frame_w = frame.image_size.width as f32;
            let frame_h = frame.image_size.height as f32;
            if frame_w <= 0.0 || frame_h <= 0.0 {
                continue;
            }

            for kp in &frame.keypoints {
                let norm_x = kp.point.x / frame_w;
                let norm_y = kp.point.y / frame_h;
                let glob_norm_x = norm_x.mul_add(frame.roi_bounds.width, frame.roi_bounds.x);
                let glob_norm_y = norm_y.mul_add(frame.roi_bounds.height, frame.roi_bounds.y);
                let glob_x = glob_norm_x * width as f32;
                let glob_y = glob_norm_y * height as f32;

                draw_keypoint_marker(
                    &mut overlay,
                    glob_x.round() as i32,
                    glob_y.round() as i32,
                    3,
                    marker_color,
                );
            }
        }

        overlay
    }

    /// Renders an annotated overlay with pairwise correspondence vectors across sub-frames.
    ///
    /// # Arguments
    /// * `image` - Source full scan strip image.
    /// * `frames` - Slice of feature frames containing keypoints and ROI geometry.
    /// * `matches` - Pairwise match sets grouped by `(FramePair, Vec<FeatureMatch>)`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn render_matches_overlay(
        image: &DynamicImage,
        frames: &[FeatureFrame],
        matches: &[(FramePair, Vec<FeatureMatch>)],
    ) -> RgbaImage {
        let mut overlay = image.to_rgba8();
        let (width, height) = overlay.dimensions();

        // Precompute global pixel coordinates for all keypoints across frames in a single vectorized pass
        let global_pts_by_frame: HashMap<usize, Vec<(i32, i32)>> = frames
            .iter()
            .map(|frame| {
                let frame_w = frame.image_size.width as f32;
                let frame_h = frame.image_size.height as f32;
                let pts = if frame_w > 0.0 && frame_h > 0.0 {
                    frame
                        .keypoints
                        .iter()
                        .map(|kp| {
                            let norm_x = kp.point.x / frame_w;
                            let norm_y = kp.point.y / frame_h;
                            let gx = norm_x.mul_add(frame.roi_bounds.width, frame.roi_bounds.x)
                                * width as f32;
                            let gy = norm_y.mul_add(frame.roi_bounds.height, frame.roi_bounds.y)
                                * height as f32;
                            (gx.round() as i32, gy.round() as i32)
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                (frame.frame_index, pts)
            })
            .collect();

        for &((idx_a, idx_b), ref match_vec) in matches {
            // Distinct colors by baseline:
            // (0, 1) -> Emerald Green
            // (1, 2) -> Cyan
            // (0, 2) -> Amber
            let line_color = match (idx_a, idx_b) {
                (0, 1) | (1, 0) => Rgba([0, 255, 128, 220]),
                (1, 2) | (2, 1) => Rgba([0, 200, 255, 220]),
                _ => Rgba([255, 190, 0, 220]),
            };

            let Some(pts_a) = global_pts_by_frame.get(&idx_a) else {
                continue;
            };
            let Some(pts_b) = global_pts_by_frame.get(&idx_b) else {
                continue;
            };

            for m in match_vec {
                if let (Some(&(x0, y0)), Some(&(x1, y1))) =
                    (pts_a.get(m.index_a), pts_b.get(m.index_b))
                {
                    // Connecting line drawing disabled to reduce visual clutter; preserved for future diagnostic toggles:
                    // draw_line(&mut overlay, x0, y0, x1, y1, line_color);
                    draw_keypoint_marker(&mut overlay, x0, y0, 2, line_color);
                    draw_keypoint_marker(&mut overlay, x1, y1, 2, line_color);
                }
            }
        }

        overlay
    }

    /// Renders an annotated overlay with depth-consistent feature triplets across all three views.
    ///
    /// Triplet vectors connecting Frame 0 -> Frame 1 -> Frame 2 are color-coded by disparity (depth Z).
    ///
    /// # Arguments
    /// * `image` - Source full scan strip image.
    /// * `frames` - Slice of feature frames containing keypoints and ROI bounds.
    /// * `triplets` - Slice of verified depth-consistent feature triplets.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn render_triplets_overlay(
        image: &DynamicImage,
        frames: &[FeatureFrame],
        triplets: &[FeatureTriplet],
    ) -> RgbaImage {
        let mut overlay = image.to_rgba8();
        let (width, height) = overlay.dimensions();

        // Precompute global pixel coordinates for all keypoints across frames in a single vectorized pass
        let global_pts_by_frame: HashMap<usize, Vec<(i32, i32)>> = frames
            .iter()
            .map(|frame| {
                let frame_w = frame.image_size.width as f32;
                let frame_h = frame.image_size.height as f32;
                let pts = if frame_w > 0.0 && frame_h > 0.0 {
                    frame
                        .keypoints
                        .iter()
                        .map(|kp| {
                            let norm_x = kp.point.x / frame_w;
                            let norm_y = kp.point.y / frame_h;
                            let gx = norm_x.mul_add(frame.roi_bounds.width, frame.roi_bounds.x)
                                * width as f32;
                            let gy = norm_y.mul_add(frame.roi_bounds.height, frame.roi_bounds.y)
                                * height as f32;
                            (gx.round() as i32, gy.round() as i32)
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                (frame.frame_index, pts)
            })
            .collect();

        // 1. Draw ALL detected keypoints across all frames as background dots
        let all_points_color = Rgba([220, 120, 255, 170]);
        for pts in global_pts_by_frame.values() {
            for &(gx, gy) in pts {
                draw_keypoint_marker(&mut overlay, gx, gy, 3, all_points_color);
            }
        }

        // Determine disparity min and max across all triplets for dynamic linear normalization,
        // using the 3-frame average disparity magnitude matching gif.rs depth clustering.
        let max_disparity = triplets
            .iter()
            .map(|t| (t.disparity_01 + t.disparity_12) * 0.5)
            .fold(0.0_f32, f32::max)
            .max(10.0_f32);

        // 2. Draw depth-colored keypoint markers for verified triplets
        let pts0 = global_pts_by_frame.get(&0);
        let pts1 = global_pts_by_frame.get(&1);
        let pts2 = global_pts_by_frame.get(&2);

        if let (Some(p0), Some(p1), Some(p2)) = (pts0, pts1, pts2) {
            for trip in triplets {
                if let (Some(&(x0, y0)), Some(&(x1, y1)), Some(&(x2, y2))) = (
                    p0.get(trip.index_0),
                    p1.get(trip.index_1),
                    p2.get(trip.index_2),
                ) {
                    // Continuous Turbo/Jet-like linear heatmap color variation matching histogram bins:
                    // Far background (small disparity) -> Deep Blue / Cyan
                    // Midground -> Emerald Green / Yellow
                    // Close foreground (high disparity) -> Orange / Vivid Crimson Red
                    let avg_disp = (trip.disparity_01 + trip.disparity_12) * 0.5;
                    let t = (avg_disp / max_disparity).clamp(0.0, 1.0);
                    let color = disparity_to_heatmap_rgba(t);

                    // Connecting lines disabled to reduce visual clutter; preserved for future diagnostic toggles:
                    // draw_thick_line(&mut overlay, x0, y0, x1, y1, color, 4);
                    // draw_thick_line(&mut overlay, x1, y1, x2, y2, color, 4);

                    // Emphasize the 3-frame consistent vertices
                    draw_keypoint_marker(&mut overlay, x0, y0, 6, color);
                    draw_keypoint_marker(&mut overlay, x1, y1, 6, color);
                    draw_keypoint_marker(&mut overlay, x2, y2, 6, color);
                }
            }
        }

        overlay
    }

    /// Draws detected face bounding boxes and crosshair landmarks directly onto an existing overlay canvas.
    ///
    /// Non-dominant faces are outlined in Vivid Cyan (`[0, 200, 255, 220]`), while the selected
    /// dominant portrait subject is highlighted in thick Emerald Green (`[0, 255, 128, 255]`).
    ///
    /// # Arguments
    /// * `overlay` - Target RGBA canvas image to draw annotations upon.
    /// * `rois` - Set of frame ROIs defining sub-frame placement on the scan strip.
    /// * `frame_faces` - Per-frame face detection records with `(frame_idx, detections, dominant_idx)`.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    pub fn draw_faces_onto_image(
        overlay: &mut RgbaImage,
        rois: &FrameRoiSet,
        frame_faces: &[FrameFaceRecord],
    ) {
        let (width, height) = overlay.dimensions();
        let strip_size = crate::geom::Size2D::new(width, height);

        let candidate_box_color = Rgba([0, 210, 255, 240]); // Vivid Electric Cyan
        let dominant_box_color = Rgba([50, 255, 50, 255]); // Vibrant Lime Green for selected dominant face
        let landmark_color = Rgba([180, 100, 255, 255]); // Royal Violet '+' for non-dominant faces
        let dominant_landmark_color = Rgba([255, 45, 45, 255]); // Crimson Red '+' for dominant face

        for &(frame_idx, ref detections, dominant_idx) in frame_faces {
            let Some(frame_roi) = rois.frames.get(frame_idx) else {
                continue;
            };
            let frame_bounds = frame_roi.bounds;

            for (face_idx, face) in detections.iter().enumerate() {
                let is_dominant = dominant_idx == Some(face_idx);
                let box_color = if is_dominant {
                    dominant_box_color
                } else {
                    candidate_box_color
                };
                let thickness = if is_dominant { 4 } else { 2 };

                // Map face NormalizedRect in local frame space [0, 1] to global scan strip pixel rect
                let glob_x = face.bbox.x.mul_add(frame_bounds.width, frame_bounds.x);
                let glob_y = face.bbox.y.mul_add(frame_bounds.height, frame_bounds.y);
                let glob_w = face.bbox.width * frame_bounds.width;
                let glob_h = face.bbox.height * frame_bounds.height;

                if let Ok(glob_norm_rect) =
                    crate::geom::NormalizedRect::new(glob_x, glob_y, glob_w, glob_h)
                {
                    let px_rect = glob_norm_rect.to_pixel_rect(strip_size);
                    draw_bounding_box(overlay, &px_rect, box_color, thickness);
                }

                // Draw 5-point crosshair '+' landmarks distinct from circular keypoints (enlarged for visibility)
                let (lm_color, lm_arm, lm_thick) = if is_dominant {
                    (dominant_landmark_color, 14, 4)
                } else {
                    (landmark_color, 10, 3)
                };

                for pt in &face.landmarks {
                    let glob_lm_x =
                        pt.x.mul_add(frame_bounds.width, frame_bounds.x) * (width as f32);
                    let glob_lm_y =
                        pt.y.mul_add(frame_bounds.height, frame_bounds.y) * (height as f32);

                    draw_crosshair_marker(
                        overlay,
                        glob_lm_x.round() as i32,
                        glob_lm_y.round() as i32,
                        lm_arm,
                        lm_thick,
                        lm_color,
                    );
                }
            }
        }
    }

    /// Renders an annotated overlay with detected face bounding boxes, 5-point crosshair landmarks, and dominant face highlighting.
    ///
    /// Non-dominant faces are outlined in Vivid Cyan (`[0, 200, 255, 220]`), while the selected
    /// dominant portrait subject is highlighted in thick Emerald Green (`[0, 255, 128, 255]`).
    ///
    /// # Arguments
    /// * `image` - Source full scan strip image.
    /// * `rois` - Set of frame ROIs defining sub-frame placement on the scan strip.
    /// * `frame_faces` - Per-frame face detection records with `(frame_idx, detections, dominant_idx)`.
    #[must_use]
    pub fn render_faces_overlay(
        image: &DynamicImage,
        rois: &FrameRoiSet,
        frame_faces: &[FrameFaceRecord],
    ) -> RgbaImage {
        let mut overlay = image.to_rgba8();
        Self::draw_faces_onto_image(&mut overlay, rois, frame_faces);
        overlay
    }
}

/// Maps a normalized scalar disparity value $t \in [0.0, 1.0]$ to a high-contrast linear Turbo/Jet heatmap RGBA color.
///
/// * $t \approx 0.0$ (Infinity / Background): Deep Indigo Blue `[30, 80, 255]`
/// * $t \approx 0.25$: Vivid Cyan `[0, 220, 240]`
/// * $t \approx 0.50$ (Midground): Lime Green `[40, 230, 80]`
/// * $t \approx 0.75$: Amber Yellow `[255, 210, 0]`
/// * $t \approx 1.0$ (Near Foreground): Vivid Crimson Red `[255, 35, 35]`
#[inline]
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn disparity_to_heatmap_rgba(t: f32) -> Rgba<u8> {
    let t = t.clamp(0.0, 1.0);

    // Multi-stop piecewise linear interpolation
    let (r, g, b) = if t < 0.25 {
        // [0.0, 0.25]: Blue -> Cyan
        let factor = t / 0.25;
        (
            30.0 * (1.0 - factor),
            140.0_f32.mul_add(factor, 80.0),
            255.0,
        )
    } else if t < 0.5 {
        // [0.25, 0.5]: Cyan -> Green
        let factor = (t - 0.25) / 0.25;
        (
            40.0 * factor,
            10.0_f32.mul_add(factor, 220.0),
            255.0_f32.mul_add(1.0 - factor, 80.0 * factor),
        )
    } else if t < 0.75 {
        // [0.5, 0.75]: Green -> Yellow
        let factor = (t - 0.5) / 0.25;
        (
            215.0_f32.mul_add(factor, 40.0),
            20.0_f32.mul_add(-factor, 230.0),
            80.0 * (1.0 - factor),
        )
    } else {
        // [0.75, 1.0]: Yellow -> Red
        let factor = (t - 0.75) / 0.25;
        (
            255.0,
            210.0_f32.mul_add(1.0 - factor, 35.0 * factor),
            35.0 * factor,
        )
    };

    Rgba([r.round() as u8, g.round() as u8, b.round() as u8, 255])
}

fn draw_bounding_box(img: &mut RgbaImage, rect: &PixelRect, color: Rgba<u8>, thickness: u32) {
    let (max_x, max_y) = img.dimensions();
    for t in 0..thickness {
        let y_top = (rect.y.saturating_add(t)).min(max_y.saturating_sub(1));
        let y_bot = (rect.y.saturating_add(rect.height))
            .saturating_sub(1 + t)
            .min(max_y.saturating_sub(1));
        let x_left = (rect.x.saturating_add(t)).min(max_x.saturating_sub(1));
        let x_right = (rect.x.saturating_add(rect.width))
            .saturating_sub(1 + t)
            .min(max_x.saturating_sub(1));

        for x in x_left..=x_right {
            img.put_pixel(x, y_top, color);
            img.put_pixel(x, y_bot, color);
        }
        for y in y_top..=y_bot {
            img.put_pixel(x_left, y, color);
            img.put_pixel(x_right, y, color);
        }
    }
}

#[allow(clippy::cast_sign_loss)]
fn draw_keypoint_marker(
    img: &mut RgbaImage,
    center_x: i32,
    center_y: i32,
    radius: i32,
    color: Rgba<u8>,
) {
    let (max_w, max_h) = img.dimensions();
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let px = center_x + dx;
            let py = center_y + dy;
            if px >= 0 && py >= 0 && (px as u32) < max_w && (py as u32) < max_h {
                img.put_pixel(px as u32, py as u32, color);
            }
        }
    }
}

#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
fn draw_crosshair_marker(
    img: &mut RgbaImage,
    center_x: i32,
    center_y: i32,
    arm_length: i32,
    thickness: u32,
    color: Rgba<u8>,
) {
    let (max_w, max_h) = img.dimensions();

    // Pass 1: High-contrast dark outline halo for clear visibility on bright or dark backgrounds
    let outline_arm = arm_length + 2;
    let outline_thick = thickness + 2;
    let outline_half = (outline_thick / 2) as i32;
    let outline_color = Rgba([0, 0, 0, 220]);

    for dx in -outline_arm..=outline_arm {
        for dy in -outline_half..=(outline_thick as i32 - 1 - outline_half) {
            let px = center_x + dx;
            let py = center_y + dy;
            if px >= 0 && py >= 0 && (px as u32) < max_w && (py as u32) < max_h {
                img.put_pixel(px as u32, py as u32, outline_color);
            }
        }
    }
    for dy in -outline_arm..=outline_arm {
        for dx in -outline_half..=(outline_thick as i32 - 1 - outline_half) {
            let px = center_x + dx;
            let py = center_y + dy;
            if px >= 0 && py >= 0 && (px as u32) < max_w && (py as u32) < max_h {
                img.put_pixel(px as u32, py as u32, outline_color);
            }
        }
    }

    // Pass 2: Colored foreground crosshair
    let half_thick = (thickness / 2) as i32;

    for dx in -arm_length..=arm_length {
        for dy in -half_thick..=(thickness as i32 - 1 - half_thick) {
            let px = center_x + dx;
            let py = center_y + dy;
            if px >= 0 && py >= 0 && (px as u32) < max_w && (py as u32) < max_h {
                img.put_pixel(px as u32, py as u32, color);
            }
        }
    }

    for dy in -arm_length..=arm_length {
        for dx in -half_thick..=(thickness as i32 - 1 - half_thick) {
            let px = center_x + dx;
            let py = center_y + dy;
            if px >= 0 && py >= 0 && (px as u32) < max_w && (py as u32) < max_h {
                img.put_pixel(px as u32, py as u32, color);
            }
        }
    }
}

/// Draws an anti-aliased or standard line between two points using Bresenham's algorithm.
#[allow(
    dead_code,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::similar_names
)]
fn draw_line(img: &mut RgbaImage, x0: i32, y0: i32, x1: i32, y1: i32, color: Rgba<u8>) {
    let (max_w, max_h) = img.dimensions();
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    let mut curr_x = x0;
    let mut curr_y = y0;

    loop {
        if curr_x >= 0 && curr_y >= 0 && (curr_x as u32) < max_w && (curr_y as u32) < max_h {
            img.put_pixel(curr_x as u32, curr_y as u32, color);
        }

        if curr_x == x1 && curr_y == y1 {
            break;
        }

        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            curr_x += sx;
        }
        if e2 <= dx {
            err += dx;
            curr_y += sy;
        }
    }
}

/// Draws a thick line between two points by sweeping parallel Bresenham strokes.
#[allow(dead_code, clippy::cast_possible_wrap)]
fn draw_thick_line(
    img: &mut RgbaImage,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: Rgba<u8>,
    thickness: u32,
) {
    if thickness <= 1 {
        draw_line(img, x0, y0, x1, y1, color);
        return;
    }

    let half = (thickness / 2) as i32;
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();

    if dx >= dy {
        // Mostly horizontal: sweep vertical offsets
        for offset in -half..=(thickness as i32 - 1 - half) {
            draw_line(img, x0, y0 + offset, x1, y1 + offset, color);
        }
    } else {
        // Mostly vertical: sweep horizontal offsets
        for offset in -half..=(thickness as i32 - 1 - half) {
            draw_line(img, x0 + offset, y0, x1 + offset, y1, color);
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::detector::{EvenSplitDetector, RoiDetectionConfig, RoiDetector};
    use image::{DynamicImage, GenericImageView, RgbaImage};

    #[test]
    fn test_visualizer_overlay_and_crops() {
        let base_img = RgbaImage::from_pixel(300, 100, Rgba([200, 200, 200, 255]));
        let dynamic_base = DynamicImage::ImageRgba8(base_img);
        let detector = EvenSplitDetector::new();
        let config = RoiDetectionConfig::default();
        let rois = detector
            .detect(&dynamic_base, &config, None)
            .expect("Detection succeeds");

        // Render overlay
        let overlay =
            RoiVisualizer::render_overlay(&dynamic_base, &rois, Rgba([255, 0, 0, 255]), 2);
        assert_eq!(overlay.dimensions(), (300, 100));
        // Top-left pixel of frame 0 border should be red
        assert_eq!(overlay.get_pixel(0, 0), &Rgba([255, 0, 0, 255]));

        // Extract crops
        let crops = RoiVisualizer::extract_frame_images(&dynamic_base, &rois)
            .expect("Crops extract successfully");
        assert_eq!(crops.len(), 3);
        for crop in crops {
            assert_eq!(crop.dimensions(), (100, 100));
        }

        // Render features overlay
        let frame0 = FeatureFrame::new(
            0,
            rois.frames[0].bounds,
            crate::geom::Size2D::new(100, 100),
            vec![crate::feature::KeyPoint::new(
                crate::geom::Point2D::new(50.0, 50.0),
                0.9,
                None,
            )],
        );
        let frame1 = FeatureFrame::new(
            1,
            rois.frames[1].bounds,
            crate::geom::Size2D::new(100, 100),
            vec![crate::feature::KeyPoint::new(
                crate::geom::Point2D::new(50.0, 50.0),
                0.9,
                None,
            )],
        );

        let feat_overlay = RoiVisualizer::render_features_overlay(
            &dynamic_base,
            &[frame0.clone(), frame1.clone()],
            Rgba([0, 255, 0, 255]),
        );
        assert_eq!(feat_overlay.dimensions(), (300, 100));

        let frame2 = FeatureFrame::new(
            2,
            rois.frames[2].bounds,
            crate::geom::Size2D::new(100, 100),
            vec![crate::feature::KeyPoint::new(
                crate::geom::Point2D::new(50.0, 50.0),
                0.9,
                None,
            )],
        );

        // Test matches overlay
        let matches = vec![
            ((0, 1), vec![FeatureMatch::new(0, 0, 0.95)]),
            ((1, 2), vec![FeatureMatch::new(0, 0, 0.90)]),
        ];
        let match_overlay = RoiVisualizer::render_matches_overlay(
            &dynamic_base,
            &[frame0.clone(), frame1.clone(), frame2.clone()],
            &matches,
        );
        assert_eq!(match_overlay.dimensions(), (300, 100));

        // Test triplets overlay
        let triplets = vec![FeatureTriplet {
            index_0: 0,
            index_1: 0,
            index_2: 0,
            confidence: 0.92,
            disparity_01: 20.0,
            disparity_12: 20.0,
            cascade_error: 0.0,
        }];
        let trip_overlay = RoiVisualizer::render_triplets_overlay(
            &dynamic_base,
            &[frame0, frame1, frame2],
            &triplets,
        );
        assert_eq!(trip_overlay.dimensions(), (300, 100));

        // Test faces overlay
        let lm = [crate::geom::Point2D::new(0.5, 0.5); 5];
        let face_candidate = FaceDetection::new(
            crate::geom::NormalizedRect::new(0.1, 0.1, 0.3, 0.3).unwrap(),
            0.85,
            lm,
        );
        let face_dominant = FaceDetection::new(
            crate::geom::NormalizedRect::new(0.4, 0.2, 0.4, 0.4).unwrap(),
            0.95,
            lm,
        );

        let frame_faces = vec![(1, vec![face_candidate, face_dominant], Some(1))];
        let face_overlay = RoiVisualizer::render_faces_overlay(&dynamic_base, &rois, &frame_faces);
        assert_eq!(face_overlay.dimensions(), (300, 100));
    }
}
