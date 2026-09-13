#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::must_use_candidate,
    clippy::manual_midpoint,
    clippy::items_after_statements,
    clippy::unreadable_literal,
    clippy::missing_const_for_fn,
    clippy::missing_errors_doc,
    clippy::or_fun_call,
    clippy::redundant_closure_for_method_calls
)]
use image::DynamicImage;
use image::math::Rect;
use rayon::prelude::*;
use crate::Result;

/// Configuration for the image splitting process.
#[derive(Debug, Clone, Copy)]
pub struct SplitConfig {
    /// Expected number of output frames (typically 3).
    pub expected_frames: usize,
    /// Percentile to use for adaptive thresholding (0.0 to 1.0).
    pub percentile: f32,
    /// Minimum width of a gutter.
    pub min_width: usize,
    /// Whether to look for light gutters (hills) instead of dark gutters (valleys).
    pub light_gutters: bool,
}

/// Context for performing the image splitting.
pub struct SplitContext {
    /// Configuration for the splitting process.
    pub config: SplitConfig,
    /// Input image to split.
    pub input_image: DynamicImage,
    /// Detected frame bounding boxes.
    pub detected_rects: Vec<Rect>,
}

impl SplitContext {
    /// Creates a new `SplitContext` with the given input image and configuration.
    pub fn new(img: DynamicImage, config: SplitConfig) -> Self {
        Self {
            config,
            input_image: img,
            detected_rects: Vec::new(),
        }
    }

    /// Runs the splitting algorithm on the input image.
    pub fn run(&mut self) -> Result<()> {
        let width = self.input_image.width() as usize;
        let height = self.input_image.height() as usize;
        if width == 0 || height == 0 {
            return Err(crate::error::Error::Splitting("Image dimensions cannot be zero".to_string()));
        }

        let is_horizontal = width >= height;

        // Calculate VPP and HPP using compute_profile
        let profile = compute_profile(&self.input_image);

        // Detect gutters along the main splitting axis (VPP for horizontal, HPP for vertical) using find_gutters
        let main_profile = if is_horizontal {
            &profile.vpp
        } else {
            &profile.hpp
        };

        let gutter_config = GutterConfig {
            expected_count: Some(self.config.expected_frames.saturating_sub(1)),
            light_gutters: self.config.light_gutters,
            min_width: self.config.min_width,
            percentile: self.config.percentile,
        };

        let gutters = find_gutters(main_profile, &gutter_config);

        // Find the active boundaries (first and last non-gutter indices) on both axes
        let get_active_bounds = |prof: &[u64], percentile: f32, light_gutters: bool| -> (usize, usize) {
            if prof.is_empty() {
                return (0, 0);
            }
            let smoothed = smooth_profile(prof, 7);
            let threshold = calculate_percentile(&smoothed, percentile);
            let is_gutter = |val: u64| {
                if light_gutters {
                    val >= threshold
                } else {
                    val <= threshold
                }
            };
            let start = smoothed.iter().position(|&val| !is_gutter(val)).unwrap_or(0);
            let end = smoothed.iter().rposition(|&val| !is_gutter(val)).unwrap_or(smoothed.len().saturating_sub(1));
            (start, end)
        };

        let (x_start, x_end) = get_active_bounds(&profile.vpp, self.config.percentile, self.config.light_gutters);
        let (y_start, y_end) = get_active_bounds(&profile.hpp, self.config.percentile, self.config.light_gutters);

        let mut rects = Vec::with_capacity(3);

        // Sort gutter centers to be safe
        let mut centers: Vec<usize> = gutters.iter().map(|g| g.center()).collect();
        centers.sort_unstable();

        if gutters.len() == 2 {
            let c1 = centers[0];
            let c2 = centers[1];

            if is_horizontal {
                let h = y_end.saturating_add(1).saturating_sub(y_start);
                // Frame 1: x = x_start, width = c1 - x_start
                rects.push(create_rect(x_start, y_start, c1.saturating_sub(x_start), h, width, height));
                // Frame 2: x = c1, width = c2 - c1
                rects.push(create_rect(c1, y_start, c2.saturating_sub(c1), h, width, height));
                // Frame 3: x = c2, width = (x_end + 1) - c2
                rects.push(create_rect(c2, y_start, x_end.saturating_add(1).saturating_sub(c2), h, width, height));
            } else {
                let w = x_end.saturating_add(1).saturating_sub(x_start);
                // Frame 1: y = y_start, height = c1 - y_start
                rects.push(create_rect(x_start, y_start, w, c1.saturating_sub(y_start), width, height));
                // Frame 2: y = c1, height = c2 - c1
                rects.push(create_rect(x_start, c1, w, c2.saturating_sub(c1), width, height));
                // Frame 3: y = c2, height = (y_end + 1) - c2
                rects.push(create_rect(x_start, c2, w, y_end.saturating_add(1).saturating_sub(c2), width, height));
            }
        } else {
            // Fallback: divide active range on main axis into 3 equal sections
            if is_horizontal {
                let active_len = x_end.saturating_add(1).saturating_sub(x_start);
                let w = active_len / 3;
                let h = y_end.saturating_add(1).saturating_sub(y_start);

                // Frame 1: [active_start, active_start + w]
                rects.push(create_rect(x_start, y_start, w, h, width, height));
                // Frame 2: [active_start + w, active_start + 2*w]
                rects.push(create_rect(x_start.saturating_add(w), y_start, w, h, width, height));
                // Frame 3: [active_start + 2*w, active_end + 1]
                rects.push(create_rect(
                    x_start.saturating_add(2 * w),
                    y_start,
                    x_end.saturating_add(1).saturating_sub(x_start.saturating_add(2 * w)),
                    h,
                    width,
                    height,
                ));
            } else {
                let active_len = y_end.saturating_add(1).saturating_sub(y_start);
                let w = active_len / 3;
                let w_h = x_end.saturating_add(1).saturating_sub(x_start);

                // Frame 1: [active_start, active_start + w]
                rects.push(create_rect(x_start, y_start, w_h, w, width, height));
                // Frame 2: [active_start + w, active_start + 2*w]
                rects.push(create_rect(x_start, y_start.saturating_add(w), w_h, w, width, height));
                // Frame 3: [active_start + 2*w, active_end + 1]
                rects.push(create_rect(
                    x_start,
                    y_start.saturating_add(2 * w),
                    w_h,
                    y_end.saturating_add(1).saturating_sub(y_start.saturating_add(2 * w)),
                    width,
                    height,
                ));
            }
        }

        self.detected_rects = rects;
        Ok(())
    }
}

fn create_rect(x: usize, y: usize, w: usize, h: usize, img_width: usize, img_height: usize) -> Rect {
    let x = x.min(img_width.saturating_sub(1)) as u32;
    let y = y.min(img_height.saturating_sub(1)) as u32;
    let max_w = (img_width as u32).saturating_sub(x);
    let max_h = (img_height as u32).saturating_sub(y);
    let width = (w as u32).min(max_w);
    let height = (h as u32).min(max_h);
    Rect { x, y, width, height }
}


/// Intermediate data for debugging.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectionProfile {
    /// Vertical Projection Profile: Sum of pixel values along each column.
    pub vpp: Vec<u64>,
    /// Horizontal Projection Profile: Sum of pixel values along each row.
    pub hpp: Vec<u64>,
    /// Width of the analyzed image.
    pub width: u32,
    /// Height of the analyzed image.
    pub height: u32,
}

/// Represents a detected gutter in a projection profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gutter {
    /// Starting index of the gutter (inclusive).
    pub start: usize,
    /// Ending index of the gutter (exclusive).
    pub end: usize,
}

impl Gutter {
    /// Returns the center position of the gutter.
    pub fn center(&self) -> usize {
        (self.start + self.end) / 2
    }
}

/// Configuration for gutter detection.
#[derive(Debug, Clone, Copy)]
pub struct GutterConfig {
    /// Expected number of gutters.
    pub expected_count: Option<usize>,
    /// Whether to look for light gutters (hills) instead of dark gutters (valleys).
    pub light_gutters: bool,
    /// Minimum width of a gutter.
    pub min_width: usize,
    /// Percentile to use for adaptive thresholding (0.0 to 1.0).
    pub percentile: f32,
}

/// Detects gutters in a 1D projection profile.
pub fn find_gutters(profile: &[u64], config: &GutterConfig) -> Vec<Gutter> {
    if profile.is_empty() {
        return Vec::new();
    }

    // 1. Smoothing (window size ~5-10)
    const WINDOW_SIZE: usize = 7;
    let profile_smoothed = smooth_profile(profile, WINDOW_SIZE);

    // 2. Adaptive Thresholding
    let threshold = calculate_percentile(&profile_smoothed, config.percentile);

    // 3. Candidate Detection
    let mut candidates = Vec::new();
    let mut cur_start = None;

    for (i, &val) in profile_smoothed.iter().enumerate() {
        let is_match = if config.light_gutters {
            val >= threshold
        } else {
            val <= threshold
        };

        if is_match {
            if cur_start.is_none() {
                cur_start = Some(i);
            }
        } else if let Some(start) = cur_start {
            candidates.push(Gutter { start, end: i });
            cur_start = None;
        }
    }
    if let Some(start) = cur_start {
        candidates.push(Gutter { start, end: profile.len() });
    }

    // 4. Filtering & Selection
    let mut gutters: Vec<_> = candidates
        .into_iter()
        .filter(|g| (g.end - g.start) >= config.min_width)
        .collect();

    if let Some(expected) = config.expected_count {
        if gutters.len() > expected {
            // Sort by "quality" (average intensity relative to threshold)
            gutters.sort_by(|a, b| {
                let avg_a = average_u64(&profile_smoothed[a.start..a.end]);
                let avg_b = average_u64(&profile_smoothed[b.start..b.end]);
                if config.light_gutters {
                    // For light gutters, higher intensity is better
                    avg_b.partial_cmp(&avg_a).unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    // For dark gutters, lower intensity is better
                    avg_a.partial_cmp(&avg_b).unwrap_or(std::cmp::Ordering::Equal)
                }
            });
            gutters.truncate(expected);
            gutters.sort_by_key(|g| g.start);
        }
    }

    gutters
}

fn smooth_profile(data: &[u64], window: usize) -> Vec<u64> {
    let n = data.len();
    let mut smoothed = Vec::with_capacity(n);
    let half = (window / 2) as isize;

    for i in 0..n {
        let start = (i as isize - half).max(0) as usize;
        let end = (i as isize + half + 1).min(n as isize) as usize;
        let sum: u64 = data[start..end].iter().sum();
        smoothed.push(sum / (end - start) as u64);
    }
    smoothed
}

fn calculate_percentile(data: &[u64], p: f32) -> u64 {
    if data.is_empty() { return 0; }
    let mut sorted = data.to_vec();
    sorted.sort_unstable();
    let idx = ((sorted.len() - 1) as f32 * p).clamp(0.0, (sorted.len() - 1) as f32) as usize;
    sorted[idx]
}

fn average_u64(data: &[u64]) -> f64 {
    if data.is_empty() { return 0.0; }
    data.iter().sum::<u64>() as f64 / data.len() as f64
}

/// Computes the vertical and horizontal projection profiles of an image.
///
/// VPP: Sum of pixel values along each column.
/// HPP: Sum of pixel values along each row.
///
/// The image is converted to grayscale (Luma8) before processing.
pub fn compute_profile(img: &DynamicImage) -> ProjectionProfile {
    let luma = img.to_luma8();
    let (width, height) = luma.dimensions();

    if width == 0 || height == 0 {
        return ProjectionProfile {
            vpp: vec![0; width as usize],
            hpp: vec![0; height as usize],
            width,
            height,
        };
    }

    let buf = luma.as_raw();
    let w = width as usize;

    // Parallel HPP computation (row-wise sum)
    let hpp: Vec<u64> = buf
        .par_chunks_exact(w)
        .map(|row| row.iter().map(|&p| u64::from(p)).sum())
        .collect();

    // Parallel VPP computation (row-wise iteration with thread-local accumulators)
    let vpp: Vec<u64> = buf
        .par_chunks_exact(w)
        .fold(
            || vec![0u64; w],
            |mut acc, row| {
                for (acc_val, &pixel) in acc.iter_mut().zip(row.iter()) {
                    *acc_val += u64::from(pixel);
                }
                acc
            },
        )
        .reduce(
            || vec![0u64; w],
            |mut a, b| {
                for (a_val, &b_val) in a.iter_mut().zip(b.iter()) {
                    *a_val += b_val;
                }
                a
            },
        );

    ProjectionProfile {
        vpp,
        hpp,
        width,
        height,
    }
}

/// Splits an image into sub-images based on the provided rects.
///
/// # Errors
///
/// This function does not currently fail, but returns a Result for API consistency.
pub fn split_image(img: &DynamicImage, rects: &[Rect]) -> Result<Vec<DynamicImage>> {
    let mut sub_images = Vec::with_capacity(rects.len());
    let (img_w, img_h) = (img.width(), img.height());

    for r in rects {
        // Clamp bounds to prevent panics in crop_imm
        let x = r.x.min(img_w.saturating_sub(1));
        let y = r.y.min(img_h.saturating_sub(1));
        let w = r.width.min(img_w.saturating_sub(x));
        let h = r.height.min(img_h.saturating_sub(y));
        
        let cropped = img.crop_imm(x, y, w, h);
        sub_images.push(cropped);
    }
    Ok(sub_images)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, GrayImage, Luma};

    #[test]
    fn test_compute_profile_white() {
        let width = 100;
        let height = 100;
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(width, height, Luma([255])));
        
        let profile = compute_profile(&img);
        
        assert_eq!(profile.width, width);
        assert_eq!(profile.height, height);
        assert_eq!(profile.vpp.len(), width as usize);
        assert_eq!(profile.hpp.len(), height as usize);
        
        // Each entry in VPP should be sum of 255 over 100 pixels in a column
        let expected_v = 255 * u64::from(height);
        // Each entry in HPP should be sum of 255 over 100 pixels in a row
        let expected_h = 255 * u64::from(width);
        
        for (i, &val) in profile.vpp.iter().enumerate() {
            assert_eq!(val, expected_v, "VPP mismatch at index {i}");
        }
        for (i, &val) in profile.hpp.iter().enumerate() {
            assert_eq!(val, expected_h, "HPP mismatch at index {i}");
        }
    }

    #[test]
    fn test_compute_profile_black() {
        let width = 50;
        let height = 50;
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(width, height, Luma([0])));
        
        let profile = compute_profile(&img);
        
        for &val in &profile.vpp {
            assert_eq!(val, 0);
        }
        for &val in &profile.hpp {
            assert_eq!(val, 0);
        }
    }

    #[test]
    fn test_compute_profile_half_white_half_black() {
        let width = 100;
        let height = 100;
        let mut img = GrayImage::new(width, height);
        
        // Left half white (255), Right half black (0)
        for x in 0..width {
            for y in 0..height {
                if x < 50 {
                    img.put_pixel(x, y, Luma([255]));
                } else {
                    img.put_pixel(x, y, Luma([0]));
                }
            }
        }
        
        let profile = compute_profile(&DynamicImage::ImageLuma8(img));
        
        // VPP: first 50 entries should be 255 * 100 = 25,500, rest 0
        for x in 0..100 {
            if x < 50 {
                assert_eq!(profile.vpp[x], 25500, "VPP mismatch at index {x}");
            } else {
                assert_eq!(profile.vpp[x], 0, "VPP mismatch at index {x}");
            }
        }
        
        // HPP: each entry should be sum of 50 white pixels (255) and 50 black pixels (0)
        // 255 * 50 = 12,750
        for y in 0..100 {
            assert_eq!(profile.hpp[y], 12750, "HPP mismatch at index {y}");
        }
    }

    #[test]
    fn test_find_gutters_standard() {
        // 3 images (100px wide), 2 gutters (20px wide)
        // Profile: [100, 20, 100, 20, 100] = 340 total width
        let mut profile = vec![1000u64; 340];
        // Gutter 1: 100..120
        profile[100..120].fill(10);
        // Gutter 2: 220..240
        profile[220..240].fill(10);

        let config = GutterConfig {
            expected_count: Some(2),
            light_gutters: false,
            min_width: 10,
            percentile: 0.1,
        };

        let gutters = find_gutters(&profile, &config);
        
        assert_eq!(gutters.len(), 2);
        // Note: With smoothing (WINDOW_SIZE=7), the edges will be slightly blurred.
        // We allow some tolerance.
        assert!(gutters[0].start >= 98 && gutters[0].start <= 102);
        assert!(gutters[0].end >= 118 && gutters[0].end <= 122);
        assert!(gutters[1].start >= 218 && gutters[1].start <= 222);
        assert!(gutters[1].end >= 238 && gutters[1].end <= 242);
    }

    #[test]
    fn test_find_gutters_with_noise() {
        // Same as standard but with small variations
        let mut profile = vec![0u64; 340];
        let mut seed = 42u64;
        let mut next_rand = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            seed % 20 // 0..20 variation
        };

        for (i, val) in profile.iter_mut().enumerate() {
            let base = if (100..120).contains(&i) || (220..240).contains(&i) {
                10
            } else {
                1000
            };
            *val = base + next_rand();
        }

        let config = GutterConfig {
            expected_count: Some(2),
            light_gutters: false,
            min_width: 10,
            percentile: 0.1,
        };

        let gutters = find_gutters(&profile, &config);
        assert_eq!(gutters.len(), 2);
        // With smoothing and a proper percentile, the detected range should be close
        // to the original 100..120 and 220..240.
        assert!(gutters[0].start >= 97 && gutters[0].start <= 103);
        assert!(gutters[0].end >= 117 && gutters[0].end <= 123);
        assert!(gutters[1].start >= 217 && gutters[1].start <= 223);
        assert!(gutters[1].end >= 237 && gutters[1].end <= 243);
    }

    #[test]
    fn test_find_gutters_light() {
        // Light gutters: hills as gutters, valleys as images (e.g. overexposed film)
        let mut profile = vec![10u64; 340];
        // Light Gutter 1: 100..120
        profile[100..120].fill(1000);
        // Light Gutter 2: 220..240
        profile[220..240].fill(1000);

        let config = GutterConfig {
            expected_count: Some(2),
            light_gutters: true,
            min_width: 10,
            percentile: 0.9,
        };

        let gutters = find_gutters(&profile, &config);
        assert_eq!(gutters.len(), 2);
        assert!(gutters[0].start >= 98 && gutters[0].start <= 102);
        assert!(gutters[0].end >= 118 && gutters[0].end <= 122);
        assert!(gutters[1].start >= 218 && gutters[1].start <= 222);
        assert!(gutters[1].end >= 238 && gutters[1].end <= 242);
    }

    #[test]
    fn test_rect_fields() {
        let r = Rect { x: 1, y: 2, width: 3, height: 4 };
        assert_eq!(r.x, 1);
        assert_eq!(r.y, 2);
        assert_eq!(r.width, 3);
        assert_eq!(r.height, 4);
    }

    #[test]
    fn test_split_context_horizontal_standard() -> Result<()> {
        let width = 300;
        let height = 100;
        let mut img = GrayImage::new(width, height);
        
        for x in 0..width {
            for y in 0..height {
                let is_gutter = (90..110).contains(&x) || (190..210).contains(&x);
                if is_gutter {
                    img.put_pixel(x, y, Luma([0]));
                } else {
                    img.put_pixel(x, y, Luma([255]));
                }
            }
        }
        let dynamic_img = DynamicImage::ImageLuma8(img);
        
        let config = SplitConfig {
            expected_frames: 3,
            percentile: 0.2,
            min_width: 10,
            light_gutters: false,
        };
        
        let mut context = SplitContext::new(dynamic_img, config);
        context.run()?;
        
        assert_eq!(context.detected_rects.len(), 3);
        
        let r1 = context.detected_rects[0];
        let r2 = context.detected_rects[1];
        let r3 = context.detected_rects[2];
        
        assert_eq!(r1.x, 0);
        assert_eq!(r1.width, 100);
        assert_eq!(r2.x, 100);
        assert_eq!(r2.width, 100);
        assert_eq!(r3.x, 200);
        assert_eq!(r3.width, 100);
        Ok(())
    }

    #[test]
    fn test_split_context_fallback() -> Result<()> {
        let width = 300;
        let height = 100;
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(width, height, Luma([255])));
        
        let config = SplitConfig {
            expected_frames: 3,
            percentile: 0.1,
            min_width: 5,
            light_gutters: false,
        };
        
        let mut context = SplitContext::new(img, config);
        context.run()?;
        
        assert_eq!(context.detected_rects.len(), 3);
        assert_eq!(context.detected_rects[0].x, 0);
        assert_eq!(context.detected_rects[0].width, 100);
        assert_eq!(context.detected_rects[1].x, 100);
        assert_eq!(context.detected_rects[1].width, 100);
        assert_eq!(context.detected_rects[2].x, 200);
        assert_eq!(context.detected_rects[2].width, 100);
        Ok(())
    }

    #[test]
    fn test_split_image_cropping() -> Result<()> {
        let width = 300;
        let height = 100;
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(width, height, Luma([255])));
        
        let rects = vec![
            Rect { x: 0, y: 0, width: 100, height: 100 },
            Rect { x: 100, y: 0, width: 100, height: 100 },
            Rect { x: 200, y: 0, width: 100, height: 100 },
        ];
        
        let sub_images = split_image(&img, &rects)?;
        assert_eq!(sub_images.len(), 3);
        for sub in sub_images {
            assert_eq!(sub.width(), 100);
            assert_eq!(sub.height(), 100);
        }
        Ok(())
    }
}
