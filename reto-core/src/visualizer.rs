//! Visual inspection overlay rendering and sub-image extraction utilities.

use crate::error::RoiError;
use crate::geom::{FrameRoiSet, PixelRect};
use image::{DynamicImage, Rgba, RgbaImage};

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
    pub fn render_overlay(
        image: &DynamicImage,
        rois: &FrameRoiSet,
        border_color: Rgba<u8>,
        border_thickness: u32,
    ) -> RgbaImage {
        let mut overlay = image.to_rgba8();
        let (width, height) = overlay.dimensions();

        // Draw bounding boxes for each detected frame
        for frame in &rois.frames {
            let px = frame.bounds.to_pixel_rect(width, height);
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
    ///
    /// # Examples
    /// ```
    /// use reto_core::{EvenSplitDetector, RoiDetectionConfig, RoiDetector, RoiVisualizer};
    /// use image::{DynamicImage, Rgba, RgbaImage};
    ///
    /// let img = DynamicImage::ImageRgba8(RgbaImage::from_pixel(300, 100, Rgba([255, 255, 255, 255])));
    /// let rois = EvenSplitDetector::new().detect(&img, &RoiDetectionConfig::default(), None).unwrap();
    /// let crops = RoiVisualizer::extract_frame_images(&img, &rois).unwrap();
    /// assert_eq!(crops.len(), 3);
    /// ```
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
    }
}
