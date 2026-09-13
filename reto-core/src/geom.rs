//! Geometric coordinate types, orientation definitions, and zero-copy frame `RoI` views.

use crate::error::RoiError;
use image::{GenericImageView, SubImage};
use serde::{Deserialize, Serialize};

/// Film strip scan orientation based on dimensional aspect ratio.
///
/// Invariant: Input images always have aspect ratio != 1.0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StripOrientation {
    /// Width > Height: A set of portrait frames stacked horizontally side-by-side.
    /// This is the primary format in practical RETO3D film strip scans.
    Horizontal,
    /// Height > Width: A set of landscape frames stacked vertically top-to-bottom.
    Vertical,
}

impl StripOrientation {
    /// Determines orientation from raw pixel dimensions.
    ///
    /// # Arguments
    /// * `width` - Source image width in pixels.
    /// * `height` - Source image height in pixels.
    ///
    /// # Errors
    /// Returns [`RoiError::ImageTooSmall`] if width or height is zero.
    /// Returns [`RoiError::SquareImageNotSupported`] if `width == height`.
    ///
    /// # Examples
    /// ```
    /// use reto_core::StripOrientation;
    ///
    /// let orientation = StripOrientation::from_dimensions(300, 100).unwrap();
    /// assert_eq!(orientation, StripOrientation::Horizontal);
    /// ```
    pub const fn from_dimensions(width: u32, height: u32) -> Result<Self, RoiError> {
        if width == 0 || height == 0 {
            return Err(RoiError::ImageTooSmall { width, height });
        }
        if width > height {
            Ok(Self::Horizontal)
        } else if height > width {
            Ok(Self::Vertical)
        } else {
            Err(RoiError::SquareImageNotSupported { width, height })
        }
    }

    /// Selects the pre-defined delegator for this orientation.
    ///
    /// # Examples
    /// ```
    /// use reto_core::StripOrientation;
    ///
    /// let delegator = StripOrientation::Horizontal.delegator();
    /// assert_eq!(delegator.major_dimension(300, 100), 300);
    /// ```
    #[inline]
    #[must_use]
    pub fn delegator(self) -> &'static dyn OrientationDelegator {
        match self {
            Self::Horizontal => &HORIZONTAL_DELEGATOR,
            Self::Vertical => &VERTICAL_DELEGATOR,
        }
    }
}

/// Pre-defined delegator contract for orientation-specific axis operations.
///
/// Eliminates scattered control-flow branching throughout detection and alignment pipelines.
pub trait OrientationDelegator: Send + Sync {
    /// Length of the major stacking dimension (width for horizontal, height for vertical).
    fn major_dimension(&self, width: u32, height: u32) -> u32;

    /// Length of the minor frame dimension (height for horizontal, width for vertical).
    fn minor_dimension(&self, width: u32, height: u32) -> u32;

    /// Constructs a `NormalizedRect` from stacking axis bounds and cross axis bounds.
    fn build_rect(
        &self,
        stack_min: f32,
        stack_len: f32,
        cross_min: f32,
        cross_len: f32,
    ) -> NormalizedRect;

    /// Maps 1D stacking coordinates `(major, cross)` to 2D image pixel coordinates `(x, y)`.
    fn to_xy(&self, major: u32, cross: u32) -> (u32, u32);

    /// Maps 2D image pixel coordinates `(x, y)` to 1D stacking coordinates `(major, cross)`.
    fn to_stack_coords(&self, x: u32, y: u32) -> (u32, u32);

    /// Returns the `(start_index, stride)` in the flat 2D buffer for the perpendicular slice at `major_idx`.
    fn slice_stride(&self, major_idx: u32, width: u32, height: u32) -> (usize, usize);
}

/// Delegator operations for horizontally-stacked portrait frames (width > height).
#[derive(Debug, Clone, Copy)]
pub struct HorizontalDelegator;

/// Delegator operations for vertically-stacked landscape frames (height > width).
#[derive(Debug, Clone, Copy)]
pub struct VerticalDelegator;

/// Static singleton delegator for horizontal strips.
pub static HORIZONTAL_DELEGATOR: HorizontalDelegator = HorizontalDelegator;

/// Static singleton delegator for vertical strips.
pub static VERTICAL_DELEGATOR: VerticalDelegator = VerticalDelegator;

impl OrientationDelegator for HorizontalDelegator {
    #[inline]
    fn major_dimension(&self, width: u32, _height: u32) -> u32 {
        width
    }

    #[inline]
    fn minor_dimension(&self, _width: u32, height: u32) -> u32 {
        height
    }

    #[inline]
    fn build_rect(
        &self,
        stack_min: f32,
        stack_len: f32,
        cross_min: f32,
        cross_len: f32,
    ) -> NormalizedRect {
        NormalizedRect {
            x: stack_min,
            y: cross_min,
            width: stack_len,
            height: cross_len,
        }
    }

    #[inline]
    fn to_xy(&self, major: u32, cross: u32) -> (u32, u32) {
        (major, cross)
    }

    #[inline]
    fn to_stack_coords(&self, x: u32, y: u32) -> (u32, u32) {
        (x, y)
    }

    #[inline]
    fn slice_stride(&self, major_idx: u32, width: u32, _height: u32) -> (usize, usize) {
        (major_idx as usize, width as usize)
    }
}

impl OrientationDelegator for VerticalDelegator {
    #[inline]
    fn major_dimension(&self, _width: u32, height: u32) -> u32 {
        height
    }

    #[inline]
    fn minor_dimension(&self, width: u32, _height: u32) -> u32 {
        width
    }

    #[inline]
    fn build_rect(
        &self,
        stack_min: f32,
        stack_len: f32,
        cross_min: f32,
        cross_len: f32,
    ) -> NormalizedRect {
        NormalizedRect {
            x: cross_min,
            y: stack_min,
            width: cross_len,
            height: stack_len,
        }
    }

    #[inline]
    fn to_xy(&self, major: u32, cross: u32) -> (u32, u32) {
        (cross, major)
    }

    #[inline]
    fn to_stack_coords(&self, x: u32, y: u32) -> (u32, u32) {
        (y, x)
    }

    #[inline]
    fn slice_stride(&self, major_idx: u32, width: u32, _height: u32) -> (usize, usize) {
        ((major_idx * width) as usize, 1)
    }
}

/// Rectangle in normalized unit coordinate space `[0.0, 1.0]` using single-precision floats (`f32`).
///
/// Origin `(0.0, 0.0)` is at the top-left corner of the source image,
/// and `(1.0, 1.0)` is at the bottom-right corner.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NormalizedRect {
    /// Horizontal coordinate of the top-left corner in unit space `[0.0, 1.0]`.
    pub x: f32,
    /// Vertical coordinate of the top-left corner in unit space `[0.0, 1.0]`.
    pub y: f32,
    /// Width of the rectangle in unit space `(0.0, 1.0]`.
    pub width: f32,
    /// Height of the rectangle in unit space `(0.0, 1.0]`.
    pub height: f32,
}

impl NormalizedRect {
    /// Constructs and validates a new `NormalizedRect`.
    ///
    /// # Arguments
    /// * `x` - Normalized X coordinate in `[0.0, 1.0]`.
    /// * `y` - Normalized Y coordinate in `[0.0, 1.0]`.
    /// * `width` - Normalized width in `(0.0, 1.0]`.
    /// * `height` - Normalized height in `(0.0, 1.0]`.
    ///
    /// # Errors
    /// Returns [`RoiError::InvalidBounds`] if coordinates fall outside `[0.0, 1.0]`
    /// or if width/height are non-positive.
    ///
    /// # Examples
    /// ```
    /// use reto_core::NormalizedRect;
    ///
    /// let rect = NormalizedRect::new(0.0, 0.0, 0.333, 1.0).unwrap();
    /// assert_eq!(rect.x, 0.0);
    /// ```
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Result<Self, RoiError> {
        if x < 0.0_f32
            || y < 0.0_f32
            || width <= 0.0_f32
            || height <= 0.0_f32
            || (x + width) > 1.0001_f32
            || (y + height) > 1.0001_f32
        {
            return Err(RoiError::InvalidBounds {
                x,
                y,
                width,
                height,
            });
        }
        Ok(Self {
            x: x.clamp(0.0_f32, 1.0_f32),
            y: y.clamp(0.0_f32, 1.0_f32),
            width: width.min(1.0_f32 - x),
            height: height.min(1.0_f32 - y),
        })
    }

    /// Converts normalized bounds to concrete pixel coordinates for an image of dimensions `(img_w, img_h)`.
    ///
    /// # Arguments
    /// * `img_w` - Total image width in pixels.
    /// * `img_h` - Total image height in pixels.
    ///
    /// # Examples
    /// ```
    /// use reto_core::NormalizedRect;
    ///
    /// let rect = NormalizedRect::new(0.0, 0.0, 0.5, 1.0).unwrap();
    /// let px = rect.to_pixel_rect(200, 100);
    /// assert_eq!(px.width, 100);
    /// assert_eq!(px.height, 100);
    /// ```
    #[inline]
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    pub fn to_pixel_rect(self, img_w: u32, img_h: u32) -> PixelRect {
        let x = (self.x * img_w as f32).round() as u32;
        let y = (self.y * img_h as f32).round() as u32;
        let width = ((self.width * img_w as f32).round() as u32).min(img_w.saturating_sub(x));
        let height = ((self.height * img_h as f32).round() as u32).min(img_h.saturating_sub(y));
        PixelRect {
            x,
            y,
            width,
            height,
        }
    }
}

/// Concrete integer pixel rectangle bounding box.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PixelRect {
    /// Horizontal pixel offset from left.
    pub x: u32,
    /// Vertical pixel offset from top.
    pub y: u32,
    /// Pixel width.
    pub width: u32,
    /// Pixel height.
    pub height: u32,
}

/// Information about a single identified film frame `RoI`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameRoi {
    /// 0-indexed position within the strip (e.g. 0 = left, 1 = middle, 2 = right).
    pub index: usize,
    /// Bounding rectangle in normalized unit space.
    pub bounds: NormalizedRect,
    /// Confidence score `[0.0, 1.0]` of boundary detection.
    pub confidence: f32,
}

/// Ordered collection of detected film frames representing a single exposure strip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameRoiSet {
    /// Source image width at detection time.
    pub source_width: u32,
    /// Source image height at detection time.
    pub source_height: u32,
    /// Strip orientation (Horizontal for portrait frames, Vertical for landscape frames).
    pub orientation: StripOrientation,
    /// Detected frame regions in spatial stacking order.
    pub frames: Vec<FrameRoi>,
}

impl FrameRoiSet {
    /// Creates a new `FrameRoiSet`.
    ///
    /// # Arguments
    /// * `source_width` - Detected image pixel width.
    /// * `source_height` - Detected image pixel height.
    /// * `orientation` - Scan orientation.
    /// * `frames` - Ordered list of detected frame regions.
    #[must_use]
    pub const fn new(
        source_width: u32,
        source_height: u32,
        orientation: StripOrientation,
        frames: Vec<FrameRoi>,
    ) -> Self {
        Self {
            source_width,
            source_height,
            orientation,
            frames,
        }
    }

    /// Returns the number of detected frames.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.frames.len()
    }

    /// Checks if no frames were detected.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Provides a zero-copy borrowed sub-image view of the requested frame.
    ///
    /// # Safety / Cost
    /// Performs zero pixel buffer copies, returning an `image::SubImage`
    /// that borrows directly from the source image reference.
    ///
    /// # Arguments
    /// * `image` - Source image buffer reference to slice.
    /// * `frame_idx` - 0-indexed frame number to view.
    ///
    /// # Errors
    /// Returns [`RoiError::FrameIndexOutOfRange`] if `frame_idx >= self.len()`.
    ///
    /// # Examples
    /// ```
    /// use reto_core::{EvenSplitDetector, RoiDetectionConfig, RoiDetector};
    /// use image::{GenericImageView, Rgba, RgbaImage};
    ///
    /// let img = RgbaImage::from_pixel(300, 100, Rgba([255, 255, 255, 255]));
    /// let detector = EvenSplitDetector::new();
    /// let rois = detector.detect(&img, &RoiDetectionConfig::default(), None).unwrap();
    /// let sub = rois.sub_image(&img, 0).unwrap();
    /// assert_eq!(sub.dimensions(), (100, 100));
    /// ```
    pub fn sub_image<'a, I: GenericImageView>(
        &self,
        image: &'a I,
        frame_idx: usize,
    ) -> Result<SubImage<&'a I>, RoiError> {
        let frame = self
            .frames
            .get(frame_idx)
            .ok_or(RoiError::FrameIndexOutOfRange {
                requested: frame_idx,
                available: self.frames.len(),
            })?;

        let (w, h) = image.dimensions();
        let pixel = frame.bounds.to_pixel_rect(w, h);
        Ok(SubImage::new(
            image,
            pixel.x,
            pixel.y,
            pixel.width,
            pixel.height,
        ))
    }
}
