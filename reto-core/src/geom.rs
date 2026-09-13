//! Geometric coordinate types, orientation definitions, and zero-copy frame `RoI` views.

use crate::error::RoiError;
use image::{GenericImageView, SubImage};
use ndarray::{ArrayView2, ArrayViewMut2, Axis};
use serde::{Deserialize, Serialize};

/// 2D dimension container representing width and height.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Size2D<T = u32> {
    /// Width dimension.
    pub width: T,
    /// Height dimension.
    pub height: T,
}

impl<T> Size2D<T> {
    /// Creates a new `Size2D` with specified width and height.
    #[inline]
    #[must_use]
    pub const fn new(width: T, height: T) -> Self {
        Self { width, height }
    }
}

/// Concrete 2D coordinate point in a specific reference space.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
pub struct Point2D<T = f32> {
    /// Horizontal X coordinate.
    pub x: T,
    /// Vertical Y coordinate.
    pub y: T,
}

impl<T> Point2D<T> {
    /// Creates a new `Point2D` with specified coordinates.
    #[inline]
    #[must_use]
    pub const fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

impl Point2D<f32> {
    /// Converts a local coordinate point (within an ROI) to the global scan space.
    ///
    /// # Arguments
    /// * `roi_bounds` - The bounding rectangle of the sub-frame in normalized unit space.
    /// * `strip_size` - Full scan strip dimensions (width, height).
    ///
    /// # Examples
    /// ```
    /// use reto_core::{NormalizedRect, Point2D, Size2D};
    ///
    /// let roi = NormalizedRect::new(0.5, 0.0, 0.5, 1.0).unwrap();
    /// let strip_size = Size2D::new(200, 100);
    /// let local_pt = Point2D::new(10.0, 20.0);
    /// let global_pt = local_pt.to_global(roi, strip_size);
    /// assert_eq!(global_pt.x, 110.0);
    /// assert_eq!(global_pt.y, 20.0);
    /// ```
    #[inline]
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn to_global(self, roi_bounds: NormalizedRect, strip_size: Size2D<u32>) -> Self {
        let px = roi_bounds.to_pixel_rect(strip_size);
        Self {
            x: self.x + px.x as f32,
            y: self.y + px.y as f32,
        }
    }

    /// Converts a global scan strip coordinate point into an ROI's local sub-frame space.
    ///
    /// # Arguments
    /// * `roi_bounds` - The bounding rectangle of the sub-frame.
    /// * `strip_size` - Full scan strip dimensions (width, height).
    ///
    /// # Returns
    /// Returns `Some(Point2D)` if the point lies inside the sub-frame bounds, else `None`.
    ///
    /// # Examples
    /// ```
    /// use reto_core::{NormalizedRect, Point2D, Size2D};
    ///
    /// let roi = NormalizedRect::new(0.5, 0.0, 0.5, 1.0).unwrap();
    /// let strip_size = Size2D::new(200, 100);
    /// let global_pt = Point2D::new(110.0, 20.0);
    /// let local_pt = global_pt.to_local(roi, strip_size).unwrap();
    /// assert_eq!(local_pt.x, 10.0);
    /// assert_eq!(local_pt.y, 20.0);
    /// ```
    #[inline]
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn to_local(self, roi_bounds: NormalizedRect, strip_size: Size2D<u32>) -> Option<Self> {
        let px = roi_bounds.to_pixel_rect(strip_size);
        let lx = self.x - px.x as f32;
        let ly = self.y - px.y as f32;
        if lx >= 0.0 && ly >= 0.0 && lx < px.width as f32 && ly < px.height as f32 {
            Some(Self { x: lx, y: ly })
        } else {
            None
        }
    }
}

/// A coordinate point anchored in the frame's local bounding box space.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LocalCoord {
    /// Relative pixel offset within the sub-frame `[0, width) x [0, height)`.
    pub pixel: Point2D<f32>,
    /// Relative unit coordinates `[0.0, 1.0]^2` within the sub-frame.
    pub normalized: Point2D<f32>,
}

impl LocalCoord {
    /// Constructs a new `LocalCoord` from sub-frame pixel and normalized coordinates.
    #[inline]
    #[must_use]
    pub const fn new(pixel: Point2D<f32>, normalized: Point2D<f32>) -> Self {
        Self { pixel, normalized }
    }

    /// Converts this local coordinate into full strip global coordinate space.
    ///
    /// # Arguments
    /// * `roi_bounds` - The bounding rectangle of the sub-frame in normalized unit space.
    /// * `strip_size` - Full scan strip dimensions.
    ///
    /// # Examples
    /// ```
    /// use reto_core::{LocalCoord, NormalizedRect, Point2D, Size2D};
    ///
    /// let roi = NormalizedRect::new(0.5, 0.0, 0.5, 1.0).unwrap();
    /// let strip_size = Size2D::new(200, 100);
    /// let local = LocalCoord::new(Point2D::new(10.0, 20.0), Point2D::new(0.1, 0.2));
    /// let global = local.to_global(roi, strip_size);
    /// assert_eq!(global.pixel.x, 110.0);
    /// assert_eq!(global.pixel.y, 20.0);
    /// assert_eq!(global.normalized.x, 0.55);
    /// assert_eq!(global.normalized.y, 0.2);
    /// ```
    #[inline]
    #[must_use]
    #[allow(clippy::suboptimal_flops)]
    pub fn to_global(self, roi_bounds: NormalizedRect, strip_size: Size2D<u32>) -> GlobalCoord {
        GlobalCoord {
            pixel: self.pixel.to_global(roi_bounds, strip_size),
            normalized: Point2D::new(
                self.normalized.x.mul_add(roi_bounds.width, roi_bounds.x),
                self.normalized.y.mul_add(roi_bounds.height, roi_bounds.y),
            ),
        }
    }
}

/// A coordinate point anchored in the full strip scan space.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GlobalCoord {
    /// Pixel coordinates `[0, strip_width) x [0, strip_height)`.
    pub pixel: Point2D<f32>,
    /// Unit coordinates `[0.0, 1.0]^2` across the entire scan strip.
    pub normalized: Point2D<f32>,
}

impl GlobalCoord {
    /// Constructs a new `GlobalCoord` from scan strip pixel and normalized coordinates.
    #[inline]
    #[must_use]
    pub const fn new(pixel: Point2D<f32>, normalized: Point2D<f32>) -> Self {
        Self { pixel, normalized }
    }

    /// Converts this global coordinate into an ROI's local sub-frame coordinate space.
    ///
    /// # Arguments
    /// * `roi_bounds` - The bounding rectangle of the sub-frame.
    /// * `strip_size` - Full scan strip dimensions.
    ///
    /// # Returns
    /// Returns `Some(LocalCoord)` if the coordinate falls inside the ROI bounds, else `None`.
    ///
    /// # Examples
    /// ```
    /// use reto_core::{GlobalCoord, NormalizedRect, Point2D, Size2D};
    ///
    /// let roi = NormalizedRect::new(0.5, 0.0, 0.5, 1.0).unwrap();
    /// let strip_size = Size2D::new(200, 100);
    /// let global = GlobalCoord::new(Point2D::new(110.0, 20.0), Point2D::new(0.55, 0.2));
    /// let local = global.to_local(roi, strip_size).unwrap();
    /// assert_eq!(local.pixel.x, 10.0);
    /// assert_eq!(local.pixel.y, 20.0);
    /// assert!((local.normalized.x - 0.1).abs() < 1e-6);
    /// assert!((local.normalized.y - 0.2).abs() < 1e-6);
    /// ```
    #[inline]
    #[must_use]
    pub fn to_local(
        self,
        roi_bounds: NormalizedRect,
        strip_size: Size2D<u32>,
    ) -> Option<LocalCoord> {
        let pixel = self.pixel.to_local(roi_bounds, strip_size)?;
        let norm_x = (self.normalized.x - roi_bounds.x) / roi_bounds.width;
        let norm_y = (self.normalized.y - roi_bounds.y) / roi_bounds.height;
        if norm_x >= 0.0 && norm_y >= 0.0 && norm_x <= 1.0001 && norm_y <= 1.0001 {
            Some(LocalCoord {
                pixel,
                normalized: Point2D::new(norm_x.clamp(0.0, 1.0), norm_y.clamp(0.0, 1.0)),
            })
        } else {
            None
        }
    }
}

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
    /// Determines the film orientation from explicit dimensions.
    ///
    /// # Errors
    /// Returns [`RoiError`] if dimensions are zero or the image is square.
    ///
    /// # Examples
    /// ```
    /// use reto_core::StripOrientation;
    ///
    /// let orientation = StripOrientation::from_dimensions(300, 100).unwrap();
    /// assert_eq!(orientation, StripOrientation::Horizontal);
    /// ```
    pub const fn from_dimensions(width: u32, height: u32) -> Result<Self, RoiError> {
        Self::from_size(Size2D::new(width, height))
    }

    /// Determines the film orientation from structured 2D dimensions.
    ///
    /// # Errors
    /// Returns [`RoiError`] if dimensions are zero or the image is square.
    pub const fn from_size(size: Size2D<u32>) -> Result<Self, RoiError> {
        let width = size.width;
        let height = size.height;
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
    /// use reto_core::{Size2D, StripOrientation};
    ///
    /// let delegator = StripOrientation::Horizontal.delegator();
    /// assert_eq!(delegator.major_dimension(Size2D::new(300, 100)), 300);
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
    fn major_dimension(&self, size: Size2D<u32>) -> u32;

    /// Length of the minor frame dimension (height for horizontal, width for vertical).
    fn minor_dimension(&self, size: Size2D<u32>) -> u32;

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
    fn slice_stride(&self, major_idx: u32, size: Size2D<u32>) -> (usize, usize);

    /// Fills the neural network inference canvas with normalized luma values according to orientation layout.
    fn fill_inference_canvas(
        &self,
        canvas: &mut [f32],
        canvas_width: u32,
        scaled_img: &image::GrayImage,
        scaled_w: u32,
        scaled_h: u32,
    );

    /// Maps raw keypoint coordinates `(kx, ky)` extracted from the neural inference canvas back to scaled frame coordinates `(rx, ry)`.
    fn unmap_canvas_coords(&self, kx: f32, ky: f32) -> (f32, f32);
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
    fn major_dimension(&self, size: Size2D<u32>) -> u32 {
        size.width
    }

    #[inline]
    fn minor_dimension(&self, size: Size2D<u32>) -> u32 {
        size.height
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
    fn slice_stride(&self, major_idx: u32, size: Size2D<u32>) -> (usize, usize) {
        (major_idx as usize, size.width as usize)
    }

    #[inline]
    #[allow(clippy::cast_precision_loss)]
    fn fill_inference_canvas(
        &self,
        canvas: &mut [f32],
        canvas_width: u32,
        scaled_img: &image::GrayImage,
        scaled_w: u32,
        scaled_h: u32,
    ) {
        let Ok(src_view) = ArrayView2::<'_, u8>::from_shape(
            (scaled_h as usize, scaled_w as usize),
            scaled_img.as_raw(),
        ) else {
            return;
        };

        // Eager vectorized tensor normalization and transpose: [scaled_h, scaled_w] -> [scaled_w, scaled_h]
        let transposed = src_view.mapv(|p| f32::from(p) / 255.0).reversed_axes();

        let effective_w = (scaled_h as usize).min(canvas_width as usize);
        let canvas_height = canvas.len() / (canvas_width as usize);
        if let Ok(mut canvas_view) =
            ArrayViewMut2::<'_, f32>::from_shape((canvas_height, canvas_width as usize), canvas)
        {
            let row_limit = (scaled_w as usize).min(canvas_height);
            for (mut dst_row, src_row) in canvas_view
                .axis_iter_mut(Axis(0))
                .zip(transposed.axis_iter(Axis(0)))
                .take(row_limit)
            {
                if let Some(dst) = dst_row.as_slice_mut() {
                    for (d, &s) in dst[..effective_w]
                        .iter_mut()
                        .zip(src_row.iter().take(effective_w))
                    {
                        *d = s;
                    }
                }
            }
        }
    }

    #[inline]
    fn unmap_canvas_coords(&self, kx: f32, ky: f32) -> (f32, f32) {
        (ky, kx)
    }
}

impl OrientationDelegator for VerticalDelegator {
    #[inline]
    fn major_dimension(&self, size: Size2D<u32>) -> u32 {
        size.height
    }

    #[inline]
    fn minor_dimension(&self, size: Size2D<u32>) -> u32 {
        size.width
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
    fn slice_stride(&self, major_idx: u32, size: Size2D<u32>) -> (usize, usize) {
        ((major_idx * size.width) as usize, 1)
    }

    #[inline]
    #[allow(clippy::cast_precision_loss)]
    fn fill_inference_canvas(
        &self,
        canvas: &mut [f32],
        canvas_width: u32,
        scaled_img: &image::GrayImage,
        scaled_w: u32,
        scaled_h: u32,
    ) {
        let Ok(src_view) = ArrayView2::<'_, u8>::from_shape(
            (scaled_h as usize, scaled_w as usize),
            scaled_img.as_raw(),
        ) else {
            return;
        };

        // Eager vectorized tensor normalization [scaled_h, scaled_w]
        let normalized = src_view.mapv(|p| f32::from(p) / 255.0);

        let effective_w = (scaled_w as usize).min(canvas_width as usize);
        let canvas_height = canvas.len() / (canvas_width as usize);
        if let Ok(mut canvas_view) =
            ArrayViewMut2::<'_, f32>::from_shape((canvas_height, canvas_width as usize), canvas)
        {
            let row_limit = (scaled_h as usize).min(canvas_height);
            for (mut dst_row, src_row) in canvas_view
                .axis_iter_mut(Axis(0))
                .zip(normalized.axis_iter(Axis(0)))
                .take(row_limit)
            {
                if let Some(dst) = dst_row.as_slice_mut() {
                    for (d, &s) in dst[..effective_w]
                        .iter_mut()
                        .zip(src_row.iter().take(effective_w))
                    {
                        *d = s;
                    }
                }
            }
        }
    }

    #[inline]
    fn unmap_canvas_coords(&self, kx: f32, ky: f32) -> (f32, f32) {
        (kx, ky)
    }
}

/// Normalized 2D bounding box representing relative unit sub-regions in `[0.0, 1.0]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NormalizedRect {
    /// Normalized horizontal start coordinate in `[0.0, 1.0]`.
    pub x: f32,
    /// Normalized vertical start coordinate in `[0.0, 1.0]`.
    pub y: f32,
    /// Normalized width in `[0.0, 1.0]`.
    pub width: f32,
    /// Normalized height in `[0.0, 1.0]`.
    pub height: f32,
}

impl NormalizedRect {
    /// Constructs and validates a new `NormalizedRect`.
    ///
    /// # Errors
    /// Returns [`RoiError`] if values are negative, not finite, or if width/height is zero.
    ///
    /// # Examples
    /// ```
    /// use reto_core::NormalizedRect;
    ///
    /// let rect = NormalizedRect::new(0.0, 0.0, 0.5, 1.0).unwrap();
    /// assert_eq!(rect.width, 0.5);
    /// ```
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Result<Self, RoiError> {
        if !x.is_finite()
            || !y.is_finite()
            || !width.is_finite()
            || !height.is_finite()
            || x < 0.0_f32
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

    /// Converts normalized bounds to concrete pixel coordinates for an image of dimensions `size`.
    ///
    /// # Arguments
    /// * `size` - Total image dimensions in pixels.
    ///
    /// # Examples
    /// ```
    /// use reto_core::{NormalizedRect, Size2D};
    ///
    /// let rect = NormalizedRect::new(0.0, 0.0, 0.5, 1.0).unwrap();
    /// let px = rect.to_pixel_rect(Size2D::new(200, 100));
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
    pub fn to_pixel_rect(self, size: Size2D<u32>) -> PixelRect {
        let x = (self.x * size.width as f32).round() as u32;
        let y = (self.y * size.height as f32).round() as u32;
        let width =
            ((self.width * size.width as f32).round() as u32).min(size.width.saturating_sub(x));
        let height =
            ((self.height * size.height as f32).round() as u32).min(size.height.saturating_sub(y));
        PixelRect {
            x,
            y,
            width,
            height,
        }
    }

    /// Checks whether a point in normalized coordinates $(px, py) \in [0.0, 1.0]^2$ is inside this bounding box.
    ///
    /// # Arguments
    /// * `px` - Normalized horizontal coordinate $x$.
    /// * `py` - Normalized vertical coordinate $y$.
    ///
    /// # Examples
    /// ```
    /// use reto_core::NormalizedRect;
    ///
    /// let rect = NormalizedRect::new(0.2, 0.2, 0.4, 0.4).unwrap();
    /// assert!(rect.contains_point(0.3, 0.3));
    /// assert!(!rect.contains_point(0.1, 0.3));
    /// ```
    #[inline]
    #[must_use]
    pub fn contains_point(&self, px: f32, py: f32) -> bool {
        px >= self.x && px <= (self.x + self.width) && py >= self.y && py <= (self.y + self.height)
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

impl PixelRect {
    /// Returns the rectangle dimensions as a `Size2D<u32>`.
    #[inline]
    #[must_use]
    pub const fn size(&self) -> Size2D<u32> {
        Size2D::new(self.width, self.height)
    }
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
        let pixel = frame.bounds.to_pixel_rect(Size2D::new(w, h));
        Ok(SubImage::new(
            image,
            pixel.x,
            pixel.y,
            pixel.width,
            pixel.height,
        ))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp, clippy::suboptimal_flops)]
mod tests {
    use super::*;

    #[test]
    fn test_point2d_transforms() {
        let roi = NormalizedRect::new(0.25, 0.1, 0.5, 0.8).unwrap();
        let strip_size = Size2D::new(1000, 500);

        let local_pt = Point2D::new(50.0, 100.0);
        let global_pt = local_pt.to_global(roi, strip_size);

        // roi.x * 1000 = 250, roi.y * 500 = 50
        assert_eq!(global_pt.x, 300.0);
        assert_eq!(global_pt.y, 150.0);

        let roundtrip = global_pt.to_local(roi, strip_size).unwrap();
        assert_eq!(roundtrip.x, local_pt.x);
        assert_eq!(roundtrip.y, local_pt.y);

        // Point outside the roi
        let outside = Point2D::new(10.0, 10.0);
        assert!(outside.to_local(roi, strip_size).is_none());
    }

    #[test]
    fn test_coord_transforms() {
        let roi = NormalizedRect::new(0.25, 0.1, 0.5, 0.8).unwrap();
        let strip_size = Size2D::new(1000, 500);

        let local = LocalCoord::new(Point2D::new(50.0, 100.0), Point2D::new(0.1, 0.25));
        let global = local.to_global(roi, strip_size);

        assert_eq!(global.pixel.x, 300.0);
        assert_eq!(global.pixel.y, 150.0);
        assert!((global.normalized.x - (0.25 + 0.1 * 0.5)).abs() < 1e-6);
        assert!((global.normalized.y - (0.1 + 0.25 * 0.8)).abs() < 1e-6);

        let local_back = global.to_local(roi, strip_size).unwrap();
        assert_eq!(local_back.pixel.x, local.pixel.x);
        assert_eq!(local_back.pixel.y, local.pixel.y);
        assert!((local_back.normalized.x - local.normalized.x).abs() < 1e-6);
        assert!((local_back.normalized.y - local.normalized.y).abs() < 1e-6);
    }
}
