#![forbid(unsafe_code)]
//! Core library for Reto-Split: 3D film camera image splitting, alignment, and parallax processing.

// Encapsulated internal modules (thin public API facade)
pub(crate) mod detector;
pub(crate) mod error;
pub(crate) mod geom;
pub(crate) mod luma;
pub(crate) mod partition;
pub(crate) mod pipeline;
pub(crate) mod stats;
pub(crate) mod visualizer;

// Public Facade Exports
pub use detector::{
    CompositeDiagnosticTap, EvenSplitDetector, NoOpDiagnosticTap, PillarStatsDetector,
    RoiDetectionConfig, RoiDetector, RoiDiagnosticTap, SaveDenoisedLumaDiagnosticTap,
    SaveLumaDiagnosticTap,
};
pub use error::{Error, Result, RoiError};
pub use geom::{
    FrameRoi, FrameRoiSet, HorizontalDelegator, NormalizedRect, OrientationDelegator, PixelRect,
    StripOrientation, VerticalDelegator, HORIZONTAL_DELEGATOR, VERTICAL_DELEGATOR,
};
pub use luma::{
    median9, Bt709LumaConverter, LumaConverter, ScaledGrayscaleStrip, SimpleGrayConverter,
    DEFAULT_CROSS_PERCENTILE, DEFAULT_INVERSE_GAMMA, DEFAULT_PROFILE_PERCENTILE,
    PROJECTION_MAX_DIMENSION,
};
pub use partition::{
    EvenSplitPartitioner, GutterPartitioner, OptimalGridPartitioner, PartitionResult,
    PartitionValidator, PrioritizedPartitionEngine, ThresholdPartitioner,
};
pub use pipeline::{
    is_supported_image, process_item, process_single_image, run_batch, BatchProcessingRequest,
    ImageItemContext, ItemProcessingOutcome, ProcessSummary, SUPPORTED_EXTENSIONS,
};
pub use stats::{
    matched_filter_1d, AxisPixelStats, AxisStatisticsProfile, GutterSpan, OptimalGridResult,
    ThresholdPartitionResult, DEFAULT_HIGH_PERCENTILE, DEFAULT_LOW_PERCENTILE,
};
pub use visualizer::RoiVisualizer;
