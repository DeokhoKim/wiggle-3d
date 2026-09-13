#![forbid(unsafe_code)]
//! Core library for Reto-Split.

/// Error types and Result alias.
pub mod error;
/// Image splitting and projection profile analysis.
pub mod splitting;

pub use error::{Error, Result};

/// A demo function to verify tracing setup.
///
/// # Errors
///
/// This function currently never returns an error, but it follows the library convention.
#[tracing::instrument]
pub fn demo_function() -> Result<()> {
    tracing::info!("Tracing is active in reto-core");
    Ok(())
}
