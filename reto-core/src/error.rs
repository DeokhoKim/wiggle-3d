use thiserror::Error;

/// Common error type for reto-core.
#[derive(Debug, Error)]
pub enum Error {
    /// IO error wrapper.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Splitting error.
    #[error("Splitting error: {0}")]
    Splitting(String),

    /// Placeholder for other errors.
    #[error("Unknown error: {0}")]
    Unknown(String),
}

/// Type alias for `Result` with `reto_core::Error`.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn test_error_from_io() {
        let io_error = io::Error::new(io::ErrorKind::NotFound, "test error");
        let error: Error = io_error.into();
        assert!(matches!(error, Error::Io(_)));
        assert!(error.to_string().contains("IO error: test error"));
    }

    #[test]
    fn test_error_unknown() {
        let error = Error::Unknown("test message".to_string());
        assert_eq!(error.to_string(), "Unknown error: test message");
    }
}
