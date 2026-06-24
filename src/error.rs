use thiserror::Error;

/// Error type for all Iris library operations.
#[derive(Debug, Error)]
pub enum IrisError {
    #[error("Image I/O or format error: {0}")]
    Image(#[from] image::ImageError),

    #[error("Standard I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Tensor operation failed: {0}")]
    Tensor(String),

    #[error("Dimension mismatch: expected {expected:?}, found {actual:?}")]
    DimensionMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Model load failure: {0}")]
    ModelLoad(String),

    #[error("Model execution or inference failure: {0}")]
    Inference(String),

    #[error("Feature not implemented: {0}")]
    FeatureNotImplemented(String),

    #[error("Generic error: {0}")]
    Generic(String),
}

/// Result type for Iris library operations.
pub type Result<T> = std::result::Result<T, IrisError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_formatting() {
        let err = IrisError::Generic("test error".to_string());
        assert_eq!(format!("{}", err), "Generic error: test error");

        let err_dim = IrisError::DimensionMismatch {
            expected: vec![1, 2, 3],
            actual: vec![1, 2],
        };
        assert!(format!("{}", err_dim).contains("Dimension mismatch"));
    }
}
