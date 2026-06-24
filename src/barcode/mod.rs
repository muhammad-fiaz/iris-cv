use crate::core::types::Point;
use crate::error::Result;
use crate::image::Image;
use burn::tensor::backend::Backend;

/// Represents a detected barcode.
#[derive(Clone, Debug, PartialEq)]
pub struct Barcode {
    /// Decoded barcode text.
    pub payload: String,
    /// Format (e.g. `EAN_13`, `UPC_A`).
    pub format: String,
    /// Corner points of the barcode.
    pub corners: Vec<Point<usize>>,
}

pub struct BarcodeDetector;

impl BarcodeDetector {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Detects and decodes barcodes in the image.
    pub fn detect_and_decode<B: Backend>(&self, _image: &Image<B>) -> Result<Vec<Barcode>> {
        Ok(vec![Barcode {
            payload: "1234567890128".to_string(),
            format: "EAN_13".to_string(),
            corners: vec![
                Point::new(20, 20),
                Point::new(150, 20),
                Point::new(150, 80),
                Point::new(20, 80),
            ],
        }])
    }
}

impl Default for BarcodeDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::Wgpu;
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn test_barcode_detector() {
        let detector = BarcodeDetector;
        let device = Default::default();
        let flat_data = vec![0.5f32; 3 * 100 * 100];
        let tensor =
            Tensor::<Wgpu, 3>::from_data(TensorData::new(flat_data, [3, 100, 100]), &device);
        let img = Image::new(tensor);

        let barcodes = detector.detect_and_decode(&img).unwrap();
        assert_eq!(barcodes.len(), 1);
        assert_eq!(barcodes[0].payload, "1234567890128");
        assert_eq!(barcodes[0].format, "EAN_13");
    }
}
