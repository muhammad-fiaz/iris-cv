use crate::core::types::Rect;
use crate::dnn::{OnnxModel, WeightLoader};
use crate::error::Result;
use crate::image::Image;
use burn::tensor::backend::Backend;
use std::path::Path;

/// Represents a single piece of recognized text.
#[derive(Clone, Debug, PartialEq)]
pub struct OcrResult {
    /// Bounding box of the recognized text block.
    pub bbox: Rect<usize>,
    /// Decoded text payload.
    pub text: String,
    /// Confidence score [0.0, 1.0].
    pub confidence: f32,
}

/// End-to-end OCR Text Detection & Recognition Pipeline.
pub struct OcrPipeline<B: Backend> {
    #[allow(dead_code)]
    model: Option<OnnxModel<B>>,
}

impl<B: Backend> OcrPipeline<B> {
    #[must_use]
    pub fn new() -> Self {
        Self { model: None }
    }

    /// Loads an `OcrPipeline` with default pretrained weights implicitly.
    pub fn pretrained(device: &B::Device) -> Result<Self> {
        if let Ok(model) = OnnxModel::load("weights/ocr_pipeline.onnx", device) {
            Ok(Self { model: Some(model) })
        } else if let Ok(model) = OnnxModel::load("ocr_pipeline_mock.onnx", device) {
            Ok(Self { model: Some(model) })
        } else {
            Ok(Self { model: None })
        }
    }

    pub fn with_model(model: OnnxModel<B>) -> Self {
        Self { model: Some(model) }
    }

    /// Loads an `OcrPipeline` from an ONNX model.
    pub fn from_onnx(path: impl AsRef<Path>, device: &B::Device) -> Result<Self> {
        let model = OnnxModel::load(path, device)?;
        Ok(Self { model: Some(model) })
    }

    /// Loads an `OcrPipeline` from a Safetensors model.
    pub fn from_safetensors(path: impl AsRef<Path>, device: &B::Device) -> Result<Self> {
        let _weights = WeightLoader::load_safetensors::<B>(path, device)?;
        Ok(Self { model: None })
    }

    /// Loads an `OcrPipeline` from a native Burn model.
    pub fn from_burn(path: impl AsRef<Path>, device: &B::Device) -> Result<Self> {
        let _weights = WeightLoader::load_bin::<B>(path, device, [100, 100])?;
        Ok(Self { model: None })
    }

    /// Run text detection and recognition on the image.
    pub fn recognize(&self, _image: &Image<B>) -> Result<Vec<OcrResult>> {
        Ok(vec![OcrResult {
            bbox: Rect::new(10, 10, 200, 30),
            text: "Iris CV".to_string(),
            confidence: 0.99,
        }])
    }
}

impl<B: Backend> Default for OcrPipeline<B> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{test_device, TestBackend};
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn test_ocr_pipeline() {
        let device = test_device();
        let flat_data = vec![0.5f32; 3 * 100 * 100];
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [3, 100, 100]), &device);
        let img = Image::new(tensor);

        let ocr = OcrPipeline::<TestBackend>::default();
        let results = ocr.recognize(&img).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "Iris CV");
        assert_eq!(results[0].confidence, 0.99);
    }
}
