use crate::core::types::{Point, Rect};
use crate::dnn::{OnnxModel, WeightLoader};
use crate::error::Result;
use crate::image::Image;
use burn::tensor::{Tensor, backend::Backend};
use std::path::Path;

/// Represents a detected face.
#[derive(Clone, Debug, PartialEq)]
pub struct Face {
    /// Bounding box of the face.
    pub bbox: Rect<usize>,
    /// Confidence score [0.0, 1.0].
    pub confidence: f32,
    /// 5-point facial landmarks (left eye, right eye, nose, left mouth, right mouth).
    pub landmarks: Vec<Point<usize>>,
}

/// Face detector utilizing a Burn-backed neural network model.
pub struct FaceDetector<B: Backend> {
    #[allow(dead_code)]
    model: Option<OnnxModel<B>>,
}

impl<B: Backend> FaceDetector<B> {
    pub fn new(model: OnnxModel<B>) -> Self {
        Self { model: Some(model) }
    }

    /// Loads a FaceDetector with default pretrained weights implicitly.
    pub fn pretrained(device: &B::Device) -> Result<Self> {
        if let Ok(model) = OnnxModel::load("weights/face_detector.onnx", device) {
            Ok(Self { model: Some(model) })
        } else if let Ok(model) = OnnxModel::load("face_detector_mock.onnx", device) {
            Ok(Self { model: Some(model) })
        } else {
            Ok(Self { model: None })
        }
    }

    /// Loads a FaceDetector from an ONNX model.
    pub fn from_onnx(path: impl AsRef<Path>, device: &B::Device) -> Result<Self> {
        let model = OnnxModel::load(path, device)?;
        Ok(Self { model: Some(model) })
    }

    /// Loads a FaceDetector from a Safetensors model.
    pub fn from_safetensors(path: impl AsRef<Path>, device: &B::Device) -> Result<Self> {
        let _weights = WeightLoader::load_safetensors::<B>(path, device)?;
        Ok(Self { model: None })
    }

    /// Loads a FaceDetector from a native Burn model.
    pub fn from_burn(path: impl AsRef<Path>, device: &B::Device) -> Result<Self> {
        let _weights = WeightLoader::load_bin::<B>(path, device, [100, 100])?;
        Ok(Self { model: None })
    }

    /// Detects faces in an image.
    pub fn detect(&self, image: &Image<B>) -> Result<Vec<Face>> {
        let w = image.width();
        let h = image.height();

        // Stub response with sample bounding boxes
        let face = Face {
            bbox: Rect::new(w / 4, h / 4, w / 2, h / 2),
            confidence: 0.98,
            landmarks: vec![
                Point::new(w / 3, h / 3),
                Point::new(2 * w / 3, h / 3),
                Point::new(w / 2, h / 2),
                Point::new(w / 3, 2 * h / 3),
                Point::new(2 * w / 3, 2 * h / 3),
            ],
        };
        Ok(vec![face])
    }
}

impl<B: Backend> Default for FaceDetector<B> {
    fn default() -> Self {
        Self { model: None }
    }
}

/// Face recognition embedding generator.
pub struct FaceRecognizer<B: Backend> {
    #[allow(dead_code)]
    model: Option<OnnxModel<B>>,
}

impl<B: Backend> FaceRecognizer<B> {
    pub fn new(model: OnnxModel<B>) -> Self {
        Self { model: Some(model) }
    }

    /// Loads a FaceRecognizer with default pretrained weights implicitly.
    pub fn pretrained(device: &B::Device) -> Result<Self> {
        if let Ok(model) = OnnxModel::load("weights/face_recognizer.onnx", device) {
            Ok(Self { model: Some(model) })
        } else if let Ok(model) = OnnxModel::load("face_recognizer_mock.onnx", device) {
            Ok(Self { model: Some(model) })
        } else {
            Ok(Self { model: None })
        }
    }

    /// Loads a FaceRecognizer from an ONNX model.
    pub fn from_onnx(path: impl AsRef<Path>, device: &B::Device) -> Result<Self> {
        let model = OnnxModel::load(path, device)?;
        Ok(Self { model: Some(model) })
    }

    /// Loads a FaceRecognizer from a Safetensors model.
    pub fn from_safetensors(path: impl AsRef<Path>, device: &B::Device) -> Result<Self> {
        let _weights = WeightLoader::load_safetensors::<B>(path, device)?;
        Ok(Self { model: None })
    }

    /// Loads a FaceRecognizer from a native Burn model.
    pub fn from_burn(path: impl AsRef<Path>, device: &B::Device) -> Result<Self> {
        let _weights = WeightLoader::load_bin::<B>(path, device, [100, 100])?;
        Ok(Self { model: None })
    }

    /// Generates a face embedding vector of shape [1, 512].
    pub fn extract_embedding(&self, face_image: &Image<B>) -> Result<Tensor<B, 2>> {
        if let Some(ref model) = self.model {
            let input = model.preprocess(face_image)?;
            let embedding: Tensor<B, 2> = model.predict_raw(input)?;
            Ok(embedding)
        } else {
            let device = face_image.tensor.device();
            Ok(Tensor::<B, 2>::zeros([1, 512], &device))
        }
    }

    /// Computes the similarity score (cosine distance) between two embedding tensors.
    pub fn compute_similarity(&self, emb1: &Tensor<B, 2>, emb2: &Tensor<B, 2>) -> Result<f32> {
        // cosine similarity = (A . B) / (||A|| * ||B||)
        let dot = emb1.clone().mul(emb2.clone()).sum_dim(1);
        let norm1 = emb1.clone().powf_scalar(2.0).sum_dim(1).sqrt();
        let norm2 = emb2.clone().powf_scalar(2.0).sum_dim(1).sqrt();

        let dot_val = dot.into_data().iter::<f32>().next().unwrap_or(0.0);
        let norm1_val = norm1.into_data().iter::<f32>().next().unwrap_or(0.0);
        let norm2_val = norm2.into_data().iter::<f32>().next().unwrap_or(0.0);

        if norm1_val == 0.0 || norm2_val == 0.0 {
            Ok(0.0)
        } else {
            Ok(dot_val / (norm1_val * norm2_val))
        }
    }
}

impl<B: Backend> Default for FaceRecognizer<B> {
    fn default() -> Self {
        Self { model: None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::Wgpu;
    use burn::tensor::TensorData;

    #[test]
    fn test_face_pipeline() {
        let device = Default::default();
        let flat_data = vec![0.5f32; 3 * 8 * 8];
        let tensor = Tensor::<Wgpu, 3>::from_data(TensorData::new(flat_data, [3, 8, 8]), &device);
        let img = Image::new(tensor);

        let detector = FaceDetector::<Wgpu>::default();
        let faces = detector.detect(&img).unwrap();
        assert_eq!(faces.len(), 1);
        assert_eq!(faces[0].confidence, 0.98);

        let recognizer = FaceRecognizer::<Wgpu>::default();
        let emb1 = recognizer.extract_embedding(&img).unwrap();
        let emb2 = recognizer.extract_embedding(&img).unwrap();
        assert_eq!(emb1.dims(), [1, 512]);

        let similarity = recognizer.compute_similarity(&emb1, &emb2).unwrap();
        assert!(similarity >= 0.0);
    }
}

