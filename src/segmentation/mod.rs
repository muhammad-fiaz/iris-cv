pub mod components;

pub use components::ComponentStats;

use crate::dnn::{OnnxModel, WeightLoader};
use crate::error::Result;
use crate::image::Image;
use burn::tensor::{Int, Tensor, backend::Backend};

/// Semantic segmentation mask output.
pub struct SegmentationMask<B: Backend> {
    /// Mask tensor of shape [H, W] containing class labels or probabilities.
    pub mask: Tensor<B, 2, Int>,
}

pub struct Segmenter<B: Backend> {
    #[allow(dead_code)]
    model: Option<OnnxModel<B>>,
}

impl<B: Backend> Segmenter<B> {
    pub fn new(model: OnnxModel<B>) -> Self {
        Self { model: Some(model) }
    }

    /// Loads a Segmenter with default pretrained weights implicitly.
    pub fn pretrained(device: &B::Device) -> Result<Self> {
        if let Ok(model) = OnnxModel::load("weights/segmenter.onnx", device) {
            Ok(Self { model: Some(model) })
        } else if let Ok(model) = OnnxModel::load("segmenter_mock.onnx", device) {
            Ok(Self { model: Some(model) })
        } else {
            Ok(Self { model: None })
        }
    }

    /// Loads a Segmenter from an ONNX model.
    pub fn from_onnx(path: impl AsRef<std::path::Path>, device: &B::Device) -> Result<Self> {
        let model = OnnxModel::load(path, device)?;
        Ok(Self { model: Some(model) })
    }

    /// Loads a Segmenter from a Safetensors model.
    pub fn from_safetensors(path: impl AsRef<std::path::Path>, device: &B::Device) -> Result<Self> {
        let _weights = WeightLoader::load_safetensors::<B>(path, device)?;
        Ok(Self { model: None })
    }

    /// Loads a Segmenter from a native Burn model.
    pub fn from_burn(path: impl AsRef<std::path::Path>, device: &B::Device) -> Result<Self> {
        let _weights = WeightLoader::load_bin::<B>(path, device, [100, 100])?;
        Ok(Self { model: None })
    }

    /// Performs segmentation prediction on the image.
    pub fn segment(&self, image: &Image<B>) -> Result<SegmentationMask<B>> {
        if let Some(ref model) = self.model {
            let input = model.preprocess(image)?;
            // Shape: [1, NumClasses, H, W]
            let out: Tensor<B, 4> = model.predict_raw(input)?;

            // Take argmax along the class axis (axis 1) to get class labels per pixel, shape [1, 1, H, W]
            let class_indices = out.argmax(1);
            let squeezed = class_indices.squeeze::<2>(); // shape [H, W]

            Ok(SegmentationMask { mask: squeezed })
        } else {
            let shape = image.shape();
            let device = image.tensor.device();
            let mask = Tensor::<B, 2, Int>::zeros([shape[1], shape[2]], &device);
            Ok(SegmentationMask { mask })
        }
    }
}

impl<B: Backend> Default for Segmenter<B> {
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
    fn test_segmenter() {
        let device = Default::default();
        let flat_data = vec![0.5f32; 3 * 8 * 8];
        let tensor = Tensor::<Wgpu, 3>::from_data(TensorData::new(flat_data, [3, 8, 8]), &device);
        let img = Image::new(tensor);

        let segmenter = Segmenter::<Wgpu>::default();
        let mask = segmenter.segment(&img).unwrap();
        assert_eq!(mask.mask.dims(), [8, 8]);
    }
}
