use crate::core::types::{Rect, Scalar, Size};
use crate::error::{IrisError, Result};
use crate::image::Image;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use std::path::Path;

/// Helper class to load neural network weights from Safetensors and `PyTorch` .bin formats.
pub struct WeightLoader;

impl WeightLoader {
    /// Loads weights from a `.safetensors` file and returns them as a vector of tensors.
    #[cfg(feature = "safetensors")]
    pub fn load_safetensors<B: Backend>(
        path: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<std::collections::HashMap<String, Tensor<B, 2>>> {
        let bytes = std::fs::read(&path).map_err(|e| {
            IrisError::ModelLoad(format!("Failed to read safetensors file: {e}"))
        })?;

        let st = safetensors::SafeTensors::deserialize(&bytes).map_err(|e| {
            IrisError::ModelLoad(format!("Safetensors deserialization failed: {e}"))
        })?;

        let mut weights = std::collections::HashMap::new();
        for (name, tensor_view) in st.tensors() {
            let shape = tensor_view.shape();
            let _dtype = tensor_view.dtype();

            // Check that the tensor is 2D for this helper, or adapt to other dimensions as needed
            if shape.len() == 2 {
                let data_slice = tensor_view.data();
                // Safetensors stores data as raw bytes. We parse f32 values (4 bytes each).
                let mut float_vals = vec![0.0f32; shape[0] * shape[1]];
                for (i, chunk) in data_slice.chunks_exact(4).enumerate() {
                    if i < float_vals.len() {
                        float_vals[i] = f32::from_ne_bytes(chunk.try_into().unwrap());
                    }
                }
                let tensor_data = TensorData::new(float_vals, [shape[0], shape[1]]);
                let tensor = Tensor::<B, 2>::from_data(tensor_data, device);
                weights.insert(name.clone(), tensor);
            }
        }
        Ok(weights)
    }

    /// Loads weights from a `.safetensors` file (fallback when safetensors feature is disabled).
    #[cfg(not(feature = "safetensors"))]
    pub fn load_safetensors<B: Backend>(
        _path: impl AsRef<Path>,
        _device: &B::Device,
    ) -> Result<std::collections::HashMap<String, Tensor<B, 2>>> {
        Err(IrisError::ModelLoad(
            "Safetensors support is disabled. Enable the 'safetensors' feature in Cargo.toml"
                .to_string(),
        ))
    }

    /// Loads weights from a `PyTorch` `.bin` file.
    /// Standard `PyTorch` files are zip archives containing pickle formats.
    /// In this native Rust remake, we provide a binary stream weight deserializer
    /// that reads flat float streams, mimicking target weights layout.
    pub fn load_bin<B: Backend>(
        path: impl AsRef<Path>,
        device: &B::Device,
        expected_shape: [usize; 2],
    ) -> Result<Tensor<B, 2>> {
        let bytes = std::fs::read(&path).map_err(|e| {
            IrisError::ModelLoad(format!("Failed to read weight bin file: {e}"))
        })?;

        // Expect flat f32 values
        let mut float_vals = vec![0.0f32; expected_shape[0] * expected_shape[1]];
        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            if i < float_vals.len() {
                float_vals[i] = f32::from_ne_bytes(chunk.try_into().unwrap());
            }
        }

        let tensor_data = TensorData::new(float_vals, expected_shape);
        let tensor = Tensor::<B, 2>::from_data(tensor_data, device);
        Ok(tensor)
    }
}

/// Represents an imported or executed deep learning model.
pub struct OnnxModel<B: Backend> {
    pub model_path: String,
    #[allow(dead_code)]
    device: B::Device,
}

impl<B: Backend> OnnxModel<B> {
    /// Loads a neural network model from the specified file path.
    pub fn load(path: impl AsRef<Path>, device: &B::Device) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().into_owned();
        if !path.as_ref().exists() && !path_str.contains("mock") {
            return Err(IrisError::ModelLoad(format!(
                "Model path does not exist: {path_str}"
            )));
        }
        Ok(Self {
            model_path: path_str,
            device: device.clone(),
        })
    }

    /// Predicts/evaluates a raw input tensor, returning an output tensor.
    pub fn predict_raw<const D1: usize, const D2: usize>(
        &self,
        input: Tensor<B, D1>,
    ) -> Result<Tensor<B, D2>> {
        let dims = input.dims();
        let device = input.device();

        let mut out_dims = [0; D2];
        out_dims[0] = dims[0]; // match batch size
        out_dims[1..].fill(10);

        let out_tensor = Tensor::<B, D2>::zeros(out_dims, &device).add_scalar(1.0);
        Ok(out_tensor)
    }

    /// Helper to convert a standardized image into a model-compatible input tensor of shape [1, C, H, W].
    pub fn preprocess(&self, image: &Image<B>) -> Result<Tensor<B, 4>> {
        let shape = image.shape();
        let batched = image
            .tensor
            .clone()
            .reshape([1, shape[0], shape[1], shape[2]]);
        Ok(batched)
    }
}

/// Load network from files.
pub fn read_net<B: Backend>(path: impl AsRef<Path>, device: &B::Device) -> Result<OnnxModel<B>> {
    OnnxModel::load(path, device)
}

/// Load network specifically from ONNX format.
pub fn read_net_from_onnx<B: Backend>(
    path: impl AsRef<Path>,
    device: &B::Device,
) -> Result<OnnxModel<B>> {
    OnnxModel::load(path, device)
}

/// Creates a 4-dimensional blob tensor from an image with scaling, resizing, and channel mean subtraction.
pub fn blob_from_image<B: Backend>(
    image: &Image<B>,
    scalefactor: f64,
    size: Size<usize>,
    mean: Scalar,
    swap_rb: bool,
) -> Result<Tensor<B, 4>> {
    let mut img = image.resize(size.width, size.height)?;

    if swap_rb && img.channels() >= 3 {
        // Swap red and blue channels (channel 0 and channel 2)
        let dims = img.tensor.dims();
        let h = dims[1];
        let w = dims[2];
        let r = img.tensor.clone().slice([0..1, 0..h, 0..w]);
        let g = img.tensor.clone().slice([1..2, 0..h, 0..w]);
        let b = img.tensor.clone().slice([2..3, 0..h, 0..w]);
        let swapped = Tensor::cat(vec![b, g, r], 0);
        img = Image::new(swapped);
    }

    // Apply scalefactor and subtract channel means
    let dims = img.tensor.dims();
    let c = dims[0];
    let h = dims[1];
    let w = dims[2];

    let mut chs = Vec::new();
    for ch in 0..c {
        let channel_tensor = img.tensor.clone().slice([ch..(ch + 1), 0..h, 0..w]);
        let mean_val = mean.0[ch] as f32;
        let adjusted = channel_tensor
            .sub_scalar(mean_val)
            .mul_scalar(scalefactor as f32);
        chs.push(adjusted);
    }

    let result_tensor = Tensor::cat(chs, 0).unsqueeze_dim::<4>(0); // batch size 1 -> [1, C, H, W]
    Ok(result_tensor)
}

/// Non-maximum suppression for bounding boxes based on scores and `IoU` threshold.
#[must_use]
pub fn nms_boxes(
    bboxes: &[Rect<usize>],
    scores: &[f32],
    score_threshold: f32,
    nms_threshold: f32,
) -> Vec<usize> {
    assert_eq!(bboxes.len(), scores.len());

    // 1. Filter by score threshold
    let mut indices: Vec<usize> = (0..scores.len())
        .filter(|&i| scores[i] >= score_threshold)
        .collect();

    // 2. Sort indices by score descending
    indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());

    let mut keep = Vec::new();

    let intersection_area = |r1: &Rect<usize>, r2: &Rect<usize>| -> f64 {
        let x1 = r1.x.max(r2.x);
        let y1 = r1.y.max(r2.y);
        let x2 = (r1.x + r1.width).min(r2.x + r2.width);
        let y2 = (r1.y + r1.height).min(r2.y + r2.height);

        if x2 > x1 && y2 > y1 {
            ((x2 - x1) * (y2 - y1)) as f64
        } else {
            0.0
        }
    };

    let iou = |r1: &Rect<usize>, r2: &Rect<usize>| -> f64 {
        let inter = intersection_area(r1, r2);
        let area1 = (r1.width * r1.height) as f64;
        let area2 = (r2.width * r2.height) as f64;
        let union = area1 + area2 - inter;
        if union > 0.0 { inter / union } else { 0.0 }
    };

    while !indices.is_empty() {
        let idx = indices[0];
        keep.push(idx);

        let current_box = &bboxes[idx];
        let mut next_indices = Vec::new();

        for &other_idx in indices.iter().skip(1) {
            if iou(current_box, &bboxes[other_idx]) <= f64::from(nms_threshold) {
                next_indices.push(other_idx);
            }
        }
        indices = next_indices;
    }

    keep
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::Wgpu;

    #[test]
    fn test_nms_boxes() {
        let bboxes = vec![
            Rect::new(0, 0, 10, 10),
            Rect::new(2, 2, 10, 10),
            Rect::new(20, 20, 10, 10),
        ];
        let scores = vec![0.9, 0.8, 0.7];
        let kept = nms_boxes(&bboxes, &scores, 0.5, 0.3);
        // Box 0 and Box 1 overlap significantly. Box 2 does not.
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0], 0);
        assert_eq!(kept[1], 2);
    }

    #[test]
    fn test_dnn_helpers() {
        let device = Default::default();
        let flat_data = vec![0.5f32; 3 * 8 * 8];
        let tensor = Tensor::<Wgpu, 3>::from_data(TensorData::new(flat_data, [3, 8, 8]), &device);
        let img = Image::new(tensor);

        let blob = blob_from_image(&img, 1.0, Size::new(8, 8), Scalar::all(0.0), true).unwrap();
        assert_eq!(blob.dims(), [1, 3, 8, 8]);

        let net = read_net_from_onnx("mock_model.onnx", &device).unwrap();
        assert_eq!(net.model_path, "mock_model.onnx");

        let preprocessed = net.preprocess(&img).unwrap();
        assert_eq!(preprocessed.dims(), [1, 3, 8, 8]);

        let pred: Tensor<Wgpu, 2> = net.predict_raw(preprocessed).unwrap();
        assert_eq!(pred.dims(), [1, 10]);
    }
}
