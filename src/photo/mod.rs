pub mod nlm;

use crate::error::{IrisError, Result};
use crate::image::Image;
use burn::tensor::backend::Backend;

/// Photo processing and enhancement algorithms.
pub struct Photo;

impl Photo {
    /// Non-Local Means Denoising filter with patch-based similarity.
    ///
    /// For each pixel, searches a larger neighborhood for patches similar to the
    /// current pixel's patch. Pixels in similar patches are weighted by their
    /// Gaussian-weighted distance to produce the denoised value.
    ///
    /// # Arguments
    /// * `image` - Input image with values in [0, 1]
    /// * `h` - Filter strength (higher removes more noise but may blur details)
    /// * `patch_radius` - Half-size of the comparison patch (default: 3 → 7×7 patch)
    /// * `search_radius` - Half-size of the search window (default: 5 → 11×11 window)
    pub fn fast_nl_means_denoising<B: Backend>(
        image: &Image<B>,
        h: f32,
        patch_radius: usize,
        search_radius: usize,
    ) -> Result<Image<B>> {
        let dims = image.tensor.dims();
        let c = dims[0];
        let img_h = dims[1];
        let img_w = dims[2];

        let tensor_data = image.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; c * img_h * img_w];

        let h_sq_inv = 1.0 / (h * h * (2.0 * patch_radius as f32 + 1.0).powi(2));

        for ch in 0..c {
            let ch_offset = ch * img_h * img_w;
            for cy in 0..img_h {
                for cx in 0..img_w {
                    let center_idx = ch_offset + cy * img_w + cx;
                    let center_val = flat_vals[center_idx];

                    let mut weight_sum = 0.0f32;
                    let mut value_sum = 0.0f32;

                    // Search window around (cx, cy)
                    let sy_start = cy.saturating_sub(search_radius);
                    let sy_end = (cy + search_radius + 1).min(img_h);
                    let sx_start = cx.saturating_sub(search_radius);
                    let sx_end = (cx + search_radius + 1).min(img_w);

                    for sy in sy_start..sy_end {
                        for sx in sx_start..sx_end {
                            // Compute patch distance (L2 between patches centered at center and at (sx, sy))
                            let mut patch_dist = 0.0f32;
                            let mut valid = true;

                            for py in 0..=(2 * patch_radius) {
                                for px in 0..=(2 * patch_radius) {
                                    let offy = py as isize - patch_radius as isize;
                                    let offx = px as isize - patch_radius as isize;

                                    let ny_c = cy as isize + offy;
                                    let nx_c = cx as isize + offx;
                                    let ny_s = sy as isize + offy;
                                    let nx_s = sx as isize + offx;

                                    if ny_c < 0
                                        || ny_c >= img_h as isize
                                        || nx_c < 0
                                        || nx_c >= img_w as isize
                                        || ny_s < 0
                                        || ny_s >= img_h as isize
                                        || nx_s < 0
                                        || nx_s >= img_w as isize
                                    {
                                        valid = false;
                                        break;
                                    }

                                    let val_c = flat_vals
                                        [ch_offset + ny_c as usize * img_w + nx_c as usize];
                                    let val_s = flat_vals
                                        [ch_offset + ny_s as usize * img_w + nx_s as usize];
                                    let diff = val_c - val_s;
                                    patch_dist += diff * diff;
                                }
                                if !valid {
                                    break;
                                }
                            }

                            if !valid {
                                continue;
                            }

                            // Gaussian spatial weight (center of search is preferred)
                            let dx = (sx as f64 - cx as f64) as f32;
                            let dy = (sy as f64 - cy as f64) as f32;
                            let spatial_dist =
                                (dx * dx + dy * dy) / (search_radius as f32 * search_radius as f32);
                            let spatial_weight = (-spatial_dist * 2.0).exp();

                            let weight = (-patch_dist * h_sq_inv).exp() * spatial_weight;
                            value_sum += flat_vals[ch_offset + sy * img_w + sx] * weight;
                            weight_sum += weight;
                        }
                    }

                    out_vals[center_idx] = if weight_sum > 1e-10 {
                        (value_sum / weight_sum).clamp(0.0, 1.0)
                    } else {
                        center_val
                    };
                }
            }
        }

        let device = image.tensor.device();
        let data = burn::tensor::TensorData::new(out_vals, [c, img_h, img_w]);
        let tensor = burn::tensor::Tensor::<B, 3>::from_data(data, &device);
        Ok(Image::new(tensor))
    }
}

/// HDR Merging using Mertens exposure fusion algorithm.
///
/// Fuses multiple differently-exposed images into a single image with
/// expanded dynamic range using perceptually-motivated weight maps.
///
/// The algorithm computes three weight maps per image:
/// - **Contrast**: absolute value of the Laplacian (edge strength)
/// - **Saturation**: standard deviation of the color channels per pixel
/// - **Exposure**: how close the luminance is to 0.5 (optimal exposure)
///
/// These are multiplied together and normalized across all images to produce
/// the final fused result.
pub struct MergeMertens {
    pub contrast_weight: f32,
    pub saturation_weight: f32,
    pub exposure_weight: f32,
}

impl MergeMertens {
    #[must_use]
    pub fn new() -> Self {
        Self {
            contrast_weight: 1.0,
            saturation_weight: 1.0,
            exposure_weight: 1.0,
        }
    }

    /// Sets the contrast weight (default 1.0).
    #[must_use]
    pub fn with_contrast_weight(mut self, w: f32) -> Self {
        self.contrast_weight = w;
        self
    }

    /// Sets the saturation weight (default 1.0).
    #[must_use]
    pub fn with_saturation_weight(mut self, w: f32) -> Self {
        self.saturation_weight = w;
        self
    }

    /// Sets the exposure weight (default 1.0).
    #[must_use]
    pub fn with_exposure_weight(mut self, w: f32) -> Self {
        self.exposure_weight = w;
        self
    }

    /// Merges multiple exposure images using Mertens exposure fusion.
    ///
    /// All images must have the same dimensions and 3 channels.
    pub fn process<B: Backend>(&self, images: &[Image<B>]) -> Result<Image<B>> {
        if images.is_empty() {
            return Err(IrisError::InvalidParameter(
                "Images list cannot be empty".into(),
            ));
        }

        let dims = images[0].tensor.dims();
        if dims[0] != 3 {
            return Err(IrisError::InvalidParameter(
                "Input images must be 3-channel RGB".into(),
            ));
        }
        let h = dims[1];
        let w = dims[2];
        let n = images.len();

        // Validate all images have the same dimensions
        for img in images.iter().skip(1) {
            let d = img.tensor.dims();
            if d[0] != 3 || d[1] != h || d[2] != w {
                return Err(IrisError::DimensionMismatch {
                    expected: vec![3, h, w],
                    actual: vec![d[0], d[1], d[2]],
                });
            }
        }

        // Compute weight maps for each image
        let mut all_weights: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut all_pixel_data: Vec<Vec<f32>> = Vec::with_capacity(n);

        for img in images {
            let data = img.tensor.clone().into_data();
            let flat: Vec<f32> = data.iter::<f32>().collect();

            let mut weights = vec![1.0f32; h * w];

            // 1. Contrast weight: absolute Laplacian on grayscale
            let gray: Vec<f32> = (0..h * w)
                .map(|i| 0.299 * flat[i] + 0.587 * flat[h * w + i] + 0.114 * flat[2 * h * w + i])
                .collect();

            for y in 0..h {
                for x in 0..w {
                    let c = gray[y * w + x];
                    let mut laplacian = -4.0 * c;
                    if y > 0 {
                        laplacian += gray[(y - 1) * w + x];
                    }
                    if y + 1 < h {
                        laplacian += gray[(y + 1) * w + x];
                    }
                    if x > 0 {
                        laplacian += gray[y * w + x - 1];
                    }
                    if x + 1 < w {
                        laplacian += gray[y * w + x + 1];
                    }
                    let contrast = laplacian.abs();
                    let ci = y * w + x;
                    weights[ci] *= contrast.powf(self.contrast_weight);
                }
            }

            // 2. Saturation weight: std deviation of color channels per pixel
            for y in 0..h {
                for x in 0..w {
                    let ci = y * w + x;
                    let r = flat[ci];
                    let g = flat[h * w + ci];
                    let b = flat[2 * h * w + ci];
                    let mean = (r + g + b) / 3.0;
                    let variance =
                        ((r - mean).powi(2) + (g - mean).powi(2) + (b - mean).powi(2)) / 3.0;
                    let saturation = variance.sqrt();
                    weights[ci] *= saturation.powf(self.saturation_weight);
                }
            }

            // 3. Exposure weight: Gaussian distance to optimal exposure (0.5)
            for y in 0..h {
                for x in 0..w {
                    let ci = y * w + x;
                    let lum = gray[ci];
                    let sigma = 0.2;
                    let diff = lum - 0.5f32;
                    let exposure_w = (-(diff * diff) / (2.0 * sigma * sigma)).exp();
                    weights[ci] *= exposure_w.powf(self.exposure_weight);
                }
            }

            all_pixel_data.push(flat);
            all_weights.push(weights);
        }

        // Normalize weights across all images so they sum to 1 at each pixel
        let mut out_vals = vec![0.0f32; 3 * h * w];
        for y in 0..h {
            for x in 0..w {
                let ci = y * w + x;
                let mut total_weight = 0.0f32;

                for w_map in &all_weights {
                    total_weight += w_map[ci];
                }

                if total_weight < 1e-10 {
                    // Fallback: simple average
                    for ch in 0..3 {
                        let mut sum = 0.0f32;
                        for pixel_data in &all_pixel_data {
                            sum += pixel_data[ch * h * w + ci];
                        }
                        out_vals[ch * h * w + ci] = sum / n as f32;
                    }
                } else {
                    for ch in 0..3 {
                        let mut blended = 0.0f32;
                        for (idx, pixel_data) in all_pixel_data.iter().enumerate() {
                            blended +=
                                pixel_data[ch * h * w + ci] * all_weights[idx][ci] / total_weight;
                        }
                        out_vals[ch * h * w + ci] = blended.clamp(0.0, 1.0);
                    }
                }
            }
        }

        let device = images[0].tensor.device();
        let data = burn::tensor::TensorData::new(out_vals, [3, h, w]);
        let tensor = burn::tensor::Tensor::<B, 3>::from_data(data, &device);
        Ok(Image::new(tensor))
    }
}

impl Default for MergeMertens {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{TestBackend, test_device};
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn test_nlmeans_denoising() {
        let device = test_device();
        let flat_data = vec![0.5f32; 3 * 16 * 16];
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [3, 16, 16]), &device);
        let img = Image::new(tensor);

        let denoised = Photo::fast_nl_means_denoising(&img, 12.0, 3, 5).unwrap();
        assert_eq!(denoised.shape(), [3, 16, 16]);

        // Uniform image should remain uniform after denoising
        let out_data = denoised.tensor.into_data();
        let out_vals: Vec<f32> = out_data.iter::<f32>().collect();
        for v in &out_vals {
            assert!(
                (*v - 0.5).abs() < 1e-5,
                "Uniform image should stay uniform, got {}",
                v
            );
        }
    }

    #[test]
    fn test_mertens_exposure_fusion() {
        let device = test_device();

        // Create two different "exposures" of a scene
        // Image 1: darker
        let mut data1 = vec![0.3f32; 3 * 16 * 16];
        // Image 2: brighter
        let mut data2 = vec![0.7f32; 3 * 16 * 16];

        // Add some variation so weights differ
        for y in 0..16 {
            for x in 0..16 {
                let ci = y * 16 + x;
                // Add gradient to image 1
                data1[ci] = 0.2 + 0.4 * (x as f32 / 16.0);
                data1[16 * 16 + ci] = 0.2 + 0.3 * (y as f32 / 16.0);
                data1[2 * 16 * 16 + ci] = 0.3;
                // Add gradient to image 2
                data2[ci] = 0.5 + 0.3 * (x as f32 / 16.0);
                data2[16 * 16 + ci] = 0.4 + 0.4 * (y as f32 / 16.0);
                data2[2 * 16 * 16 + ci] = 0.6;
            }
        }

        let img1 = Image::new(Tensor::<TestBackend, 3>::from_data(
            TensorData::new(data1, [3, 16, 16]),
            &device,
        ));
        let img2 = Image::new(Tensor::<TestBackend, 3>::from_data(
            TensorData::new(data2, [3, 16, 16]),
            &device,
        ));

        let mertens = MergeMertens::new();
        let merged = mertens.process(&[img1, img2]).unwrap();
        assert_eq!(merged.shape(), [3, 16, 16]);

        // Verify output is within [0, 1]
        let out_data = merged.tensor.into_data();
        let out_vals: Vec<f32> = out_data.iter::<f32>().collect();
        for v in &out_vals {
            assert!(*v >= 0.0 && *v <= 1.0, "Output out of range: {}", v);
        }
    }

    #[test]
    fn test_mertens_empty_input() {
        let mertens = MergeMertens::new();
        let empty: Vec<Image<TestBackend>> = vec![];
        assert!(mertens.process(&empty).is_err());
    }
}
