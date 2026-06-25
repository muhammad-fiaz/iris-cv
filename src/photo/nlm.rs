use crate::error::Result;
use crate::image::Image;
use burn::tensor::{Tensor, TensorData, backend::Backend};

/// Non-Local Means denoising using patch-based similarity weighting.
///
/// For each pixel, searches a neighborhood window for similar patches.
/// The denoised value is a weighted average of all search-window pixels,
/// where weights decay exponentially with patch dissimilarity.
///
/// - `h`: Filter strength (higher = more smoothing). Controls the Gaussian
///   decay of the similarity weights.
/// - `template_size`: Half-size of the comparison patch (actual patch is
///   `(2*template_size+1)^2`).
/// - `search_size`: Half-size of the search window around each pixel.
pub fn fast_nl_means_denoising<B: Backend>(
    image: &Image<B>,
    h: f32,
    template_size: usize,
    search_size: usize,
) -> Result<Image<B>> {
    let dims = image.tensor.dims();
    let c = dims[0];
    let h_img = dims[1];
    let w_img = dims[2];

    let device = image.tensor.device();
    let tensor_data = image.tensor.clone().into_data();
    let flat: Vec<f32> = tensor_data.iter::<f32>().collect();
    let mut out_vals = vec![0.0f32; c * h_img * w_img];

    let patch_r = template_size as isize;
    let search_r = search_size as isize;
    let total_pixels = h_img * w_img;
    let patch_area = (2.0 * patch_r as f32 + 1.0).powi(2);
    let h_sq_inv = 1.0 / (h * h * patch_area * c as f32);

    for ch in 0..c {
        let ch_offset = ch * total_pixels;

        for py in 0..h_img {
            for px in 0..w_img {
                let mut weighted_sum = 0.0f32;
                let mut weight_total = 0.0f32;

                let sy_min = (py as isize - search_r).max(0) as usize;
                let sy_max = ((py as isize + search_r) as usize).min(h_img - 1);
                let sx_min = (px as isize - search_r).max(0) as usize;
                let sx_max = ((px as isize + search_r) as usize).min(w_img - 1);

                for qy in sy_min..=sy_max {
                    for qx in sx_min..=sx_max {
                        let mut ssd = 0.0f32;

                        for dy in -patch_r..=patch_r {
                            for dx in -patch_r..=patch_r {
                                let ry1 = (py as isize + dy).clamp(0, h_img as isize - 1) as usize;
                                let rx1 = (px as isize + dx).clamp(0, w_img as isize - 1) as usize;
                                let ry2 = (qy as isize + dy).clamp(0, h_img as isize - 1) as usize;
                                let rx2 = (qx as isize + dx).clamp(0, w_img as isize - 1) as usize;

                                let v1 = flat[ch_offset + ry1 * w_img + rx1];
                                let v2 = flat[ch_offset + ry2 * w_img + rx2];
                                let diff = v1 - v2;
                                ssd += diff * diff;
                            }
                        }

                        let weight = (-ssd * h_sq_inv).exp();
                        weighted_sum += flat[ch_offset + qy * w_img + qx] * weight;
                        weight_total += weight;
                    }
                }

                out_vals[ch_offset + py * w_img + px] = if weight_total > 0.0 {
                    (weighted_sum / weight_total).clamp(0.0, 1.0)
                } else {
                    flat[ch_offset + py * w_img + px]
                };
            }
        }
    }

    let new_data = TensorData::new(out_vals, [c, h_img, w_img]);
    let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
    Ok(Image::new(new_tensor))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{TestBackend, test_device};

    #[test]
    fn test_nl_means_denoising() {
        let device = test_device();
        let flat_data = vec![0.5f32; 3 * 12 * 12];
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [3, 12, 12]), &device);
        let img = Image::new(tensor);

        let denoised = fast_nl_means_denoising::<TestBackend>(&img, 10.0, 1, 2).unwrap();
        assert_eq!(denoised.shape(), [3, 12, 12]);

        // Uniform image should remain unchanged
        let vals: Vec<f32> = denoised.tensor.into_data().iter::<f32>().collect();
        for v in &vals {
            assert!((*v - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn test_nl_means_reduces_noise() {
        let device = test_device();
        let mut flat_data = vec![0.5f32; 16 * 16];
        flat_data[7 * 16 + 7] = 0.9;
        flat_data[7 * 16 + 8] = 0.1;
        flat_data[8 * 16 + 7] = 0.1;
        flat_data[8 * 16 + 8] = 0.9;

        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [1, 16, 16]), &device);
        let img = Image::new(tensor);

        let denoised = fast_nl_means_denoising::<TestBackend>(&img, 15.0, 1, 3).unwrap();
        assert_eq!(denoised.shape(), [1, 16, 16]);

        let vals: Vec<f32> = denoised.tensor.into_data().iter::<f32>().collect();
        let center_val =
            (vals[7 * 16 + 7] + vals[7 * 16 + 8] + vals[8 * 16 + 7] + vals[8 * 16 + 8]) / 4.0;
        assert!(center_val < 0.7 && center_val > 0.3);
    }
}
