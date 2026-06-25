use crate::error::{IrisError, Result};
use crate::image::Image;
use burn::tensor::{Tensor, TensorData, backend::Backend};

/// Histogram of Oriented Gradients descriptor.
///
/// Computes a feature descriptor based on gradient orientation histograms
/// localized to fine-spatial regions (cells), grouped into blocks for contrast normalization.
pub struct HogDescriptor {
    cell_size: usize,
    block_size: usize,
    nbins: usize,
}

impl HogDescriptor {
    /// Creates a new HOG descriptor.
    ///
    /// - `cell_size`: Size of each cell in pixels (e.g., 8).
    /// - `block_size`: Number of cells per block (e.g., 2).
    /// - `nbins`: Number of orientation bins (e.g., 9).
    pub fn new(cell_size: usize, block_size: usize, nbins: usize) -> Self {
        Self {
            cell_size,
            block_size,
            nbins,
        }
    }

    /// Computes the HOG descriptor for the given image.
    ///
    /// The image is converted to grayscale, gradient magnitude and direction are computed,
    /// orientation histograms are built per cell, and blocks are L2-normalized.
    /// Returns a 1D tensor containing the concatenated block descriptors.
    pub fn compute<B: Backend>(&self, image: &Image<B>) -> Result<Tensor<B, 1>> {
        let gray = image.grayscale()?;
        let dims = gray.tensor.dims();
        let h = dims[1];
        let w = dims[2];

        if h < self.cell_size || w < self.cell_size {
            return Err(IrisError::InvalidParameter(format!(
                "Image {}x{} too small for cell_size {}",
                w, h, self.cell_size
            )));
        }

        let device = gray.tensor.device();
        let tensor_data = gray.tensor.clone().into_data();
        let flat: Vec<f32> = tensor_data.iter::<f32>().collect();

        // --- Compute gradient magnitude and direction ---
        let mut magnitude = vec![0.0f32; h * w];
        let mut direction = vec![0.0f32; h * w];

        for y in 0..h {
            for x in 0..w {
                // Horizontal gradient (Sobel-like)
                let gx = if x == 0 {
                    flat[y * w + 1] - flat[y * w]
                } else if x == w - 1 {
                    flat[y * w + w - 1] - flat[y * w + w - 2]
                } else {
                    flat[y * w + x + 1] - flat[y * w + x - 1]
                };

                // Vertical gradient
                let gy = if y == 0 {
                    flat[w + x] - flat[x]
                } else if y == h - 1 {
                    flat[(h - 1) * w + x] - flat[(h - 2) * w + x]
                } else {
                    flat[(y + 1) * w + x] - flat[(y - 1) * w + x]
                };

                magnitude[y * w + x] = (gx * gx + gy * gy).sqrt();
                direction[y * w + x] = gy.atan2(gx); // range [-pi, pi]
            }
        }

        // --- Build orientation histograms per cell ---
        let n_cells_y = h / self.cell_size;
        let n_cells_x = w / self.cell_size;
        let bin_width = std::f32::consts::PI / self.nbins as f32;

        // cell_hists[cell_y][cell_x][bin]
        let mut cell_hists = vec![vec![vec![0.0f32; self.nbins]; n_cells_x]; n_cells_y];

        for cy in 0..n_cells_y {
            for cx in 0..n_cells_x {
                let y_start = cy * self.cell_size;
                let x_start = cx * self.cell_size;

                for dy in 0..self.cell_size {
                    for dx in 0..self.cell_size {
                        let y = y_start + dy;
                        let x = x_start + dx;
                        let mag = magnitude[y * w + x];
                        let ang = direction[y * w + x]; // [-pi, pi]

                        // Map angle to [0, pi] (unsigned gradients)
                        let angle = if ang < 0.0 { ang + std::f32::consts::PI } else { ang };

                        let bin_f = angle / bin_width;
                        let bin0 = (bin_f as usize) % self.nbins;
                        let bin1 = (bin0 + 1) % self.nbins;
                        let frac = bin_f - bin0 as f32;

                        cell_hists[cy][cx][bin0] += mag * (1.0 - frac);
                        cell_hists[cy][cx][bin1] += mag * frac;
                    }
                }
            }
        }

        // --- Normalize blocks and concatenate ---
        let n_blocks_y = n_cells_y.saturating_sub(self.block_size - 1);
        let n_blocks_x = n_cells_x.saturating_sub(self.block_size - 1);
        let block_desc_len = self.block_size * self.block_size * self.nbins;
        let total_len = n_blocks_y * n_blocks_x * block_desc_len;

        let mut descriptor = vec![0.0f32; total_len];

        for by in 0..n_blocks_y {
            for bx in 0..n_blocks_x {
                let mut block_vec: Vec<f32> = Vec::with_capacity(block_desc_len);
                for dy in 0..self.block_size {
                    for dx in 0..self.block_size {
                        let hist = &cell_hists[by + dy][bx + dx];
                        block_vec.extend_from_slice(hist);
                    }
                }

                // L2 normalize the block
                let eps = 1e-6f32;
                let norm: f32 = block_vec.iter().map(|v| v * v).sum::<f32>().sqrt() + eps;
                for v in &mut block_vec {
                    *v /= norm;
                }

                let block_idx = (by * n_blocks_x + bx) * block_desc_len;
                descriptor[block_idx..block_idx + block_desc_len].copy_from_slice(&block_vec);
            }
        }

        let data = TensorData::new(descriptor, [total_len]);
        let tensor = Tensor::<B, 1>::from_data(data, &device);
        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{TestBackend, test_device};

    #[test]
    fn test_hog_descriptor_computation() {
        let device = test_device();
        // 32x32 grayscale image
        let flat_data = vec![0.5f32; 32 * 32];
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [1, 32, 32]), &device);
        let img = Image::new(tensor);

        let hog = HogDescriptor::new(8, 2, 9);
        let descriptor = hog.compute(&img).unwrap();

        // n_cells = 32/8 = 4, n_blocks = (4-2+1)^2 = 9
        // block_desc_len = 2*2*9 = 36
        // total = 9 * 36 = 324
        assert_eq!(descriptor.dims(), [324]);
    }

    #[test]
    fn test_hog_too_small_image() {
        let device = test_device();
        let flat_data = vec![0.5f32; 4 * 4];
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [1, 4, 4]), &device);
        let img = Image::new(tensor);

        let hog = HogDescriptor::new(8, 2, 9);
        assert!(hog.compute(&img).is_err());
    }
}
