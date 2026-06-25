use crate::error::Result;
use crate::image::Image;
use burn::tensor::{Tensor, TensorData, backend::Backend};

impl<B: Backend> Image<B> {
    /// Applies Scharr operator to find horizontal and vertical gradients.
    pub fn scharr(&self) -> Result<Self> {
        let gray = self.grayscale()?;
        let dims = gray.tensor.dims();
        let h = dims[1];
        let w = dims[2];

        let device = gray.tensor.device();
        let tensor_data = gray.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; h * w];

        let kx = [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]];
        let ky = [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]];

        {
            use rayon::prelude::*;
            out_vals
                .par_chunks_exact_mut(w)
                .enumerate()
                .skip(1)
                .take(h - 2)
                .for_each(|(y, row)| {
                    for x in 1..(w - 1) {
                        let mut gx = 0.0f32;
                        let mut gy = 0.0f32;

                        for dy in -1..=1 {
                            for dx in -1..=1 {
                                let val = flat_vals
                                    [(y as isize + dy) as usize * w + (x as isize + dx) as usize];
                                gx += val * kx[(dy + 1) as usize][(dx + 1) as usize] as f32;
                                gy += val * ky[(dy + 1) as usize][(dx + 1) as usize] as f32;
                            }
                        }
                        row[x] = (gx * gx + gy * gy).sqrt();
                    }
                });
        }

        let new_data = TensorData::new(out_vals, [1, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Computes the Laplacian of an image.
    pub fn laplacian(&self) -> Result<Self> {
        let gray = self.grayscale()?;
        let dims = gray.tensor.dims();
        let h = dims[1];
        let w = dims[2];

        let device = gray.tensor.device();
        let tensor_data = gray.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; h * w];

        // Standard 3x3 Laplacian kernel
        let kernel = [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]];

        {
            use rayon::prelude::*;
            out_vals
                .par_chunks_exact_mut(w)
                .enumerate()
                .skip(1)
                .take(h - 2)
                .for_each(|(y, row)| {
                    for x in 1..(w - 1) {
                        let mut sum = 0.0f32;
                        for dy in -1..=1 {
                            for dx in -1..=1 {
                                let val = flat_vals
                                    [(y as isize + dy) as usize * w + (x as isize + dx) as usize];
                                sum += val * kernel[(dy + 1) as usize][(dx + 1) as usize] as f32;
                            }
                        }
                        row[x] = sum.abs();
                    }
                });
        }

        let new_data = TensorData::new(out_vals, [1, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{TestBackend, test_device};

    #[test]
    fn test_gradients() {
        let device = test_device();
        let flat_data = vec![0.5f32; 3 * 8 * 8];
        let tensor_data = TensorData::new(flat_data, [3, 8, 8]);
        let img = Image::new(Tensor::<TestBackend, 3>::from_data(tensor_data, &device));

        let scharr_img = img.scharr().unwrap();
        assert_eq!(scharr_img.shape(), [1, 8, 8]);

        let laplace_img = img.laplacian().unwrap();
        assert_eq!(laplace_img.shape(), [1, 8, 8]);
    }
}
