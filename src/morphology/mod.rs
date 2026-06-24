pub mod ops;

pub use ops::{MorphOp, MorphShape, Morphology};

use crate::error::{IrisError, Result};
use crate::image::Image;
use burn::tensor::{Tensor, TensorData, backend::Backend};

impl<B: Backend> Image<B> {
    /// Dilates the image by using a rectangular structuring element of the given size.
    /// For each pixel, it takes the maximum value in the neighborhood.
    pub fn dilate(self, kernel_size: usize) -> Result<Self> {
        if kernel_size.is_multiple_of(2) {
            return Err(IrisError::InvalidParameter(
                "Kernel size must be odd".into(),
            ));
        }

        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let device = self.tensor.device();
        let tensor_data = self.tensor.into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; c * h * w];

        let rad = (kernel_size / 2) as isize;

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            out_vals
                .par_chunks_exact_mut(w)
                .enumerate()
                .for_each(|(idx, row)| {
                    let ch = idx / h;
                    let y = idx % h;

                    for x in 0..w {
                        let mut max_val = f32::MIN;
                        for ky in -rad..=rad {
                            let py = y as isize + ky;
                            if py >= 0 && py < h as isize {
                                for kx in -rad..=rad {
                                    let px = x as isize + kx;
                                    if px >= 0 && px < w as isize {
                                        let val = flat_vals
                                            [ch * h * w + (py as usize) * w + (px as usize)];
                                        if val > max_val {
                                            max_val = val;
                                        }
                                    }
                                }
                            }
                        }
                        row[x] = max_val;
                    }
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for ch in 0..c {
                for y in 0..h {
                    for x in 0..w {
                        let mut max_val = f32::MIN;
                        for ky in -rad..=rad {
                            let py = y as isize + ky;
                            if py >= 0 && py < h as isize {
                                for kx in -rad..=rad {
                                    let px = x as isize + kx;
                                    if px >= 0 && px < w as isize {
                                        let val = flat_vals
                                            [ch * h * w + (py as usize) * w + (px as usize)];
                                        if val > max_val {
                                            max_val = val;
                                        }
                                    }
                                }
                            }
                        }
                        out_vals[ch * h * w + y * w + x] = max_val;
                    }
                }
            }
        }

        let new_data = TensorData::new(out_vals, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Erodes the image by using a rectangular structuring element of the given size.
    /// For each pixel, it takes the minimum value in the neighborhood.
    pub fn erode(self, kernel_size: usize) -> Result<Self> {
        if kernel_size.is_multiple_of(2) {
            return Err(IrisError::InvalidParameter(
                "Kernel size must be odd".into(),
            ));
        }

        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let device = self.tensor.device();
        let tensor_data = self.tensor.into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; c * h * w];

        let rad = (kernel_size / 2) as isize;

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            out_vals
                .par_chunks_exact_mut(w)
                .enumerate()
                .for_each(|(idx, row)| {
                    let ch = idx / h;
                    let y = idx % h;

                    for x in 0..w {
                        let mut min_val = f32::MAX;
                        for ky in -rad..=rad {
                            let py = y as isize + ky;
                            if py >= 0 && py < h as isize {
                                for kx in -rad..=rad {
                                    let px = x as isize + kx;
                                    if px >= 0 && px < w as isize {
                                        let val = flat_vals
                                            [ch * h * w + (py as usize) * w + (px as usize)];
                                        if val < min_val {
                                            min_val = val;
                                        }
                                    }
                                }
                            }
                        }
                        row[x] = min_val;
                    }
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for ch in 0..c {
                for y in 0..h {
                    for x in 0..w {
                        let mut min_val = f32::MAX;
                        for ky in -rad..=rad {
                            let py = y as isize + ky;
                            if py >= 0 && py < h as isize {
                                for kx in -rad..=rad {
                                    let px = x as isize + kx;
                                    if px >= 0 && px < w as isize {
                                        let val = flat_vals
                                            [ch * h * w + (py as usize) * w + (px as usize)];
                                        if val < min_val {
                                            min_val = val;
                                        }
                                    }
                                }
                            }
                        }
                        out_vals[ch * h * w + y * w + x] = min_val;
                    }
                }
            }
        }

        let new_data = TensorData::new(out_vals, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Morphological opening (erosion followed by dilation).
    pub fn morph_open(self, kernel_size: usize) -> Result<Self> {
        self.erode(kernel_size)?.dilate(kernel_size)
    }

    /// Morphological closing (dilation followed by erosion).
    pub fn morph_close(self, kernel_size: usize) -> Result<Self> {
        self.dilate(kernel_size)?.erode(kernel_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{TestBackend, test_device};

    #[test]
    fn test_morphology() {
        let device = test_device();
        let flat_data = vec![0.5f32; 3 * 8 * 8];
        let tensor_data = TensorData::new(flat_data, [3, 8, 8]);
        let img = Image::new(Tensor::<TestBackend, 3>::from_data(tensor_data, &device));

        let dilated = img.clone().dilate(3).unwrap();
        assert_eq!(dilated.shape(), [3, 8, 8]);

        let eroded = img.clone().erode(3).unwrap();
        assert_eq!(eroded.shape(), [3, 8, 8]);

        let opened = img.clone().morph_open(3).unwrap();
        assert_eq!(opened.shape(), [3, 8, 8]);

        let closed = img.clone().morph_close(3).unwrap();
        assert_eq!(closed.shape(), [3, 8, 8]);
    }
}
