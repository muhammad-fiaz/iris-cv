use crate::error::Result;
use crate::image::Image;
use burn::tensor::{Tensor, TensorData, backend::Backend};

/// Different types of thresholding operations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ThresholdType {
    Binary,
    BinaryInv,
    Trunc,
    ToZero,
    ToZeroInv,
}

impl<B: Backend> Image<B> {
    /// Applies a fixed-level threshold to each array element.
    pub fn threshold(&self, thresh: f32, maxval: f32, thresh_type: ThresholdType) -> Result<Self> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let device = self.tensor.device();
        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; c * h * w];

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            out_vals
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, out_val)| {
                    let val = flat_vals[i];
                    *out_val = match thresh_type {
                        ThresholdType::Binary => {
                            if val > thresh {
                                maxval
                            } else {
                                0.0
                            }
                        }
                        ThresholdType::BinaryInv => {
                            if val > thresh {
                                0.0
                            } else {
                                maxval
                            }
                        }
                        ThresholdType::Trunc => {
                            if val > thresh {
                                thresh
                            } else {
                                val
                            }
                        }
                        ThresholdType::ToZero => {
                            if val > thresh {
                                val
                            } else {
                                0.0
                            }
                        }
                        ThresholdType::ToZeroInv => {
                            if val > thresh {
                                0.0
                            } else {
                                val
                            }
                        }
                    };
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..(c * h * w) {
                let val = flat_vals[i];
                out_vals[i] = match thresh_type {
                    ThresholdType::Binary => {
                        if val > thresh {
                            maxval
                        } else {
                            0.0
                        }
                    }
                    ThresholdType::BinaryInv => {
                        if val > thresh {
                            0.0
                        } else {
                            maxval
                        }
                    }
                    ThresholdType::Trunc => {
                        if val > thresh {
                            thresh
                        } else {
                            val
                        }
                    }
                    ThresholdType::ToZero => {
                        if val > thresh {
                            val
                        } else {
                            0.0
                        }
                    }
                    ThresholdType::ToZeroInv => {
                        if val > thresh {
                            0.0
                        } else {
                            val
                        }
                    }
                };
            }
        }

        let new_data = TensorData::new(out_vals, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Automatically computes a threshold using Otsu's method and applies binary thresholding.
    pub fn threshold_otsu(&self, maxval: f32) -> Result<Self> {
        let gray = self.grayscale()?;
        let dims = gray.tensor.dims();
        let h = dims[1];
        let w = dims[2];

        let tensor_data = gray.tensor.into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

        // 1. Calculate 256-bin histogram
        let mut hist = [0u32; 256];
        for &val in &flat_vals {
            let bin = (val.clamp(0.0, 1.0) * 255.0) as usize;
            hist[bin] += 1;
        }

        let total = (h * w) as f32;
        let mut sum = 0.0f32;
        for (i, &count) in hist.iter().enumerate() {
            sum += (i as f32) * (count as f32);
        }

        let mut sum_b = 0.0f32;
        let mut w_b = 0.0f32;
        let mut max_var = 0.0f32;
        let mut threshold = 0;

        // 2. Iterate to find threshold maximizing between-class variance
        for (t, &count) in hist.iter().enumerate() {
            w_b += count as f32;
            if w_b == 0.0 {
                continue;
            }
            let w_f = total - w_b;
            if w_f == 0.0 {
                break;
            }

            sum_b += (t as f32) * (count as f32);
            let m_b = sum_b / w_b;
            let m_f = (sum - sum_b) / w_f;

            let var_between = w_b * w_f * (m_b - m_f) * (m_b - m_f);
            if var_between > max_var {
                max_var = var_between;
                threshold = t;
            }
        }

        let thresh_float = (threshold as f32) / 255.0;
        self.threshold(thresh_float, maxval, ThresholdType::Binary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{TestBackend, test_device};

    #[test]
    fn test_threshold() {
        let device = test_device();
        let flat_data = vec![0.5f32; 3 * 8 * 8];
        let tensor_data = TensorData::new(flat_data, [3, 8, 8]);
        let img = Image::new(Tensor::<TestBackend, 3>::from_data(tensor_data, &device));

        let thresh = img.threshold(0.4, 1.0, ThresholdType::Binary).unwrap();
        assert_eq!(thresh.shape(), [3, 8, 8]);

        let otsu = img.threshold_otsu(1.0).unwrap();
        assert_eq!(otsu.shape(), [3, 8, 8]);
    }
}
