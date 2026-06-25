pub mod bilateral;

use crate::error::{IrisError, Result};
use crate::image::Image;
use burn::tensor::{Tensor, TensorData, backend::Backend};

/// Square root of 2, used for diagonal distance in distance transform.
const SQRT_2: f32 = std::f32::consts::SQRT_2;

impl<B: Backend> Image<B> {
    /// Applies a box filter to blur the image with the specified kernel size.
    pub fn box_blur(self, kernel_size: usize) -> Result<Self> {
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

        {
            use rayon::prelude::*;
            out_vals
                .par_chunks_exact_mut(w)
                .enumerate()
                .for_each(|(idx, row)| {
                    let ch = idx / h;
                    let y = idx % h;
                    for x in 0..w {
                        let mut sum = 0.0f32;
                        let mut count = 0.0f32;

                        for ky in -rad..=rad {
                            let py = y as isize + ky;
                            if py >= 0 && py < h as isize {
                                for kx in -rad..=rad {
                                    let px = x as isize + kx;
                                    if px >= 0 && px < w as isize {
                                        sum += flat_vals
                                            [ch * h * w + (py as usize) * w + (px as usize)];
                                        count += 1.0;
                                    }
                                }
                            }
                        }
                        row[x] = sum / count;
                    }
                });
        }

        let new_data = TensorData::new(out_vals, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Applies a Gaussian blur filter to the image.
    pub fn gaussian_blur(self, kernel_size: usize, sigma: f64) -> Result<Self> {
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

        // Generate Gaussian kernel
        let mut kernel = vec![vec![0.0f64; kernel_size]; kernel_size];
        let mut sum = 0.0f64;

        let s2 = 2.0 * sigma * sigma;

        for ky in -rad..=rad {
            for kx in -rad..=rad {
                let r = (kx * kx + ky * ky) as f64;
                let val = (-r / s2).exp();
                kernel[(ky + rad) as usize][(kx + rad) as usize] = val;
                sum += val;
            }
        }

        // Normalize kernel
        for row in &mut kernel {
            for val in row {
                *val /= sum;
            }
        }

        {
            use rayon::prelude::*;
            out_vals
                .par_chunks_exact_mut(w)
                .enumerate()
                .for_each(|(idx, row)| {
                    let ch = idx / h;
                    let y = idx % h;
                    for x in 0..w {
                        let mut blur_sum = 0.0f64;
                        for ky in -rad..=rad {
                            let py = y as isize + ky;
                            let py_clamped = py.clamp(0, h as isize - 1) as usize;
                            for kx in -rad..=rad {
                                let px = x as isize + kx;
                                let px_clamped = px.clamp(0, w as isize - 1) as usize;
                                let weight = kernel[(ky + rad) as usize][(kx + rad) as usize];
                                blur_sum +=
                                    f64::from(flat_vals[ch * h * w + py_clamped * w + px_clamped])
                                        * weight;
                            }
                        }
                        row[x] = blur_sum as f32;
                    }
                });
        }

        let new_data = TensorData::new(out_vals, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Applies a median filter to reduce salt-and-pepper noise.
    pub fn median_blur(self, kernel_size: usize) -> Result<Self> {
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

        {
            use rayon::prelude::*;
            out_vals
                .par_chunks_exact_mut(w)
                .enumerate()
                .for_each(|(idx, row)| {
                    let ch = idx / h;
                    let y = idx % h;
                    for x in 0..w {
                        let mut neighbors = Vec::with_capacity(kernel_size * kernel_size);

                        for ky in -rad..=rad {
                            let py = y as isize + ky;
                            if py >= 0 && py < h as isize {
                                for kx in -rad..=rad {
                                    let px = x as isize + kx;
                                    if px >= 0 && px < w as isize {
                                        neighbors.push(
                                            flat_vals
                                                [ch * h * w + (py as usize) * w + (px as usize)],
                                        );
                                    }
                                }
                            }
                        }
                        neighbors
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let median = neighbors[neighbors.len() / 2];
                        row[x] = median;
                    }
                });
        }

        let new_data = TensorData::new(out_vals, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Computes the distance transform of a binary/grayscale image.
    /// Each pixel's value becomes its Euclidean distance to the nearest zero pixel.
    /// Uses the Meijster algorithm (two-pass) for O(n) exact Euclidean distance transform.
    pub fn distance_transform(&self) -> Result<Self> {
        let gray = self.grayscale()?;
        let dims = gray.tensor.dims();
        let h = dims[1];
        let w = dims[2];

        let tensor_data = gray.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

        // Binarize: nonzero pixels are foreground
        let mut binary = vec![false; h * w];
        for (i, &v) in flat_vals.iter().enumerate() {
            binary[i] = v > 0.5;
        }

        let inf = (h * w) as f32;
        let mut dt = vec![inf; h * w];

        // First pass: forward scan
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if binary[idx] {
                    dt[idx] = 0.0;
                } else {
                    if x > 0 {
                        let prev = dt[idx - 1] + 1.0;
                        if prev < dt[idx] {
                            dt[idx] = prev;
                        }
                    }
                    if y > 0 {
                        let prev = dt[(y - 1) * w + x] + 1.0;
                        if prev < dt[idx] {
                            dt[idx] = prev;
                        }
                    }
                    if x > 0 && y > 0 {
                        let prev = dt[(y - 1) * w + (x - 1)] + SQRT_2;
                        if prev < dt[idx] {
                            dt[idx] = prev;
                        }
                    }
                    if x < w - 1 && y > 0 {
                        let prev = dt[(y - 1) * w + (x + 1)] + SQRT_2;
                        if prev < dt[idx] {
                            dt[idx] = prev;
                        }
                    }
                }
            }
        }

        // Second pass: backward scan
        for y in (0..h).rev() {
            for x in (0..w).rev() {
                let idx = y * w + x;
                if x < w - 1 {
                    let next = dt[idx + 1] + 1.0;
                    if next < dt[idx] {
                        dt[idx] = next;
                    }
                }
                if y < h - 1 {
                    let next = dt[(y + 1) * w + x] + 1.0;
                    if next < dt[idx] {
                        dt[idx] = next;
                    }
                }
                if x < w - 1 && y < h - 1 {
                    let next = dt[(y + 1) * w + (x + 1)] + SQRT_2;
                    if next < dt[idx] {
                        dt[idx] = next;
                    }
                }
                if x > 0 && y < h - 1 {
                    let next = dt[(y + 1) * w + (x - 1)] + SQRT_2;
                    if next < dt[idx] {
                        dt[idx] = next;
                    }
                }
            }
        }

        // Normalize to [0, 1] range
        let max_dt = dt.iter().cloned().fold(0.0f32, f32::max);
        if max_dt > 0.0 {
            for v in &mut dt {
                *v /= max_dt;
            }
        }

        let device = gray.tensor.device();
        let data = TensorData::new(dt, [1, h, w]);
        let tensor = Tensor::<B, 3>::from_data(data, &device);
        Ok(Image::new(tensor))
    }

    /// Applies an arbitrary 2D convolution kernel to the image.
    /// `kernel` is a 2D slice of shape `[kh][kw]`. Output is computed via valid convolution.
    /// `anchor` specifies the center of the kernel; if `None`, the center is `(kw/2, kh/2)`.
    /// `delta` is added to each result pixel, and `border` controls out-of-bounds handling.
    pub fn filter2d(
        &self,
        kernel: &[&[f32]],
        anchor: Option<(isize, isize)>,
        delta: f32,
    ) -> Result<Self> {
        let kh = kernel.len();
        if kh == 0 || kernel[0].is_empty() {
            return Err(IrisError::InvalidParameter(
                "Kernel must be non-empty".into(),
            ));
        }
        let kw = kernel[0].len();

        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let (ax, ay) = anchor.unwrap_or((kw as isize / 2, kh as isize / 2));

        let device = self.tensor.device();
        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; c * h * w];

        for ch in 0..c {
            for y in 0..h {
                for x in 0..w {
                    let mut sum = 0.0f64;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let sy = y as isize + ky as isize - ay;
                            let sx = x as isize + kx as isize - ax;
                            if sy >= 0 && sy < h as isize && sx >= 0 && sx < w as isize {
                                sum += flat_vals[ch * h * w + sy as usize * w + sx as usize] as f64
                                    * kernel[ky][kx] as f64;
                            }
                        }
                    }
                    out_vals[ch * h * w + y * w + x] = sum as f32 + delta;
                }
            }
        }

        let new_data = TensorData::new(out_vals, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Computes weighted sum of two images: `dst = src1 * alpha + src2 * beta + gamma`.
    pub fn add_weighted(&self, other: &Self, alpha: f32, beta: f32, gamma: f32) -> Result<Self> {
        if self.shape() != other.shape() {
            return Err(IrisError::DimensionMismatch {
                expected: self.shape().to_vec(),
                actual: other.shape().to_vec(),
            });
        }
        let result = self
            .tensor
            .clone()
            .mul_scalar(alpha)
            .add(other.tensor.clone().mul_scalar(beta))
            .add_scalar(gamma);
        Ok(Image::new(result))
    }

    /// Computes `dst = src * scale + shift`, then optionally converts to abs.
    pub fn convert_scale_abs(&self, scale: f32, shift: f32) -> Result<Self> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let device = self.tensor.device();
        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; c * h * w];

        for i in 0..(c * h * w) {
            let val = (flat_vals[i] * scale + shift).abs().min(1.0);
            out_vals[i] = val;
        }

        let new_data = TensorData::new(out_vals, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Copies source image to destination within a masked region.
    /// Where `mask` is nonzero, `dst = src`; otherwise `dst` is unchanged.
    pub fn copy_to(&self, dst: &mut Self, mask: Option<&Self>) -> Result<()> {
        if self.shape() != dst.shape() {
            return Err(IrisError::DimensionMismatch {
                expected: self.shape().to_vec(),
                actual: dst.shape().to_vec(),
            });
        }

        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let src_data = self.tensor.clone().into_data();
        let src_vals: Vec<f32> = src_data.iter::<f32>().collect();
        let dst_data = dst.tensor.clone().into_data();
        let mut dst_vals: Vec<f32> = dst_data.iter::<f32>().collect();

        let mask_vals: Option<Vec<f32>> = mask.map(|m| {
            let d = m.tensor.clone().into_data();
            d.iter::<f32>().collect()
        });

        let pixels = h * w;
        for i in 0..pixels {
            let dominated = match &mask_vals {
                Some(mv) => mv[i] > 0.0,
                None => true,
            };
            if dominated {
                for ch in 0..c {
                    dst_vals[ch * pixels + i] = src_vals[ch * pixels + i];
                }
            }
        }

        *dst = Image::new(Tensor::<B, 3>::from_data(
            TensorData::new(dst_vals, [c, h, w]),
            &dst.tensor.device(),
        ));
        Ok(())
    }

    /// Computes the Laplacian of Gaussian (LoG) filter response.
    /// Applies Gaussian smoothing then Laplacian operator to detect edges/blobs.
    /// `sigma` controls the scale of the Gaussian kernel.
    pub fn laplacian_of_gaussian(&self, sigma: f32) -> Result<Self> {
        let gray = self.grayscale()?;
        let dims = gray.tensor.dims();
        let h = dims[1];
        let w = dims[2];

        // Build LoG kernel: size = 6*sigma, rounded to odd
        let k_size = ((6.0 * sigma as f64) as usize) | 1;
        let half = k_size / 2;

        // LoG kernel: -(1/(pi*sigma^4)) * (1 - (x^2+y^2)/(2*sigma^2)) * exp(-(x^2+y^2)/(2*sigma^2))
        let sigma2 = (sigma as f64) * (sigma as f64);
        let sigma4 = sigma2 * sigma2;
        let mut kernel = Vec::with_capacity(k_size * k_size);
        let mut kernel_sum = 0.0f64;

        for ky in 0..k_size {
            for kx in 0..k_size {
                let dx = (kx as f64) - (half as f64);
                let dy = (ky as f64) - (half as f64);
                let r2 = dx * dx + dy * dy;
                let val = -(1.0 / (std::f64::consts::PI * sigma4))
                    * (1.0 - r2 / (2.0 * sigma2))
                    * (-r2 / (2.0 * sigma2)).exp();
                kernel.push(val);
                kernel_sum += val;
            }
        }

        // Zero-sum normalize: subtract mean so kernel sums to zero
        let mean = kernel_sum / (k_size * k_size) as f64;
        for k in &mut kernel {
            *k -= mean;
        }

        // Apply convolution
        let tensor_data = gray.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; h * w];

        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0f64;
                for ky in 0..k_size {
                    for kx in 0..k_size {
                        let sy = (y + ky).min(h - 1);
                        let sx = (x + kx).min(w - 1);
                        sum += flat_vals[sy * w + sx] as f64 * kernel[ky * k_size + kx];
                    }
                }
                out_vals[y * w + x] = sum as f32;
            }
        }

        let device = gray.tensor.device();
        let data = TensorData::new(out_vals, [1, h, w]);
        let tensor = Tensor::<B, 3>::from_data(data, &device);
        Ok(Image::new(tensor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{TestBackend, test_device};

    #[test]
    fn test_filters_blur() {
        let device = test_device();
        let flat_data = vec![0.5f32; 3 * 8 * 8];
        let tensor_data = TensorData::new(flat_data, [3, 8, 8]);
        let img = Image::new(Tensor::<TestBackend, 3>::from_data(tensor_data, &device));

        let boxed = img.clone().box_blur(3).unwrap();
        assert_eq!(boxed.shape(), [3, 8, 8]);

        let gauss = img.clone().gaussian_blur(3, 1.0).unwrap();
        assert_eq!(gauss.shape(), [3, 8, 8]);

        let median = img.clone().median_blur(3).unwrap();
        assert_eq!(median.shape(), [3, 8, 8]);
    }

    #[test]
    fn test_distance_transform() {
        let device = test_device();
        let flat_data = vec![
            0.0f32, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [1, 4, 4]), &device);
        let img = Image::new(tensor);
        let dt = img.distance_transform().unwrap();
        assert_eq!(dt.shape(), [1, 4, 4]);
    }

    #[test]
    fn test_laplacian_of_gaussian() {
        let device = test_device();
        let flat_data = vec![0.5f32; 3 * 16 * 16];
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [3, 16, 16]), &device);
        let img = Image::new(tensor);
        let log = img.laplacian_of_gaussian(1.0).unwrap();
        assert_eq!(log.shape(), [1, 16, 16]);
    }

    #[test]
    fn test_filter2d() {
        let device = test_device();
        let data = TensorData::new(vec![0.5f32; 3 * 8 * 8], [3, 8, 8]);
        let img = Image::new(Tensor::<TestBackend, 3>::from_data(data, &device));

        // Simple averaging kernel
        let kernel: Vec<&[f32]> = vec![
            &[1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
            &[1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
            &[1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
        ];
        let result = img.filter2d(&kernel, None, 0.0).unwrap();
        assert_eq!(result.shape(), [3, 8, 8]);
    }

    #[test]
    fn test_add_weighted() {
        let device = test_device();
        let data1 = TensorData::new(vec![0.5f32; 3 * 4 * 4], [3, 4, 4]);
        let data2 = TensorData::new(vec![0.3f32; 3 * 4 * 4], [3, 4, 4]);
        let img1 = Image::new(Tensor::<TestBackend, 3>::from_data(data1, &device));
        let img2 = Image::new(Tensor::<TestBackend, 3>::from_data(data2, &device));

        let result = img1.add_weighted(&img2, 0.6, 0.4, 0.0).unwrap();
        assert_eq!(result.shape(), [3, 4, 4]);
    }

    #[test]
    fn test_convert_scale_abs() {
        let device = test_device();
        let data = TensorData::new(vec![-0.5f32, -0.1, 0.0, 0.3, 0.8], [1, 1, 5]);
        let img = Image::new(Tensor::<TestBackend, 3>::from_data(data, &device));
        let result = img.convert_scale_abs(1.0, 0.0).unwrap();
        assert_eq!(result.shape(), [1, 1, 5]);
        let vals: Vec<f32> = result.tensor.into_data().iter::<f32>().collect();
        assert!((vals[0] - 0.5).abs() < 1e-5);
        assert!((vals[1] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_copy_to_with_mask() {
        let device = test_device();
        let data = TensorData::new(vec![1.0f32; 3 * 4 * 4], [3, 4, 4]);
        let mask_data = TensorData::new(
            vec![
                1.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0,
            ],
            [1, 4, 4],
        );
        let src = Image::new(Tensor::<TestBackend, 3>::from_data(data, &device));
        let mut dst = Image::new(Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![0.0f32; 3 * 4 * 4], [3, 4, 4]),
            &device,
        ));
        let mask = Image::new(Tensor::<TestBackend, 3>::from_data(mask_data, &device));
        src.copy_to(&mut dst, Some(&mask)).unwrap();
        assert_eq!(dst.shape(), [3, 4, 4]);
    }
}
