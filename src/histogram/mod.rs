use crate::error::{IrisError, Result};
use crate::image::Image;
use burn::tensor::{Tensor, TensorData, backend::Backend};

impl<B: Backend> Image<B> {
    /// Computes the 256-bin histogram for each channel.
    /// Returns a vector of vectors (one per channel), each containing 256 counts.
    pub fn calc_hist(&self) -> Result<Vec<Vec<u32>>> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut histograms = vec![vec![0u32; 256]; c];

        for ch in 0..c {
            for y in 0..h {
                for x in 0..w {
                    let val = flat_vals[ch * h * w + y * w + x];
                    let bin = (val.clamp(0.0, 1.0) * 255.0) as usize;
                    histograms[ch][bin] += 1;
                }
            }
        }

        Ok(histograms)
    }

    /// Performs histogram equalization on a grayscale image to enhance contrast.
    pub fn equalize_hist(&self) -> Result<Self> {
        let gray = self.grayscale()?;
        let dims = gray.tensor.dims();
        let h = dims[1];
        let w = dims[2];

        let device = gray.tensor.device();
        let tensor_data = gray.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; h * w];

        // 1. Compute histogram
        let mut hist = [0u32; 256];
        for &val in &flat_vals {
            let bin = (val.clamp(0.0, 1.0) * 255.0) as usize;
            hist[bin] += 1;
        }

        // 2. Compute cumulative distribution function (CDF)
        let mut cdf = [0u32; 256];
        let mut sum = 0u32;
        for i in 0..256 {
            sum += hist[i];
            cdf[i] = sum;
        }

        // 3. Find CDF minimum non-zero value
        let cdf_min = cdf.iter().find(|&&x| x > 0).copied().unwrap_or(0) as f32;
        let total = (h * w) as f32;

        // 4. Equalize mapping
        let mut lut = [0.0f32; 256];
        if total > cdf_min {
            for i in 0..256 {
                lut[i] = ((cdf[i] as f32 - cdf_min) / (total - cdf_min) * 255.0).round() / 255.0;
            }
        }

        for i in 0..(h * w) {
            let bin = (flat_vals[i].clamp(0.0, 1.0) * 255.0) as usize;
            out_vals[i] = lut[bin];
        }

        let new_data = TensorData::new(out_vals, [1, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Performs histogram equalization on a color image by converting to YCrCb,
    /// equalizing the Y channel, and converting back.
    pub fn equalize_hist_color(&self) -> Result<Self> {
        let dims = self.tensor.dims();
        if dims[0] != 3 {
            return Err(IrisError::InvalidParameter(
                "Input must be a 3-channel RGB image".into(),
            ));
        }

        // Convert RGB to YCrCb
        let ycrcb = self.rgb_to_ycrcb()?;

        // Extract Y channel as single-channel image
        let y_channel = ycrcb.tensor.clone().slice([0..1, 0..dims[1], 0..dims[2]]);
        let y_img = Image::new(y_channel);

        // Equalize histogram on Y channel
        let y_equalized = y_img.equalize_hist()?;

        // Reconstruct YCrCb with equalized Y
        let cr = ycrcb.tensor.clone().slice([1..2, 0..dims[1], 0..dims[2]]);
        let cb = ycrcb.tensor.clone().slice([2..3, 0..dims[1], 0..dims[2]]);
        let ycrcb_equalized =
            Image::merge_channels(&[y_equalized, Image::new(cr), Image::new(cb)])?;

        // Convert back to RGB
        ycrcb_equalized.ycrcb_to_rgb()
    }

    /// Contrast Limited Adaptive Histogram Equalization (CLAHE).
    /// Divides the image into `grid_size x grid_size` tiles and applies
    /// histogram equalization to each tile independently with clip limit.
    pub fn clahe(&self, clip_limit: f32, grid_size: usize) -> Result<Self> {
        if grid_size == 0 {
            return Err(IrisError::InvalidParameter("grid_size must be > 0".into()));
        }

        let gray = self.grayscale()?;
        let dims = gray.tensor.dims();
        let h = dims[1];
        let w = dims[2];

        let device = gray.tensor.device();
        let tensor_data = gray.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = flat_vals.clone();

        let tile_h = h / grid_size;
        let tile_w = w / grid_size;

        if tile_h == 0 || tile_w == 0 {
            return Err(IrisError::InvalidParameter(
                "Image too small for given grid_size".into(),
            ));
        }

        for ty in 0..grid_size {
            for tx in 0..grid_size {
                let y0 = ty * tile_h;
                let x0 = tx * tile_w;
                let y1 = if ty == grid_size - 1 { h } else { y0 + tile_h };
                let x1 = if tx == grid_size - 1 { w } else { x0 + tile_w };

                let tile_pixels = (y1 - y0) * (x1 - x0);

                // Compute histogram for this tile
                let mut hist = [0u32; 256];
                for y in y0..y1 {
                    for x in x0..x1 {
                        let bin = (flat_vals[y * w + x].clamp(0.0, 1.0) * 255.0) as usize;
                        hist[bin] += 1;
                    }
                }

                // Apply clip limit: redistribute excess bins
                if clip_limit > 0.0 {
                    let limit = (clip_limit * tile_pixels as f32 / 256.0) as u32;
                    let mut excess = 0u32;
                    for bin in 0..256 {
                        if hist[bin] > limit {
                            excess += hist[bin] - limit;
                            hist[bin] = limit;
                        }
                    }
                    // Redistribute excess evenly
                    let avg_inc = excess / 256;
                    let rem = excess % 256;
                    for bin in 0..256 {
                        hist[bin] += avg_inc;
                        if bin < rem as usize {
                            hist[bin] += 1;
                        }
                    }
                }

                // Compute CDF
                let mut cdf = [0u32; 256];
                let mut sum = 0u32;
                for i in 0..256 {
                    sum += hist[i];
                    cdf[i] = sum;
                }

                let cdf_min = cdf.iter().find(|&&x| x > 0).copied().unwrap_or(0) as f32;
                let total = tile_pixels as f32;

                let mut lut = [0.0f32; 256];
                if total > cdf_min {
                    for i in 0..256 {
                        lut[i] =
                            ((cdf[i] as f32 - cdf_min) / (total - cdf_min) * 255.0).round() / 255.0;
                    }
                }

                // Apply LUT to tile
                for y in y0..y1 {
                    for x in x0..x1 {
                        let bin = (flat_vals[y * w + x].clamp(0.0, 1.0) * 255.0) as usize;
                        out_vals[y * w + x] = lut[bin];
                    }
                }
            }
        }

        let new_data = TensorData::new(out_vals, [1, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Applies a lookup table (LUT) to map pixel values.
    /// `lut` must have 256 entries mapping each possible pixel value (0..255) to a new value.
    pub fn apply_lut(&self, lut: &[f32; 256]) -> Result<Self> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let device = self.tensor.device();
        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; c * h * w];

        for i in 0..(c * h * w) {
            let bin = (flat_vals[i].clamp(0.0, 1.0) * 255.0) as usize;
            out_vals[i] = lut[bin];
        }

        let new_data = TensorData::new(out_vals, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Compares two histograms using the specified method.
    /// Both histograms must have the same length.
    pub fn compare_hist(hist_a: &[f32], hist_b: &[f32], method: &str) -> Result<f64> {
        if hist_a.len() != hist_b.len() {
            return Err(IrisError::DimensionMismatch {
                expected: vec![hist_a.len()],
                actual: vec![hist_b.len()],
            });
        }

        match method {
            "correlation" => {
                let n = hist_a.len() as f64;
                let mean_a: f64 = hist_a.iter().map(|&x| x as f64).sum::<f64>() / n;
                let mean_b: f64 = hist_b.iter().map(|&x| x as f64).sum::<f64>() / n;

                let mut num = 0.0;
                let mut den_a = 0.0;
                let mut den_b = 0.0;
                for i in 0..hist_a.len() {
                    let da = hist_a[i] as f64 - mean_a;
                    let db = hist_b[i] as f64 - mean_b;
                    num += da * db;
                    den_a += da * da;
                    den_b += db * db;
                }
                let den = (den_a * den_b).sqrt();
                Ok(if den.abs() < 1e-10 { 0.0 } else { num / den })
            }
            "chi_square" => {
                let mut sum = 0.0;
                for i in 0..hist_a.len() {
                    let a = hist_a[i] as f64;
                    let b = hist_b[i] as f64;
                    if a + b > 0.0 {
                        sum += (a - b).powi(2) / (a + b);
                    }
                }
                Ok(sum)
            }
            "intersection" => {
                let sum: f64 = hist_a
                    .iter()
                    .zip(hist_b.iter())
                    .map(|(&a, &b)| (a as f64).min(b as f64))
                    .sum();
                Ok(sum)
            }
            "hellinger" => {
                let mut sum = 0.0;
                for i in 0..hist_a.len() {
                    let a = (hist_a[i] as f64).sqrt();
                    let b = (hist_b[i] as f64).sqrt();
                    sum += (a - b).powi(2);
                }
                Ok((sum / 2.0).sqrt())
            }
            _ => Err(IrisError::InvalidParameter(format!(
                "Unknown comparison method: {method}. Use correlation, chi_square, intersection, or hellinger"
            ))),
        }
    }

    /// Compares two color histograms using the specified method.
    /// Returns per-channel comparison results.
    pub fn compare_hist_color(
        hist_a: &[Vec<f32>],
        hist_b: &[Vec<f32>],
        method: &str,
    ) -> Result<Vec<f64>> {
        if hist_a.len() != hist_b.len() {
            return Err(IrisError::DimensionMismatch {
                expected: vec![hist_a.len()],
                actual: vec![hist_b.len()],
            });
        }

        let mut results = Vec::with_capacity(hist_a.len());
        for (a, b) in hist_a.iter().zip(hist_b.iter()) {
            let score = Self::compare_hist(a, b, method)?;
            results.push(score);
        }
        Ok(results)
    }

    /// Computes a 2D histogram over two channels of a multi-channel image.
    ///
    /// `channel_x` and `channel_y` select which channels to histogram (0-indexed).
    /// The result is a 2D tensor of shape `[bins, bins]` with counts normalized
    /// so that the maximum bin value equals 1.0.
    pub fn calc_hist_2d(
        &self,
        channel_x: usize,
        channel_y: usize,
        bins: usize,
    ) -> Result<Tensor<B, 2>> {
        if bins == 0 {
            return Err(IrisError::InvalidParameter("bins must be > 0".into()));
        }

        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        if channel_x >= c || channel_y >= c {
            return Err(IrisError::DimensionMismatch {
                expected: vec![c],
                actual: vec![channel_x.max(channel_y) + 1],
            });
        }

        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

        let mut hist = vec![0u32; bins * bins];

        for y in 0..h {
            for x in 0..w {
                let val_x = flat_vals[channel_x * h * w + y * w + x];
                let val_y = flat_vals[channel_y * h * w + y * w + x];

                let bin_x =
                    ((val_x.clamp(0.0, 1.0) * (bins as f32 - 1.0)).round() as usize).min(bins - 1);
                let bin_y =
                    ((val_y.clamp(0.0, 1.0) * (bins as f32 - 1.0)).round() as usize).min(bins - 1);

                hist[bin_y * bins + bin_x] += 1;
            }
        }

        // Normalize to [0, 1] range
        let max_val = hist.iter().copied().max().unwrap_or(1) as f32;
        let hist_f32: Vec<f32> = hist.iter().map(|&v| v as f32 / max_val).collect();

        let device = self.tensor.device();
        let new_data = TensorData::new(hist_f32, [bins, bins]);
        let new_tensor = Tensor::<B, 2>::from_data(new_data, &device);
        Ok(new_tensor)
    }

    /// Adaptive histogram equalization (non-CLAHE version).
    ///
    /// Divides the image into `grid_size x grid_size` tiles, equalizes each tile,
    /// and blends neighboring tiles using bilinear interpolation to avoid artifacts
    /// at tile boundaries.
    pub fn equalize_hist_adaptive(&self, clip_limit: f32, grid_size: usize) -> Result<Self> {
        if grid_size == 0 {
            return Err(IrisError::InvalidParameter("grid_size must be > 0".into()));
        }

        let gray = self.grayscale()?;
        let dims = gray.tensor.dims();
        let h = dims[1];
        let w = dims[2];

        let device = gray.tensor.device();
        let tensor_data = gray.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

        let tile_h = h / grid_size;
        let tile_w = w / grid_size;

        if tile_h == 0 || tile_w == 0 {
            return Err(IrisError::InvalidParameter(
                "Image too small for given grid_size".into(),
            ));
        }

        // Compute LUT for each tile
        let mut tile_luts: Vec<Vec<f32>> = Vec::with_capacity(grid_size * grid_size);

        for ty in 0..grid_size {
            for tx in 0..grid_size {
                let y0 = ty * tile_h;
                let x0 = tx * tile_w;
                let y1 = if ty == grid_size - 1 { h } else { y0 + tile_h };
                let x1 = if tx == grid_size - 1 { w } else { x0 + tile_w };

                let tile_pixels = (y1 - y0) * (x1 - x0);

                // Compute histogram for this tile
                let mut hist = [0u32; 256];
                for y in y0..y1 {
                    for x in x0..x1 {
                        let bin = (flat_vals[y * w + x].clamp(0.0, 1.0) * 255.0) as usize;
                        hist[bin] += 1;
                    }
                }

                // Apply clip limit
                if clip_limit > 0.0 {
                    let limit = (clip_limit * tile_pixels as f32 / 256.0) as u32;
                    let mut excess = 0u32;
                    for bin in 0..256 {
                        if hist[bin] > limit {
                            excess += hist[bin] - limit;
                            hist[bin] = limit;
                        }
                    }
                    let avg_inc = excess / 256;
                    let rem = excess % 256;
                    for bin in 0..256 {
                        hist[bin] += avg_inc;
                        if bin < rem as usize {
                            hist[bin] += 1;
                        }
                    }
                }

                // Compute CDF and LUT
                let mut cdf = [0u32; 256];
                let mut sum = 0u32;
                for i in 0..256 {
                    sum += hist[i];
                    cdf[i] = sum;
                }

                let cdf_min = cdf.iter().find(|&&x| x > 0).copied().unwrap_or(0) as f32;
                let total = tile_pixels as f32;

                let mut lut = [0.0f32; 256];
                if total > cdf_min {
                    for i in 0..256 {
                        lut[i] =
                            ((cdf[i] as f32 - cdf_min) / (total - cdf_min) * 255.0).round() / 255.0;
                    }
                }

                tile_luts.push(lut.to_vec());
            }
        }

        // Apply LUTs with bilinear interpolation at tile boundaries
        let mut out_vals = vec![0.0f32; h * w];

        for y in 0..h {
            for x in 0..w {
                // Determine which tile center this pixel is nearest to
                let tx = (x as f32 / tile_w as f32 - 0.5).clamp(0.0, (grid_size - 1) as f32);
                let ty = (y as f32 / tile_h as f32 - 0.5).clamp(0.0, (grid_size - 1) as f32);

                let tx0 = tx.floor() as usize;
                let ty0 = ty.floor() as usize;
                let tx1 = (tx0 + 1).min(grid_size - 1);
                let ty1 = (ty0 + 1).min(grid_size - 1);

                let fx = tx - tx0 as f32;
                let fy = ty - ty0 as f32;

                let bin = (flat_vals[y * w + x].clamp(0.0, 1.0) * 255.0) as usize;

                let lut00 = &tile_luts[ty0 * grid_size + tx0];
                let lut10 = &tile_luts[ty0 * grid_size + tx1];
                let lut01 = &tile_luts[ty1 * grid_size + tx0];
                let lut11 = &tile_luts[ty1 * grid_size + tx1];

                let v00 = lut00[bin];
                let v10 = lut10[bin];
                let v01 = lut01[bin];
                let v11 = lut11[bin];

                // Bilinear interpolation
                let val = v00 * (1.0 - fx) * (1.0 - fy)
                    + v10 * fx * (1.0 - fy)
                    + v01 * (1.0 - fx) * fy
                    + v11 * fx * fy;

                out_vals[y * w + x] = val;
            }
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
    fn test_histogram_operations() {
        let device = test_device();
        let flat_data = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [1, 2, 4]), &device);
        let img = Image::new(tensor);

        let hists = img.calc_hist().unwrap();
        assert_eq!(hists.len(), 1);
        assert_eq!(hists[0].len(), 256);

        let eq = img.equalize_hist().unwrap();
        assert_eq!(eq.shape(), [1, 2, 4]);
    }

    #[test]
    fn test_equalize_hist_color() {
        let device = test_device();
        let flat_data = vec![
            0.2f32, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7, 0.9, 0.0, 0.2, 0.4,
        ];
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [3, 2, 2]), &device);
        let img = Image::new(tensor);

        let eq = img.equalize_hist_color().unwrap();
        assert_eq!(eq.shape(), [3, 2, 2]);
    }

    #[test]
    fn test_compare_hist_color() {
        let hist_a = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 3.0, 4.0, 5.0],
            vec![3.0, 4.0, 5.0, 6.0],
        ];
        let hist_b = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 3.0, 4.0, 5.0],
            vec![3.0, 4.0, 5.0, 6.0],
        ];

        let results =
            Image::<TestBackend>::compare_hist_color(&hist_a, &hist_b, "correlation").unwrap();
        assert_eq!(results.len(), 3);
        for r in results {
            assert!((r - 1.0).abs() < 1e-5);
        }

        let chi_results =
            Image::<TestBackend>::compare_hist_color(&hist_a, &hist_b, "chi_square").unwrap();
        for r in chi_results {
            assert!(r.abs() < 1e-5);
        }
    }

    #[test]
    fn test_clahe() {
        let device = test_device();
        let data: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();
        let tensor = Tensor::<TestBackend, 3>::from_data(TensorData::new(data, [1, 8, 8]), &device);
        let img = Image::new(tensor);
        let result = img.clahe(2.0, 4).unwrap();
        assert_eq!(result.shape(), [1, 8, 8]);
    }

    #[test]
    fn test_apply_lut() {
        let device = test_device();
        let data = vec![0.0f32, 0.5, 1.0];
        let tensor = Tensor::<TestBackend, 3>::from_data(TensorData::new(data, [1, 1, 3]), &device);
        let img = Image::new(tensor);

        let mut lut = [0.0f32; 256];
        for i in 0..256 {
            lut[i] = 1.0 - (i as f32) / 255.0; // Invert
        }
        let result = img.apply_lut(&lut).unwrap();
        assert_eq!(result.shape(), [1, 1, 3]);
        let vals: Vec<f32> = result.tensor.into_data().iter::<f32>().collect();
        assert!((vals[0] - 1.0).abs() < 1e-5); // 0.0 -> 1.0
        assert!((vals[2] - 0.0).abs() < 1e-5); // 1.0 -> 0.0
    }

    #[test]
    fn test_compare_hist() {
        let hist_a = vec![1.0, 2.0, 3.0, 4.0];
        let hist_b = vec![1.0, 2.0, 3.0, 4.0];
        let corr = Image::<TestBackend>::compare_hist(&hist_a, &hist_b, "correlation").unwrap();
        assert!((corr - 1.0).abs() < 1e-5); // Identical histograms => correlation = 1

        let chi = Image::<TestBackend>::compare_hist(&hist_a, &hist_b, "chi_square").unwrap();
        assert!((chi).abs() < 1e-5); // Identical => chi_square = 0

        let inter = Image::<TestBackend>::compare_hist(&hist_a, &hist_b, "intersection").unwrap();
        assert!((inter - 10.0).abs() < 1e-5); // sum of min(a, b) = 1+2+3+4 = 10

        let hel = Image::<TestBackend>::compare_hist(&hist_a, &hist_b, "hellinger").unwrap();
        assert!((hel).abs() < 1e-5);
    }

    #[test]
    fn test_compare_hist_invalid() {
        let hist_a = vec![1.0, 2.0];
        let hist_b = vec![1.0, 2.0];
        assert!(Image::<TestBackend>::compare_hist(&hist_a, &hist_b, "invalid").is_err());
    }

    #[test]
    fn test_calc_hist_2d() {
        let device = test_device();
        // Create a 2-channel image of shape [2, 4, 4]
        let mut flat_data = Vec::new();
        // Channel 0: values from 0 to 1
        for y in 0..4 {
            for x in 0..4 {
                flat_data.push((y * 4 + x) as f32 / 15.0);
            }
        }
        // Channel 1: values from 1 to 0 (reversed)
        for y in 0..4 {
            for x in 0..4 {
                flat_data.push(1.0 - (y * 4 + x) as f32 / 15.0);
            }
        }
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [2, 4, 4]), &device);
        let img = Image::new(tensor);

        let hist_2d = img.calc_hist_2d(0, 1, 4).unwrap();
        let dims = hist_2d.dims();
        assert_eq!(dims, [4, 4]);
        let vals: Vec<f32> = hist_2d.into_data().iter::<f32>().collect();
        // All bins should be >= 0 and at least one should be > 0
        assert!(vals.iter().all(|&v| v >= 0.0));
        assert!(vals.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn test_equalize_hist_adaptive() {
        let device = test_device();
        let data: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();
        let tensor = Tensor::<TestBackend, 3>::from_data(TensorData::new(data, [1, 8, 8]), &device);
        let img = Image::new(tensor);

        let result = img.equalize_hist_adaptive(2.0, 2).unwrap();
        assert_eq!(result.shape(), [1, 8, 8]);
        let vals: Vec<f32> = result.tensor.into_data().iter::<f32>().collect();
        // All output values should be in [0, 1]
        assert!(vals.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }
}
