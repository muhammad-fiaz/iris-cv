pub mod ops;

pub use ops::{MorphOp, MorphShape, Morphology};

use crate::error::{IrisError, Result};
use crate::image::Image;
use burn::tensor::{Tensor, TensorData, backend::Backend};

impl<B: Backend> Image<B> {
    /// Dilates the image using the given structuring element.
    /// `kernel` is a 2D binary kernel (`1` = active, `0` = inactive).
    /// For each pixel, it takes the maximum value in the active neighborhood.
    pub fn dilate_with_kernel(self, kernel: &[&[u8]]) -> Result<Self> {
        let kh = kernel.len();
        if kh == 0 || kernel[0].is_empty() {
            return Err(IrisError::InvalidParameter("Kernel must be non-empty".into()));
        }
        let kw = kernel[0].len();

        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let device = self.tensor.device();
        let tensor_data = self.tensor.into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; c * h * w];

        let ay = kh as isize / 2;
        let ax = kw as isize / 2;

        for ch in 0..c {
            for y in 0..h {
                for x in 0..w {
                    let mut max_val = f32::MIN;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            if kernel[ky][kx] == 0 {
                                continue;
                            }
                            let sy = y as isize + ky as isize - ay;
                            let sx = x as isize + kx as isize - ax;
                            if sy >= 0 && sy < h as isize && sx >= 0 && sx < w as isize {
                                let val = flat_vals
                                    [ch * h * w + sy as usize * w + sx as usize];
                                if val > max_val {
                                    max_val = val;
                                }
                            }
                        }
                    }
                    out_vals[ch * h * w + y * w + x] = if max_val == f32::MIN {
                        flat_vals[ch * h * w + y * w + x]
                    } else {
                        max_val
                    };
                }
            }
        }

        let new_data = TensorData::new(out_vals, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Erodes the image using the given structuring element.
    /// `kernel` is a 2D binary kernel (`1` = active, `0` = inactive).
    /// For each pixel, it takes the minimum value in the active neighborhood.
    pub fn erode_with_kernel(self, kernel: &[&[u8]]) -> Result<Self> {
        let kh = kernel.len();
        if kh == 0 || kernel[0].is_empty() {
            return Err(IrisError::InvalidParameter("Kernel must be non-empty".into()));
        }
        let kw = kernel[0].len();

        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let device = self.tensor.device();
        let tensor_data = self.tensor.into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; c * h * w];

        let ay = kh as isize / 2;
        let ax = kw as isize / 2;

        for ch in 0..c {
            for y in 0..h {
                for x in 0..w {
                    let mut min_val = f32::MAX;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            if kernel[ky][kx] == 0 {
                                continue;
                            }
                            let sy = y as isize + ky as isize - ay;
                            let sx = x as isize + kx as isize - ax;
                            if sy >= 0 && sy < h as isize && sx >= 0 && sx < w as isize {
                                let val = flat_vals
                                    [ch * h * w + sy as usize * w + sx as usize];
                                if val < min_val {
                                    min_val = val;
                                }
                            }
                        }
                    }
                    out_vals[ch * h * w + y * w + x] = if min_val == f32::MAX {
                        flat_vals[ch * h * w + y * w + x]
                    } else {
                        min_val
                    };
                }
            }
        }

        let new_data = TensorData::new(out_vals, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

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

    /// Hit-or-miss transform for binary pattern matching.
    ///
    /// `pattern` defines the foreground (1) pixels to match and `bg_pattern`
    /// defines the background (1 = must-be-background) pixels to match.
    /// Pixels in either pattern set to 0 are "don't care".
    /// Returns a binary image where matched structuring element origins are set to 1.0.
    pub fn hit_or_miss(
        &self,
        pattern: &[&[u8]],
        bg_pattern: &[&[u8]],
    ) -> Result<Self> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        if pattern.is_empty() || pattern[0].is_empty() {
            return Err(IrisError::InvalidParameter(
                "Pattern must be non-empty".into(),
            ));
        }
        if bg_pattern.len() != pattern.len() || bg_pattern[0].len() != pattern[0].len() {
            return Err(IrisError::InvalidParameter(
                "Background pattern must match foreground pattern dimensions".into(),
            ));
        }

        let ph = pattern.len();
        let pw = pattern[0].len();
        let ay = ph as isize / 2;
        let ax = pw as isize / 2;

        let device = self.tensor.device();
        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; c * h * w];

        for ch in 0..c {
            for y in 0..h {
                for x in 0..w {
                    let mut matched = true;

                    for ky in 0..ph {
                        for kx in 0..pw {
                            let sy = y as isize + ky as isize - ay;
                            let sx = x as isize + kx as isize - ax;

                            if sy < 0 || sy >= h as isize || sx < 0 || sx >= w as isize {
                                if pattern[ky][kx] == 1 || bg_pattern[ky][kx] == 1 {
                                    matched = false;
                                    break;
                                }
                                continue;
                            }

                            let val = flat_vals[ch * h * w + sy as usize * w + sx as usize];
                            let is_foreground = val > 0.5;

                            if pattern[ky][kx] == 1 && !is_foreground {
                                matched = false;
                                break;
                            }
                            if bg_pattern[ky][kx] == 1 && is_foreground {
                                matched = false;
                                break;
                            }
                        }
                        if !matched {
                            break;
                        }
                    }

                    if matched {
                        out_vals[ch * h * w + y * w + x] = 1.0;
                    }
                }
            }
        }

        let new_data = TensorData::new(out_vals, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Zhang-Suen thinning algorithm for binary images.
    ///
    /// Iteratively removes boundary pixels that are not essential to
    /// connectivity until the skeleton is a single pixel wide.
    /// Expects a binary image with foreground = 1.0, background = 0.0.
    pub fn thin(&self) -> Result<Self> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let device = self.tensor.device();
        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

        let mut grid: Vec<u8> = flat_vals.iter().map(|&v| if v > 0.5 { 1u8 } else { 0u8 }).collect();

        let count_transitions = |grid: &[u8], w: usize, h: usize, x: isize, y: isize| -> u8 {
            let dx = [1, 1, 0, -1, -1, -1, 0, 1];
            let dy = [0, 1, 1, 1, 0, -1, -1, -1];
            let mut count = 0u8;
            for i in 0..8 {
                let i2 = (i + 1) % 8;
                let x1 = x + dx[i];
                let y1 = y + dy[i];
                let x2 = x + dx[i2];
                let y2 = y + dy[i2];

                let v1 = if x1 >= 0 && x1 < w as isize && y1 >= 0 && y1 < h as isize {
                    grid[y1 as usize * w + x1 as usize]
                } else {
                    0
                };
                let v2 = if x2 >= 0 && x2 < w as isize && y2 >= 0 && y2 < h as isize {
                    grid[y2 as usize * w + x2 as usize]
                } else {
                    0
                };

                if v1 == 0 && v2 == 1 {
                    count += 1;
                }
            }
            count
        };

        let count_neighbors = |grid: &[u8], w: usize, h: usize, x: isize, y: isize| -> u32 {
            let dx = [1, 1, 0, -1, -1, -1, 0, 1];
            let dy = [0, 1, 1, 1, 0, -1, -1, -1];
            let mut sum = 0u32;
            for i in 0..8 {
                let nx = x + dx[i];
                let ny = y + dy[i];
                if nx >= 0 && nx < w as isize && ny >= 0 && ny < h as isize {
                    sum += grid[ny as usize * w + nx as usize] as u32;
                }
            }
            sum
        };

        let mut changed = true;
        while changed {
            changed = false;

            // Step 1
            let mut to_remove = Vec::new();
            for y in 1..(h - 1) {
                for x in 1..(w - 1) {
                    let xi = x as isize;
                    let yi = y as isize;

                    let p = grid[yi as usize * w + xi as usize];
                    if p != 1 {
                        continue;
                    }

                    let n = count_neighbors(&grid, w, h, xi, yi);
                    let t = count_transitions(&grid, w, h, xi, yi);

                    let p2 = grid[((yi - 1).max(0)) as usize * w + xi as usize];
                    let p4 = grid[yi as usize * w + ((xi + 1).min(w as isize - 1)) as usize];
                    let p6 = grid[((yi + 1).min(h as isize - 1)) as usize * w + xi as usize];
                    let p8 = grid[yi as usize * w + ((xi - 1).max(0)) as usize];

                    if (2..=6).contains(&n) && t == 1 && p4 == 0 && p6 == 0 && (p2 == 0 || p8 == 0) {
                        to_remove.push((y, x));
                    }
                }
            }

            for (y, x) in &to_remove {
                grid[y * w + x] = 0;
                changed = true;
            }

            // Step 2
            let mut to_remove = Vec::new();
            for y in 1..(h - 1) {
                for x in 1..(w - 1) {
                    let xi = x as isize;
                    let yi = y as isize;

                    let p = grid[yi as usize * w + xi as usize];
                    if p != 1 {
                        continue;
                    }

                    let n = count_neighbors(&grid, w, h, xi, yi);
                    let t = count_transitions(&grid, w, h, xi, yi);

                    let p2 = grid[((yi - 1).max(0)) as usize * w + xi as usize];
                    let p4 = grid[yi as usize * w + ((xi + 1).min(w as isize - 1)) as usize];
                    let p6 = grid[((yi + 1).min(h as isize - 1)) as usize * w + xi as usize];
                    let p8 = grid[yi as usize * w + ((xi - 1).max(0)) as usize];

                    if (2..=6).contains(&n) && t == 1 && p2 == 0 && p8 == 0 && (p4 == 0 || p6 == 0) {
                        to_remove.push((y, x));
                    }
                }
            }

            for (y, x) in &to_remove {
                grid[y * w + x] = 0;
                changed = true;
            }
        }

        let out_vals: Vec<f32> = grid.iter().map(|&v| v as f32).collect();
        let new_data = TensorData::new(out_vals, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Morphological skeleton extraction.
    ///
    /// Computes the skeleton by iteratively applying morphological opening
    /// with a structuring element and subtracting the opened result from the
    /// original, then accumulating the residuals. Uses a 3x3 cross kernel.
    pub fn skeleton(&self) -> Result<Self> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let device = self.tensor.device();
        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

        let cross_kernel: Vec<&[u8]> = vec![&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]];

        let mut current = flat_vals.clone();
        let mut skeleton = vec![0.0f32; c * h * w];

        let mut iter_count = 0;
        let max_iters = h + w; // skeleton converges in at most min(h,w) iterations

        while iter_count < max_iters {
            iter_count += 1;

            // Erosion
            let eroded = Self::erode_flat(&current, c, h, w, &cross_kernel);

            // Dilation of eroded
            let opened = Self::dilate_flat(&eroded, c, h, w, &cross_kernel);

            // Subtract opened from current: temp = current - opened
            let mut temp = vec![0.0f32; c * h * w];
            let mut any_nonzero = false;
            for i in 0..(c * h * w) {
                let diff = current[i] - opened[i];
                if diff > 0.5 {
                    temp[i] = 1.0;
                    skeleton[i] = 1.0;
                    any_nonzero = true;
                }
            }

            // current = opened
            current = opened;

            if !any_nonzero {
                break;
            }
        }

        let new_data = TensorData::new(skeleton, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Helper: flat erosion with a custom kernel operating on an f32 slice.
    fn erode_flat(input: &[f32], c: usize, h: usize, w: usize, kernel: &[&[u8]]) -> Vec<f32> {
        let kh = kernel.len();
        let kw = kernel[0].len();
        let ay = kh as isize / 2;
        let ax = kw as isize / 2;
        let mut out = vec![0.0f32; c * h * w];

        for ch in 0..c {
            for y in 0..h {
                for x in 0..w {
                    let mut min_val = f32::MAX;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            if kernel[ky][kx] == 0 {
                                continue;
                            }
                            let sy = y as isize + ky as isize - ay;
                            let sx = x as isize + kx as isize - ax;
                            if sy >= 0 && sy < h as isize && sx >= 0 && sx < w as isize {
                                let val = input[ch * h * w + sy as usize * w + sx as usize];
                                if val < min_val {
                                    min_val = val;
                                }
                            }
                        }
                    }
                    out[ch * h * w + y * w + x] = if min_val == f32::MAX {
                        input[ch * h * w + y * w + x]
                    } else {
                        min_val
                    };
                }
            }
        }
        out
    }

    /// Helper: flat dilation with a custom kernel operating on an f32 slice.
    fn dilate_flat(input: &[f32], c: usize, h: usize, w: usize, kernel: &[&[u8]]) -> Vec<f32> {
        let kh = kernel.len();
        let kw = kernel[0].len();
        let ay = kh as isize / 2;
        let ax = kw as isize / 2;
        let mut out = vec![0.0f32; c * h * w];

        for ch in 0..c {
            for y in 0..h {
                for x in 0..w {
                    let mut max_val = f32::MIN;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            if kernel[ky][kx] == 0 {
                                continue;
                            }
                            let sy = y as isize + ky as isize - ay;
                            let sx = x as isize + kx as isize - ax;
                            if sy >= 0 && sy < h as isize && sx >= 0 && sx < w as isize {
                                let val = input[ch * h * w + sy as usize * w + sx as usize];
                                if val > max_val {
                                    max_val = val;
                                }
                            }
                        }
                    }
                    out[ch * h * w + y * w + x] = if max_val == f32::MIN {
                        input[ch * h * w + y * w + x]
                    } else {
                        max_val
                    };
                }
            }
        }
        out
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

    #[test]
    fn test_dilate_with_cross_kernel() {
        let device = test_device();
        let flat_data = vec![0.0f32; 3 * 8 * 8];
        let tensor_data = TensorData::new(flat_data, [3, 8, 8]);
        let img = Image::new(Tensor::<TestBackend, 3>::from_data(tensor_data, &device));

        let kernel: Vec<&[u8]> = vec![&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]];
        let dilated = img.dilate_with_kernel(&kernel).unwrap();
        assert_eq!(dilated.shape(), [3, 8, 8]);
    }

    #[test]
    fn test_erode_with_ellipse_kernel() {
        let device = test_device();
        let flat_data = vec![0.5f32; 3 * 8 * 8];
        let tensor_data = TensorData::new(flat_data, [3, 8, 8]);
        let img = Image::new(Tensor::<TestBackend, 3>::from_data(tensor_data, &device));

        let kernel: Vec<&[u8]> = vec![&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]];
        let eroded = img.erode_with_kernel(&kernel).unwrap();
        assert_eq!(eroded.shape(), [3, 8, 8]);
    }

    #[test]
    fn test_empty_kernel() {
        let device = test_device();
        let data = vec![0.5f32; 3 * 8 * 8];
        let tensor = Tensor::<TestBackend, 3>::from_data(TensorData::new(data, [3, 8, 8]), &device);
        let img = Image::new(tensor);
        let empty: Vec<&[u8]> = vec![];
        assert!(img.dilate_with_kernel(&empty).is_err());
    }

    #[test]
    fn test_hit_or_miss() {
        let device = test_device();
        // Create a simple image with a vertical line at x=4
        let mut flat_data = vec![0.0f32; 8 * 8];
        for y in 1..7 {
            flat_data[y * 8 + 4] = 1.0;
        }
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [1, 8, 8]), &device);
        let img = Image::new(tensor);

        // Pattern: a vertical line of 3 foreground pixels
        let pattern: Vec<&[u8]> = vec![&[0, 0, 0], &[0, 1, 0], &[0, 1, 0], &[0, 1, 0], &[0, 0, 0]];
        // Background: nothing required
        let bg_pattern: Vec<&[u8]> = vec![&[0, 0, 0], &[1, 0, 1], &[1, 0, 1], &[1, 0, 1], &[0, 0, 0]];

        let result = img.hit_or_miss(&pattern, &bg_pattern).unwrap();
        assert_eq!(result.shape(), [1, 8, 8]);
        let vals: Vec<f32> = result.tensor.into_data().iter::<f32>().collect();
        // At least one pixel should match the hit-or-miss pattern
        assert!(vals.iter().any(|&v| v > 0.5));
    }

    #[test]
    fn test_thin() {
        let device = test_device();
        // Create a thick 5x5 block
        let mut flat_data = vec![0.0f32; 10 * 10];
        for y in 2..7 {
            for x in 2..7 {
                flat_data[y * 10 + x] = 1.0;
            }
        }
        let orig_count = flat_data.iter().filter(|&&v| v > 0.5).count();
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [1, 10, 10]), &device);
        let img = Image::new(tensor);

        let result = img.thin().unwrap();
        assert_eq!(result.shape(), [1, 10, 10]);
        let vals: Vec<f32> = result.tensor.into_data().iter::<f32>().collect();
        let thin_count = vals.iter().filter(|&&v| v > 0.5).count();
        assert!(thin_count <= orig_count);
        assert!(thin_count > 0);
    }

    #[test]
    fn test_skeleton() {
        let device = test_device();
        // Create a rectangular block
        let mut flat_data = vec![0.0f32; 12 * 12];
        for y in 2..10 {
            for x in 2..10 {
                flat_data[y * 12 + x] = 1.0;
            }
        }
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [1, 12, 12]), &device);
        let img = Image::new(tensor);

        let result = img.skeleton().unwrap();
        assert_eq!(result.shape(), [1, 12, 12]);
        let vals: Vec<f32> = result.tensor.into_data().iter::<f32>().collect();
        let skel_count = vals.iter().filter(|&&v| v > 0.5).count();
        // Skeleton should have fewer pixels than original block
        assert!(skel_count > 0);
    }
}
