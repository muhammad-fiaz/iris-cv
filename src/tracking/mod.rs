pub mod subtractor;

pub use subtractor::BackgroundSubtractor;

use crate::core::types::Rect;
use crate::error::{IrisError, Result};
use crate::image::Image;
use burn::tensor::backend::Backend;

/// Standard tracker algorithms.
pub enum TrackerType {
    KCF,
    CSRT,
    MOSSE,
}

/// MOSSE (Minimum Output Sum of Squared Errors) tracker.
///
/// MOSSE is a correlation filter-based tracker that adapts to appearance changes.
/// It uses element-wise multiplication in the frequency domain for fast correlation.
struct MosseState {
    /// Pre-trained filter in frequency domain
    filter_freq: Vec<f64>,
    /// Learning rate for online adaptation
    learning_rate: f64,
    /// Target size (height, width)
    target_size: (usize, usize),
    /// Previous frame grayscale data
    prev_gray: Option<Vec<f32>>,
    /// Previous bounding box
    prev_bbox: Option<Rect<usize>>,
    /// Width of search window
    search_w: usize,
    /// Height of search window
    search_h: usize,
}

impl MosseState {
    fn new() -> Self {
        Self {
            filter_freq: Vec::new(),
            learning_rate: 0.125,
            target_size: (0, 0),
            prev_gray: None,
            prev_bbox: None,
            search_w: 0,
            search_h: 0,
        }
    }

    fn init(&mut self, gray_data: &[f32], img_w: usize, bbox: Rect<usize>) {
        let cy = bbox.y + bbox.height / 2;
        let cx = bbox.x + bbox.width / 2;
        let patch_h = bbox.height.max(4);
        let patch_w = bbox.width.max(4);

        // Extract and resize patch to fixed size (use bbox size)
        let size = patch_h.max(patch_w);
        self.target_size = (size, size);
        self.search_w = size * 2;
        self.search_h = size * 2;

        // Extract patch centered on bbox
        let patch = extract_patch(gray_data, img_w, img_w, cx, cy, size, size);

        // Compute FFT of patch (simplified - use DFT)
        let patch_freq = simple_dft_2d(&patch, size, size);

        // Create desired output (Gaussian centered at target)
        let mut target = vec![0.0f64; size * size];
        let center = size as f64 / 2.0;
        let sigma = size as f64 / 4.0;
        for y in 0..size {
            for x in 0..size {
                let dx = x as f64 - center;
                let dy = y as f64 - center;
                target[y * size + x] = (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
            }
        }
        let target_freq = simple_dft_2d(&target, size, size);

        // Initialize filter: H = G / (A + epsilon) where A accumulates
        let mut filter = Vec::with_capacity(size * size);
        let epsilon = 1e-4;
        for i in 0..size * size {
            let a_re = patch_freq[i * 2];
            let a_im = patch_freq[i * 2 + 1];
            let g_re = target_freq[i * 2];
            let g_im = target_freq[i * 2 + 1];
            let denom = a_re * a_re + a_im * a_im + epsilon;
            filter.push((g_re * a_re + g_im * a_im) / denom);
            filter.push((g_im * a_re - g_re * a_im) / denom);
        }

        self.filter_freq = filter;
        self.prev_gray = Some(gray_data.to_vec());
        self.prev_bbox = Some(bbox);
    }

    fn update(&mut self, gray_data: &[f32], img_w: usize) -> Option<Rect<usize>> {
        let prev_bbox = self.prev_bbox?;
        let (size, _) = self.target_size;

        let cy = prev_bbox.y + prev_bbox.height / 2;
        let cx = prev_bbox.x + prev_bbox.width / 2;

        // Search in a window around previous position
        let search_cx = cx;
        let search_cy = cy;

        let patch = extract_patch(gray_data, img_w, img_w, search_cx, search_cy, size, size);
        let patch_freq = simple_dft_2d(&patch, size, size);

        // Correlate: response = IFFT(H * conj(P))
        let mut response = vec![0.0f64; size * size];
        let mut max_val = f64::NEG_INFINITY;
        let mut max_idx = 0;

        for i in 0..size * size {
            let p_re = patch_freq[i * 2];
            let p_im = patch_freq[i * 2 + 1];
            let h_re = self.filter_freq[i * 2];
            let h_im = self.filter_freq[i * 2 + 1];
            // H * conj(P)
            let re = h_re * p_re + h_im * p_im;
            let _im = h_im * p_re - h_re * p_im;
            response[i] = re; // Real part of IFFT (simplified)
            if re > max_val {
                max_val = re;
                max_idx = i;
            }
        }

        // Find peak location
        let peak_y = max_idx / size;
        let peak_x = max_idx % size;
        let dy = peak_y as i32 - size as i32 / 2;
        let dx = peak_x as i32 - size as i32 / 2;

        let new_cx = (cx as i32 + dx).max(0) as usize;
        let new_cy = (cy as i32 + dy).max(0) as usize;

        // Update filter with new appearance
        let new_patch = extract_patch(gray_data, img_w, img_w, new_cx, new_cy, size, size);
        let new_freq = simple_dft_2d(&new_patch, size, size);

        let alpha = self.learning_rate;
        for i in 0..size * size {
            self.filter_freq[i * 2] =
                self.filter_freq[i * 2] * (1.0 - alpha) + new_freq[i * 2] * alpha;
            self.filter_freq[i * 2 + 1] =
                self.filter_freq[i * 2 + 1] * (1.0 - alpha) + new_freq[i * 2 + 1] * alpha;
        }

        let new_bbox = Rect::new(
            new_cx.saturating_sub(prev_bbox.width / 2),
            new_cy.saturating_sub(prev_bbox.height / 2),
            prev_bbox.width,
            prev_bbox.height,
        );

        self.prev_gray = Some(gray_data.to_vec());
        self.prev_bbox = Some(new_bbox);

        Some(new_bbox)
    }
}

/// Extract a patch from a grayscale image centered at (cx, cy).
fn extract_patch(
    data: &[f32],
    img_w: usize,
    img_h: usize,
    cx: usize,
    cy: usize,
    patch_w: usize,
    patch_h: usize,
) -> Vec<f64> {
    let mut patch = vec![0.0f64; patch_h * patch_w];
    let half_w = patch_w / 2;
    let half_h = patch_h / 2;

    for py in 0..patch_h {
        for px in 0..patch_w {
            let sx = cx as i32 + px as i32 - half_w as i32;
            let sy = cy as i32 + py as i32 - half_h as i32;
            if sx >= 0 && sx < img_w as i32 && sy >= 0 && sy < img_h as i32 {
                patch[py * patch_w + px] = data[sy as usize * img_w + sx as usize] as f64;
            }
        }
    }
    patch
}

/// Simple 2D DFT (not FFT - uses O(N²) but correct).
/// Returns interleaved real/imaginary pairs.
fn simple_dft_2d(data: &[f64], w: usize, h: usize) -> Vec<f64> {
    let mut result = vec![0.0f64; w * h * 2];
    let n = w * h;

    for u in 0..h {
        for v in 0..w {
            let mut sum_re = 0.0f64;
            let mut sum_im = 0.0f64;

            for y in 0..h {
                for x in 0..w {
                    let angle =
                        -2.0 * std::f64::consts::PI * ((u * y) as f64 / h as f64 + (v * x) as f64 / w as f64);
                    let val = data[y * w + x];
                    sum_re += val * angle.cos();
                    sum_im += val * angle.sin();
                }
            }

            result[(u * w + v) * 2] = sum_re / (n as f64).sqrt();
            result[(u * w + v) * 2 + 1] = sum_im / (n as f64).sqrt();
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Mean-Shift Tracker
// ---------------------------------------------------------------------------

/// A non-parametric kernel-based tracker that locates the densest region
/// (the target) in successive frames.
///
/// The algorithm builds a colour-histogram model of the target region at
/// initialisation time and then iteratively shifts the window towards the
/// weighted mean (mean-shift) in each subsequent frame until convergence.
pub struct MeanShiftTracker {
    /// Current bounding box.
    bbox: Option<Rect<usize>>,
    /// Target colour histogram (normalised, bins per channel).
    model_hist: Vec<f32>,
    /// Number of bins per colour channel (quantised 0..bins).
    bins: usize,
    /// Spatial bandwidth of the Epanechnikov kernel (radius in pixels).
    spatial_radius: f32,
    /// Maximum mean-shift iterations per update.
    max_iter: usize,
    /// Convergence threshold: stop when the shift is below this value (pixels).
    epsilon: f32,
    /// Width of the last processed image.
    last_w: usize,
    /// Height of the last processed image.
    last_h: usize,
}

impl Default for MeanShiftTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl MeanShiftTracker {
    /// Creates a new tracker with sensible defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bbox: None,
            model_hist: Vec::new(),
            bins: 16,
            spatial_radius: 0.0,
            max_iter: 30,
            epsilon: 0.5,
            last_w: 0,
            last_h: 0,
        }
    }

    /// Sets the number of histogram bins per channel.
    #[must_use]
    pub fn with_bins(mut self, bins: usize) -> Self {
        self.bins = bins;
        self
    }

    /// Sets the maximum number of mean-shift iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Initialises the tracker with a known region-of-interest in `image`.
    pub fn init<B: Backend>(&mut self, image: &Image<B>, roi: Rect<usize>) -> Result<()> {
        let dims = image.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        if c < 3 {
            return Err(IrisError::InvalidParameter(
                "MeanShift requires at least a 3-channel image".into(),
            ));
        }

        let tensor_data = image.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

        let cx = roi.x + roi.width / 2;
        let cy = roi.y + roi.height / 2;
        self.spatial_radius =
            ((roi.width as f32).powi(2) + (roi.height as f32).powi(2)).sqrt() / 2.0;

        // Quantise each RGB channel into `bins` buckets and build the
        // normalised histogram of the ROI.
        let total_bins = self.bins * self.bins * self.bins;
        let mut hist = vec![0.0f32; total_bins];

        for y in roi.y..(roi.y + roi.height).min(h) {
            for x in roi.x..(roi.x + roi.width).min(w) {
                let idx = (y * w + x) * 3;
                let r = flat_vals[idx];
                let g = flat_vals[idx + 1];
                let b = flat_vals[idx + 2];

                let ri = ((r * self.bins as f32) as usize).min(self.bins - 1);
                let gi = ((g * self.bins as f32) as usize).min(self.bins - 1);
                let bi = ((b * self.bins as f32) as usize).min(self.bins - 1);
                let bin = ri * self.bins * self.bins + gi * self.bins + bi;

                // Epanechnikov spatial weight (centred on ROI centre)
                let dx = x as f32 - cx as f32;
                let dy = y as f32 - cy as f32;
                let dist = (dx * dx + dy * dy).sqrt();
                let weight = if dist <= self.spatial_radius {
                    1.0 - (dist / self.spatial_radius).powi(2)
                } else {
                    0.0
                };
                hist[bin] += weight;
            }
        }

        // Normalise the histogram
        let sum: f32 = hist.iter().sum();
        if sum > 1e-10 {
            for v in hist.iter_mut() {
                *v /= sum;
            }
        }

        self.model_hist = hist;
        self.bbox = Some(roi);
        self.last_w = w;
        self.last_h = h;

        Ok(())
    }

    /// Runs one mean-shift iteration and returns the updated bounding box.
    pub fn update<B: Backend>(&mut self, image: &Image<B>) -> Result<Rect<usize>> {
        let dims = image.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let cur_bbox = self.bbox.ok_or_else(|| {
            IrisError::Generic("MeanShiftTracker not initialised. Call init first.".into())
        })?;

        if c < 3 {
            return Err(IrisError::InvalidParameter(
                "MeanShift requires at least a 3-channel image".into(),
            ));
        }

        let tensor_data = image.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

        let mut cx = cur_bbox.x as f32 + cur_bbox.width as f32 / 2.0;
        let mut cy = cur_bbox.y as f32 + cur_bbox.height as f32 / 2.0;
        let hw = cur_bbox.width as f32 / 2.0;
        let hh = cur_bbox.height as f32 / 2.0;

        for _ in 0..self.max_iter {
            // Accumulate numerator and denominator of the mean-shift vector
            let mut sum_wx = 0.0f64;
            let mut sum_wy = 0.0f64;
            let mut sum_w = 0.0f64;

            let y_min = (cy - hh).max(0.0) as usize;
            let y_max = (cy + hh).min(h as f32 - 1.0) as usize;
            let x_min = (cx - hw).max(0.0) as usize;
            let x_max = (cx + hw).min(w as f32 - 1.0) as usize;

            for y in y_min..=y_max {
                for x in x_min..=x_max {
                    let idx = (y * w + x) * 3;
                    let r = flat_vals[idx];
                    let g = flat_vals[idx + 1];
                    let b = flat_vals[idx + 2];

                    let ri = ((r * self.bins as f32) as usize).min(self.bins - 1);
                    let gi = ((g * self.bins as f32) as usize).min(self.bins - 1);
                    let bi = ((b * self.bins as f32) as usize).min(self.bins - 1);
                    let bin = ri * self.bins * self.bins + gi * self.bins + bi;

                    let model_val = self.model_hist[bin];
                    if model_val < 1e-10 {
                        continue;
                    }

                    // Kernel weight (Epanechnikov)
                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    let dist = (dx * dx + dy * dy).sqrt();
                    let kernel_weight = if dist <= self.spatial_radius {
                        1.0 - (dist / self.spatial_radius).powi(2)
                    } else {
                        0.0
                    };

                    let w_i = model_val * kernel_weight;
                    sum_wx += w_i as f64 * x as f64;
                    sum_wy += w_i as f64 * y as f64;
                    sum_w += w_i as f64;
                }
            }

            if sum_w < 1e-10 {
                break;
            }

            let new_cx = (sum_wx / sum_w) as f32;
            let new_cy = (sum_wy / sum_w) as f32;
            let shift_x = new_cx - cx;
            let shift_y = new_cy - cy;

            cx = new_cx;
            cy = new_cy;

            if (shift_x * shift_x + shift_y * shift_y).sqrt() < self.epsilon {
                break;
            }
        }

        let new_bbox = Rect::new(
            (cx - hw).round().max(0.0) as usize,
            (cy - hh).round().max(0.0) as usize,
            cur_bbox.width,
            cur_bbox.height,
        );

        self.bbox = Some(new_bbox);
        self.last_w = w;
        self.last_h = h;

        Ok(new_bbox)
    }
}

/// Object tracker pipeline.
pub struct Tracker<B: Backend> {
    pub tracker_type: TrackerType,
    pub bbox: Option<Rect<usize>>,
    mosse_state: MosseState,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> Tracker<B> {
    /// Creates a new Tracker.
    #[must_use]
    pub fn new(tracker_type: TrackerType) -> Self {
        Self {
            tracker_type,
            bbox: None,
            mosse_state: MosseState::new(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Initializes the tracker with a known bounding box of the target object.
    pub fn init(&mut self, image: &Image<B>, bbox: Rect<usize>) -> Result<()> {
        self.bbox = Some(bbox);

        // Initialize MOSSE state
        let gray = image.grayscale()?;
        let dims = gray.tensor.dims();
        let _h = dims[1];
        let w = dims[2];
        let tensor_data = gray.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        self.mosse_state.init(&flat_vals, w, bbox);

        Ok(())
    }

    /// Updates the tracker, finding the new location of the target in the frame.
    pub fn update(&mut self, image: &Image<B>) -> Result<Rect<usize>> {
        let current = self.bbox.ok_or_else(|| {
            crate::error::IrisError::Generic("Tracker not initialized. Call init first.".into())
        })?;

        // Try MOSSE update
        let gray = image.grayscale()?;
        let dims = gray.tensor.dims();
        let _h = dims[1];
        let w = dims[2];
        let tensor_data = gray.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

        if let Some(new_bbox) = self.mosse_state.update(&flat_vals, w) {
            self.bbox = Some(new_bbox);
            return Ok(new_bbox);
        }

        // Fallback: return current position unchanged
        Ok(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{TestBackend, test_device};
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn test_mosse_tracker() {
        let device = test_device();
        let flat_data = vec![0.5f32; 3 * 32 * 32];
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [3, 32, 32]), &device);
        let img = Image::new(tensor);

        let mut tracker = Tracker::new(TrackerType::MOSSE);
        let init_bbox = Rect::new(8, 8, 16, 16);
        tracker.init(&img, init_bbox).unwrap();

        let updated = tracker.update(&img).unwrap();
        // Should return a valid bounding box
        assert!(updated.width > 0);
        assert!(updated.height > 0);
    }

    #[test]
    fn test_meanshift_tracker_init_and_update() {
        let device = test_device();

        // Create an 8-channel (RGB) image with a distinct coloured region
        // (bright red) somewhere that the tracker should lock onto.
        let mut flat_data = vec![0.2f32; 3 * 32 * 32];
        // Paint a red block at (12, 12) – (20, 20)
        for y in 12..20 {
            for x in 12..20 {
                let idx = (y * 32 + x) * 3;
                flat_data[idx] = 1.0; // R
                flat_data[idx + 1] = 0.0; // G
                flat_data[idx + 2] = 0.0; // B
            }
        }
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [3, 32, 32]), &device);
        let img = Image::new(tensor);

        let mut tracker = MeanShiftTracker::new().with_bins(8).with_max_iter(10);
        let roi = Rect::new(10, 10, 12, 12);
        tracker.init(&img, roi).unwrap();

        // On the same image, the tracker should converge back to the red block
        let updated = tracker.update(&img).unwrap();
        assert!(updated.width > 0);
        assert!(updated.height > 0);
        // The centre should be near the red block centre (16, 16)
        let ucx = updated.x + updated.width / 2;
        let ucy = updated.y + updated.height / 2;
        assert!(
            (ucx as isize - 16).unsigned_abs() <= 2 && (ucy as isize - 16).unsigned_abs() <= 2,
            "Expected centre near (16,16), got ({ucx},{ucy})"
        );
    }

    #[test]
    fn test_tracker_api() {
        let device = test_device();
        let flat_data = vec![0.5f32; 3 * 16 * 16];
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [3, 16, 16]), &device);
        let img = Image::new(tensor);

        let mut tracker = Tracker::new(TrackerType::KCF);
        let init_bbox = Rect::new(2, 2, 8, 8);
        tracker.init(&img, init_bbox).unwrap();

        let updated = tracker.update(&img).unwrap();
        assert_eq!(updated.width, 8);
        assert_eq!(updated.height, 8);
    }
}
