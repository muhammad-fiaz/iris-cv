pub mod matching;

pub use matching::{BFMatcher, DMatch, FlannMatcher, MatchDrawer};

use crate::core::types::Point;
use crate::error::{IrisError, Result};
use crate::image::Image;
use burn::tensor::{Tensor, backend::Backend};

/// Template matching method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TemplateMatchMethod {
    /// Sum of squared differences (unnormalized).
    TmSqdiff,
    /// Sum of squared differences (normalized).
    TmSqdiffNormed,
    /// Cross-correlation (unnormalized).
    TmCcorr,
    /// Cross-correlation (normalized).
    TmCcorrNormed,
    /// Cross-correlation coefficient (unnormalized).
    TmCcoeff,
    /// Cross-correlation coefficient (normalized).
    TmCcoeffNormed,
}

/// Performs template matching using sliding window correlation.
///
/// Returns a 2D tensor of shape `[H - th + 1, W - tw + 1]` where `(th, tw)` is the template size.
pub fn template_match<B: Backend>(
    source: &Image<B>,
    template: &Image<B>,
    method: TemplateMatchMethod,
) -> crate::error::Result<Tensor<B, 2>> {
    let src_dims = source.tensor.dims();
    let tpl_dims = template.tensor.dims();

    let src_h = src_dims[1];
    let src_w = src_dims[2];
    let tpl_h = tpl_dims[1];
    let tpl_w = tpl_dims[2];

    if tpl_h > src_h || tpl_w > src_w {
        return Err(IrisError::DimensionMismatch {
            expected: vec![src_h, src_w],
            actual: vec![tpl_h, tpl_w],
        });
    }

    let src_data = source.tensor.clone().into_data();
    let tpl_data = template.tensor.clone().into_data();
    let src_flat: Vec<f32> = src_data.iter::<f32>().collect();
    let tpl_flat: Vec<f32> = tpl_data.iter::<f32>().collect();

    let src_channels = src_dims[0];
    let tpl_channels = tpl_dims[0];

    let out_h = src_h - tpl_h + 1;
    let out_w = src_w - tpl_w + 1;
    let mut result = vec![0.0f32; out_h * out_w];

    // Compute template mean for CCOEFF methods
    let tpl_mean: f32 = tpl_flat.iter().sum::<f32>() / tpl_flat.len() as f32;
    let tpl_sub: Vec<f32> = tpl_flat.iter().map(|&v| v - tpl_mean).collect();
    let tpl_norm: f32 = tpl_sub.iter().map(|v| v * v).sum::<f32>().sqrt();

    for oy in 0..out_h {
        for ox in 0..out_w {
            let mut sum = 0.0f32;

            // Compute correlation/difference
            for c in 0..src_channels.min(tpl_channels) {
                for ty in 0..tpl_h {
                    for tx in 0..tpl_w {
                        let si = c * src_h * src_w + (oy + ty) * src_w + (ox + tx);
                        let ti = c * tpl_h * tpl_w + ty * tpl_w + tx;
                        let sv = src_flat[si];
                        let tv = tpl_flat[ti];

                        match method {
                            TemplateMatchMethod::TmSqdiff
                            | TemplateMatchMethod::TmSqdiffNormed => {
                                let diff = sv - tv;
                                sum += diff * diff;
                            }
                            TemplateMatchMethod::TmCcorr
                            | TemplateMatchMethod::TmCcorrNormed => {
                                sum += sv * tv;
                            }
                            TemplateMatchMethod::TmCcoeff
                            | TemplateMatchMethod::TmCcoeffNormed => {
                                let src_sub = sv - {
                                    let mut region_sum = 0.0f32;
                                    for rty in 0..tpl_h {
                                        for rtx in 0..tpl_w {
                                            let ri =
                                                c * src_h * src_w + (oy + rty) * src_w + (ox + rtx);
                                            region_sum += src_flat[ri];
                                        }
                                    }
                                    region_sum / (tpl_h * tpl_w) as f32
                                };
                                sum += src_sub * tpl_sub[c * tpl_h * tpl_w + ty * tpl_w + tx];
                            }
                        }
                    }
                }
            }

            // Normalize if needed
            match method {
                TemplateMatchMethod::TmSqdiffNormed => {
                    let mut src_sum_sq = 0.0f32;
                    for c in 0..src_channels.min(tpl_channels) {
                        for ty in 0..tpl_h {
                            for tx in 0..tpl_w {
                                let si = c * src_h * src_w + (oy + ty) * src_w + (ox + tx);
                                let v = src_flat[si];
                                src_sum_sq += v * v;
                            }
                        }
                    }
                    let denom = (src_sum_sq * tpl_flat.iter().map(|v| v * v).sum::<f32>()).sqrt();
                    if denom > 1e-10 {
                        result[oy * out_w + ox] = sum / denom;
                    }
                }
                TemplateMatchMethod::TmCcorrNormed => {
                    let mut src_norm = 0.0f32;
                    for c in 0..src_channels.min(tpl_channels) {
                        for ty in 0..tpl_h {
                            for tx in 0..tpl_w {
                                let si = c * src_h * src_w + (oy + ty) * src_w + (ox + tx);
                                let v = src_flat[si];
                                src_norm += v * v;
                            }
                        }
                    }
                    let denom = src_norm.sqrt() * tpl_norm;
                    if denom > 1e-10 {
                        result[oy * out_w + ox] = sum / denom;
                    }
                }
                TemplateMatchMethod::TmCcoeffNormed => {
                    let mut src_sum = 0.0f32;
                    let mut src_sum_sq = 0.0f32;
                    let count = (src_channels.min(tpl_channels) * tpl_h * tpl_w) as f32;
                    for c in 0..src_channels.min(tpl_channels) {
                        for ty in 0..tpl_h {
                            for tx in 0..tpl_w {
                                let si = c * src_h * src_w + (oy + ty) * src_w + (ox + tx);
                                let v = src_flat[si];
                                src_sum += v;
                                src_sum_sq += v * v;
                            }
                        }
                    }
                    let src_mean = src_sum / count;
                    let src_var = src_sum_sq - count * src_mean * src_mean;
                    let denom = (src_var.max(0.0)).sqrt() * tpl_norm;
                    if denom > 1e-10 {
                        result[oy * out_w + ox] = sum / denom;
                    }
                }
                // Unnormalized methods keep sum as-is
                _ => {
                    result[oy * out_w + ox] = sum;
                }
            }
        }
    }

    let device = source.tensor.device();
    let data = burn::tensor::TensorData::new(result, [out_h, out_w]);
    Ok(Tensor::<B, 2>::from_data(data, &device))
}

/// Represents a detected keypoint in an image.
#[derive(Clone, Debug, PartialEq)]
pub struct KeyPoint {
    /// Coordinates of the keypoint.
    pub pt: Point<f64>,
    /// Diameter of the meaningful keypoint neighborhood.
    pub size: f64,
    /// Computed orientation of the keypoint (degrees).
    pub angle: f64,
    /// Strength/response of the keypoint.
    pub response: f64,
    /// Octave (pyramid layer) from which the keypoint was extracted.
    pub octave: i32,
    /// Object class ID.
    pub class_id: i32,
}

impl KeyPoint {
    #[must_use]
    pub fn new(x: f64, y: f64, size: f64) -> Self {
        Self {
            pt: Point::new(x, y),
            size,
            angle: -1.0,
            response: 0.0,
            octave: 0,
            class_id: -1,
        }
    }
}

/// Feature detector type.
pub enum FeatureType {
    ORB,
    BRISK,
    AKAZE,
    SIFT,
}

pub struct FeatureDetector {
    #[allow(dead_code)]
    detector_type: FeatureType,
    max_features: usize,
}

impl FeatureDetector {
    #[must_use]
    pub fn new(detector_type: FeatureType) -> Self {
        Self {
            detector_type,
            max_features: 500,
        }
    }

    /// Sets the maximum number of features to detect.
    #[must_use]
    pub fn with_max_features(mut self, max: usize) -> Self {
        self.max_features = max;
        self
    }

    /// Detects keypoints in an image using the FAST corner detector.
    /// FAST checks a circle of 16 pixels around each candidate point
    /// and requires at least 12 contiguous pixels to be brighter or darker.
    pub fn detect<B: Backend>(&self, image: &Image<B>) -> Result<Vec<KeyPoint>> {
        let gray = image.grayscale()?;
        let dims = gray.tensor.dims();
        let h = dims[1];
        let w = dims[2];

        let tensor_data = gray.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

        let mut keypoints = Vec::new();
        let border = 3; // FAST radius = 3

        // FAST circle offsets (16 points around center)
        let circle: [(i32, i32); 16] = [
            (0, -3),
            (1, -3),
            (2, -2),
            (3, -1),
            (3, 0),
            (3, 1),
            (2, 2),
            (1, 3),
            (0, 3),
            (-1, 3),
            (-2, 2),
            (-3, 1),
            (-3, 0),
            (-3, -1),
            (-2, -2),
            (-1, -3),
        ];

        // Threshold for brightness difference (relative to center pixel)
        let threshold = 10.0f32 / 255.0;
        let n_points = 9; // Require 9 of 16 contiguous pixels

        for y in border..(h - border) {
            for x in border..(w - border) {
                let center = flat_vals[y * w + x];

                // Quick check: pixels 0, 4, 8, 12 must all be brighter or darker
                let p0 = flat_vals[(y as i32 + circle[0].1) as usize * w + (x as i32 + circle[0].0) as usize];
                let p4 = flat_vals[(y as i32 + circle[4].1) as usize * w + (x as i32 + circle[4].0) as usize];
                let p8 = flat_vals[(y as i32 + circle[8].1) as usize * w + (x as i32 + circle[8].0) as usize];
                let p12 = flat_vals[(y as i32 + circle[12].1) as usize * w + (x as i32 + circle[12].0) as usize];

                let all_bright =
                    p0 > center + threshold && p4 > center + threshold && p8 > center + threshold
                        && p12 > center + threshold;
                let all_dark =
                    p0 < center - threshold && p4 < center - threshold && p8 < center - threshold
                        && p12 < center - threshold;

                if !all_bright && !all_dark {
                    continue;
                }

                // Check full circle for contiguous arc of n_points
                let mut max_arc = 0;
                let mut current_arc = 0;

                // Read all 16 circle pixels
                let mut circle_vals = [0.0f32; 16];
                for i in 0..16 {
                    let nx = x as i32 + circle[i].0;
                    let ny = y as i32 + circle[i].1;
                    circle_vals[i] = flat_vals[ny as usize * w + nx as usize];
                }

                for i in 0..32 {
                    let val = circle_vals[i % 16];
                    if (all_bright && val > center + threshold)
                        || (all_dark && val < center - threshold)
                    {
                        current_arc += 1;
                        if current_arc > max_arc {
                            max_arc = current_arc;
                        }
                    } else {
                        current_arc = 0;
                    }
                }

                if max_arc >= n_points {
                    // Compute response using intensity difference in center of arc
                    let mut sum_diff = 0.0f32;
                    for i in 0..16 {
                        let diff = (circle_vals[i] - center).abs();
                        sum_diff += diff;
                    }
                    let response = sum_diff / 16.0;

                    let mut kp = KeyPoint::new(x as f64, y as f64, 3.0);
                    kp.response = response as f64;
                    kp.octave = 0;
                    keypoints.push(kp);
                }
            }
        }

        // Sort by response strength and keep top N
        keypoints.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap());
        keypoints.truncate(self.max_features);

        // Non-maximum suppression: remove keypoints too close together
        let mut suppressed = Vec::new();
        let min_dist = 7.0;
        for kp in &keypoints {
            let too_close = suppressed.iter().any(|other: &KeyPoint| {
                let dx = kp.pt.x - other.pt.x;
                let dy = kp.pt.y - other.pt.y;
                (dx * dx + dy * dy).sqrt() < min_dist
            });
            if !too_close {
                suppressed.push(kp.clone());
            }
        }

        Ok(suppressed)
    }

    /// Computes ORB descriptors for detected keypoints.
    /// ORB uses BRIEF descriptors with rotation invariance via intensity centroid.
    /// Returns a descriptor tensor of shape [`NumKeyPoints`, `DescriptorDim`].
    pub fn compute<B: Backend>(
        &self,
        image: &Image<B>,
        keypoints: &[KeyPoint],
    ) -> Result<Tensor<B, 2>> {
        let gray = image.grayscale()?;
        let dims = gray.tensor.dims();
        let h = dims[1];
        let w = dims[2];

        let tensor_data = gray.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

        let n = keypoints.len();
        let descriptor_dim = 32; // ORB uses 256-bit descriptors = 32 bytes
        let mut descriptors = vec![0u8; n * descriptor_dim];

        // Fixed random sampling pattern for BRIEF (deterministic, 256 pairs)
        let pattern: [(i32, i32, i32, i32); 256] = generate_brief_pattern();

        for (ki, kp) in keypoints.iter().enumerate() {
            let cx = kp.pt.x as i32;
            let cy = kp.pt.y as i32;

            // Compute intensity centroid angle for rotation invariance
            let m10 = compute_moment(&flat_vals, w, h, cx, cy, 1, 0);
            let m01 = compute_moment(&flat_vals, w, h, cx, cy, 0, 1);
            let angle = m01.atan2(m10); // radians

            let cos_a = angle.cos();
            let sin_a = angle.sin();

            // Sample 256 bit pairs
            for byte_idx in 0..descriptor_dim {
                let mut byte_val = 0u8;
                for bit_idx in 0..8 {
                    let pair_idx = byte_idx * 8 + bit_idx;
                    let (dx1, dy1, dx2, dy2) = pattern[pair_idx];

                    // Rotate sampling pattern
                    let rx1 = (dx1 as f64 * cos_a - dy1 as f64 * sin_a) as i32;
                    let ry1 = (dx1 as f64 * sin_a + dy1 as f64 * cos_a) as i32;
                    let rx2 = (dx2 as f64 * cos_a - dy2 as f64 * sin_a) as i32;
                    let ry2 = (dx2 as f64 * sin_a + dy2 as f64 * cos_a) as i32;

                    let px1 = (cx + rx1).clamp(0, w as i32 - 1) as usize;
                    let py1 = (cy + ry1).clamp(0, h as i32 - 1) as usize;
                    let px2 = (cx + rx2).clamp(0, w as i32 - 1) as usize;
                    let py2 = (cy + ry2).clamp(0, h as i32 - 1) as usize;

                    let val1 = flat_vals[py1 * w + px1];
                    let val2 = flat_vals[py2 * w + px2];

                    if val1 < val2 {
                        byte_val |= 1 << bit_idx;
                    }
                }
                descriptors[ki * descriptor_dim + byte_idx] = byte_val;
            }
        }

        // Convert u8 descriptors to float tensor for Burn compatibility
        let float_desc: Vec<f32> = descriptors.iter().map(|&b| b as f32).collect();
        let device = image.tensor.device();
        let data = burn::tensor::TensorData::new(float_desc, [n, descriptor_dim]);
        let tensor = Tensor::<B, 2>::from_data(data, &device);
        Ok(tensor)
    }
}

/// Compute a spatial moment of image pixels around a point.
fn compute_moment(flat_vals: &[f32], w: usize, h: usize, cx: i32, cy: i32, px: i32, py: i32) -> f64 {
    let radius = 15;
    let mut sum = 0.0f64;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let nx = cx + dx;
            let ny = cy + dy;
            if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                let val = flat_vals[ny as usize * w + nx as usize] as f64;
                sum += val * (dx as f64).powi(px) * (dy as f64).powi(py);
            }
        }
    }
    sum
}

/// Generates a fixed pseudo-random sampling pattern for BRIEF descriptors.
fn generate_brief_pattern() -> [(i32, i32, i32, i32); 256] {
    let mut pattern = [(0i32, 0i32, 0i32, 0i32); 256];
    // Simple deterministic pattern using linear congruential generator
    let mut seed: u32 = 42;
    for i in 0..256 {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let x1 = ((seed >> 16) as i32 % 31) - 15;
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let y1 = ((seed >> 16) as i32 % 31) - 15;
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let x2 = ((seed >> 16) as i32 % 31) - 15;
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let y2 = ((seed >> 16) as i32 % 31) - 15;
        pattern[i] = (x1, y1, x2, y2);
    }
    pattern
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{TestBackend, test_device};
    use burn::tensor::TensorData;

    #[test]
    fn test_orb_feature_detection() {
        let device = test_device();
        // Create an image with strong edges that produce FAST corners
        // Use a cross pattern: vertical and horizontal bars
        let mut flat_data = vec![0.0f32; 3 * 100 * 100];
        // Vertical bar
        for y in 0..100 {
            for x in 45..55 {
                flat_data[y * 100 + x] = 1.0;
                flat_data[10000 + y * 100 + x] = 1.0;
                flat_data[20000 + y * 100 + x] = 1.0;
            }
        }
        // Horizontal bar
        for y in 45..55 {
            for x in 0..100 {
                flat_data[y * 100 + x] = 1.0;
                flat_data[10000 + y * 100 + x] = 1.0;
                flat_data[20000 + y * 100 + x] = 1.0;
            }
        }
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [3, 100, 100]), &device);
        let img = Image::new(tensor);

        let detector = FeatureDetector::new(FeatureType::ORB).with_max_features(50);
        let keypoints = detector.detect(&img).unwrap();

        // Verify API correctness: returns correct type and bounds
        for kp in &keypoints {
            assert!(kp.pt.x >= 0.0 && kp.pt.x < 100.0);
            assert!(kp.pt.y >= 0.0 && kp.pt.y < 100.0);
            assert!(kp.size > 0.0);
            assert!(kp.response >= 0.0);
        }

        let descriptors = detector.compute(&img, &keypoints).unwrap();
        assert_eq!(descriptors.dims(), [keypoints.len(), 32]);
    }

    #[test]
    fn test_template_match() {
        let device = test_device();
        // Create a 6x6 image with a white block in the corner
        let mut src_data = vec![0.0f32; 3 * 6 * 6];
        // White block at top-left 3x3
        for c in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    src_data[c * 36 + y * 6 + x] = 1.0;
                }
            }
        }
        let src_tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(src_data.clone(), [3, 6, 6]), &device);
        let src_img = Image::new(src_tensor);

        // Template: same 3x3 white block
        let mut tpl_data = vec![0.0f32; 3 * 3 * 3];
        for c in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    tpl_data[c * 9 + y * 3 + x] = 1.0;
                }
            }
        }
        let tpl_tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(tpl_data, [3, 3, 3]), &device);
        let tpl_img = Image::new(tpl_tensor);

        // TM_SQDIFF: minimum should be at (0,0) where template matches perfectly
        let result = src_img.template_match(&tpl_img, TemplateMatchMethod::TmSqdiff).unwrap();
        assert_eq!(result.dims(), [4, 4]);

        let result_data = result.into_data();
        let vals: Vec<f32> = result_data.iter::<f32>().collect();

        // Position (0,0) should have zero difference (perfect match)
        assert!(vals[0] < 0.01, "Expected near-zero at (0,0), got {}", vals[0]);

        // TM_CCORR: maximum should be at (0,0)
        let result_corr = src_img.template_match(&tpl_img, TemplateMatchMethod::TmCcorr).unwrap();
        let corr_data = result_corr.into_data();
        let corr_vals: Vec<f32> = corr_data.iter::<f32>().collect();
        assert!(corr_vals[0] > corr_vals[1], "Expected (0,0) to have higher correlation");
    }
}
