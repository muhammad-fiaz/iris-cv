use crate::error::{IrisError, Result};
use crate::image::Image;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

const INFINITY: f32 = 1.0e10;

/// Wrapper around `f32` that implements `Ord` (panics on NaN).
#[derive(Clone, Copy, PartialEq)]
struct OrdF32(f32);

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Telea (Fast Marching Method) image inpainting.
///
/// Replaces pixels where `mask > 0` by propagating values from the boundary
/// of the known region inward, weighted by distance and gradient.
///
/// * `image` – input image of shape `[C, H, W]`, values in `[0, 1]`.
/// * `mask`  – single-channel mask of shape `[1, H, W]` where `0` = known, `>0` = to inpaint.
/// * `radius` – maximum propagation radius in pixels; known pixels farther than this are ignored.
pub fn inpaint<B: Backend>(image: &Image<B>, mask: &Image<B>, radius: f32) -> Result<Image<B>> {
    let img_dims = image.tensor.dims();
    let mask_dims = mask.tensor.dims();

    if img_dims.len() != 3 || mask_dims.len() != 3 {
        return Err(IrisError::InvalidParameter(
            "image must be [C,H,W] and mask must be [1,H,W]".into(),
        ));
    }
    if img_dims[1] != mask_dims[1] || img_dims[2] != mask_dims[2] {
        return Err(IrisError::DimensionMismatch {
            expected: vec![img_dims[0], mask_dims[1], mask_dims[2]],
            actual: img_dims.to_vec(),
        });
    }
    if mask_dims[0] != 1 {
        return Err(IrisError::InvalidParameter(
            "mask must have exactly 1 channel".into(),
        ));
    }

    let c = img_dims[0];
    let h = img_dims[1];
    let w = img_dims[2];
    let pixels = h * w;

    let img_data = image.tensor.clone().into_data();
    let img_vals: Vec<f32> = img_data.iter::<f32>().collect();
    let mask_data = mask.tensor.clone().into_data();
    let mask_vals: Vec<f32> = mask_data.iter::<f32>().collect();

    let mut inpaint_buf: Vec<Vec<f32>> = (0..c)
        .map(|ch| img_vals[ch * pixels..(ch + 1) * pixels].to_vec())
        .collect();

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum State {
        Known,
        Band,
        Unknown,
    }

    let mut state = vec![State::Unknown; pixels];
    let mut dist = vec![INFINITY; pixels];

    let neighbours = |y: usize, x: usize| -> [(isize, isize); 4] {
        [
            (y as isize - 1, x as isize),
            (y as isize + 1, x as isize),
            (y as isize, x as isize - 1),
            (y as isize, x as isize + 1),
        ]
    };

    let mut heap: BinaryHeap<(Reverse<OrdF32>, usize)> = BinaryHeap::new();

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if mask_vals[idx] == 0.0 {
                state[idx] = State::Known;
                dist[idx] = 0.0;
                for (ny, nx) in neighbours(y, x) {
                    if ny >= 0 && ny < h as isize && nx >= 0 && nx < w as isize {
                        let ni = ny as usize * w + nx as usize;
                        if state[ni] == State::Unknown {
                            state[ni] = State::Band;
                            dist[ni] = 1.0;
                            heap.push((Reverse(OrdF32(1.0)), ni));
                        }
                    }
                }
            }
        }
    }

    // FMM narrow-band neighbours helper: returns indices of known/band 4-neighbours
    let band_neighbours = |y: usize, x: usize, st: &[State], w_: usize, h_: usize| -> Vec<usize> {
        let mut result = Vec::with_capacity(4);
        for (ny, nx) in neighbours(y, x) {
            if ny < 0 || ny >= h_ as isize || nx < 0 || nx >= w_ as isize {
                continue;
            }
            let ni = ny as usize * w_ + nx as usize;
            if st[ni] != State::Unknown {
                result.push(ni);
            }
        }
        result
    };

    while let Some((Reverse(OrdF32(d)), idx)) = heap.pop() {
        if state[idx] == State::Known {
            continue;
        }
        if d > radius {
            break;
        }

        let y = idx / w;
        let x = idx % w;

        // Gather known/band neighbours for weighted interpolation
        let known_nbrs = band_neighbours(y, x, &state, w, h);

        let mut sum_weight = 0.0f64;
        let mut sum_vals: Vec<f64> = vec![0.0; c];

        // Pre-compute local gradient magnitude (averaged over known neighbours)
        // using finite differences of the known/band neighbour values.
        let mut grad_mag: f64 = 0.0;
        for &ni in &known_nbrs {
            let ny = ni / w;
            let nx = ni % w;
            let dy_dir = ny as f64 - y as f64;
            let dx_dir = nx as f64 - x as f64;
            let spatial_dist = (dy_dir * dy_dir + dx_dir * dx_dir).sqrt().max(1.0e-6);

            // Compute gradient of the image at this known neighbour using its own known neighbours
            let mut g_magnitude: f64 = 0.0;
            let nnbrs = band_neighbours(ny, nx, &state, w, h);
            for &nni in &nnbrs {
                let nny = nni / w;
                let nnx = nni % w;
                let ddy = nny as f64 - ny as f64;
                let ddx = nnx as f64 - nx as f64;
                let ndist = (ddy * ddy + ddx * ddx).sqrt().max(1.0e-6);
                for ch in 0..c {
                    let diff = inpaint_buf[ch][nni] as f64 - inpaint_buf[ch][ni] as f64;
                    g_magnitude += diff * diff;
                }
                let _ = ndist;
            }
            grad_mag = grad_mag.max(g_magnitude.sqrt());

            // Spatial weight (inverse distance squared)
            let w_spatial = 1.0 / (spatial_dist * spatial_dist);

            // Directional weight: prefer propagation along edges (perpendicular to gradient).
            // The Telea anisotropic term: if the propagation direction is along the gradient,
            // reduce weight (we want to propagate across edges, not along them).
            // beta is high when propagation is perpendicular to the gradient.
            let beta = if grad_mag > 1.0e-6 {
                // Compute gradient direction components at this known neighbour
                let mut gx: f64 = 0.0;
                let mut gy: f64 = 0.0;
                for &nni in &nnbrs {
                    let nny = nni / w;
                    let nnx = nni % w;
                    let ddy = nny as f64 - ny as f64;
                    let ddx = nnx as f64 - nx as f64;
                    for ch in 0..c {
                        let diff = inpaint_buf[ch][nni] as f64 - inpaint_buf[ch][ni] as f64;
                        gx += diff * ddx;
                        gy += diff * ddy;
                    }
                }
                let gnorm = (gx * gx + gy * gy).sqrt().max(1.0e-12);
                gx /= gnorm;
                gy /= gnorm;

                // Propagation direction (from known neighbour toward current pixel)
                let px = x as f64 - nx as f64;
                let py = y as f64 - ny as f64;
                let pnorm = (px * px + py * py).sqrt().max(1.0e-12);
                let pxn = px / pnorm;
                let pyn = py / pnorm;

                // Cross product magnitude = sin(angle between gradient and propagation)
                // High value means propagation is perpendicular to gradient (along edge) -> good
                (gx * pyn - gy * pxn).abs()
            } else {
                0.5 // no gradient info, neutral weight
            };

            let weight = w_spatial * (1.0 + beta);

            for ch in 0..c {
                sum_vals[ch] += inpaint_buf[ch][ni] as f64 * weight;
            }
            sum_weight += weight;
        }

        if sum_weight > 1.0e-12 {
            for ch in 0..c {
                inpaint_buf[ch][idx] = (sum_vals[ch] / sum_weight) as f32;
            }
        }

        state[idx] = State::Known;

        for (ny, nx) in neighbours(y, x) {
            if ny < 0 || ny >= h as isize || nx < 0 || nx >= w as isize {
                continue;
            }
            let ni = ny as usize * w + nx as usize;
            if state[ni] == State::Known {
                continue;
            }
            let new_dist = dist[idx] + 1.0;
            if new_dist < dist[ni] {
                dist[ni] = new_dist;
            }
            if state[ni] == State::Unknown {
                state[ni] = State::Band;
            }
            heap.push((Reverse(OrdF32(dist[ni])), ni));
        }
    }

    for idx in 0..pixels {
        if state[idx] != State::Known {
            for ch in 0..c {
                inpaint_buf[ch][idx] = img_vals[ch * pixels + idx];
            }
        }
    }

    let mut flat = vec![0.0f32; c * pixels];
    for ch in 0..c {
        flat[ch * pixels..(ch + 1) * pixels].copy_from_slice(&inpaint_buf[ch]);
    }

    let device = image.tensor.device();
    let new_data = TensorData::new(flat, [c, h, w]);
    let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
    Ok(Image::new(new_tensor))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{TestBackend, test_device};
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn test_inpaint_no_mask() {
        let device = test_device();
        let data = vec![0.25f32; 3 * 8 * 8];
        let img = Image::new(Tensor::<TestBackend, 3>::from_data(
            TensorData::new(data, [3, 8, 8]),
            &device,
        ));
        let mask = Image::new(Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![0.0f32; 8 * 8], [1, 8, 8]),
            &device,
        ));

        let result = inpaint(&img, &mask, 5.0).unwrap();
        assert_eq!(result.shape(), [3, 8, 8]);
        let vals: Vec<f32> = result.tensor.into_data().iter::<f32>().collect();
        for v in vals {
            assert!((v - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_inpaint_center_region() {
        let device = test_device();
        let h = 10usize;
        let w = 10usize;

        let mut img_vals = vec![0.0f32; h * w];
        for y in 0..h {
            for x in 0..w {
                img_vals[y * w + x] = if x < w / 2 { 0.0 } else { 1.0 };
            }
        }
        let img = Image::new(Tensor::<TestBackend, 3>::from_data(
            TensorData::new(img_vals.clone(), [1, h, w]),
            &device,
        ));

        let mut mask_vals = vec![0.0f32; h * w];
        mask_vals[4 * w + 4] = 1.0;
        mask_vals[4 * w + 5] = 1.0;
        mask_vals[5 * w + 4] = 1.0;
        mask_vals[5 * w + 5] = 1.0;
        let mask = Image::new(Tensor::<TestBackend, 3>::from_data(
            TensorData::new(mask_vals, [1, h, w]),
            &device,
        ));

        let result = inpaint(&img, &mask, 5.0).unwrap();
        assert_eq!(result.shape(), [1, h, w]);

        let out: Vec<f32> = result.tensor.into_data().iter::<f32>().collect();

        for y in 4..=5 {
            for x in 4..=5 {
                let v = out[y * w + x];
                assert!((0.0..=1.0).contains(&v), "pixel ({},{}) = {}", x, y, v);
            }
        }
        assert!((out[4 * w + 3] - img_vals[4 * w + 3]).abs() < 1e-6);
        assert!((out[5 * w + 6] - img_vals[5 * w + 6]).abs() < 1e-6);
    }

    #[test]
    fn test_inpaint_rgb_channel_independence() {
        let device = test_device();
        let h = 6usize;
        let w = 6usize;

        let mut img_vals = vec![0.0f32; 3 * h * w];
        for y in 0..h {
            for x in 0..w {
                img_vals[y * w + x] = 0.1;
                img_vals[h * w + y * w + x] = 0.5;
                img_vals[2 * h * w + y * w + x] = 0.9;
            }
        }
        let img = Image::new(Tensor::<TestBackend, 3>::from_data(
            TensorData::new(img_vals, [3, h, w]),
            &device,
        ));

        let mut mask_vals = vec![0.0f32; h * w];
        mask_vals[3 * w + 3] = 1.0;
        let mask = Image::new(Tensor::<TestBackend, 3>::from_data(
            TensorData::new(mask_vals, [1, h, w]),
            &device,
        ));

        let result = inpaint(&img, &mask, 10.0).unwrap();
        let out: Vec<f32> = result.tensor.into_data().iter::<f32>().collect();

        let r = out[3 * w + 3];
        let g = out[h * w + 3 * w + 3];
        let b = out[2 * h * w + 3 * w + 3];
        assert!((r - 0.1).abs() < 0.01, "R={}", r);
        assert!((g - 0.5).abs() < 0.01, "G={}", g);
        assert!((b - 0.9).abs() < 0.01, "B={}", b);
    }

    #[test]
    fn test_inpaint_dimension_mismatch() {
        let device = test_device();
        let img = Image::new(Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![0.5f32; 3 * 8 * 8], [3, 8, 8]),
            &device,
        ));
        let mask = Image::new(Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![0.0f32; 6 * 6], [1, 6, 6]),
            &device,
        ));
        let result = inpaint(&img, &mask, 5.0);
        assert!(result.is_err());
    }
}
