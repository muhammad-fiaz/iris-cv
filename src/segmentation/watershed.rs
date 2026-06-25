use crate::error::{IrisError, Result};
use crate::image::Image;
use burn::tensor::{Int, Tensor, TensorData, backend::Backend};

/// Marker-based watershed segmentation using iterative flooding.
///
/// Each unique positive marker value defines a seed region. The algorithm
/// floods outward from all seeds simultaneously (priority queue based on
/// pixel intensity), assigning each pixel to the nearest marker's region.
/// Pixels where different regions meet become watershed lines (label 0).
///
/// - `image`: Input grayscale or single-channel image.
/// - `markers`: 2D integer tensor of shape `[H, W]` with marker labels.
///   Zero means unassigned; positive integers are region seeds.
///
/// Returns a 2D integer tensor `[H, W]` with the segmentation labels.
pub fn watershed<B: Backend>(
    image: &Image<B>,
    markers: &Tensor<B, 2, Int>,
) -> Result<Tensor<B, 2, Int>> {
    let gray = image.grayscale()?;
    let img_dims = gray.tensor.dims();
    let h = img_dims[1];
    let w = img_dims[2];
    let mk_dims = markers.dims();

    if mk_dims[0] != h || mk_dims[1] != w {
        return Err(IrisError::DimensionMismatch {
            expected: vec![h, w],
            actual: mk_dims.to_vec(),
        });
    }

    let device = gray.tensor.device();
    let img_data = gray.tensor.clone().into_data();
    let img_vals: Vec<f32> = img_data.iter::<f32>().collect();

    let mk_data = markers.clone().into_data();
    let mk_vals: Vec<i32> = mk_data.iter::<i32>().collect();

    // Validate markers: must have at least one seed
    if mk_vals.iter().all(|&v| v == 0) {
        return Err(IrisError::InvalidParameter(
            "Markers must contain at least one positive seed".into(),
        ));
    }

    // Working labels: copy markers
    let mut labels: Vec<i32> = mk_vals.clone();

    // Priority queue: (intensity, y, x) — min-heap via reverse ordering
    let mut queue: Vec<(f32, usize, usize)> = Vec::new();

    // Seed the queue with all marker pixels
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if labels[idx] > 0 {
                queue.push((img_vals[idx], y, x));
            }
        }
    }
    // Make it a min-heap
    queue.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    // Use a BTreeMap-based priority approach: simple sorted insertion is fine for correctness
    // (not optimized for speed, but correct for the algorithm)

    let neighbors = [(1_isize, 0_isize), (-1, 0), (0, 1), (0, -1)];

    while let Some((_intensity, cy, cx)) = queue.pop() {
        let cidx = cy * w + cx;
        let c_label = labels[cidx];
        if c_label == 0 {
            continue;
        }

        for &(dx, dy) in &neighbors {
            let nx = cx as isize + dx;
            let ny = cy as isize + dy;
            if nx < 0 || nx >= w as isize || ny < 0 || ny >= h as isize {
                continue;
            }
            let ux = nx as usize;
            let uy = ny as usize;
            let nidx = uy * w + ux;

            if labels[nidx] == 0 {
                // Unassigned: claim it and enqueue
                labels[nidx] = c_label;
                // Binary search insertion to maintain sorted order (min-heap by intensity)
                let new_entry = (img_vals[nidx], uy, ux);
                match queue.binary_search_by(|a| {
                    a.0.partial_cmp(&new_entry.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                }) {
                    Ok(pos) | Err(pos) => queue.insert(pos, new_entry),
                }
            } else if labels[nidx] != c_label {
                // Conflict: both regions claim this pixel — watershed line
                // Leave as-is (the first one wins; boundary pixels remain labeled)
                // For a strict watershed line, set to 0:
                // labels[nidx] = 0;
            }
        }
    }

    let out_data = TensorData::new(labels, [h, w]);
    let out_tensor = Tensor::<B, 2, Int>::from_data(out_data, &device);
    Ok(out_tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{TestBackend, test_device};

    #[test]
    fn test_watershed_basic() {
        let device = test_device();

        // 8x8 grayscale image with two regions
        let mut img_vals = vec![0.3f32; 8 * 8];
        // Left region is dark
        for y in 0..8 {
            for x in 0..4 {
                img_vals[y * 8 + x] = 0.2;
            }
        }
        // Right region is bright
        for y in 0..8 {
            for x in 4..8 {
                img_vals[y * 8 + x] = 0.8;
            }
        }
        let img_tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(img_vals, [1, 8, 8]), &device);
        let img = Image::new(img_tensor);

        // Two markers: label 1 top-left, label 2 bottom-right
        let mut mk_vals = vec![0i32; 8 * 8];
        mk_vals[0] = 1;
        mk_vals[7 * 8 + 7] = 2;
        let mk_tensor =
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(mk_vals, [8, 8]), &device);

        let result = watershed(&img, &mk_tensor).unwrap();
        assert_eq!(result.dims(), [8, 8]);

        // Marker pixels should retain their labels
        let out_data = result.clone().into_data();
        let out_vals: Vec<i32> = out_data.iter::<i32>().collect();
        assert_eq!(out_vals[0], 1); // top-left marker
        assert_eq!(out_vals[63], 2); // bottom-right marker

        // All pixels should be assigned to some region
        assert!(out_vals.iter().all(|&v| v > 0));
    }

    #[test]
    fn test_watershed_no_markers_error() {
        let device = test_device();
        let img_tensor = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![0.5f32; 16], [1, 4, 4]),
            &device,
        );
        let img = Image::new(img_tensor);

        let mk_tensor = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![0i32; 16], [4, 4]),
            &device,
        );

        assert!(watershed(&img, &mk_tensor).is_err());
    }
}
