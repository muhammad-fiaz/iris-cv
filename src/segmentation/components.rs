use crate::core::types::Rect;
use crate::error::Result;
use crate::image::Image;
use burn::tensor::{Int, Tensor, TensorData, backend::Backend};

/// Component statistics output.
#[derive(Clone, Debug, PartialEq)]
pub struct ComponentStats {
    pub label: usize,
    pub bbox: Rect<usize>,
    pub area: usize,
    pub centroid: (f64, f64),
}

impl<B: Backend> Image<B> {
    /// Identifies connected components in a binary image and calculates statistics for each component.
    /// Returns labeled image tensor of shape [H, W] and stats list.
    pub fn connected_components_with_stats(
        &self,
    ) -> Result<(Tensor<B, 2, Int>, Vec<ComponentStats>)> {
        let gray = self.grayscale()?;
        let dims = gray.tensor.dims();
        let h = dims[1];
        let w = dims[2];

        let device = gray.tensor.device();
        let tensor_data = gray.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

        let mut labels = vec![0usize; h * w];
        let mut stats = Vec::new();
        let mut current_label = 0;

        // Depth-First Search Labeling
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if flat_vals[idx] > 0.5 && labels[idx] == 0 {
                    current_label += 1;

                    // Connected component statistics tracker
                    let mut area = 0;
                    let mut min_x = x;
                    let mut max_x = x;
                    let mut min_y = y;
                    let mut max_y = y;
                    let mut sum_x = 0;
                    let mut sum_y = 0;

                    // BFS queue
                    let mut queue = std::collections::VecDeque::new();
                    queue.push_back((x, y));
                    labels[idx] = current_label;

                    while let Some((cx, cy)) = queue.pop_front() {
                        area += 1;
                        min_x = min_x.min(cx);
                        max_x = max_x.max(cx);
                        min_y = min_y.min(cy);
                        max_y = max_y.max(cy);
                        sum_x += cx;
                        sum_y += cy;

                        // 4-connectivity neighbors
                        let neighbors = [
                            (cx as isize + 1, cy as isize),
                            (cx as isize - 1, cy as isize),
                            (cx as isize, cy as isize + 1),
                            (cx as isize, cy as isize - 1),
                        ];

                        for &(nx, ny) in &neighbors {
                            if nx >= 0 && nx < w as isize && ny >= 0 && ny < h as isize {
                                let nidx = (ny as usize) * w + (nx as usize);
                                if flat_vals[nidx] > 0.5 && labels[nidx] == 0 {
                                    labels[nidx] = current_label;
                                    queue.push_back((nx as usize, ny as usize));
                                }
                            }
                        }
                    }

                    stats.push(ComponentStats {
                        label: current_label,
                        bbox: Rect::new(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1),
                        area,
                        centroid: (sum_x as f64 / area as f64, sum_y as f64 / area as f64),
                    });
                }
            }
        }

        // Reconstruct labeled tensor as integer tensor
        let labels_i32: Vec<i32> = labels.iter().map(|&l| l as i32).collect();
        let labels_data = TensorData::new(labels_i32, [h, w]);
        let labels_tensor = Tensor::<B, 2, Int>::from_data(labels_data, &device);

        Ok((labels_tensor, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::Wgpu;

    #[test]
    fn test_connected_components() {
        let device = Default::default();
        // 5x5 image with two separated components
        let mut flat_data = vec![0.0f32; 1 * 5 * 5];
        flat_data[0] = 1.0; flat_data[1] = 1.0;
        flat_data[23] = 1.0; flat_data[24] = 1.0;

        let tensor = Tensor::<Wgpu, 3>::from_data(TensorData::new(flat_data, [1, 5, 5]), &device);
        let img = Image::new(tensor);

        let (labels, stats) = img.connected_components_with_stats().unwrap();
        assert_eq!(labels.dims(), [5, 5]);
        assert_eq!(stats.len(), 2);
        assert_eq!(stats[0].area, 2);
        assert_eq!(stats[1].area, 2);
    }
}

