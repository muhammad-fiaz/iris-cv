use crate::error::{IrisError, Result};
use crate::image::Image;
use burn::tensor::{Int, Tensor, TensorData, backend::Backend};

impl<B: Backend> Image<B> {
    /// Resizes the image to the specified width and height using nearest-neighbor interpolation.
    /// This runs fully in parallel on the GPU/CPU backend using Burn's tensor indexing.
    pub fn resize(&self, new_width: usize, new_height: usize) -> Result<Self> {
        let dims = self.tensor.dims();
        let _c = dims[0];
        let h = dims[1];
        let w = dims[2];

        if new_width == 0 || new_height == 0 {
            return Err(IrisError::InvalidParameter(
                "Dimensions must be greater than zero".into(),
            ));
        }

        let device = &self.tensor.device();

        // Calculate Y mapping indices
        let y_indices_vec: Vec<i32> = (0..new_height)
            .map(|y| ((y * h) / new_height) as i32)
            .collect();
        let y_indices =
            Tensor::<B, 1, Int>::from_data(TensorData::new(y_indices_vec, [new_height]), device);

        // Calculate X mapping indices
        let x_indices_vec: Vec<i32> = (0..new_width)
            .map(|x| ((x * w) / new_width) as i32)
            .collect();
        let x_indices =
            Tensor::<B, 1, Int>::from_data(TensorData::new(x_indices_vec, [new_width]), device);

        // Perform fast index selections on the tensor
        let resized = self
            .tensor
            .clone()
            .select(1, y_indices)
            .select(2, x_indices);

        Ok(Image::new(resized))
    }

    /// Crops a rectangular region from the image.
    pub fn crop(&self, x: usize, y: usize, width: usize, height: usize) -> Result<Self> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        if x + width > w || y + height > h {
            return Err(IrisError::DimensionMismatch {
                expected: vec![c, height, width],
                actual: vec![c, h, w],
            });
        }

        // Slice the tensor using ranges
        let cropped = self
            .tensor
            .clone()
            .slice([0..c, y..(y + height), x..(x + width)]);
        Ok(Image::new(cropped))
    }

    /// Flips the image.
    /// - horizontal: Flip along the width dimension.
    /// - vertical: Flip along the height dimension.
    pub fn flip(&self, horizontal: bool, vertical: bool) -> Result<Self> {
        let mut flipped = self.tensor.clone();
        if vertical {
            // Dimension 1 is height
            flipped = flipped.flip([1]);
        }
        if horizontal {
            // Dimension 2 is width
            flipped = flipped.flip([2]);
        }
        Ok(Image::new(flipped))
    }

    /// Rotates the image by 90, 180, or 270 degrees clockwise.
    pub fn rotate(&self, angle_degrees: u32) -> Result<Self> {
        match angle_degrees {
            0 | 360 => Ok(self.clone()),
            90 => {
                // Swap height & width, then flip horizontally
                let transposed = self.tensor.clone().swap_dims(1, 2);
                let rotated = transposed.flip([2]);
                Ok(Image::new(rotated))
            }
            180 => {
                // Flip both vertically and horizontally
                let rotated = self.tensor.clone().flip([1, 2]);
                Ok(Image::new(rotated))
            }
            270 => {
                // Swap height & width, then flip vertically
                let transposed = self.tensor.clone().swap_dims(1, 2);
                let rotated = transposed.flip([1]);
                Ok(Image::new(rotated))
            }
            _ => Err(IrisError::InvalidParameter(
                "Only 90, 180, 270 degrees rotations are supported".into(),
            )),
        }
    }

    /// Converts the image to grayscale using standard ITU-R BT.601 luma weights:
    /// Y = 0.299*R + 0.587*G + 0.114*B
    pub fn grayscale(&self) -> Result<Self> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        if c == 1 {
            return Ok(self.clone());
        }

        if c < 3 {
            return Err(IrisError::Tensor(
                "Cannot convert image with less than 3 channels to grayscale".into(),
            ));
        }

        // Slice R, G, B channels
        let r = self.tensor.clone().slice([0..1, 0..h, 0..w]);
        let g = self.tensor.clone().slice([1..2, 0..h, 0..w]);
        let b = self.tensor.clone().slice([2..3, 0..h, 0..w]);

        let gray = r
            .mul_scalar(0.299)
            .add(g.mul_scalar(0.587))
            .add(b.mul_scalar(0.114));

        Ok(Image::new(gray))
    }

    /// Converts a single-channel grayscale image to a 3-channel RGB image.
    pub fn to_rgb(&self) -> Result<Self> {
        let dims = self.tensor.dims();
        let c = dims[0];
        if c == 3 {
            return Ok(self.clone());
        }
        if c != 1 {
            return Err(IrisError::Tensor(
                "Input image must be single-channel to convert to RGB".into(),
            ));
        }

        // Concatenate along the channel axis (dimension 0)
        let rgb = Tensor::cat(
            vec![
                self.tensor.clone(),
                self.tensor.clone(),
                self.tensor.clone(),
            ],
            0,
        );
        Ok(Image::new(rgb))
    }

    /// Builds a Gaussian pyramid with the specified number of levels.
    /// Each level is half the size of the previous one, smoothed with Gaussian blur.
    pub fn gaussian_pyramid(&self, levels: usize) -> Result<Vec<Self>> {
        let mut pyramid = Vec::with_capacity(levels);
        pyramid.push(self.clone());

        let mut current = self.clone();
        for _ in 1..levels {
            let dims = current.tensor.dims();
            let h = dims[1];
            let w = dims[2];
            let new_h = h / 2;
            let new_w = w / 2;
            if new_h == 0 || new_w == 0 {
                break;
            }
            // Gaussian blur before downsampling to prevent aliasing
            let blurred = current.gaussian_blur(3, 1.0)?;
            let downsampled = blurred.resize(new_w, new_h)?;
            pyramid.push(downsampled);
            current = pyramid.last().cloned().unwrap();
        }

        Ok(pyramid)
    }

    /// Computes the integral image (summed area table) of the grayscale channel.
    /// Returns a tensor of shape [1, H+1, W+1] where integral[y][x] is the sum
    /// of all pixels in the rectangle (0,0) to (x-1, y-1).
    pub fn integral_image(&self) -> Result<Image<B>> {
        let gray = self.grayscale()?;
        let dims = gray.tensor.dims();
        let h = dims[1];
        let w = dims[2];

        let tensor_data = gray.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

        let mut integral = vec![0.0f32; (h + 1) * (w + 1)];

        for y in 0..h {
            let mut row_sum = 0.0f32;
            for x in 0..w {
                row_sum += flat_vals[y * w + x];
                integral[(y + 1) * (w + 1) + (x + 1)] =
                    integral[y * (w + 1) + (x + 1)] + row_sum;
            }
        }

        let device = gray.tensor.device();
        let data = TensorData::new(integral, [1, h + 1, w + 1]);
        let tensor = Tensor::<B, 3>::from_data(data, &device);
        Ok(Image::new(tensor))
    }

    /// Performs a flood fill operation starting from seed point (x, y).
    /// Fills connected pixels with `fill_value` if they are within `lo_diff` and `hi_diff`
    /// of the seed pixel value.
    pub fn flood_fill(
        &self,
        seed_x: usize,
        seed_y: usize,
        fill_value: f32,
        lo_diff: f32,
        hi_diff: f32,
    ) -> Result<Self> {
        let gray = self.grayscale()?;
        let dims = gray.tensor.dims();
        let h = dims[1];
        let w = dims[2];

        if seed_x >= w || seed_y >= h {
            return Err(IrisError::InvalidParameter(
                "Seed point is outside image bounds".into(),
            ));
        }

        let tensor_data = gray.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = flat_vals.clone();

        let seed_val = flat_vals[seed_y * w + seed_x];
        let lo = seed_val - lo_diff;
        let hi = seed_val + hi_diff;

        // BFS-based flood fill
        let mut visited = vec![false; h * w];
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((seed_x, seed_y));
        visited[seed_y * w + seed_x] = true;

        let dx = [1, 0, -1, 0];
        let dy = [0, 1, 0, -1];

        while let Some((cx, cy)) = queue.pop_front() {
            out_vals[cy * w + cx] = fill_value;

            for d in 0..4 {
                let nx = cx as isize + dx[d];
                let ny = cy as isize + dy[d];
                if nx >= 0 && nx < w as isize && ny >= 0 && ny < h as isize {
                    let ux = nx as usize;
                    let uy = ny as usize;
                    let idx = uy * w + ux;
                    if !visited[idx] {
                        let pixel = flat_vals[idx];
                        if pixel >= lo && pixel <= hi {
                            visited[idx] = true;
                            queue.push_back((ux, uy));
                        }
                    }
                }
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
    use burn::tensor::TensorData;

    #[test]
    fn test_image_conversions() {
        let device = test_device();
        let flat_data = vec![
            0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
        ];
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [3, 2, 2]), &device);
        let img = Image::new(tensor);

        let gray = img.grayscale().unwrap();
        assert_eq!(gray.shape(), [1, 2, 2]);

        let rgb = gray.to_rgb().unwrap();
        assert_eq!(rgb.shape(), [3, 2, 2]);
    }
}
