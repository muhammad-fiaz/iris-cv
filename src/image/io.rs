use crate::error::Result;
use crate::image::Image;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use std::path::Path;

/// Loads an image from the specified file path.
/// The image is automatically converted to RGB/Grayscale and represented as a [C, H, W] float tensor.
pub fn load_image<B: Backend>(path: impl AsRef<Path>, device: &B::Device) -> Result<Image<B>> {
    let img = image::open(path)?;
    let img = img.to_rgb8();
    let (width, height) = img.dimensions();
    let w = width as usize;
    let h = height as usize;
    let c = 3usize;

    let mut flat_data = vec![0.0f32; c * h * w];
    let pixels = img.as_flat_samples();
    let slice = pixels.as_slice();

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            // Map HWC (file layout) to CHW (standard deep learning tensor layout)
            flat_data[y * w + x] = f32::from(slice[idx]) / 255.0; // R
            flat_data[h * w + y * w + x] = f32::from(slice[idx + 1]) / 255.0; // G
            flat_data[2 * h * w + y * w + x] = f32::from(slice[idx + 2]) / 255.0; // B
        }
    }

    let tensor_data = TensorData::new(flat_data, [c, h, w]);
    let tensor = Tensor::<B, 3>::from_data(tensor_data, device);
    Ok(Image::new(tensor))
}

/// Saves the image tensor to the specified file path.
/// Assumes image tensor layout is [C, H, W] with float values in [0.0, 1.0].
pub fn save_image<B: Backend>(image: &Image<B>, path: impl AsRef<Path>) -> Result<()> {
    let dims = image.tensor.dims();
    let c = dims[0];
    let h = dims[1];
    let w = dims[2];

    // Read the tensor data back to the host CPU
    // Under Burn 0.21.0, into_data() fetches the tensor data synchronously on the CPU.
    let tensor_data = image.tensor.clone().into_data();
    let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

    let mut img_buf = image::ImageBuffer::new(w as u32, h as u32);

    for y in 0..h {
        for x in 0..w {
            let r_val = flat_vals[y * w + x];
            let g_val = if c > 1 {
                flat_vals[h * w + y * w + x]
            } else {
                r_val
            };
            let b_val = if c > 2 {
                flat_vals[2 * h * w + y * w + x]
            } else {
                r_val
            };

            let r = (r_val.clamp(0.0, 1.0) * 255.0) as u8;
            let g = (g_val.clamp(0.0, 1.0) * 255.0) as u8;
            let b = (b_val.clamp(0.0, 1.0) * 255.0) as u8;

            img_buf.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }

    img_buf.save(path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::Wgpu;

    #[test]
    fn test_image_io() {
        let device = Default::default();
        let flat_data = vec![0.5f32; 3 * 8 * 8];
        let tensor = Tensor::<Wgpu, 3>::from_data(TensorData::new(flat_data, [3, 8, 8]), &device);
        let img = Image::new(tensor);

        let temp_path = "temp_test_io.png";
        save_image(&img, temp_path).unwrap();

        let loaded = load_image::<Wgpu>(temp_path, &device).unwrap();
        assert_eq!(loaded.shape(), [3, 8, 8]);

        let _ = std::fs::remove_file(temp_path);
    }
}
