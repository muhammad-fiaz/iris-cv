pub mod geometric;
pub mod io;
pub mod ops;

pub use geometric::GeometricTransform;

use burn::tensor::{Tensor, backend::Backend};

/// Represents an image represented by a 3D Burn Tensor of shape [C, H, W] (Channels, Height, Width).
/// The pixel values are floats, normally scaled to [0.0, 1.0].
#[derive(Clone, Debug)]
pub struct Image<B: Backend> {
    /// The underlying 3D tensor of shape [C, H, W].
    pub tensor: Tensor<B, 3>,
}

impl<B: Backend> Image<B> {
    /// Creates a new Image from a 3D Burn Tensor.
    pub fn new(tensor: Tensor<B, 3>) -> Self {
        Self { tensor }
    }

    /// Loads an image from a file path using the default/given backend device.
    pub fn open(
        path: impl AsRef<std::path::Path>,
        device: &B::Device,
    ) -> crate::error::Result<Self> {
        io::load_image(path, device)
    }

    /// Saves the image to a file path.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> crate::error::Result<()> {
        io::save_image(self, path)
    }

    /// Returns the shape [C, H, W] of the image tensor.
    pub fn shape(&self) -> [usize; 3] {
        let dims = self.tensor.dims();
        [dims[0], dims[1], dims[2]]
    }

    /// Returns the number of channels (C) of the image.
    pub fn channels(&self) -> usize {
        self.tensor.dims()[0]
    }

    /// Returns the height (H) of the image.
    pub fn height(&self) -> usize {
        self.tensor.dims()[1]
    }

    /// Returns the width (W) of the image.
    pub fn width(&self) -> usize {
        self.tensor.dims()[2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    use burn::tensor::TensorData;

    type TestBackend = Wgpu;

    fn get_test_device() -> WgpuDevice {
        Default::default()
    }

    #[test]
    fn test_image_creation() {
        let device = get_test_device();
        let data = TensorData::new(vec![0.5f32; 3 * 10 * 10], [3, 10, 10]);
        let tensor = Tensor::<TestBackend, 3>::from_data(data, &device);
        let img = Image::new(tensor);

        assert_eq!(img.channels(), 3);
        assert_eq!(img.height(), 10);
        assert_eq!(img.width(), 10);
        assert_eq!(img.shape(), [3, 10, 10]);
    }

    #[test]
    fn test_image_ops() {
        let device = get_test_device();
        let data = TensorData::new(vec![0.5f32; 3 * 8 * 8], [3, 8, 8]);
        let tensor = Tensor::<TestBackend, 3>::from_data(data, &device);
        let img = Image::new(tensor);

        // Crop
        let cropped = img.crop(2, 2, 4, 4).unwrap();
        assert_eq!(cropped.shape(), [3, 4, 4]);

        // Flip
        let flipped = img.flip(true, false).unwrap();
        assert_eq!(flipped.shape(), [3, 8, 8]);

        // Rotate
        let rotated = img.rotate(90).unwrap();
        assert_eq!(rotated.shape(), [3, 8, 8]); // width and height same in this case

        // Grayscale
        let gray = img.grayscale().unwrap();
        assert_eq!(gray.channels(), 1);
        assert_eq!(gray.height(), 8);
        assert_eq!(gray.width(), 8);

        // To RGB
        let rgb = gray.to_rgb().unwrap();
        assert_eq!(rgb.channels(), 3);
    }
}
