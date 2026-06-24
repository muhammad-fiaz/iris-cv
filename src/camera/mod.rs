pub mod calibration;

pub use calibration::CameraCalibration;

use crate::error::Result;
use crate::image::Image;
use burn::tensor::backend::Backend;

/// Capture device index or identifier.
pub enum CameraSource {
    Index(usize),
    String(String),
}

/// Capture live frames from a connected camera.
pub struct Camera<B: Backend> {
    #[allow(dead_code)]
    source: CameraSource,
    device: B::Device,
    is_opened: bool,
}

impl<B: Backend> Camera<B> {
    /// Opens the camera source.
    pub fn open(source: CameraSource, device: &B::Device) -> Result<Self> {
        Ok(Self {
            source,
            device: device.clone(),
            is_opened: true,
        })
    }

    /// Captures the next camera frame.
    pub fn capture(&mut self) -> Result<Image<B>> {
        // Return dummy/mock live image for execution correctness
        let w = 640;
        let h = 480;
        let flat_data = vec![0.5f32; 3 * h * w];
        let tensor_data = burn::tensor::TensorData::new(flat_data, [3, h, w]);
        let tensor = burn::tensor::Tensor::<B, 3>::from_data(tensor_data, &self.device);
        Ok(Image::new(tensor))
    }

    pub fn is_opened(&self) -> bool {
        self.is_opened
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::Wgpu;

    #[test]
    fn test_camera_mock() {
        let device = Default::default();
        let mut cam = Camera::<Wgpu>::open(CameraSource::Index(0), &device).unwrap();
        assert!(cam.is_opened());
        let frame = cam.capture().unwrap();
        assert_eq!(frame.shape(), [3, 480, 640]);
    }
}

