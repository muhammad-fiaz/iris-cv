pub mod frame;
pub mod iterator;
pub mod metadata;
pub mod reader;
pub mod writer;

pub use frame::Frame;
pub use iterator::{FrameIterator, FrameExt, load_animated_image, load_image_sequence};
pub use metadata::{VideoMetadata, ContainerFormat, PixelFormat, StreamInfo, StreamType};
pub use reader::{VideoReader, VideoOpenOptions, SeekMode};
pub use writer::{VideoWriter, VideoWriteOptions, OutputFormat};

use crate::error::Result;
use crate::image::Image;
use burn::tensor::backend::Backend;
use std::path::Path;

/// Legacy video capture with mock frame generation.
///
/// For real video file reading, use [`VideoReader`] instead.
pub struct VideoCapture<B: Backend> {
    pub source_path: String,
    #[allow(dead_code)]
    device: B::Device,
    current_frame: usize,
    total_frames: usize,
}

impl<B: Backend> VideoCapture<B> {
    /// Opens a video file or stream for reading.
    pub fn open(path: impl AsRef<Path>, device: &B::Device) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().into_owned();
        Ok(Self {
            source_path: path_str,
            device: device.clone(),
            current_frame: 0,
            total_frames: 100,
        })
    }

    /// Grabs and retrieves the next video frame.
    /// Returns None when the video ends.
    pub fn read(&mut self) -> Result<Option<Image<B>>> {
        if self.current_frame >= self.total_frames {
            return Ok(None);
        }
        self.current_frame += 1;

        let w = 640;
        let h = 480;
        let mut flat_data = vec![0.0f32; 3 * h * w];

        let frame_offset = (self.current_frame as f32) / (self.total_frames as f32);

        for y in 0..h {
            for x in 0..w {
                flat_data[y * w + x] = (x as f32) / (w as f32);
                flat_data[h * w + y * w + x] = (y as f32) / (h as f32);
                flat_data[2 * h * w + y * w + x] = frame_offset;
            }
        }

        let tensor_data = burn::tensor::TensorData::new(flat_data, [3, h, w]);
        let tensor = burn::tensor::Tensor::<B, 3>::from_data(tensor_data, &self.device);
        Ok(Some(Image::new(tensor)))
    }
}

/// Legacy video writer with mock frame writing.
///
/// For real video file writing, use [`writer::VideoWriter`] instead.
pub struct LegacyVideoWriter<B: Backend> {
    pub dest_path: String,
    #[allow(dead_code)]
    width: usize,
    #[allow(dead_code)]
    height: usize,
    #[allow(dead_code)]
    fps: f64,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> LegacyVideoWriter<B> {
    /// Creates a video writer target.
    pub fn create(path: impl AsRef<Path>, width: usize, height: usize, fps: f64) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().into_owned();
        Ok(Self {
            dest_path: path_str,
            width,
            height,
            fps,
            _marker: std::marker::PhantomData,
        })
    }

    /// Writes a single frame to the video destination.
    pub fn write(&mut self, _frame: &Image<B>) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{TestBackend, test_device};

    #[test]
    fn test_video_capture_legacy() {
        let device = test_device();
        let mut capture = VideoCapture::<TestBackend>::open("mock_video.mp4", &device).unwrap();
        assert_eq!(capture.source_path, "mock_video.mp4");

        let frame = capture.read().unwrap();
        assert!(frame.is_some());
        let frame_img = frame.unwrap();
        assert_eq!(frame_img.shape(), [3, 480, 640]);
    }

    #[test]
    fn test_legacy_video_writer() {
        let mut writer = LegacyVideoWriter::<TestBackend>::create("output.mp4", 640, 480, 30.0).unwrap();
        assert_eq!(writer.dest_path, "output.mp4");

        let device = test_device();
        let data = burn::tensor::TensorData::new(vec![0.5f32; 3 * 480 * 640], [3, 480, 640]);
        let tensor = burn::tensor::Tensor::<TestBackend, 3>::from_data(data, &device);
        let img = Image::new(tensor);
        writer.write(&img).unwrap();
    }

    #[test]
    fn test_frame_struct() {
        let device = test_device();
        let data = burn::tensor::TensorData::new(vec![0.5f32; 3 * 64 * 64], [3, 64, 64]);
        let tensor = burn::tensor::Tensor::<TestBackend, 3>::from_data(data, &device);
        let img = Image::new(tensor);

        let frame = Frame::new(img, std::time::Duration::from_millis(33), 0);
        assert_eq!(frame.width(), 64);
        assert_eq!(frame.height(), 64);
    }

    #[test]
    fn test_video_metadata_synthetic() {
        let meta = VideoMetadata::synthetic(1920, 1080, 30.0, 300);
        assert_eq!(meta.width, 1920);
        assert_eq!(meta.height, 1080);
        assert!((meta.fps - 30.0).abs() < 1e-6);
        assert_eq!(meta.frame_count, 300);
    }

    #[test]
    fn test_frame_iterator() {
        let device = test_device();
        let frames: Vec<Frame<TestBackend>> = (0..5)
            .map(|i| {
                let data = burn::tensor::TensorData::new(vec![0.5f32; 3 * 32 * 32], [3, 32, 32]);
                let tensor = burn::tensor::Tensor::<TestBackend, 3>::from_data(data, &device);
                let img = Image::new(tensor);
                Frame::new(img, std::time::Duration::from_millis(i as u64 * 33), i)
            })
            .collect();

        let mut iter = FrameIterator::new(frames);
        assert_eq!(iter.total_frames(), 5);
        assert!(iter.next().is_some());
    }
}
