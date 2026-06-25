use crate::error::{IrisError, Result};
use crate::video::frame::Frame;
use crate::video::iterator::FrameIterator;
use crate::video::metadata::{ContainerFormat, VideoMetadata};
use burn::tensor::backend::Backend;
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::iterator::{load_animated_image, load_image_sequence};

/// Seek mode for random access within a video.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeekMode {
    /// Seek to the nearest keyframe (fast, may not be precise).
    ByKeyframe,
    /// Seek to the exact frame (slower, precise).
    Exact,
    /// Seek to the nearest frame at the given timestamp.
    ByTimestamp,
}

/// Configuration for opening a video source.
#[derive(Clone, Debug)]
pub struct VideoOpenOptions {
    /// Seek mode for random access.
    pub seek_mode: SeekMode,
    /// Maximum number of frames to load (0 = unlimited).
    pub max_frames: usize,
    /// Whether to preload all frames into memory.
    pub preload_all: bool,
    /// Target width to resize frames to (0 = keep original).
    pub target_width: usize,
    /// Target height to resize frames to (0 = keep original).
    pub target_height: usize,
    /// Frame rate for image sequences (default: 30.0).
    pub sequence_fps: f64,
    /// Pattern to match in image sequence filenames.
    pub sequence_pattern: String,
}

impl Default for VideoOpenOptions {
    fn default() -> Self {
        Self {
            seek_mode: SeekMode::Exact,
            max_frames: 0,
            preload_all: true,
            target_width: 0,
            target_height: 0,
            sequence_fps: 30.0,
            sequence_pattern: String::new(),
        }
    }
}

/// A high-level video reader that loads frames into memory.
///
/// Supports animated images (GIF, APNG), image sequences, and basic
/// container formats. All frames are cached for fast random access.
pub struct VideoReader<B: Backend> {
    frames: Vec<Frame<B>>,
    metadata: VideoMetadata,
    source_path: Option<PathBuf>,
}

impl<B: Backend> VideoReader<B> {
    /// Opens a video file and loads all frames into memory.
    pub fn open(path: impl AsRef<Path>, options: &VideoOpenOptions) -> Result<Self> {
        let path = path.as_ref();
        let path_str = path.to_string_lossy().to_string();
        let format = ContainerFormat::from_path(&path_str);

        if !path.exists() {
            return Err(IrisError::Video(format!(
                "Video file not found: {}",
                path.display()
            )));
        }

        let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

        let device = B::Device::default();
        let mut frames = match format {
            ContainerFormat::Gif | ContainerFormat::Png | ContainerFormat::WebP => {
                load_animated_image(path, &device)?
            }
            _ => {
                if path.is_dir() {
                    load_image_sequence(
                        path,
                        &options.sequence_pattern,
                        &device,
                        options.sequence_fps,
                    )?
                } else {
                    let img = image::open(path)
                        .map_err(|e| IrisError::Video(format!("Failed to open image: {e}")))?;

                    let rgb = img.to_rgb8();
                    let (w, h) = rgb.dimensions();
                    let raw: Vec<f32> = rgb
                        .pixels()
                        .flat_map(|p| {
                            let [r, g, b] = p.0;
                            [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]
                        })
                        .collect();

                    let tensor = burn::tensor::Tensor::<B, 3>::from_data(
                        burn::tensor::TensorData::new(raw, [3, h as usize, w as usize]),
                        &device,
                    );
                    let image = crate::image::Image::new(tensor);
                    vec![Frame::new(image, Duration::ZERO, 0)]
                }
            }
        };

        if options.max_frames > 0 && frames.len() > options.max_frames {
            frames.truncate(options.max_frames);
        }

        let metadata = if !frames.is_empty() {
            let w = frames[0].width();
            let h = frames[0].height();
            let fps = if frames.len() > 1 {
                let total_time: f64 = frames.iter().map(|f| f.duration.as_secs_f64()).sum();
                if total_time > 0.0 {
                    frames.len() as f64 / total_time
                } else {
                    30.0
                }
            } else {
                30.0
            };

            VideoMetadata {
                format,
                duration: frames.iter().map(|f| f.duration).sum(),
                fps,
                width: w,
                height: h,
                frame_count: frames.len(),
                video_codec: format!("{format:?}"),
                pixel_format: super::metadata::PixelFormat::Rgb8,
                rotation: 0,
                bit_rate: 0,
                streams: Vec::new(),
                has_audio: false,
                has_subtitles: false,
                file_size,
            }
        } else {
            return Err(IrisError::Video(
                "No frames found in video file".to_string(),
            ));
        };

        Ok(Self {
            frames,
            metadata,
            source_path: Some(path.to_path_buf()),
        })
    }

    /// Creates a reader from pre-loaded frames and metadata.
    ///
    /// Useful for in-memory video processing pipelines where frames
    /// are generated or loaded programmatically.
    #[must_use]
    pub fn from_frames(frames: Vec<Frame<B>>, metadata: VideoMetadata) -> Self {
        Self {
            frames,
            metadata,
            source_path: None,
        }
    }

    /// Returns a reference to all loaded frames.
    #[must_use]
    pub fn frames(&self) -> &[Frame<B>] {
        &self.frames
    }

    /// Returns the total number of loaded frames.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Returns video metadata.
    #[must_use]
    pub fn metadata(&self) -> &VideoMetadata {
        &self.metadata
    }

    /// Returns the source file path.
    #[must_use]
    pub fn source_path(&self) -> Option<&Path> {
        self.source_path.as_deref()
    }

    /// Returns a frame iterator for sequential access.
    #[must_use]
    pub fn iter(&self) -> FrameIterator<B> {
        FrameIterator::new(self.frames.clone())
    }

    /// Returns a looping frame iterator.
    #[must_use]
    pub fn loop_iter(&self) -> FrameIterator<B> {
        FrameIterator::new(self.frames.clone()).with_loop()
    }

    /// Gets a specific frame by index.
    pub fn get_frame(&self, index: usize) -> Result<&Frame<B>> {
        self.frames.get(index).ok_or_else(|| {
            IrisError::InvalidParameter(format!(
                "Frame index {index} out of range [0, {})",
                self.frames.len()
            ))
        })
    }

    /// Gets frames in a range.
    pub fn get_range(&self, start: usize, end: usize) -> Result<&[Frame<B>]> {
        if start >= self.frames.len() || end > self.frames.len() || start >= end {
            return Err(IrisError::InvalidParameter(format!(
                "Invalid range [{start}, {end}) for {} frames",
                self.frames.len()
            )));
        }
        Ok(&self.frames[start..end])
    }

    /// Seeks to a specific timestamp and returns the frame.
    pub fn seek_to_time(&self, time: Duration) -> Result<&Frame<B>> {
        let frame_index = (time.as_secs_f64() * self.metadata.fps).round() as usize;
        self.get_frame(frame_index)
    }

    /// Converts frames to a 4D tensor [B, C, H, W] (batch dimension).
    pub fn to_batch_tensor(&self) -> Result<burn::tensor::Tensor<B, 4>> {
        if self.frames.is_empty() {
            return Err(IrisError::Video("No frames to batch".to_string()));
        }

        let tensors: Vec<burn::tensor::Tensor<B, 4>> = self
            .frames
            .iter()
            .map(|f| f.image.tensor.clone().unsqueeze())
            .collect();

        Ok(burn::tensor::Tensor::cat(tensors, 0))
    }

    /// Computes per-frame differences (absolute difference between consecutive frames).
    pub fn frame_differences(&self) -> Result<Vec<burn::tensor::Tensor<B, 3>>> {
        if self.frames.len() < 2 {
            return Err(IrisError::Video("Need at least 2 frames".to_string()));
        }

        let diffs = self
            .frames
            .windows(2)
            .map(|w| {
                let diff = w[1].image.tensor.clone() - w[0].image.tensor.clone();
                diff.abs()
            })
            .collect();

        Ok(diffs)
    }

    /// Computes optical flow magnitude (sum of absolute differences) between consecutive frames.
    pub fn motion_magnitudes(&self) -> Result<Vec<f32>> {
        let diffs = self.frame_differences()?;
        let mut mags = Vec::with_capacity(diffs.len());
        for d in &diffs {
            let data: Vec<f32> =
                d.to_data().convert::<f32>().into_vec().map_err(|e| {
                    IrisError::Tensor(format!("Failed to convert tensor data: {e}"))
                })?;
            let sum: f32 = data.iter().sum();
            mags.push(sum);
        }
        Ok(mags)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{TestBackend, test_device};
    use burn::tensor::TensorData;

    fn make_test_frame(index: usize) -> Frame<TestBackend> {
        let device = test_device();
        let data = TensorData::new(vec![0.5f32; 3 * 32 * 32], [3, 32, 32]);
        let tensor = burn::tensor::Tensor::<TestBackend, 3>::from_data(data, &device);
        let img = crate::image::Image::new(tensor);
        Frame::new(img, Duration::from_millis(index as u64 * 33), index)
            .with_duration(Duration::from_millis(33))
    }

    #[test]
    fn test_video_reader_metadata() {
        let frames: Vec<_> = (0..30).map(make_test_frame).collect();
        let metadata = VideoMetadata::synthetic(32, 32, 30.0, 30);
        let reader = VideoReader {
            frames,
            metadata,
            source_path: None,
        };

        assert_eq!(reader.frame_count(), 30);
        assert_eq!(reader.metadata().width, 32);
        assert!((reader.metadata().fps - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_video_reader_get_frame() {
        let frames: Vec<_> = (0..5).map(make_test_frame).collect();
        let metadata = VideoMetadata::synthetic(32, 32, 30.0, 5);
        let reader = VideoReader {
            frames,
            metadata,
            source_path: None,
        };

        let frame = reader.get_frame(2).unwrap();
        assert_eq!(frame.index, 2);

        assert!(reader.get_frame(10).is_err());
    }

    #[test]
    fn test_video_reader_get_range() {
        let frames: Vec<_> = (0..10).map(make_test_frame).collect();
        let metadata = VideoMetadata::synthetic(32, 32, 30.0, 10);
        let reader = VideoReader {
            frames,
            metadata,
            source_path: None,
        };

        let range = reader.get_range(2, 5).unwrap();
        assert_eq!(range.len(), 3);
        assert_eq!(range[0].index, 2);
    }

    #[test]
    fn test_video_reader_seek_to_time() {
        let frames: Vec<_> = (0..60).map(make_test_frame).collect();
        let metadata = VideoMetadata::synthetic(32, 32, 30.0, 60);
        let reader = VideoReader {
            frames,
            metadata,
            source_path: None,
        };

        let frame = reader.seek_to_time(Duration::from_secs(1)).unwrap();
        assert_eq!(frame.index, 30);
    }

    #[test]
    fn test_video_reader_to_batch_tensor() {
        let frames: Vec<_> = (0..3).map(make_test_frame).collect();
        let metadata = VideoMetadata::synthetic(32, 32, 30.0, 3);
        let reader = VideoReader {
            frames,
            metadata,
            source_path: None,
        };

        let batch = reader.to_batch_tensor().unwrap();
        assert_eq!(batch.dims(), [3, 3, 32, 32]);
    }

    #[test]
    fn test_video_reader_frame_differences() {
        let frames: Vec<_> = (0..5).map(make_test_frame).collect();
        let metadata = VideoMetadata::synthetic(32, 32, 30.0, 5);
        let reader = VideoReader {
            frames,
            metadata,
            source_path: None,
        };

        let diffs = reader.frame_differences().unwrap();
        assert_eq!(diffs.len(), 4);
    }

    #[test]
    fn test_video_reader_motion_magnitudes() {
        let frames: Vec<_> = (0..5).map(make_test_frame).collect();
        let metadata = VideoMetadata::synthetic(32, 32, 30.0, 5);
        let reader = VideoReader {
            frames,
            metadata,
            source_path: None,
        };

        let mags = reader.motion_magnitudes().unwrap();
        assert_eq!(mags.len(), 4);
        assert!(mags.iter().all(|&m| m.abs() < 1e-6));
    }

    #[test]
    fn test_video_reader_iter() {
        let frames: Vec<_> = (0..3).map(make_test_frame).collect();
        let metadata = VideoMetadata::synthetic(32, 32, 30.0, 3);
        let reader = VideoReader {
            frames,
            metadata,
            source_path: None,
        };

        let mut iter = reader.iter();
        assert_eq!(iter.len(), 3);
        assert!(iter.next().is_some());
    }
}
