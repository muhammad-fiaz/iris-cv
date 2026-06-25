use crate::error::{IrisError, Result};
use crate::video::frame::Frame;
use burn::tensor::backend::Backend;
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::metadata::ContainerFormat;

/// Iterator over frames from a video file.
pub struct FrameIterator<B: Backend> {
    frames: Vec<Frame<B>>,
    current: usize,
    loop_playback: bool,
}

impl<B: Backend> FrameIterator<B> {
    /// Creates a new iterator from a pre-loaded set of frames.
    #[must_use]
    pub fn new(frames: Vec<Frame<B>>) -> Self {
        Self {
            frames,
            current: 0,
            loop_playback: false,
        }
    }

    /// Enables loop playback.
    #[must_use]
    pub fn with_loop(mut self) -> Self {
        self.loop_playback = true;
        self
    }

    /// Returns the total number of frames.
    #[must_use]
    pub fn total_frames(&self) -> usize {
        self.frames.len()
    }

    /// Returns the current frame index.
    #[must_use]
    pub fn current_index(&self) -> usize {
        self.current
    }

    /// Seeks to a specific frame index.
    pub fn seek(&mut self, index: usize) -> Result<()> {
        if index >= self.frames.len() {
            return Err(IrisError::InvalidParameter(format!(
                "Frame index {} out of range [0, {})",
                index,
                self.frames.len()
            )));
        }
        self.current = index;
        Ok(())
    }

    /// Seeks to the frame at the given timestamp.
    pub fn seek_to_time(&mut self, time: Duration, fps: f64) -> Result<()> {
        let frame_index = (time.as_secs_f64() * fps).round() as usize;
        self.seek(frame_index)
    }

    /// Returns the remaining frames count.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.frames.len().saturating_sub(self.current)
    }

    /// Returns the total duration of all frames.
    #[must_use]
    pub fn total_duration(&self) -> Duration {
        self.frames
            .iter()
            .map(|f| {
                if f.duration.is_zero() {
                    Duration::from_secs_f64(1.0 / 30.0)
                } else {
                    f.duration
                }
            })
            .sum()
    }
}

impl<B: Backend> Iterator for FrameIterator<B> {
    type Item = Frame<B>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.frames.len() {
            if self.loop_playback && !self.frames.is_empty() {
                self.current = 0;
            } else {
                return None;
            }
        }
        let frame = self.frames[self.current].clone();
        self.current += 1;
        Some(frame)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining();
        (remaining, Some(remaining))
    }
}

impl<B: Backend> ExactSizeIterator for FrameIterator<B> {}

/// Loads all frames from an animated GIF file.
pub fn load_animated_image<B: Backend>(path: &Path, device: &B::Device) -> Result<Vec<Frame<B>>> {
    use image::AnimationDecoder;
    use image::codecs::gif::GifDecoder;

    let path_str = path.to_string_lossy().to_string();
    let format = ContainerFormat::from_path(&path_str);

    let file = std::fs::File::open(path)
        .map_err(|e| IrisError::Video(format!("Failed to open video file: {e}")))?;

    let reader = std::io::BufReader::new(file);

    match format {
        ContainerFormat::Gif => {
            let decoder = GifDecoder::new(reader)
                .map_err(|e| IrisError::Video(format!("Failed to decode GIF: {e}")))?;

            let frames_iter = decoder.into_frames();
            let mut frames = Vec::new();

            for (i, result) in frames_iter.enumerate() {
                let frame = result
                    .map_err(|e| IrisError::Video(format!("Failed to read GIF frame {i}: {e}")))?;

                let delay = frame.delay();
                let (numer, denom) = delay.numer_denom_ms();
                let duration_ms = if denom > 0 {
                    numer as u64 * 1000 / denom as u64
                } else {
                    33
                };
                let duration = Duration::from_millis(duration_ms);

                let img = frame.buffer();
                let (w, h) = img.dimensions();
                let raw: Vec<f32> = img
                    .pixels()
                    .flat_map(|p| {
                        let [r, g, b, _a] = p.0;
                        [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]
                    })
                    .collect();

                let tensor = burn::tensor::Tensor::<B, 3>::from_data(
                    burn::tensor::TensorData::new(raw, [3, h as usize, w as usize]),
                    device,
                );
                let image = crate::image::Image::new(tensor);

                let pts = Duration::from_secs_f64(i as f64 * duration.as_secs_f64());
                let frame = Frame::new(image, pts, i)
                    .with_duration(duration)
                    .with_keyframe(i == 0 || i % 30 == 0);

                frames.push(frame);
            }

            Ok(frames)
        }
        ContainerFormat::Png => {
            use image::codecs::png::PngDecoder;

            let file2 = std::fs::File::open(path)
                .map_err(|e| IrisError::Video(format!("Failed to open PNG file: {e}")))?;
            let reader2 = std::io::BufReader::new(file2);

            let decoder = PngDecoder::new(reader2)
                .map_err(|e| IrisError::Video(format!("Failed to decode PNG: {e}")))?;

            let apng = decoder
                .apng()
                .map_err(|e| IrisError::Video(format!("Failed to parse APNG: {e}")))?;

            let frames_iter = apng.into_frames();
            let mut frames = Vec::new();

            for (i, result) in frames_iter.enumerate() {
                let frame = result
                    .map_err(|e| IrisError::Video(format!("Failed to read APNG frame {i}: {e}")))?;

                let delay = frame.delay();
                let (numer, denom) = delay.numer_denom_ms();
                let duration_ms = if denom > 0 {
                    numer as u64 * 1000 / denom as u64
                } else {
                    33
                };
                let duration = Duration::from_millis(duration_ms);

                let img = frame.buffer();
                let (w, h) = img.dimensions();
                let raw: Vec<f32> = img
                    .pixels()
                    .flat_map(|p| {
                        let [r, g, b, _a] = p.0;
                        [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]
                    })
                    .collect();

                let tensor = burn::tensor::Tensor::<B, 3>::from_data(
                    burn::tensor::TensorData::new(raw, [3, h as usize, w as usize]),
                    device,
                );
                let image = crate::image::Image::new(tensor);

                let pts = Duration::from_secs_f64(i as f64 * duration.as_secs_f64());
                let frame = Frame::new(image, pts, i)
                    .with_duration(duration)
                    .with_keyframe(i == 0 || i % 30 == 0);

                frames.push(frame);
            }

            Ok(frames)
        }
        _ => {
            // For unsupported animated formats, try as single image
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
                device,
            );
            let image = crate::image::Image::new(tensor);
            Ok(vec![Frame::new(image, Duration::ZERO, 0)])
        }
    }
}

/// Loads numbered image frames from a directory.
pub fn load_image_sequence<B: Backend>(
    dir: &Path,
    pattern: &str,
    device: &B::Device,
    fps: f64,
) -> Result<Vec<Frame<B>>> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| IrisError::Video(format!("Failed to read directory: {e}")))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.contains(pattern))
                .unwrap_or(false)
        })
        .filter(|p| {
            p.extension().is_some_and(|ext| {
                matches!(
                    ext.to_str(),
                    Some("png" | "jpg" | "jpeg" | "bmp" | "tiff" | "webp")
                )
            })
        })
        .collect();

    entries.sort();

    let mut frames = Vec::with_capacity(entries.len());
    for (i, entry) in entries.iter().enumerate() {
        let img = image::open(entry)
            .map_err(|e| IrisError::Video(format!("Failed to open frame {}: {e}", i + 1)))?;

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
            device,
        );
        let image = crate::image::Image::new(tensor);
        let pts = Duration::from_secs_f64(i as f64 / fps);
        let frame = Frame::new(image, pts, i)
            .with_duration(Duration::from_secs_f64(1.0 / fps))
            .with_keyframe(i == 0 || i % 30 == 0);

        frames.push(frame);
    }

    Ok(frames)
}

/// Trait for extending `Frame` with convenience methods.
pub trait FrameExt {
    /// Sets the keyframe flag.
    fn with_keyframe(self, is_keyframe: bool) -> Self;
}

impl<B: Backend> FrameExt for Frame<B> {
    fn with_keyframe(mut self, is_keyframe: bool) -> Self {
        self.is_keyframe = is_keyframe;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{TestBackend, test_device};
    use burn::tensor::TensorData;

    fn make_frame(index: usize) -> Frame<TestBackend> {
        let device = test_device();
        let data = TensorData::new(vec![0.5f32; 3 * 32 * 32], [3, 32, 32]);
        let tensor = burn::tensor::Tensor::<TestBackend, 3>::from_data(data, &device);
        let img = crate::image::Image::new(tensor);
        Frame::new(img, Duration::from_millis(index as u64 * 33), index)
            .with_duration(Duration::from_millis(33))
    }

    #[test]
    fn test_frame_iterator_sequential() {
        let frames: Vec<_> = (0..5).map(make_frame).collect();
        let mut iter = FrameIterator::new(frames);
        assert_eq!(iter.total_frames(), 5);
        assert_eq!(iter.remaining(), 5);

        let first = iter.next().unwrap();
        assert_eq!(first.index, 0);
        assert_eq!(iter.current_index(), 1);
        assert_eq!(iter.remaining(), 4);
    }

    #[test]
    fn test_frame_iterator_exact_size() {
        let frames: Vec<_> = (0..10).map(make_frame).collect();
        let iter = FrameIterator::new(frames);
        assert_eq!(iter.len(), 10);
    }

    #[test]
    fn test_frame_iterator_seek() {
        let frames: Vec<_> = (0..5).map(make_frame).collect();
        let mut iter = FrameIterator::new(frames);
        iter.seek(3).unwrap();
        assert_eq!(iter.current_index(), 3);
        let frame = iter.next().unwrap();
        assert_eq!(frame.index, 3);
    }

    #[test]
    fn test_frame_iterator_seek_out_of_bounds() {
        let frames: Vec<_> = (0..5).map(make_frame).collect();
        let mut iter = FrameIterator::new(frames);
        assert!(iter.seek(10).is_err());
    }

    #[test]
    fn test_frame_iterator_loop() {
        let frames: Vec<_> = (0..3).map(make_frame).collect();
        let mut iter = FrameIterator::new(frames).with_loop();
        for _ in 0..7 {
            assert!(iter.next().is_some());
        }
    }

    #[test]
    fn test_frame_iterator_total_duration() {
        let frames: Vec<_> = (0..30).map(make_frame).collect();
        let iter = FrameIterator::new(frames);
        let dur = iter.total_duration();
        assert!((dur.as_secs_f64() - 1.0).abs() < 0.02);
    }

    #[test]
    fn test_frame_iterator_seek_to_time() {
        let frames: Vec<_> = (0..60).map(make_frame).collect();
        let mut iter = FrameIterator::new(frames);
        iter.seek_to_time(Duration::from_secs(1), 30.0).unwrap();
        assert_eq!(iter.current_index(), 30);
    }
}
