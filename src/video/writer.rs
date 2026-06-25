use crate::error::{IrisError, Result};
use crate::video::frame::Frame;
use burn::tensor::backend::Backend;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Output format for the video writer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    /// Animated GIF
    Gif,
    /// PNG image sequence
    PngSequence,
    /// JPEG image sequence
    JpegSequence,
    /// QOI image sequence
    QoiSequence,
}

impl OutputFormat {
    /// Returns the file extension for this format.
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Gif => "gif",
            Self::PngSequence => "png",
            Self::JpegSequence => "jpg",
            Self::QoiSequence => "qoi",
        }
    }

    /// Detects the output format from a file path.
    #[must_use]
    pub fn from_path(path: &str) -> Self {
        let lower = path.to_lowercase();
        if lower.ends_with(".gif") {
            Self::Gif
        } else if lower.ends_with(".qoi") {
            Self::QoiSequence
        } else if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
            Self::JpegSequence
        } else {
            Self::PngSequence
        }
    }
}

/// Configuration for the video writer.
#[derive(Clone, Debug)]
pub struct VideoWriteOptions {
    /// Output format.
    pub format: OutputFormat,
    /// Frames per second for the output.
    pub fps: f64,
    /// Number of loops for GIF (0 = infinite).
    pub gif_loops: u32,
    /// JPEG quality (1-100).
    pub jpeg_quality: u8,
}

impl Default for VideoWriteOptions {
    fn default() -> Self {
        Self {
            format: OutputFormat::Gif,
            fps: 30.0,
            gif_loops: 0,
            jpeg_quality: 90,
        }
    }
}

/// A high-level video writer that buffers frames and writes to disk.
///
/// Supports GIF output and image sequences. Frames are buffered in
/// memory and flushed on `finish()` or `Drop`.
pub struct VideoWriter<B: Backend> {
    frames: Vec<Frame<B>>,
    output_path: PathBuf,
    options: VideoWriteOptions,
    width: usize,
    height: usize,
    finished: bool,
}

impl<B: Backend> VideoWriter<B> {
    /// Creates a new video writer for the given output path and dimensions.
    pub fn create(
        output_path: impl AsRef<Path>,
        width: usize,
        height: usize,
        options: &VideoWriteOptions,
    ) -> Result<Self> {
        let output_path = output_path.as_ref().to_path_buf();

        if let Some(parent) = output_path.parent().filter(|p| !p.exists()) {
            std::fs::create_dir_all(parent)
                .map_err(|e| IrisError::Video(format!("Failed to create output directory: {e}")))?;
        }

        Ok(Self {
            frames: Vec::new(),
            output_path,
            options: options.clone(),
            width,
            height,
            finished: false,
        })
    }

    /// Writes a single frame to the buffer.
    pub fn write_frame(&mut self, frame: &Frame<B>) -> Result<()> {
        if self.finished {
            return Err(IrisError::Video("Writer already finished".to_string()));
        }

        if frame.width() != self.width || frame.height() != self.height {
            return Err(IrisError::DimensionMismatch {
                expected: vec![3, self.height, self.width],
                actual: vec![3, frame.height(), frame.width()],
            });
        }

        self.frames.push(frame.clone());
        Ok(())
    }

    /// Returns the number of frames currently buffered.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Returns the current buffer duration.
    #[must_use]
    pub fn duration(&self) -> Duration {
        self.frames
            .iter()
            .map(|f| {
                if f.duration.is_zero() {
                    Duration::from_secs_f64(1.0 / self.options.fps)
                } else {
                    f.duration
                }
            })
            .sum()
    }

    /// Finishes writing and flushes all frames to disk.
    pub fn finish(&mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }

        if self.frames.is_empty() {
            return Err(IrisError::Video("No frames to write".to_string()));
        }

        match self.options.format {
            OutputFormat::Gif => self.write_gif()?,
            OutputFormat::PngSequence => self.write_png_sequence()?,
            OutputFormat::JpegSequence => self.write_jpeg_sequence()?,
            OutputFormat::QoiSequence => self.write_qoi_sequence()?,
        }

        self.finished = true;
        Ok(())
    }

    /// Converts a Burn tensor frame to an `image` crate RGB image.
    fn frame_to_image(&self, frame: &Frame<B>) -> image::RgbImage {
        let [c, h, w] = frame.shape();
        assert!(c == 3, "Only RGB frames supported for output");

        let raw_data: Vec<f32> = frame.image.tensor.to_data().into_vec()
            .expect("Tensor data should be valid");
        let mut img = image::RgbImage::new(w as u32, h as u32);

        let hw = h * w;
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let r = (raw_data[idx] * 255.0).clamp(0.0, 255.0) as u8;
                let g = (raw_data[hw + idx] * 255.0).clamp(0.0, 255.0) as u8;
                let b = (raw_data[2 * hw + idx] * 255.0).clamp(0.0, 255.0) as u8;
                img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
            }
        }

        img
    }

    /// Writes frames as an animated GIF.
    fn write_gif(&self) -> Result<()> {
        use image::codecs::gif::{GifEncoder, Repeat};

        let file = std::fs::File::create(&self.output_path)
            .map_err(|e| IrisError::Video(format!("Failed to create GIF file: {e}")))?;

        let mut encoder = GifEncoder::new(file);
        encoder.set_repeat(Repeat::Infinite)
            .map_err(|e| IrisError::Video(format!("Failed to set GIF repeat: {e}")))?;

        let delay = image::Delay::from_saturating_duration(Duration::from_secs_f64(1.0 / self.options.fps));

        let frames: Vec<image::Frame> = self.frames.iter()
            .map(|f| {
                let img = self.frame_to_image(f);
                let dyn_img = image::DynamicImage::ImageRgb8(img);
                let rgba = dyn_img.to_rgba8();
                image::Frame::from_parts(rgba, 0, 0, delay)
            })
            .collect();

        encoder.encode_frames(frames)
            .map_err(|e| IrisError::Video(format!("Failed to write GIF: {e}")))?;

        Ok(())
    }

    /// Writes frames as a PNG image sequence.
    fn write_png_sequence(&self) -> Result<()> {
        let base = self.output_path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("frame");
        let parent = self.output_path.parent()
            .unwrap_or(Path::new("."));

        for (i, frame) in self.frames.iter().enumerate() {
            let img = self.frame_to_image(frame);
            let path = parent.join(format!("{base}_{i:06}.png"));
            img.save(&path)
                .map_err(|e| IrisError::Video(format!("Failed to save PNG frame {i}: {e}")))?;
        }

        Ok(())
    }

    /// Writes frames as a JPEG image sequence.
    fn write_jpeg_sequence(&self) -> Result<()> {
        use image::codecs::jpeg::JpegEncoder;

        let base = self.output_path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("frame");
        let parent = self.output_path.parent()
            .unwrap_or(Path::new("."));

        for (i, frame) in self.frames.iter().enumerate() {
            let img = self.frame_to_image(frame);
            let path = parent.join(format!("{base}_{i:06}.jpg"));
            let file = std::fs::File::create(&path)
                .map_err(|e| IrisError::Video(format!("Failed to create JPEG frame {i}: {e}")))?;
            let encoder = JpegEncoder::new_with_quality(file, self.options.jpeg_quality);
            img.write_with_encoder(encoder)
                .map_err(|e| IrisError::Video(format!("Failed to write JPEG frame {i}: {e}")))?;
        }

        Ok(())
    }

    /// Writes frames as a QOI image sequence.
    fn write_qoi_sequence(&self) -> Result<()> {
        let base = self.output_path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("frame");
        let parent = self.output_path.parent()
            .unwrap_or(Path::new("."));

        for (i, frame) in self.frames.iter().enumerate() {
            let img = self.frame_to_image(frame);
            let raw = img.into_raw();
            let path = parent.join(format!("{base}_{i:06}.qoi"));

            let mut qoi_data = Vec::new();
            qoi_data.extend_from_slice(b"qoif");
            qoi_data.extend_from_slice(&(self.width as u32).to_be_bytes());
            qoi_data.extend_from_slice(&(self.height as u32).to_be_bytes());
            qoi_data.push(3);
            qoi_data.push(0);

            let mut offset = 0;
            while offset < raw.len() {
                let r = raw[offset];
                let g = raw[offset + 1];
                let b = raw[offset + 2];
                qoi_data.push(0xFF);
                qoi_data.push(r);
                qoi_data.push(g);
                qoi_data.push(b);
                offset += 3;
            }

            qoi_data.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 1]);

            std::fs::write(&path, &qoi_data)
                .map_err(|e| IrisError::Video(format!("Failed to save QOI frame {i}: {e}")))?;
        }

        Ok(())
    }

    /// Returns whether the writer has been finished.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Returns a reference to the output path.
    #[must_use]
    pub fn output_path(&self) -> &Path {
        &self.output_path
    }
}

impl<B: Backend> Drop for VideoWriter<B> {
    fn drop(&mut self) {
        if !self.finished && !self.frames.is_empty() {
            let _ = self.finish();
        }
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
    fn test_video_writer_create() {
        let opts = VideoWriteOptions::default();
        let writer = VideoWriter::<TestBackend>::create("/tmp/test.gif", 32, 32, &opts);
        assert!(writer.is_ok());
        let writer = writer.unwrap();
        assert_eq!(writer.frame_count(), 0);
        assert!(!writer.is_finished());
    }

    #[test]
    fn test_video_writer_write_frame() {
        let opts = VideoWriteOptions::default();
        let mut writer = VideoWriter::<TestBackend>::create("/tmp/test.gif", 32, 32, &opts).unwrap();

        let frame = make_test_frame(0);
        writer.write_frame(&frame).unwrap();
        assert_eq!(writer.frame_count(), 1);

        let frame = make_test_frame(1);
        writer.write_frame(&frame).unwrap();
        assert_eq!(writer.frame_count(), 2);
    }

    #[test]
    fn test_video_writer_dimension_mismatch() {
        let opts = VideoWriteOptions::default();
        let mut writer = VideoWriter::<TestBackend>::create("/tmp/test.gif", 64, 64, &opts).unwrap();

        let device = test_device();
        let data = TensorData::new(vec![0.5f32; 3 * 32 * 32], [3, 32, 32]);
        let tensor = burn::tensor::Tensor::<TestBackend, 3>::from_data(data, &device);
        let img = crate::image::Image::new(tensor);
        let frame = Frame::new(img, Duration::ZERO, 0);

        assert!(writer.write_frame(&frame).is_err());
    }

    #[test]
    fn test_video_writer_duration() {
        let opts = VideoWriteOptions { fps: 30.0, ..Default::default() };
        let mut writer = VideoWriter::<TestBackend>::create("/tmp/test.gif", 32, 32, &opts).unwrap();

        for i in 0..30 {
            writer.write_frame(&make_test_frame(i)).unwrap();
        }

        let dur = writer.duration();
        assert!((dur.as_secs_f64() - 1.0).abs() < 0.02);
    }

    #[test]
    fn test_output_format_detection() {
        assert_eq!(OutputFormat::from_path("out.gif"), OutputFormat::Gif);
        assert_eq!(OutputFormat::from_path("out.png"), OutputFormat::PngSequence);
        assert_eq!(OutputFormat::from_path("out.jpg"), OutputFormat::JpegSequence);
        assert_eq!(OutputFormat::from_path("out.qoi"), OutputFormat::QoiSequence);
    }
}
