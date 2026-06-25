use std::time::Duration;

/// Supported video container formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ContainerFormat {
    /// Animated GIF
    Gif,
    /// Portable Network Graphics (animated/APNG)
    Png,
    /// JPEG (animated)
    Jpeg,
    /// WebP (animated)
    WebP,
    /// QOI format
    Qoi,
    /// Image sequence (directory of numbered images)
    ImageSequence,
    /// AVI container
    Avi,
    /// MP4/MOV container
    Mp4,
    /// MKV/Matroska container
    Mkv,
    /// WebM container
    Webm,
    /// Unknown format
    Unknown,
}

impl ContainerFormat {
    /// Returns the typical file extension for this format.
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Gif => "gif",
            Self::Png => "png",
            Self::Jpeg => "jpg",
            Self::WebP => "webp",
            Self::Qoi => "qoi",
            Self::ImageSequence => "",
            Self::Avi => "avi",
            Self::Mp4 => "mp4",
            Self::Mkv => "mkv",
            Self::Webm => "webm",
            Self::Unknown => "",
        }
    }

    /// Detects the container format from a file path extension.
    #[must_use]
    pub fn from_path(path: &str) -> Self {
        let lower = path.to_lowercase();
        if lower.ends_with(".gif") {
            Self::Gif
        } else if lower.ends_with(".apng") || lower.ends_with(".png") {
            Self::Png
        } else if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
            Self::Jpeg
        } else if lower.ends_with(".webp") {
            Self::WebP
        } else if lower.ends_with(".qoi") {
            Self::Qoi
        } else if lower.ends_with(".avi") {
            Self::Avi
        } else if lower.ends_with(".mp4") || lower.ends_with(".mov") {
            Self::Mp4
        } else if lower.ends_with(".mkv") {
            Self::Mkv
        } else if lower.ends_with(".webm") {
            Self::Webm
        } else {
            Self::Unknown
        }
    }
}

/// Pixel format of video frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    /// 8-bit RGB (3 channels)
    Rgb8,
    /// 8-bit RGBA (4 channels)
    Rgba8,
    /// 8-bit Grayscale (1 channel)
    Gray8,
    /// 16-bit RGB
    Rgb16,
    /// 16-bit RGBA
    Rgba16,
    /// 32-bit float RGB
    Rgb32F,
    /// 32-bit float RGBA
    Rgba32F,
}

impl PixelFormat {
    /// Returns the number of channels for this pixel format.
    #[must_use]
    pub fn channels(&self) -> usize {
        match self {
            Self::Gray8 => 1,
            Self::Rgb8 | Self::Rgb16 | Self::Rgb32F => 3,
            Self::Rgba8 | Self::Rgba16 | Self::Rgba32F => 4,
        }
    }
}

/// Information about a single stream within a video file.
#[derive(Clone, Debug)]
pub struct StreamInfo {
    /// Stream index (0-based).
    pub index: usize,
    /// Stream type.
    pub stream_type: StreamType,
    /// Codec name.
    pub codec: String,
    /// Width in pixels (for video streams).
    pub width: usize,
    /// Height in pixels (for video streams).
    pub height: usize,
    /// Frame rate in frames per second.
    pub fps: f64,
    /// Duration of the stream.
    pub duration: Duration,
    /// Total number of frames.
    pub frame_count: usize,
    /// Pixel format.
    pub pixel_format: PixelFormat,
    /// Rotation angle in degrees.
    pub rotation: u32,
    /// Bit rate in bits per second.
    pub bit_rate: u64,
}

/// Type of stream within a video file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamType {
    /// Video stream.
    Video,
    /// Audio stream.
    Audio,
    /// Subtitle stream.
    Subtitle,
}

/// Complete metadata for a video file.
#[derive(Clone, Debug)]
pub struct VideoMetadata {
    /// Container format.
    pub format: ContainerFormat,
    /// Video duration.
    pub duration: Duration,
    /// Frame rate in frames per second.
    pub fps: f64,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Total number of frames.
    pub frame_count: usize,
    /// Video codec.
    pub video_codec: String,
    /// Pixel format.
    pub pixel_format: PixelFormat,
    /// Rotation angle in degrees.
    pub rotation: u32,
    /// Total bit rate.
    pub bit_rate: u64,
    /// All streams in the file.
    pub streams: Vec<StreamInfo>,
    /// Whether the video has audio.
    pub has_audio: bool,
    /// Whether the video has subtitles.
    pub has_subtitles: bool,
    /// File size in bytes.
    pub file_size: u64,
}

impl VideoMetadata {
    /// Creates metadata for a synthetic/test video.
    #[must_use]
    pub fn synthetic(width: usize, height: usize, fps: f64, frame_count: usize) -> Self {
        let duration_secs = if fps > 0.0 {
            frame_count as f64 / fps
        } else {
            0.0
        };
        Self {
            format: ContainerFormat::Unknown,
            duration: Duration::from_secs_f64(duration_secs),
            fps,
            width,
            height,
            frame_count,
            video_codec: "unknown".to_string(),
            pixel_format: PixelFormat::Rgb8,
            rotation: 0,
            bit_rate: 0,
            streams: Vec::new(),
            has_audio: false,
            has_subtitles: false,
            file_size: 0,
        }
    }

    /// Returns the aspect ratio (width / height).
    #[must_use]
    pub fn aspect_ratio(&self) -> f64 {
        if self.height == 0 {
            return 0.0;
        }
        self.width as f64 / self.height as f64
    }

    /// Returns the total number of video streams.
    #[must_use]
    pub fn video_stream_count(&self) -> usize {
        self.streams
            .iter()
            .filter(|s| s.stream_type == StreamType::Video)
            .count()
    }

    /// Returns the total number of audio streams.
    #[must_use]
    pub fn audio_stream_count(&self) -> usize {
        self.streams
            .iter()
            .filter(|s| s.stream_type == StreamType::Audio)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_container_format_detection() {
        assert_eq!(ContainerFormat::from_path("test.gif"), ContainerFormat::Gif);
        assert_eq!(ContainerFormat::from_path("test.mp4"), ContainerFormat::Mp4);
        assert_eq!(ContainerFormat::from_path("test.MKV"), ContainerFormat::Mkv);
        assert_eq!(
            ContainerFormat::from_path("test.webp"),
            ContainerFormat::WebP
        );
        assert_eq!(
            ContainerFormat::from_path("test.unknown"),
            ContainerFormat::Unknown
        );
    }

    #[test]
    fn test_video_metadata() {
        let meta = VideoMetadata::synthetic(1920, 1080, 30.0, 300);
        assert_eq!(meta.width, 1920);
        assert_eq!(meta.height, 1080);
        assert!((meta.fps - 30.0).abs() < 1e-6);
        assert_eq!(meta.frame_count, 300);
        assert!((meta.aspect_ratio() - 16.0 / 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_pixel_format_channels() {
        assert_eq!(PixelFormat::Rgb8.channels(), 3);
        assert_eq!(PixelFormat::Rgba8.channels(), 4);
        assert_eq!(PixelFormat::Gray8.channels(), 1);
    }
}
