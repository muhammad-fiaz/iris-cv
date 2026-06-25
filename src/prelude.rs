pub use crate::aruco::{ArucoDetector, ArucoDict, ArucoMarker};
pub use crate::barcode::{Barcode, BarcodeDetector};
pub use crate::burn::BurnUtils;
pub use crate::camera::{Camera, CameraCalibration, CameraSource};
pub use crate::contours::{Contour, Moments, RotatedRect, ShapeAnalysis, ConvexityDefect, RetrievalMode};
pub use crate::core::{Mat, Point, Rect, Rng, Scalar, Size};
pub use crate::dnn::{
    OnnxModel, WeightLoader, blob_from_image, nms_boxes, read_net, read_net_from_onnx,
};
pub use crate::drawing::MarkerType;
pub use crate::error::{IrisError, Result};
pub use crate::face::{Face, FaceDetector, FaceRecognizer};
pub use crate::features::{BFMatcher, DMatch, FeatureDetector, FeatureType, KeyPoint, MatchDrawer, TemplateMatchMethod, template_match, FlannMatcher};
#[cfg(feature = "gpui")]
pub use crate::gui::Gui;
pub use crate::hog::HogDescriptor;
pub use crate::image::{GeometricTransform, Image};
pub use crate::inpaint::inpaint;
pub use crate::ml::KMeans;
pub use crate::morphology::{MorphOp, MorphShape, Morphology};
pub use crate::object_detection::{Detection, ObjectDetector};
pub use crate::ocr::{OcrPipeline, OcrResult};
pub use crate::optical_flow::OpticalFlow;
pub use crate::photo::{MergeMertens, Photo};
pub use crate::photo::nlm;
pub use crate::qr::{QrCode, QrDetector};
pub use crate::segmentation::{ComponentStats, SegmentationMask, Segmenter, watershed};
pub use crate::stereo::StereoBlockMatcher;
pub use crate::kalman::KalmanFilter;
pub use crate::stitching::Stitcher;
pub use crate::threshold::{AdaptiveMethod, ThresholdType};
pub use crate::tracking::{BackgroundSubtractor, Tracker, TrackerType, MeanShiftTracker};
pub use crate::video::{
    VideoCapture, LegacyVideoWriter,
    Frame, FrameIterator, FrameExt, VideoReader, VideoWriter as VideoFileWriter,
    VideoMetadata, ContainerFormat, PixelFormat, StreamInfo, StreamType,
    VideoOpenOptions, VideoWriteOptions, OutputFormat, SeekMode,
    load_animated_image, load_image_sequence,
};

// Re-export common Burn items
pub use burn::prelude::Device;
pub use burn::tensor::{Int, Tensor, TensorData, backend::Backend};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prelude_imports() {
        // Just verify we can access structural items exported by prelude
        let _pt = Point::new(0.0, 0.0);
        let _sz = Size::new(640, 480);
        let _err = IrisError::Generic("test".into());
    }
}
