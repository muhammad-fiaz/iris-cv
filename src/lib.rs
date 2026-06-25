#![recursion_limit = "256"]
#![allow(clippy::needless_range_loop)]
// Pedantic lints not applicable to a computer vision library
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::module_name_repetitions,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::unreadable_literal,
    clippy::bool_to_int_with_if,
    clippy::manual_midpoint,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::needless_pass_by_value,
    clippy::no_effect_underscore_binding,
    clippy::single_match_else,
    clippy::uninlined_format_args
)]

//! Iris
//! A fast computer vision library framework in Rust.

pub mod error;
pub mod prelude;

pub mod aruco;
pub mod barcode;
pub mod burn;
pub mod camera;
pub mod color;
pub mod contours;
pub mod core;
pub mod dnn;
pub mod drawing;
pub mod edges;
pub mod face;
pub mod features;
pub mod filters;
pub mod gpu;
/// GUI and Window Management.
///
/// Requires the `gpui` feature.
#[cfg(feature = "gpui")]
pub mod gui;
pub mod histogram;
pub mod hog;
pub mod image;
pub mod inpaint;
pub mod ml;
pub mod morphology;
pub mod noise;
pub mod object_detection;
pub mod ocr;
pub mod optical_flow;
pub mod photo;
pub mod qr;
pub mod segmentation;
pub mod simd;
pub mod stitching;
pub mod threshold;
pub mod tracking;
pub mod stereo;
pub mod kalman;
pub mod utils;
pub mod video;

#[cfg(test)]
mod tests {
    #[test]
    fn test_lib_compiles() {}
}

#[cfg(test)]
pub(crate) mod test_helpers {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    pub type TestBackend = NdArray;

    /// Returns an NdArray CPU device. Always available, no GPU required.
    pub fn test_device() -> NdArrayDevice {
        NdArrayDevice::default()
    }
}
