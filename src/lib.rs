#![recursion_limit = "256"]
#![allow(clippy::needless_range_loop)]

//! Observers
//! A fast computer vision library framework in Rust.

pub mod error;
pub mod prelude;

pub mod aruco;
pub mod barcode;
pub mod burn;
pub mod camera;
pub mod contours;
pub mod core;
pub mod dnn;
pub mod drawing;
pub mod edges;
pub mod face;
pub mod features;
pub mod filters;
pub mod gpu;
pub mod gui;
pub mod histogram;
pub mod image;
pub mod ml;
pub mod morphology;
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
pub mod utils;
pub mod video;

#[cfg(test)]
mod tests {
    #[test]
    fn test_lib_compiles() {
        assert!(true);
    }
}
