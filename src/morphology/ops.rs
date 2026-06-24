use crate::core::types::Size;
use crate::error::Result;
use crate::image::Image;
use burn::tensor::backend::Backend;

/// Types of morphological operations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MorphOp {
    Opening,
    Closing,
    Gradient,
    TopHat,
    BlackHat,
}

/// Shapes of structuring elements.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MorphShape {
    Rect,
    Cross,
    Ellipse,
}

impl<B: Backend> Image<B> {
    /// Performs advanced morphological transformations.
    pub fn morphology_ex(&self, op: MorphOp, kernel_size: usize) -> Result<Self> {
        match op {
            MorphOp::Opening => self.clone().morph_open(kernel_size),
            MorphOp::Closing => self.clone().morph_close(kernel_size),
            MorphOp::Gradient => {
                let dilated = self.clone().dilate(kernel_size)?;
                let eroded = self.clone().erode(kernel_size)?;
                dilated.subtract(&eroded)
            }
            MorphOp::TopHat => {
                let opened = self.clone().morph_open(kernel_size)?;
                self.subtract(&opened)
            }
            MorphOp::BlackHat => {
                let closed = self.clone().morph_close(kernel_size)?;
                closed.subtract(self)
            }
        }
    }
}

pub struct Morphology;

impl Morphology {
    /// Creates a structuring element (kernel matrix) for morphology.
    pub fn get_structuring_element(shape: MorphShape, ksize: Size<usize>) -> Vec<Vec<u8>> {
        let w = ksize.width;
        let h = ksize.height;
        let mut element = vec![vec![0; w]; h];

        let xc = (w / 2) as f64;
        let yc = (h / 2) as f64;

        for (y, row) in element.iter_mut().enumerate() {
            for (x, val) in row.iter_mut().enumerate() {
                *val = match shape {
                    MorphShape::Rect => 1,
                    MorphShape::Cross => {
                        if x == w / 2 || y == h / 2 {
                            1
                        } else {
                            0
                        }
                    }
                    MorphShape::Ellipse => {
                        let dx = x as f64 - xc;
                        let dy = y as f64 - yc;
                        let r_x = w as f64 / 2.0;
                        let r_y = h as f64 / 2.0;

                        if r_x > 0.0 && r_y > 0.0 {
                            let term = (dx * dx) / (r_x * r_x) + (dy * dy) / (r_y * r_y);
                            if term <= 1.05 { 1 } else { 0 }
                        } else {
                            1
                        }
                    }
                };
            }
        }

        element
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structuring_elements() {
        let size = Size::new(5, 5);
        let rect = Morphology::get_structuring_element(MorphShape::Rect, size);
        assert_eq!(rect[0][0], 1);

        let cross = Morphology::get_structuring_element(MorphShape::Cross, size);
        assert_eq!(cross[0][0], 0);
        assert_eq!(cross[2][2], 1);

        let ellipse = Morphology::get_structuring_element(MorphShape::Ellipse, size);
        assert_eq!(ellipse[0][0], 0);
    }
}

