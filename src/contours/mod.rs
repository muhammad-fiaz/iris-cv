pub mod shape_analysis;

pub use shape_analysis::{RotatedRect, ShapeAnalysis};

use crate::core::types::Point;
use crate::error::Result;
use crate::image::Image;
use burn::tensor::backend::Backend;

/// Represents a contour outline as a list of points.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Contour {
    pub points: Vec<Point<usize>>,
}

/// Image moments representing spatial distribution.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Moments {
    pub m00: f64, // Area
    pub m10: f64,
    pub m01: f64,
    pub m20: f64,
    pub m02: f64,
    pub m11: f64,
}

impl Moments {
    /// Computes the centroid (center of mass) from the moments.
    #[must_use]
    pub fn centroid(&self) -> Option<Point<f64>> {
        if self.m00.abs() < 1e-9 {
            None
        } else {
            Some(Point::new(self.m10 / self.m00, self.m01 / self.m00))
        }
    }
}

impl Contour {
    /// Creates a new Contour with given points.
    #[must_use]
    pub fn new(points: Vec<Point<usize>>) -> Self {
        Self { points }
    }

    /// Computes the convex hull of the contour using Andrew's Monotone Chain algorithm.
    #[must_use]
    pub fn convex_hull(&self) -> Self {
        let mut pts = self.points.clone();
        if pts.len() <= 3 {
            return Self::new(pts);
        }

        // Sort points lexicographically by x, then y
        pts.sort_by(|a, b| {
            if a.x == b.x {
                a.y.cmp(&b.y)
            } else {
                a.x.cmp(&b.x)
            }
        });

        fn cross(o: &Point<usize>, a: &Point<usize>, b: &Point<usize>) -> isize {
            (a.x as isize - o.x as isize) * (b.y as isize - o.y as isize)
                - (a.y as isize - o.y as isize) * (b.x as isize - o.x as isize)
        }

        let mut lower = Vec::new();
        for p in &pts {
            while lower.len() >= 2
                && cross(&lower[lower.len() - 2], &lower[lower.len() - 1], p) <= 0
            {
                lower.pop();
            }
            lower.push(*p);
        }

        let mut upper = Vec::new();
        for p in pts.iter().rev() {
            while upper.len() >= 2
                && cross(&upper[upper.len() - 2], &upper[upper.len() - 1], p) <= 0
            {
                upper.pop();
            }
            upper.push(*p);
        }

        lower.pop();
        upper.pop();
        lower.extend(upper);

        Self::new(lower)
    }

    /// Computes the polygon moments using the Shoelace formula and Green's Theorem.
    #[must_use]
    pub fn moments(&self) -> Moments {
        let pts = &self.points;
        let n = pts.len();
        if n < 3 {
            return Moments::default();
        }

        let mut m00 = 0.0;
        let mut m10 = 0.0;
        let mut m01 = 0.0;
        let mut m20 = 0.0;
        let mut m02 = 0.0;
        let mut m11 = 0.0;

        for i in 0..n {
            let p0 = pts[i];
            let p1 = pts[(i + 1) % n];

            let xi = p0.x as f64;
            let yi = p0.y as f64;
            let xi1 = p1.x as f64;
            let yi1 = p1.y as f64;

            let cross = xi * yi1 - xi1 * yi;
            m00 += cross;
            m10 += (xi + xi1) * cross;
            m01 += (yi + yi1) * cross;
            m20 += (xi * xi + xi * xi1 + xi1 * xi1) * cross;
            m02 += (yi * yi + yi * yi1 + yi1 * yi1) * cross;
            m11 += (2.0 * xi * yi + xi * yi1 + xi1 * yi + 2.0 * xi1 * yi1) * cross;
        }

        m00 /= 2.0;
        m10 /= 6.0;
        m01 /= 6.0;
        m20 /= 12.0;
        m02 /= 12.0;
        m11 /= 24.0;

        Moments {
            m00: m00.abs(),
            m10: m10.abs(),
            m01: m01.abs(),
            m20: m20.abs(),
            m02: m02.abs(),
            m11: m11.abs(),
        }
    }
}

impl<B: Backend> Image<B> {
    /// Scans a binary image (grayscale, thresholded) to find contours (connected components).
    /// Uses a basic boundary-following algorithm to find contiguous shapes.
    pub fn find_contours(&self) -> Result<Vec<Contour>> {
        let gray = self.grayscale()?;
        let dims = gray.tensor.dims();
        let h = dims[1];
        let w = dims[2];

        let tensor_data = gray.tensor.into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

        let mut visited = vec![false; h * w];
        let mut contours = Vec::new();

        // 8-neighbor offsets
        let dx = [1, 1, 0, -1, -1, -1, 0, 1];
        let dy = [0, 1, 1, 1, 0, -1, -1, -1];

        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                let idx = y * w + x;
                if flat_vals[idx] > 0.5 && !visited[idx] {
                    // Start of a boundary
                    let mut pts = Vec::new();
                    let mut cx = x;
                    let mut cy = y;
                    let mut dir = 0;

                    pts.push(Point::new(cx, cy));
                    visited[idx] = true;

                    // Trace boundary
                    let mut loop_count = 0;
                    loop {
                        let mut found = false;
                        for i in 0..8 {
                            let ndir = (dir + i) % 8;
                            let nx = cx as isize + dx[ndir];
                            let ny = cy as isize + dy[ndir];

                            if nx >= 0 && nx < w as isize && ny >= 0 && ny < h as isize {
                                let nidx = (ny as usize) * w + (nx as usize);
                                if flat_vals[nidx] > 0.5 {
                                    cx = nx as usize;
                                    cy = ny as usize;
                                    visited[nidx] = true;
                                    pts.push(Point::new(cx, cy));
                                    dir = (ndir + 5) % 8; // Backtrack direction
                                    found = true;
                                    break;
                                }
                            }
                        }

                        if !found || (cx == x && cy == y) || loop_count > 10000 {
                            break;
                        }
                        loop_count += 1;
                    }

                    if pts.len() >= 3 {
                        contours.push(Contour::new(pts));
                    }
                }
            }
        }

        Ok(contours)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{TestBackend, test_device};
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn test_contours_and_moments() {
        let pts = vec![
            Point::new(0, 0),
            Point::new(10, 0),
            Point::new(10, 10),
            Point::new(0, 10),
        ];
        let contour = Contour::new(pts);
        let hull = contour.convex_hull();
        assert_eq!(hull.points.len(), 4);

        let m = contour.moments();
        assert!(m.m00 > 0.0);
        let centroid = m.centroid().unwrap();
        assert!(centroid.x > 0.0);

        let device = test_device();
        // Create an image with a single pixel set to 1.0 (binary mask)
        let mut flat_data = vec![0.0f32; 10 * 10];
        // Set a 3x3 block to 1.0
        for y in 2..5 {
            for x in 2..5 {
                flat_data[y * 10 + x] = 1.0;
            }
        }
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [1, 10, 10]), &device);
        let img = Image::new(tensor);
        let found = img.find_contours().unwrap();
        assert!(!found.is_empty());
    }
}
