pub mod font;

use crate::core::types::{Point, Rect, Scalar};
use crate::error::Result;
use crate::image::Image;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use font::FONT_5X7;

impl<B: Backend> Image<B> {
    /// Draws a line from p1 to p2 on the image.
    /// This operates on CPU by reading tensor, rasterizing, and uploading.
    pub fn draw_line(self, p1: Point<usize>, p2: Point<usize>, color: Scalar) -> Result<Self> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let device = self.tensor.device();
        let tensor_data = self.tensor.into_data();
        let mut flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

        let mut x0 = p1.x as isize;
        let mut y0 = p1.y as isize;
        let x1 = p2.x as isize;
        let y1 = p2.y as isize;

        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;

        loop {
            if x0 >= 0 && x0 < w as isize && y0 >= 0 && y0 < h as isize {
                for ch in 0..c {
                    let val = color.0[ch] as f32;
                    flat_vals[ch * h * w + (y0 as usize) * w + (x0 as usize)] = val;
                }
            }
            if x0 == x1 && y0 == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                y0 += sy;
            }
        }

        let new_data = TensorData::new(flat_vals, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Draws a rectangle on the image. If thickness < 0, the rectangle is filled.
    pub fn draw_rectangle(self, rect: Rect<usize>, color: Scalar, thickness: i32) -> Result<Self> {
        if thickness >= 0 {
            // Draw four borders
            let p1 = Point::new(rect.x, rect.y);
            let p2 = Point::new(rect.x + rect.width, rect.y);
            let p3 = Point::new(rect.x + rect.width, rect.y + rect.height);
            let p4 = Point::new(rect.x, rect.y + rect.height);

            self.draw_line(p1, p2, color)?
                .draw_line(p2, p3, color)?
                .draw_line(p3, p4, color)?
                .draw_line(p4, p1, color)
        } else {
            // Filled rectangle
            let dims = self.tensor.dims();
            let c = dims[0];
            let h = dims[1];
            let w = dims[2];

            let device = self.tensor.device();
            let tensor_data = self.tensor.into_data();
            let mut flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

            let x_start = rect.x;
            let y_start = rect.y;
            let x_end = (rect.x + rect.width).min(w);
            let y_end = (rect.y + rect.height).min(h);

            for y in y_start..y_end {
                for x in x_start..x_end {
                    for ch in 0..c {
                        flat_vals[ch * h * w + y * w + x] = color.0[ch] as f32;
                    }
                }
            }

            let new_data = TensorData::new(flat_vals, [c, h, w]);
            let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
            Ok(Image::new(new_tensor))
        }
    }

    /// Draws a circle on the image. If thickness < 0, the circle is filled.
    pub fn draw_circle(
        self,
        center: Point<usize>,
        radius: usize,
        color: Scalar,
        thickness: i32,
    ) -> Result<Self> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let device = self.tensor.device();
        let tensor_data = self.tensor.into_data();
        let mut flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

        let xc = center.x as isize;
        let yc = center.y as isize;
        let r = radius as isize;

        let draw_pixel = |px: isize, py: isize, vals: &mut [f32]| {
            if px >= 0 && px < w as isize && py >= 0 && py < h as isize {
                for ch in 0..c {
                    vals[ch * h * w + (py as usize) * w + (px as usize)] = color.0[ch] as f32;
                }
            }
        };

        if thickness >= 0 {
            // Midpoint Circle Algorithm
            let mut x = 0isize;
            let mut y = r;
            let mut d = 3 - 2 * r;

            let draw_sym = |x_s: isize, y_s: isize, vals: &mut [f32]| {
                draw_pixel(xc + x_s, yc + y_s, vals);
                draw_pixel(xc - x_s, yc + y_s, vals);
                draw_pixel(xc + x_s, yc - y_s, vals);
                draw_pixel(xc - x_s, yc - y_s, vals);
                draw_pixel(xc + y_s, yc + x_s, vals);
                draw_pixel(xc - y_s, yc + x_s, vals);
                draw_pixel(xc + y_s, yc - x_s, vals);
                draw_pixel(xc - y_s, yc - x_s, vals);
            };

            draw_sym(x, y, &mut flat_vals);
            while y >= x {
                x += 1;
                if d > 0 {
                    y -= 1;
                    d = d + 4 * (x - y) + 10;
                } else {
                    d = d + 4 * x + 6;
                }
                draw_sym(x, y, &mut flat_vals);
            }
        } else {
            // Filled Circle
            for y in 0..h {
                for x in 0..w {
                    let dx = x as isize - xc;
                    let dy = y as isize - yc;
                    if dx * dx + dy * dy <= r * r {
                        for ch in 0..c {
                            flat_vals[ch * h * w + y * w + x] = color.0[ch] as f32;
                        }
                    }
                }
            }
        }

        let new_data = TensorData::new(flat_vals, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Draws a text label on the image using the built-in 5x7 font.
    pub fn draw_text(
        self,
        text: &str,
        org: Point<usize>,
        scale: usize,
        color: Scalar,
    ) -> Result<Self> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let device = self.tensor.device();
        let tensor_data = self.tensor.into_data();
        let mut flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

        let scale = scale.max(1);
        let mut cursor_x = org.x;

        for byte in text.bytes() {
            let char_idx = (byte as usize).min(127);
            let bitmap = FONT_5X7[char_idx];

            for (col, &col_data) in bitmap.iter().enumerate() {
                for row in 0..7 {
                    if (col_data & (1 << row)) != 0 {
                        // Draw a pixel block based on scale
                        let px_start = cursor_x + col * scale;
                        let py_start = org.y + row * scale;

                        for sy in 0..scale {
                            for sx in 0..scale {
                                let x = px_start + sx;
                                let y = py_start + sy;
                                if x < w && y < h {
                                    for ch in 0..c {
                                        flat_vals[ch * h * w + y * w + x] = color.0[ch] as f32;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            cursor_x += 6 * scale; // 5 columns + 1 space column
        }

        let new_data = TensorData::new(flat_vals, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Draws an ellipse on the image.
    /// `center` is the center of the ellipse, `axes` is (semi_major, semi_minor).
    /// `angle` is the rotation angle in degrees. `start_angle` and `end_angle` in degrees.
    /// If `thickness < 0`, the ellipse is filled.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_ellipse(
        self,
        center: Point<usize>,
        axes: (usize, usize),
        angle: f32,
        start_angle: f32,
        end_angle: f32,
        color: Scalar,
        thickness: i32,
    ) -> Result<Self> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let device = self.tensor.device();
        let tensor_data = self.tensor.into_data();
        let mut flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

        let cx = center.x as f32;
        let cy = center.y as f32;
        let (a, b) = (axes.0 as f32, axes.1 as f32);
        let angle_rad = angle * std::f32::consts::PI / 180.0;
        let start_rad = start_angle * std::f32::consts::PI / 180.0;
        let end_rad = end_angle * std::f32::consts::PI / 180.0;

        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();

        let draw_px = |px: isize, py: isize, vals: &mut [f32]| {
            if px >= 0 && px < w as isize && py >= 0 && py < h as isize {
                for ch in 0..c {
                    vals[ch * h * w + py as usize * w + px as usize] = color.0[ch] as f32;
                }
            }
        };

        if thickness >= 0 {
            // Draw ellipse outline
            for t_idx in 0..720 {
                let t = start_rad + (end_rad - start_rad) * t_idx as f32 / 720.0;
                let ex = a * t.cos();
                let ey = b * t.sin();
                let rx = ex * cos_a - ey * sin_a;
                let ry = ex * sin_a + ey * cos_a;
                draw_px((cx + rx) as isize, (cy + ry) as isize, &mut flat_vals);
            }
        } else {
            // Filled ellipse using scanline
            let max_r = a.max(b) as usize;
            let x0 = (cx as isize - max_r as isize).max(0) as usize;
            let x1 = (cx as usize + max_r).min(w);
            let y0 = (cy as isize - max_r as isize).max(0) as usize;
            let y1 = (cy as usize + max_r).min(h);

            for py in y0..y1 {
                for px in x0..x1 {
                    let dx = px as f32 - cx;
                    let dy = py as f32 - cy;
                    // Transform point into ellipse coordinate system
                    let tx = dx * cos_a + dy * sin_a;
                    let ty = -dx * sin_a + dy * cos_a;
                    if a > 0.0 && b > 0.0 && (tx / a).powi(2) + (ty / b).powi(2) <= 1.0 {
                        for ch in 0..c {
                            flat_vals[ch * h * w + py * w + px] = color.0[ch] as f32;
                        }
                    }
                }
            }
        }

        let new_data = TensorData::new(flat_vals, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Draws a polyline (connected line segments) on the image.
    pub fn draw_polyline(
        self,
        points: &[Point<usize>],
        color: Scalar,
        _thickness: i32,
    ) -> Result<Self> {
        if points.len() < 2 {
            return Ok(self);
        }
        let mut current = self;
        for i in 0..points.len() - 1 {
            current = current.draw_line(points[i], points[i + 1], color)?;
        }
        Ok(current)
    }

    /// Draws a filled polygon on the image using scanline fill.
    pub fn fill_poly(self, points: &[Point<usize>], color: Scalar) -> Result<Self> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let device = self.tensor.device();
        let tensor_data = self.tensor.into_data();
        let mut flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

        if points.len() < 3 {
            return Ok(Image::new(Tensor::<B, 3>::from_data(
                TensorData::new(flat_vals, [c, h, w]),
                &device,
            )));
        }

        // Find bounding box
        let min_y = points.iter().map(|p| p.y).min().unwrap_or(0);
        let max_y = points.iter().map(|p| p.y).max().unwrap_or(h - 1);
        let max_y = max_y.min(h - 1);

        // Scanline fill
        for y in min_y..=max_y {
            let mut intersections = Vec::new();
            let n = points.len();
            for i in 0..n {
                let j = (i + 1) % n;
                let (p1, p2) = (points[i], points[j]);
                if (p1.y <= y && p2.y > y) || (p2.y <= y && p1.y > y) {
                    let x_intersect =
                        p1.x as f64 + (y as f64 - p1.y as f64) * (p2.x as f64 - p1.x as f64)
                            / (p2.y as f64 - p1.y as f64);
                    intersections.push(x_intersect as usize);
                }
            }
            intersections.sort_unstable();

            for pair in intersections.chunks(2) {
                if pair.len() == 2 {
                    let x_start = pair[0];
                    let x_end = pair[1].min(w - 1);
                    for x in x_start..=x_end {
                        for ch in 0..c {
                            flat_vals[ch * h * w + y * w + x] = color.0[ch] as f32;
                        }
                    }
                }
            }
        }

        let new_data = TensorData::new(flat_vals, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Draws an arrowed line from p1 to p2 with an arrowhead.
    pub fn draw_arrowed_line(
        self,
        p1: Point<usize>,
        p2: Point<usize>,
        color: Scalar,
        _thickness: i32,
        tip_length: f32,
    ) -> Result<Self> {
        let img = self.draw_line(p1, p2, color)?;

        // Compute arrowhead
        let dx = p2.x as f64 - p1.x as f64;
        let dy = p2.y as f64 - p1.y as f64;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1.0 {
            return Ok(img);
        }

        let ux = dx / len;
        let uy = dy / len;
        let tip_size = len * tip_length as f64;
        let angle = std::f64::consts::FRAC_PI_6; // 30 degrees

        let left = (
            (p2.x as f64 - tip_size * (ux * angle.cos() - uy * angle.sin())) as usize,
            (p2.y as f64 - tip_size * (uy * angle.cos() + ux * angle.sin())) as usize,
        );
        let right = (
            (p2.x as f64 - tip_size * (ux * angle.cos() + uy * angle.sin())) as usize,
            (p2.y as f64 - tip_size * (uy * angle.cos() - ux * angle.sin())) as usize,
        );

        img.draw_line(
            p2,
            Point::new(left.0, left.1),
            color,
        )?
        .draw_line(p2, Point::new(right.0, right.1), color)
    }

    /// Draws a marker symbol at a point.
    pub fn draw_marker(
        self,
        center: Point<usize>,
        color: Scalar,
        marker_type: MarkerType,
        marker_size: usize,
    ) -> Result<Self> {
        match marker_type {
            MarkerType::Cross => {
                let half = marker_size / 2;
                self.draw_line(
                    Point::new(center.x.saturating_sub(half), center.y),
                    Point::new(center.x + half, center.y),
                    color,
                )?
                .draw_line(
                    Point::new(center.x, center.y.saturating_sub(half)),
                    Point::new(center.x, center.y + half),
                    color,
                )
            }
            MarkerType::TiltedCross => {
                let half = marker_size / 2;
                self.draw_line(
                    Point::new(center.x.saturating_sub(half), center.y.saturating_sub(half)),
                    Point::new(center.x + half, center.y + half),
                    color,
                )?
                .draw_line(
                    Point::new(center.x + half, center.y.saturating_sub(half)),
                    Point::new(center.x.saturating_sub(half), center.y + half),
                    color,
                )
            }
            MarkerType::Diamond => {
                let half = marker_size / 2;
                self.draw_polyline(
                    &[
                        Point::new(center.x, center.y.saturating_sub(half)),
                        Point::new(center.x + half, center.y),
                        Point::new(center.x, center.y + half),
                        Point::new(center.x.saturating_sub(half), center.y),
                        Point::new(center.x, center.y.saturating_sub(half)),
                    ],
                    color,
                    1,
                )
            }
            MarkerType::Square => self.draw_rectangle(
                Rect::new(
                    center.x.saturating_sub(marker_size / 2),
                    center.y.saturating_sub(marker_size / 2),
                    marker_size,
                    marker_size,
                ),
                color,
                1,
            ),
            MarkerType::Circle => {
                self.draw_circle(center, marker_size / 2, color, 1)
            }
            MarkerType::Filled => self.draw_circle(center, marker_size / 2, color, -1),
        }
    }
}

/// Types of drawing markers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MarkerType {
    Cross,
    TiltedCross,
    Diamond,
    Square,
    Circle,
    Filled,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{TestBackend, test_device};

    #[test]
    fn test_drawing_operations() {
        let device = test_device();
        let flat_data = vec![0.0f32; 3 * 100 * 100];
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [3, 100, 100]), &device);
        let img = Image::new(tensor);

        let img = img
            .draw_line(Point::new(10, 10), Point::new(90, 90), Scalar::all(1.0))
            .unwrap();
        let img = img
            .draw_rectangle(Rect::new(20, 20, 30, 40), Scalar::all(0.5), 1)
            .unwrap();
        let img = img
            .draw_circle(Point::new(50, 50), 20, Scalar::all(0.8), -1)
            .unwrap();
        let img = img
            .draw_text("Hello", Point::new(10, 80), 2, Scalar::all(0.9))
            .unwrap();

        assert_eq!(img.shape(), [3, 100, 100]);
    }

    #[test]
    fn test_draw_ellipse() {
        let device = test_device();
        let data = vec![0.0f32; 3 * 60 * 60];
        let tensor = Tensor::<TestBackend, 3>::from_data(TensorData::new(data, [3, 60, 60]), &device);
        let img = Image::new(tensor);

        // Outline
        let img = img
            .draw_ellipse(
                Point::new(30, 30),
                (15, 10),
                30.0,
                0.0,
                360.0,
                Scalar::all(1.0),
                1,
            )
            .unwrap();
        assert_eq!(img.shape(), [3, 60, 60]);

        // Filled
        let img = img
            .draw_ellipse(
                Point::new(30, 30),
                (15, 10),
                30.0,
                0.0,
                360.0,
                Scalar::all(0.5),
                -1,
            )
            .unwrap();
        assert_eq!(img.shape(), [3, 60, 60]);
    }

    #[test]
    fn test_draw_polyline() {
        let device = test_device();
        let data = vec![0.0f32; 3 * 50 * 50];
        let tensor = Tensor::<TestBackend, 3>::from_data(TensorData::new(data, [3, 50, 50]), &device);
        let img = Image::new(tensor);

        let points = vec![
            Point::new(10, 10),
            Point::new(40, 10),
            Point::new(40, 40),
            Point::new(10, 40),
            Point::new(10, 10),
        ];
        let img = img.draw_polyline(&points, Scalar::all(1.0), 1).unwrap();
        assert_eq!(img.shape(), [3, 50, 50]);
    }

    #[test]
    fn test_fill_poly() {
        let device = test_device();
        let data = vec![0.0f32; 3 * 50 * 50];
        let tensor = Tensor::<TestBackend, 3>::from_data(TensorData::new(data, [3, 50, 50]), &device);
        let img = Image::new(tensor);

        let points = vec![
            Point::new(10, 10),
            Point::new(40, 10),
            Point::new(40, 40),
            Point::new(10, 40),
        ];
        let img = img.fill_poly(&points, Scalar::all(0.8)).unwrap();
        assert_eq!(img.shape(), [3, 50, 50]);
    }

    #[test]
    fn test_draw_arrowed_line() {
        let device = test_device();
        let data = vec![0.0f32; 3 * 50 * 50];
        let tensor = Tensor::<TestBackend, 3>::from_data(TensorData::new(data, [3, 50, 50]), &device);
        let img = Image::new(tensor);

        let img = img
            .draw_arrowed_line(
                Point::new(10, 10),
                Point::new(40, 40),
                Scalar::all(1.0),
                1,
                0.3,
            )
            .unwrap();
        assert_eq!(img.shape(), [3, 50, 50]);
    }

    #[test]
    fn test_draw_marker() {
        let device = test_device();
        let data = vec![0.0f32; 3 * 50 * 50];
        let tensor = Tensor::<TestBackend, 3>::from_data(TensorData::new(data, [3, 50, 50]), &device);
        let img = Image::new(tensor);

        let img = img
            .draw_marker(Point::new(25, 25), Scalar::all(1.0), MarkerType::Cross, 10)
            .unwrap();
        let img = img
            .draw_marker(Point::new(25, 25), Scalar::all(0.5), MarkerType::Circle, 10)
            .unwrap();
        let img = img
            .draw_marker(Point::new(25, 25), Scalar::all(0.8), MarkerType::Filled, 10)
            .unwrap();
        assert_eq!(img.shape(), [3, 50, 50]);
    }
}
