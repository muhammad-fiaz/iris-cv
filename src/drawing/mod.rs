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
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::Wgpu;

    #[test]
    fn test_drawing_operations() {
        let device = Default::default();
        let flat_data = vec![0.0f32; 3 * 100 * 100];
        let tensor =
            Tensor::<Wgpu, 3>::from_data(TensorData::new(flat_data, [3, 100, 100]), &device);
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
}
