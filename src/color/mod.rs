use crate::error::{IrisError, Result};
use crate::image::Image;
use burn::tensor::{Tensor, TensorData, backend::Backend};

/// Color space conversion utilities for images.
impl<B: Backend> Image<B> {
    /// Converts an RGB image to HSV (Hue, Saturation, Value) color space.
    /// Input must be a 3-channel image with values in [0, 1].
    /// Returns a 3-channel image where H is in [0, 360]/360 (normalized to [0, 1]),
    /// S is in [0, 1], and V is in [0, 1].
    pub fn rgb_to_hsv(&self) -> Result<Self> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        if c != 3 {
            return Err(IrisError::InvalidParameter(
                "Input must be a 3-channel RGB image".into(),
            ));
        }

        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; 3 * h * w];

        let pixels = h * w;

        for i in 0..pixels {
            let r = flat_vals[i];
            let g = flat_vals[pixels + i];
            let b = flat_vals[2 * pixels + i];

            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            let delta = max - min;

            // Value
            out_vals[2 * pixels + i] = max;

            // Saturation
            out_vals[pixels + i] = if max.abs() < 1e-6 { 0.0 } else { delta / max };

            // Hue
            let hue = if delta.abs() < 1e-6 {
                0.0
            } else if (max - r).abs() < 1e-6 {
                60.0 * (((g - b) / delta) % 6.0)
            } else if (max - g).abs() < 1e-6 {
                60.0 * (((b - r) / delta) + 2.0)
            } else {
                60.0 * (((r - g) / delta) + 4.0)
            };

            // Normalize hue to [0, 1]
            let hue_norm = if hue < 0.0 { (hue + 360.0) / 360.0 } else { hue / 360.0 };
            out_vals[i] = hue_norm;
        }

        let device = self.tensor.device();
        let data = TensorData::new(out_vals, [3, h, w]);
        let tensor = Tensor::<B, 3>::from_data(data, &device);
        Ok(Image::new(tensor))
    }

    /// Converts an HSV image to RGB color space.
    /// Input must be a 3-channel image where H is in [0, 1] (normalized from 360),
    /// S is in [0, 1], and V is in [0, 1].
    pub fn hsv_to_rgb(&self) -> Result<Self> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h_dim = dims[1];
        let w = dims[2];

        if c != 3 {
            return Err(IrisError::InvalidParameter(
                "Input must be a 3-channel HSV image".into(),
            ));
        }

        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; 3 * h_dim * w];

        let pixels = h_dim * w;

        for i in 0..pixels {
            let hue = flat_vals[i] * 360.0; // Denormalize hue
            let sat = flat_vals[pixels + i];
            let val = flat_vals[2 * pixels + i];

            let c_val = val * sat;
            let x = c_val * (1.0 - ((hue / 60.0) % 2.0 - 1.0).abs());
            let m = val - c_val;

            let (r, g, b) = if hue < 60.0 {
                (c_val, x, 0.0)
            } else if hue < 120.0 {
                (x, c_val, 0.0)
            } else if hue < 180.0 {
                (0.0, c_val, x)
            } else if hue < 240.0 {
                (0.0, x, c_val)
            } else if hue < 300.0 {
                (x, 0.0, c_val)
            } else {
                (c_val, 0.0, x)
            };

            out_vals[i] = r + m;
            out_vals[pixels + i] = g + m;
            out_vals[2 * pixels + i] = b + m;
        }

        let device = self.tensor.device();
        let data = TensorData::new(out_vals, [3, h_dim, w]);
        let tensor = Tensor::<B, 3>::from_data(data, &device);
        Ok(Image::new(tensor))
    }

    /// Converts an RGB image to HLS (Hue, Lightness, Saturation) color space.
    /// H is normalized to [0, 1] (from 360 degrees), L and S are in [0, 1].
    pub fn rgb_to_hls(&self) -> Result<Self> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        if c != 3 {
            return Err(IrisError::InvalidParameter(
                "Input must be a 3-channel RGB image".into(),
            ));
        }

        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; 3 * h * w];

        let pixels = h * w;

        for i in 0..pixels {
            let r = flat_vals[i];
            let g = flat_vals[pixels + i];
            let b = flat_vals[2 * pixels + i];

            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            let delta = max - min;

            // Lightness
            let l = (max + min) / 2.0;
            out_vals[pixels + i] = l;

            // Saturation
            out_vals[2 * pixels + i] = if delta.abs() < 1e-6 {
                0.0
            } else if l < 0.5 {
                delta / (max + min)
            } else {
                delta / (2.0 - max - min)
            };

            // Hue
            let hue = if delta.abs() < 1e-6 {
                0.0
            } else if (max - r).abs() < 1e-6 {
                60.0 * (((g - b) / delta) % 6.0)
            } else if (max - g).abs() < 1e-6 {
                60.0 * (((b - r) / delta) + 2.0)
            } else {
                60.0 * (((r - g) / delta) + 4.0)
            };

            let hue_norm = if hue < 0.0 { (hue + 360.0) / 360.0 } else { hue / 360.0 };
            out_vals[i] = hue_norm;
        }

        let device = self.tensor.device();
        let data = TensorData::new(out_vals, [3, h, w]);
        let tensor = Tensor::<B, 3>::from_data(data, &device);
        Ok(Image::new(tensor))
    }

    /// Converts an HLS image to RGB color space.
    pub fn hls_to_rgb(&self) -> Result<Self> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h_dim = dims[1];
        let w = dims[2];

        if c != 3 {
            return Err(IrisError::InvalidParameter(
                "Input must be a 3-channel HLS image".into(),
            ));
        }

        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; 3 * h_dim * w];

        let pixels = h_dim * w;

        for i in 0..pixels {
            let hue = flat_vals[i] * 360.0;
            let l = flat_vals[pixels + i];
            let s = flat_vals[2 * pixels + i];

            let c_val = (1.0 - (2.0 * l - 1.0).abs()) * s;
            let x = c_val * (1.0 - ((hue / 60.0) % 2.0 - 1.0).abs());
            let m = l - c_val / 2.0;

            let (r, g, b) = if hue < 60.0 {
                (c_val, x, 0.0)
            } else if hue < 120.0 {
                (x, c_val, 0.0)
            } else if hue < 180.0 {
                (0.0, c_val, x)
            } else if hue < 240.0 {
                (0.0, x, c_val)
            } else if hue < 300.0 {
                (x, 0.0, c_val)
            } else {
                (c_val, 0.0, x)
            };

            out_vals[i] = r + m;
            out_vals[pixels + i] = g + m;
            out_vals[2 * pixels + i] = b + m;
        }

        let device = self.tensor.device();
        let data = TensorData::new(out_vals, [3, h_dim, w]);
        let tensor = Tensor::<B, 3>::from_data(data, &device);
        Ok(Image::new(tensor))
    }

    /// Splits a multi-channel image into individual single-channel images.
    pub fn split_channels(&self) -> Result<Vec<Self>> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let pixels = h * w;

        let mut channels = Vec::with_capacity(c);
        for ch in 0..c {
            let start = ch * pixels;
            let channel_data = flat_vals[start..start + pixels].to_vec();
            let data = TensorData::new(channel_data, [1, h, w]);
            let tensor = Tensor::<B, 3>::from_data(data, &self.tensor.device());
            channels.push(Image::new(tensor));
        }

        Ok(channels)
    }

    /// Merges single-channel images into a multi-channel image.
    pub fn merge_channels(channels: &[Image<B>]) -> Result<Self> {
        if channels.is_empty() {
            return Err(IrisError::InvalidParameter(
                "At least one channel is required".into(),
            ));
        }

        let dims = channels[0].tensor.dims();
        let h = dims[1];
        let w = dims[2];
        let c = channels.len();

        let mut all_vals = Vec::with_capacity(c * h * w);
        for ch in channels {
            let ch_dims = ch.tensor.dims();
            if ch_dims[1] != h || ch_dims[2] != w {
                return Err(IrisError::DimensionMismatch {
                    expected: vec![1, h, w],
                    actual: vec![ch_dims[0], ch_dims[1], ch_dims[2]],
                });
            }
            let data = ch.tensor.clone().into_data();
            let vals: Vec<f32> = data.iter::<f32>().collect();
            all_vals.extend_from_slice(&vals);
        }

        let device = channels[0].tensor.device();
        let data = TensorData::new(all_vals, [c, h, w]);
        let tensor = Tensor::<B, 3>::from_data(data, &device);
        Ok(Image::new(tensor))
    }

    /// Converts RGB to CIE XYZ color space.
    /// Uses sRGB D65 illuminant matrix.
    pub fn rgb_to_xyz(&self) -> Result<Self> {
        let dims = self.tensor.dims();
        if dims[0] != 3 {
            return Err(IrisError::InvalidParameter(
                "Input must be 3-channel RGB".into(),
            ));
        }
        let h = dims[1];
        let w = dims[2];
        let data = self.tensor.clone().into_data();
        let flat: Vec<f32> = data.iter::<f32>().collect();
        let pixels = h * w;
        let mut out = vec![0.0f32; 3 * pixels];

        for i in 0..pixels {
            // sRGB gamma linearization
            let r_lin = linearize(flat[i]);
            let g_lin = linearize(flat[pixels + i]);
            let b_lin = linearize(flat[2 * pixels + i]);

            // sRGB to XYZ (D65)
            out[i] = 0.412_456_4 * r_lin + 0.357_576_1 * g_lin + 0.180_437_5 * b_lin;
            out[pixels + i] = 0.212_672_9 * r_lin + 0.715_152_2 * g_lin + 0.072_175_0 * b_lin;
            out[2 * pixels + i] = 0.019_333_9 * r_lin + 0.119_192 * g_lin + 0.950_304_1 * b_lin;
        }

        Ok(Image::new(Tensor::<B, 3>::from_data(
            TensorData::new(out, [3, h, w]),
            &self.tensor.device(),
        )))
    }

    /// Converts CIE XYZ to RGB color space.
    pub fn xyz_to_rgb(&self) -> Result<Self> {
        let dims = self.tensor.dims();
        if dims[0] != 3 {
            return Err(IrisError::InvalidParameter(
                "Input must be 3-channel XYZ".into(),
            ));
        }
        let h = dims[1];
        let w = dims[2];
        let data = self.tensor.clone().into_data();
        let flat: Vec<f32> = data.iter::<f32>().collect();
        let pixels = h * w;
        let mut out = vec![0.0f32; 3 * pixels];

        for i in 0..pixels {
            let x = flat[i];
            let y = flat[pixels + i];
            let z = flat[2 * pixels + i];

            // XYZ to linear sRGB
            let r_lin = 3.240_454_2 * x - 1.537_138_5 * y - 0.498_531_4 * z;
            let g_lin = -0.969_266 * x + 1.876_010_8 * y + 0.041_556_0 * z;
            let b_lin = 0.055_643_4 * x - 0.204_025_9 * y + 1.057_225_2 * z;

            // Gamma encoding
            out[i] = delinearize(r_lin);
            out[pixels + i] = delinearize(g_lin);
            out[2 * pixels + i] = delinearize(b_lin);
        }

        Ok(Image::new(Tensor::<B, 3>::from_data(
            TensorData::new(out, [3, h, w]),
            &self.tensor.device(),
        )))
    }

    /// Converts RGB to CIE L*a*b* color space.
    /// Pipeline: RGB -> XYZ -> L*a*b* (D65 white point).
    pub fn rgb_to_lab(&self) -> Result<Self> {
        let xyz = self.rgb_to_xyz()?;
        let dims = xyz.tensor.dims();
        let h = dims[1];
        let w = dims[2];
        let data = xyz.tensor.clone().into_data();
        let flat: Vec<f32> = data.iter::<f32>().collect();
        let pixels = h * w;
        let mut out = vec![0.0f32; 3 * pixels];

        // D65 white point
        let xn = 0.950_47_f64;
        let yn = 1.0_f64;
        let zn = 1.088_83_f64;

        for i in 0..pixels {
            let x = flat[i] as f64 / xn;
            let y = flat[pixels + i] as f64 / yn;
            let z = flat[2 * pixels + i] as f64 / zn;

            let fx = lab_f(x);
            let fy = lab_f(y);
            let fz = lab_f(z);

            let l = 116.0 * fy - 16.0;
            let a = 500.0 * (fx - fy);
            let b = 200.0 * (fy - fz);

            out[i] = (l / 100.0) as f32; // L in [0, 1]
            out[pixels + i] = ((a + 128.0) / 255.0) as f32; // a in [0, 1]
            out[2 * pixels + i] = ((b + 128.0) / 255.0) as f32; // b in [0, 1]
        }

        Ok(Image::new(Tensor::<B, 3>::from_data(
            TensorData::new(out, [3, h, w]),
            &self.tensor.device(),
        )))
    }

    /// Converts CIE L*a*b* to RGB color space.
    pub fn lab_to_rgb(&self) -> Result<Self> {
        let dims = self.tensor.dims();
        if dims[0] != 3 {
            return Err(IrisError::InvalidParameter("Input must be 3-channel LAB".into()));
        }
        let h = dims[1];
        let w = dims[2];
        let data = self.tensor.clone().into_data();
        let flat: Vec<f32> = data.iter::<f32>().collect();
        let pixels = h * w;
        let mut xyz_vals = vec![0.0f32; 3 * pixels];

        let xn = 0.950_47_f64;
        let yn = 1.0_f64;
        let zn = 1.088_83_f64;

        for i in 0..pixels {
            let l = flat[i] as f64 * 100.0;
            let a = flat[pixels + i] as f64 * 255.0 - 128.0;
            let b = flat[2 * pixels + i] as f64 * 255.0 - 128.0;

            let fy = (l + 16.0) / 116.0;
            let fx = a / 500.0 + fy;
            let fz = fy - b / 200.0;

            let x = lab_f_inv(fx) * xn;
            let y = lab_f_inv(fy) * yn;
            let z = lab_f_inv(fz) * zn;

            xyz_vals[i] = x as f32;
            xyz_vals[pixels + i] = y as f32;
            xyz_vals[2 * pixels + i] = z as f32;
        }

        let xyz_img = Image::new(Tensor::<B, 3>::from_data(
            TensorData::new(xyz_vals, [3, h, w]),
            &self.tensor.device(),
        ));
        xyz_img.xyz_to_rgb()
    }

    /// Converts RGB to YUV (BT.601) color space.
    pub fn rgb_to_yuv(&self) -> Result<Self> {
        let dims = self.tensor.dims();
        if dims[0] != 3 {
            return Err(IrisError::InvalidParameter("Input must be 3-channel RGB".into()));
        }
        let h = dims[1];
        let w = dims[2];
        let data = self.tensor.clone().into_data();
        let flat: Vec<f32> = data.iter::<f32>().collect();
        let pixels = h * w;
        let mut out = vec![0.0f32; 3 * pixels];

        for i in 0..pixels {
            let r = flat[i] as f64;
            let g = flat[pixels + i] as f64;
            let b = flat[2 * pixels + i] as f64;

            let y = 0.299 * r + 0.587 * g + 0.114 * b;
            let u = -0.147_13 * r - 0.288_86 * g + 0.436 * b + 0.5;
            let v = 0.615 * r - 0.514_99 * g - 0.100_01 * b + 0.5;

            out[i] = y.clamp(0.0, 1.0) as f32;
            out[pixels + i] = u.clamp(0.0, 1.0) as f32;
            out[2 * pixels + i] = v.clamp(0.0, 1.0) as f32;
        }

        Ok(Image::new(Tensor::<B, 3>::from_data(
            TensorData::new(out, [3, h, w]),
            &self.tensor.device(),
        )))
    }

    /// Converts YUV (BT.601) to RGB color space.
    pub fn yuv_to_rgb(&self) -> Result<Self> {
        let dims = self.tensor.dims();
        if dims[0] != 3 {
            return Err(IrisError::InvalidParameter("Input must be 3-channel YUV".into()));
        }
        let h = dims[1];
        let w = dims[2];
        let data = self.tensor.clone().into_data();
        let flat: Vec<f32> = data.iter::<f32>().collect();
        let pixels = h * w;
        let mut out = vec![0.0f32; 3 * pixels];

        for i in 0..pixels {
            let y = flat[i] as f64;
            let u = flat[pixels + i] as f64 - 0.5;
            let v = flat[2 * pixels + i] as f64 - 0.5;

            let r = y + 1.139_83 * v;
            let g = y - 0.394_65 * u - 0.580_60 * v;
            let b = y + 2.032_11 * u;

            out[i] = r.clamp(0.0, 1.0) as f32;
            out[pixels + i] = g.clamp(0.0, 1.0) as f32;
            out[2 * pixels + i] = b.clamp(0.0, 1.0) as f32;
        }

        Ok(Image::new(Tensor::<B, 3>::from_data(
            TensorData::new(out, [3, h, w]),
            &self.tensor.device(),
        )))
    }

    /// Converts RGB to YCrCb (BT.601) color space.
    pub fn rgb_to_ycrcb(&self) -> Result<Self> {
        let dims = self.tensor.dims();
        if dims[0] != 3 {
            return Err(IrisError::InvalidParameter("Input must be 3-channel RGB".into()));
        }
        let h = dims[1];
        let w = dims[2];
        let data = self.tensor.clone().into_data();
        let flat: Vec<f32> = data.iter::<f32>().collect();
        let pixels = h * w;
        let mut out = vec![0.0f32; 3 * pixels];

        for i in 0..pixels {
            let r = flat[i] as f64;
            let g = flat[pixels + i] as f64;
            let b = flat[2 * pixels + i] as f64;

            let y = 0.299 * r + 0.587 * g + 0.114 * b;
            let cr = 0.713 * (r - y) + 0.5;
            let cb = 0.564 * (b - y) + 0.5;

            out[i] = y.clamp(0.0, 1.0) as f32;
            out[pixels + i] = cr.clamp(0.0, 1.0) as f32;
            out[2 * pixels + i] = cb.clamp(0.0, 1.0) as f32;
        }

        Ok(Image::new(Tensor::<B, 3>::from_data(
            TensorData::new(out, [3, h, w]),
            &self.tensor.device(),
        )))
    }

    /// Converts an RGB image to CMYK (Cyan, Magenta, Yellow, Key/Black) color space.
    /// Input must be a 3-channel image with values in [0, 1].
    /// Returns a 4-channel image with values in [0, 1].
    pub fn rgb_to_cmyk(&self) -> Result<Self> {
        let dims = self.tensor.dims();
        if dims[0] != 3 {
            return Err(IrisError::InvalidParameter(
                "Input must be a 3-channel RGB image".into(),
            ));
        }
        let h = dims[1];
        let w = dims[2];
        let pixels = h * w;

        let data = self.tensor.clone().into_data();
        let flat: Vec<f32> = data.iter::<f32>().collect();
        let mut out = vec![0.0f32; 4 * pixels];

        for i in 0..pixels {
            let r = flat[i];
            let g = flat[pixels + i];
            let b = flat[2 * pixels + i];

            let k = 1.0f32 - r.max(g).max(b);
            if k < 1.0 - 1e-6 {
                let inv = 1.0 / (1.0 - k);
                out[i] = (1.0 - r - k) * inv;                    // Cyan
                out[pixels + i] = (1.0 - g - k) * inv;           // Magenta
                out[2 * pixels + i] = (1.0 - b - k) * inv;       // Yellow
            } else {
                out[i] = 0.0;
                out[pixels + i] = 0.0;
                out[2 * pixels + i] = 0.0;
            }
            out[3 * pixels + i] = k; // Black
        }

        let device = self.tensor.device();
        let tensor = Tensor::<B, 3>::from_data(TensorData::new(out, [4, h, w]), &device);
        Ok(Image::new(tensor))
    }

    /// Converts a CMYK image to RGB color space.
    /// Input must be a 4-channel image with values in [0, 1].
    /// Returns a 3-channel RGB image with values in [0, 1].
    pub fn cmyk_to_rgb(&self) -> Result<Self> {
        let dims = self.tensor.dims();
        if dims[0] != 4 {
            return Err(IrisError::InvalidParameter(
                "Input must be a 4-channel CMYK image".into(),
            ));
        }
        let h = dims[1];
        let w = dims[2];
        let pixels = h * w;

        let data = self.tensor.clone().into_data();
        let flat: Vec<f32> = data.iter::<f32>().collect();
        let mut out = vec![0.0f32; 3 * pixels];

        for i in 0..pixels {
            let c = flat[i];
            let m = flat[pixels + i];
            let y = flat[2 * pixels + i];
            let k = flat[3 * pixels + i];

            out[i] = (1.0 - c) * (1.0 - k);                        // Red
            out[pixels + i] = (1.0 - m) * (1.0 - k);               // Green
            out[2 * pixels + i] = (1.0 - y) * (1.0 - k);           // Blue
        }

        let device = self.tensor.device();
        let tensor = Tensor::<B, 3>::from_data(TensorData::new(out, [3, h, w]), &device);
        Ok(Image::new(tensor))
    }

    /// Converts an RGB image to HSL (Hue, Saturation, Lightness) color space.
    /// Input must be a 3-channel image with values in [0, 1].
    /// H is normalized to [0, 1] (from 360 degrees), S and L are in [0, 1].
    pub fn rgb_to_hsl(&self) -> Result<Self> {
        let dims = self.tensor.dims();
        if dims[0] != 3 {
            return Err(IrisError::InvalidParameter(
                "Input must be a 3-channel RGB image".into(),
            ));
        }
        let h = dims[1];
        let w = dims[2];
        let pixels = h * w;

        let data = self.tensor.clone().into_data();
        let flat: Vec<f32> = data.iter::<f32>().collect();
        let mut out = vec![0.0f32; 3 * pixels];

        for i in 0..pixels {
            let r = flat[i] as f64;
            let g = flat[pixels + i] as f64;
            let b = flat[2 * pixels + i] as f64;

            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            let l = (max + min) / 2.0;
            let delta = max - min;

            // Saturation
            let s = if delta.abs() < 1e-10 {
                0.0
            } else if l < 0.5 {
                delta / (max + min)
            } else {
                delta / (2.0 - max - min)
            };

            // Hue
            let hue_deg = if delta.abs() < 1e-10 {
                0.0
            } else if (max - r).abs() < 1e-10 {
                60.0 * (((g - b) / delta) % 6.0)
            } else if (max - g).abs() < 1e-10 {
                60.0 * (((b - r) / delta) + 2.0)
            } else {
                60.0 * (((r - g) / delta) + 4.0)
            };

            let hue_norm = if hue_deg < 0.0 {
                (hue_deg + 360.0) / 360.0
            } else {
                hue_deg / 360.0
            };

            out[i] = hue_norm as f32;
            out[pixels + i] = s.clamp(0.0, 1.0) as f32;
            out[2 * pixels + i] = l.clamp(0.0, 1.0) as f32;
        }

        let device = self.tensor.device();
        let tensor = Tensor::<B, 3>::from_data(TensorData::new(out, [3, h, w]), &device);
        Ok(Image::new(tensor))
    }

    /// Converts an HSL image to RGB color space.
    /// Input must be a 3-channel image where H is in [0, 1] (from 360 degrees),
    /// S and L are in [0, 1].
    /// Returns a 3-channel RGB image with values in [0, 1].
    pub fn hsl_to_rgb(&self) -> Result<Self> {
        let dims = self.tensor.dims();
        if dims[0] != 3 {
            return Err(IrisError::InvalidParameter(
                "Input must be a 3-channel HSL image".into(),
            ));
        }
        let h_dim = dims[1];
        let w = dims[2];
        let pixels = h_dim * w;

        let data = self.tensor.clone().into_data();
        let flat: Vec<f32> = data.iter::<f32>().collect();
        let mut out = vec![0.0f32; 3 * pixels];

        for i in 0..pixels {
            let hue_deg = flat[i] as f64 * 360.0;
            let s = flat[pixels + i] as f64;
            let l = flat[2 * pixels + i] as f64;

            let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
            let x = c * (1.0 - ((hue_deg / 60.0) % 2.0 - 1.0).abs());
            let m = l - c / 2.0;

            let (r, g, b) = if hue_deg < 60.0 {
                (c, x, 0.0)
            } else if hue_deg < 120.0 {
                (x, c, 0.0)
            } else if hue_deg < 180.0 {
                (0.0, c, x)
            } else if hue_deg < 240.0 {
                (0.0, x, c)
            } else if hue_deg < 300.0 {
                (x, 0.0, c)
            } else {
                (c, 0.0, x)
            };

            out[i] = (r + m).clamp(0.0, 1.0) as f32;
            out[pixels + i] = (g + m).clamp(0.0, 1.0) as f32;
            out[2 * pixels + i] = (b + m).clamp(0.0, 1.0) as f32;
        }

        let device = self.tensor.device();
        let tensor = Tensor::<B, 3>::from_data(TensorData::new(out, [3, h_dim, w]), &device);
        Ok(Image::new(tensor))
    }

    /// Converts YCrCb to RGB color space.
    pub fn ycrcb_to_rgb(&self) -> Result<Self> {
        let dims = self.tensor.dims();
        if dims[0] != 3 {
            return Err(IrisError::InvalidParameter("Input must be 3-channel YCrCb".into()));
        }
        let h = dims[1];
        let w = dims[2];
        let data = self.tensor.clone().into_data();
        let flat: Vec<f32> = data.iter::<f32>().collect();
        let pixels = h * w;
        let mut out = vec![0.0f32; 3 * pixels];

        for i in 0..pixels {
            let y = flat[i] as f64;
            let cr = flat[pixels + i] as f64 - 0.5;
            let cb = flat[2 * pixels + i] as f64 - 0.5;

            let r = y + 1.402 * cr;
            let g = y - 0.714 * cr - 0.344 * cb;
            let b = y + 1.772 * cb;

            out[i] = r.clamp(0.0, 1.0) as f32;
            out[pixels + i] = g.clamp(0.0, 1.0) as f32;
            out[2 * pixels + i] = b.clamp(0.0, 1.0) as f32;
        }

        Ok(Image::new(Tensor::<B, 3>::from_data(
            TensorData::new(out, [3, h, w]),
            &self.tensor.device(),
        )))
    }
}

// Helper functions for color space conversions

fn linearize(srgb: f32) -> f32 {
    let v = srgb as f64;
    if v <= 0.040_45 {
        (v / 12.92) as f32
    } else {
        ((v + 0.055) / 1.055).powf(2.4) as f32
    }
}

fn delinearize(lin: f32) -> f32 {
    let v = lin as f64;
    if v <= 0.003_130_8 {
        (12.92 * v).clamp(0.0, 1.0) as f32
    } else {
        (1.055 * v.powf(1.0 / 2.4) - 0.055).clamp(0.0, 1.0) as f32
    }
}

fn lab_f(t: f64) -> f64 {
    let eps = 216.0 / 24_389.0;
    let kappa = 24_389.0 / 27.0;
    if t > eps {
        t.cbrt()
    } else {
        (kappa * t + 16.0) / 116.0
    }
}

fn lab_f_inv(t: f64) -> f64 {
    let eps = 216.0 / 24_389.0;
    let kappa = 24_389.0 / 27.0;
    let t3 = t * t * t;
    if t3 > eps {
        t3
    } else {
        (116.0 * t - 16.0) / kappa
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{TestBackend, test_device};
    use burn::tensor::TensorData;

    #[test]
    fn test_hsv_roundtrip() {
        let device = test_device();
        let flat_data = vec![
            1.0, 0.0, 0.0, // Red
            0.0, 1.0, 0.0, // Green
            0.0, 0.0, 1.0, // Blue
            1.0, 1.0, 0.0, // Yellow
        ];
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [3, 2, 2]), &device);
        let rgb = Image::new(tensor);

        let hsv = rgb.rgb_to_hsv().unwrap();
        assert_eq!(hsv.shape(), [3, 2, 2]);

        let back_rgb = hsv.hsv_to_rgb().unwrap();
        assert_eq!(back_rgb.shape(), [3, 2, 2]);
    }

    #[test]
    fn test_hls_roundtrip() {
        let device = test_device();
        let flat_data = vec![0.5f32; 3 * 4 * 4];
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [3, 4, 4]), &device);
        let rgb = Image::new(tensor);

        let hls = rgb.rgb_to_hls().unwrap();
        assert_eq!(hls.shape(), [3, 4, 4]);

        let back_rgb = hls.hls_to_rgb().unwrap();
        assert_eq!(back_rgb.shape(), [3, 4, 4]);
    }

    #[test]
    fn test_split_merge() {
        let device = test_device();
        let flat_data = vec![0.3, 0.6, 0.9, 0.1, 0.4, 0.7];
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [3, 1, 2]), &device);
        let img = Image::new(tensor);

        let channels = img.split_channels().unwrap();
        assert_eq!(channels.len(), 3);

        let merged = Image::merge_channels(&channels).unwrap();
        assert_eq!(merged.shape(), [3, 1, 2]);
    }

    #[test]
    fn test_xyz_roundtrip() {
        let device = test_device();
        let data = vec![0.5f32; 3 * 4 * 4];
        let tensor = Tensor::<TestBackend, 3>::from_data(TensorData::new(data, [3, 4, 4]), &device);
        let rgb = Image::new(tensor);
        let xyz = rgb.rgb_to_xyz().unwrap();
        assert_eq!(xyz.shape(), [3, 4, 4]);
        let back = xyz.xyz_to_rgb().unwrap();
        assert_eq!(back.shape(), [3, 4, 4]);
    }

    #[test]
    fn test_lab_roundtrip() {
        let device = test_device();
        let data = vec![0.5f32; 3 * 4 * 4];
        let tensor = Tensor::<TestBackend, 3>::from_data(TensorData::new(data, [3, 4, 4]), &device);
        let rgb = Image::new(tensor);
        let lab = rgb.rgb_to_lab().unwrap();
        assert_eq!(lab.shape(), [3, 4, 4]);
        let back = lab.lab_to_rgb().unwrap();
        assert_eq!(back.shape(), [3, 4, 4]);
    }

    #[test]
    fn test_yuv_roundtrip() {
        let device = test_device();
        let data = vec![0.5f32; 3 * 4 * 4];
        let tensor = Tensor::<TestBackend, 3>::from_data(TensorData::new(data, [3, 4, 4]), &device);
        let rgb = Image::new(tensor);
        let yuv = rgb.rgb_to_yuv().unwrap();
        assert_eq!(yuv.shape(), [3, 4, 4]);
        let back = yuv.yuv_to_rgb().unwrap();
        assert_eq!(back.shape(), [3, 4, 4]);
    }

    #[test]
    fn test_ycrcb_roundtrip() {
        let device = test_device();
        let data = vec![0.5f32; 3 * 4 * 4];
        let tensor = Tensor::<TestBackend, 3>::from_data(TensorData::new(data, [3, 4, 4]), &device);
        let rgb = Image::new(tensor);
        let ycrcb = rgb.rgb_to_ycrcb().unwrap();
        assert_eq!(ycrcb.shape(), [3, 4, 4]);
        let back = ycrcb.ycrcb_to_rgb().unwrap();
        assert_eq!(back.shape(), [3, 4, 4]);
    }

    #[test]
    fn test_cmyk_roundtrip() {
        let device = test_device();
        let flat_data = vec![
            // R channel (4 pixels)
            1.0, 0.0, 0.0, 0.5,
            // G channel (4 pixels)
            0.0, 1.0, 0.0, 0.5,
            // B channel (4 pixels)
            0.0, 0.0, 1.0, 0.5,
        ];
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [3, 2, 2]), &device);
        let rgb = Image::new(tensor);

        let cmyk = rgb.rgb_to_cmyk().unwrap();
        assert_eq!(cmyk.shape(), [4, 2, 2]);

        let back_rgb = cmyk.cmyk_to_rgb().unwrap();
        assert_eq!(back_rgb.shape(), [3, 2, 2]);

        // Verify roundtrip values
        let orig_data = rgb.tensor.into_data();
        let back_data = back_rgb.tensor.into_data();
        let orig_vals: Vec<f32> = orig_data.iter::<f32>().collect();
        let back_vals: Vec<f32> = back_data.iter::<f32>().collect();
        for (a, b) in orig_vals.iter().zip(back_vals.iter()) {
            assert!((a - b).abs() < 1e-5, "CMYK roundtrip mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_hsl_roundtrip() {
        let device = test_device();
        let flat_data = vec![
            1.0, 0.0, 0.0, // Red
            0.0, 1.0, 0.0, // Green
            0.0, 0.0, 1.0, // Blue
            0.5, 0.5, 0.5, // Gray
        ];
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [3, 2, 2]), &device);
        let rgb = Image::new(tensor);

        let hsl = rgb.rgb_to_hsl().unwrap();
        assert_eq!(hsl.shape(), [3, 2, 2]);

        let back_rgb = hsl.hsl_to_rgb().unwrap();
        assert_eq!(back_rgb.shape(), [3, 2, 2]);

        // Verify roundtrip values
        let orig_data = rgb.tensor.into_data();
        let back_data = back_rgb.tensor.into_data();
        let orig_vals: Vec<f32> = orig_data.iter::<f32>().collect();
        let back_vals: Vec<f32> = back_data.iter::<f32>().collect();
        for (a, b) in orig_vals.iter().zip(back_vals.iter()) {
            assert!((a - b).abs() < 1e-5, "HSL roundtrip mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_color_invalid_channel() {
        let device = test_device();
        let data = vec![0.5f32; 4 * 4 * 4]; // 4 channels
        let tensor = Tensor::<TestBackend, 3>::from_data(TensorData::new(data, [4, 4, 4]), &device);
        let img = Image::new(tensor);
        assert!(img.rgb_to_hsv().is_err());
        assert!(img.rgb_to_xyz().is_err());
        assert!(img.rgb_to_cmyk().is_err());
        assert!(img.rgb_to_hsl().is_err());

        // cmyk_to_rgb requires 4 channels, so 3-channel should fail
        let data3 = vec![0.5f32; 3 * 4 * 4];
        let tensor3 = Tensor::<TestBackend, 3>::from_data(TensorData::new(data3, [3, 4, 4]), &device);
        let img3 = Image::new(tensor3);
        assert!(img3.cmyk_to_rgb().is_err());

        // hsl_to_rgb requires 3 channels, so 4-channel should fail
        assert!(img.hsl_to_rgb().is_err());
    }
}
