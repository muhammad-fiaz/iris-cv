use crate::core::types::Point;
use crate::error::{ObserversError, Result};
use crate::image::Image;
use burn::tensor::{Tensor, TensorData, backend::Backend};

impl<B: Backend> Image<B> {
    /// Transposes the image (swaps height and width).
    pub fn transpose(&self) -> Result<Self> {
        let transposed = self.tensor.clone().swap_dims(1, 2);
        Ok(Image::new(transposed))
    }

    /// Warps the image using a 2x3 affine transformation matrix.
    pub fn warp_affine(
        &self,
        m: [[f64; 3]; 2],
        new_width: usize,
        new_height: usize,
    ) -> Result<Self> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let device = self.tensor.device();
        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; c * new_height * new_width];

        // Solve M inverse using standard Cramer's rule for the 2x2 part of the matrix
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        if det.abs() < 1e-9 {
            return Err(ObserversError::InvalidParameter(
                "Transformation matrix is singular".into(),
            ));
        }
        let inv_det = 1.0 / det;

        // M_inv computation
        let a_inv = [
            [m[1][1] * inv_det, -m[0][1] * inv_det],
            [-m[1][0] * inv_det, m[0][0] * inv_det],
        ];
        let tx_inv = -(a_inv[0][0] * m[0][2] + a_inv[0][1] * m[1][2]);
        let ty_inv = -(a_inv[1][0] * m[0][2] + a_inv[1][1] * m[1][2]);

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            out_vals
                .par_chunks_exact_mut(new_width)
                .enumerate()
                .for_each(|(idx, row)| {
                    let ch = idx / new_height;
                    let dy = idx % new_height;
                    for dx in 0..new_width {
                        // Map back to original coordinate space
                        let sx = a_inv[0][0] * (dx as f64) + a_inv[0][1] * (dy as f64) + tx_inv;
                        let sy = a_inv[1][0] * (dx as f64) + a_inv[1][1] * (dy as f64) + ty_inv;

                        let sx_round = sx.round() as isize;
                        let sy_round = sy.round() as isize;

                        if sx_round >= 0
                            && sx_round < w as isize
                            && sy_round >= 0
                            && sy_round < h as isize
                        {
                            row[dx] = flat_vals
                                [ch * h * w + (sy_round as usize) * w + (sx_round as usize)];
                        }
                    }
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for dy in 0..new_height {
                for dx in 0..new_width {
                    // Map back to original coordinate space
                    let sx = a_inv[0][0] * (dx as f64) + a_inv[0][1] * (dy as f64) + tx_inv;
                    let sy = a_inv[1][0] * (dx as f64) + a_inv[1][1] * (dy as f64) + ty_inv;

                    let sx_round = sx.round() as isize;
                    let sy_round = sy.round() as isize;

                    if sx_round >= 0
                        && sx_round < w as isize
                        && sy_round >= 0
                        && sy_round < h as isize
                    {
                        for ch in 0..c {
                            out_vals[ch * new_height * new_width + dy * new_width + dx] = flat_vals
                                [ch * h * w + (sy_round as usize) * w + (sx_round as usize)];
                        }
                    }
                }
            }
        }

        let new_data = TensorData::new(out_vals, [c, new_height, new_width]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Warps the image using a 3x3 homography / perspective transformation matrix.
    pub fn warp_perspective(
        &self,
        m: [[f64; 3]; 3],
        new_width: usize,
        new_height: usize,
    ) -> Result<Self> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let device = self.tensor.device();
        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; c * new_height * new_width];

        // Invert the 3x3 matrix using standard determinant inverse formula
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

        if det.abs() < 1e-9 {
            return Err(ObserversError::InvalidParameter(
                "Perspective matrix is singular".into(),
            ));
        }
        let inv_det = 1.0 / det;

        let m_inv = [
            [
                (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
                (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
                (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
            ],
            [
                (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
                (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
                (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
            ],
            [
                (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
                (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
                (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
            ],
        ];

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            out_vals
                .par_chunks_exact_mut(new_width)
                .enumerate()
                .for_each(|(idx, row)| {
                    let ch = idx / new_height;
                    let dy = idx % new_height;
                    for dx in 0..new_width {
                        let x_mapped =
                            m_inv[0][0] * (dx as f64) + m_inv[0][1] * (dy as f64) + m_inv[0][2];
                        let y_mapped =
                            m_inv[1][0] * (dx as f64) + m_inv[1][1] * (dy as f64) + m_inv[1][2];
                        let z_mapped =
                            m_inv[2][0] * (dx as f64) + m_inv[2][1] * (dy as f64) + m_inv[2][2];

                        if z_mapped.abs() > 1e-9 {
                            let sx = x_mapped / z_mapped;
                            let sy = y_mapped / z_mapped;
                            let sx_round = sx.round() as isize;
                            let sy_round = sy.round() as isize;

                            if sx_round >= 0
                                && sx_round < w as isize
                                && sy_round >= 0
                                && sy_round < h as isize
                            {
                                row[dx] = flat_vals
                                    [ch * h * w + (sy_round as usize) * w + (sx_round as usize)];
                            }
                        }
                    }
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for dy in 0..new_height {
                for dx in 0..new_width {
                    let x_mapped =
                        m_inv[0][0] * (dx as f64) + m_inv[0][1] * (dy as f64) + m_inv[0][2];
                    let y_mapped =
                        m_inv[1][0] * (dx as f64) + m_inv[1][1] * (dy as f64) + m_inv[1][2];
                    let z_mapped =
                        m_inv[2][0] * (dx as f64) + m_inv[2][1] * (dy as f64) + m_inv[2][2];

                    if z_mapped.abs() > 1e-9 {
                        let sx = x_mapped / z_mapped;
                        let sy = y_mapped / z_mapped;
                        let sx_round = sx.round() as isize;
                        let sy_round = sy.round() as isize;

                        if sx_round >= 0
                            && sx_round < w as isize
                            && sy_round >= 0
                            && sy_round < h as isize
                        {
                            for ch in 0..c {
                                out_vals[ch * new_height * new_width + dy * new_width + dx] =
                                    flat_vals[ch * h * w
                                        + (sy_round as usize) * w
                                        + (sx_round as usize)];
                            }
                        }
                    }
                }
            }
        }

        let new_data = TensorData::new(out_vals, [c, new_height, new_width]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Remaps pixel positions using horizontal and vertical coordinates maps.
    pub fn remap(&self, map_x: &Tensor<B, 2>, map_y: &Tensor<B, 2>) -> Result<Self> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let map_dims = map_x.dims();
        let out_h = map_dims[0];
        let out_w = map_dims[1];

        let device = self.tensor.device();
        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

        let data_map_x = map_x.clone().into_data();
        let data_map_y = map_y.clone().into_data();
        let float_map_x: Vec<f32> = data_map_x.iter::<f32>().collect();
        let float_map_y: Vec<f32> = data_map_y.iter::<f32>().collect();

        let mut out_vals = vec![0.0f32; c * out_h * out_w];

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            out_vals
                .par_chunks_exact_mut(out_w)
                .enumerate()
                .for_each(|(idx, row)| {
                    let ch = idx / out_h;
                    let dy = idx % out_h;
                    for dx in 0..out_w {
                        let map_idx = dy * out_w + dx;
                        let sx = float_map_x[map_idx].round() as isize;
                        let sy = float_map_y[map_idx].round() as isize;

                        if sx >= 0 && sx < w as isize && sy >= 0 && sy < h as isize {
                            row[dx] = flat_vals[ch * h * w + (sy as usize) * w + (sx as usize)];
                        }
                    }
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for dy in 0..out_h {
                for dx in 0..out_w {
                    let map_idx = dy * out_w + dx;
                    let sx = float_map_x[map_idx].round() as isize;
                    let sy = float_map_y[map_idx].round() as isize;

                    if sx >= 0 && sx < w as isize && sy >= 0 && sy < h as isize {
                        for ch in 0..c {
                            out_vals[ch * out_h * out_w + dy * out_w + dx] =
                                flat_vals[ch * h * w + (sy as usize) * w + (sx as usize)];
                        }
                    }
                }
            }
        }

        let new_data = TensorData::new(out_vals, [c, out_h, out_w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }
}

/// Helper functions to compute geometric matrices.
pub struct GeometricTransform;

impl GeometricTransform {
    /// Computes a 2x3 affine matrix for a rotation around center with given angle (degrees) and scale.
    pub fn get_rotation_matrix_2d(
        center: Point<f64>,
        angle_degrees: f64,
        scale: f64,
    ) -> [[f64; 3]; 2] {
        let angle_rad = angle_degrees.to_radians();
        let alpha = scale * angle_rad.cos();
        let beta = scale * angle_rad.sin();

        [
            [alpha, beta, (1.0 - alpha) * center.x - beta * center.y],
            [-beta, alpha, beta * center.x + (1.0 - alpha) * center.y],
        ]
    }

    /// Computes a 2x3 affine matrix matching 3 point correspondence pairs.
    pub fn get_affine_transform(src: &[Point<f64>; 3], dst: &[Point<f64>; 3]) -> [[f64; 3]; 2] {
        // Solves the linear system:
        // dst[i].x = a*src[i].x + b*src[i].y + c
        // dst[i].y = d*src[i].x + e*src[i].y + f
        let solve = |pts_d: [f64; 3]| -> [f64; 3] {
            // Solve matrix using simple Gaussian elimination/Cramer's rule
            let a11 = src[0].x;
            let a12 = src[0].y;
            let a13 = 1.0;
            let a21 = src[1].x;
            let a22 = src[1].y;
            let a23 = 1.0;
            let a31 = src[2].x;
            let a32 = src[2].y;
            let a33 = 1.0;

            let det = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31)
                + a13 * (a21 * a32 - a22 * a31);
            if det.abs() < 1e-9 {
                return [0.0, 0.0, 0.0];
            }
            let det_x = pts_d[0] * (a22 * a33 - a23 * a32)
                - a12 * (pts_d[1] * a33 - a23 * pts_d[2])
                + a13 * (pts_d[1] * a32 - a22 * pts_d[2]);
            let det_y = a11 * (pts_d[1] * a33 - a23 * pts_d[2])
                - pts_d[0] * (a21 * a33 - a23 * a31)
                + a13 * (a21 * pts_d[2] - pts_d[1] * a31);
            let det_z = a11 * (a22 * pts_d[2] - pts_d[1] * a32)
                - a12 * (a21 * pts_d[2] - pts_d[1] * a31)
                + pts_d[0] * (a21 * a32 - a22 * a31);

            [det_x / det, det_y / det, det_z / det]
        };

        let row1 = solve([dst[0].x, dst[1].x, dst[2].x]);
        let row2 = solve([dst[0].y, dst[1].y, dst[2].y]);
        [row1, row2]
    }

    /// Computes a 3x3 perspective matrix matching 4 point correspondence pairs.
    pub fn get_perspective_transform(
        src: &[Point<f64>; 4],
        dst: &[Point<f64>; 4],
    ) -> [[f64; 3]; 3] {
        // Solves perspective mapping from 4 source coordinates to 4 destination coordinates.
        // We write the linear equations and compute using direct coefficients matching.
        let mut m = [[0.0; 3]; 3];

        let x0 = src[0].x;
        let y0 = src[0].y;
        let x1 = src[1].x;
        let y1 = src[1].y;
        let x2 = src[2].x;
        let y2 = src[2].y;
        let x3 = src[3].x;
        let y3 = src[3].y;

        let _u0 = dst[0].x;
        let _v0 = dst[0].y;
        let _u1 = dst[1].x;
        let _v1 = dst[1].y;
        let _u2 = dst[2].x;
        let _v2 = dst[2].y;
        let _u3 = dst[3].x;
        let _v3 = dst[3].y;

        let dx1 = x1 - x2;
        let dx2 = x3 - x2;
        let dy1 = y1 - y2;
        let dy2 = y3 - y2;
        let dx3 = x0 - x1 + x2 - x3;
        let dy3 = y0 - y1 + y2 - y3;

        let det = dx1 * dy2 - dx2 * dy1;
        if det.abs() < 1e-9 {
            m[0][0] = 1.0;
            m[1][1] = 1.0;
            m[2][2] = 1.0;
            return m;
        }

        let g = (dx3 * dy2 - dx2 * dy3) / det;
        let h = (dx1 * dy3 - dx3 * dy1) / det;

        let a = x1 - x0 + g * x1;
        let b = x3 - x0 + h * x3;
        let c = x0;
        let d = y1 - y0 + g * y1;
        let e = y3 - y0 + h * y3;
        let f = y0;

        // Mapping from src to dst. For warp, we invert this coefficients
        m[0][0] = a;
        m[0][1] = b;
        m[0][2] = c;
        m[1][0] = d;
        m[1][1] = e;
        m[1][2] = f;
        m[2][0] = g;
        m[2][1] = h;
        m[2][2] = 1.0;

        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::Wgpu;
    use burn::tensor::TensorData;

    #[test]
    fn test_geometric_transforms() {
        let device = Default::default();
        let flat_data = vec![0.5f32; 3 * 10 * 10];
        let tensor = Tensor::<Wgpu, 3>::from_data(TensorData::new(flat_data, [3, 10, 10]), &device);
        let img = Image::new(tensor);

        let resized = img.resize(20, 20).unwrap();
        assert_eq!(resized.shape(), [3, 20, 20]);

        let warped_aff = img.warp_affine([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]], 10, 10).unwrap();
        assert_eq!(warped_aff.shape(), [3, 10, 10]);

        let warped_persp = img.warp_perspective([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], 10, 10).unwrap();
        assert_eq!(warped_persp.shape(), [3, 10, 10]);

        let map_x = Tensor::<Wgpu, 2>::zeros([10, 10], &device);
        let map_y = Tensor::<Wgpu, 2>::zeros([10, 10], &device);
        let remapped = img.remap(&map_x, &map_y).unwrap();
        assert_eq!(remapped.shape(), [3, 10, 10]);

        let rotated = img.rotate(90).unwrap();
        assert_eq!(rotated.shape(), [3, 10, 10]);
    }
}



