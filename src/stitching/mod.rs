use crate::error::{IrisError, Result};
use crate::features::{BFMatcher, FeatureDetector, FeatureType};
use crate::image::Image;
use burn::tensor::backend::Backend;

/// Image Stitcher for panorama creation.
///
/// Uses ORB feature detection, brute-force descriptor matching, and
/// homography estimation via DLT with RANSAC to warp and blend images.
pub struct Stitcher;

impl Stitcher {
    /// Stitches a list of images into a single panorama.
    ///
    /// - If only one image is provided, it is returned as-is.
    /// - For two or more images, computes homography between consecutive pairs
    ///   and warps them onto a common canvas using the first image as anchor.
    pub fn stitch<B: Backend>(&self, images: &[Image<B>]) -> Result<Image<B>> {
        if images.is_empty() {
            return Err(IrisError::InvalidParameter(
                "Images list cannot be empty".into(),
            ));
        }
        if images.len() == 1 {
            return Ok(images[0].clone());
        }

        // Compute homographies between consecutive pairs
        let mut homographies: Vec<[[f64; 3]; 3]> = Vec::new();
        for i in 0..images.len() - 1 {
            let h = compute_homography(&images[i], &images[i + 1])?;
            homographies.push(h);
        }

        // Accumulate homographies: image[i] -> image[0] (the anchor)
        // accumulated[0] = identity
        let mut accumulated: Vec<[[f64; 3]; 3]> = vec![
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ];
        for h in &homographies {
            let prev = accumulated.last().unwrap();
            accumulated.push(multiply_3x3(prev, h));
        }

        // Find the bounding box of the warped canvas
        let mut min_x = 0.0_f64;
        let mut min_y = 0.0_f64;
        let mut max_x = 0.0_f64;
        let mut max_y = 0.0_f64;

        for (idx, img) in images.iter().enumerate() {
            let h = img.height() as f64;
            let w = img.width() as f64;
            let corners = [
                (0.0, 0.0),
                (w, 0.0),
                (w, h),
                (0.0, h),
            ];

            let h_mat = &accumulated[idx];
            for &(cx, cy) in &corners {
                let denom = h_mat[2][0] * cx + h_mat[2][1] * cy + h_mat[2][2];
                if denom.abs() > 1e-10 {
                    let wx = (h_mat[0][0] * cx + h_mat[0][1] * cy + h_mat[0][2]) / denom;
                    let wy = (h_mat[1][0] * cx + h_mat[1][1] * cy + h_mat[1][2]) / denom;
                    min_x = min_x.min(wx);
                    min_y = min_y.min(wy);
                    max_x = max_x.max(wx);
                    max_y = max_y.max(wy);
                }
            }
        }

        // Compute canvas dimensions and translation to shift all into positive coords
        let canvas_w = (max_x - min_x).ceil() as usize;
        let canvas_h = (max_y - min_y).ceil() as usize;
        let tx = -min_x;
        let ty = -min_y;

        // Translation matrix
        let t = [
            [1.0, 0.0, tx],
            [0.0, 1.0, ty],
            [0.0, 0.0, 1.0],
        ];

        // Accumulate weight map for blending (for overlap regions)
        let mut weight_canvas = vec![0.0f32; canvas_h * canvas_w];
        let mut out_vals = vec![0.0f32; 3 * canvas_h * canvas_w];

        for (idx, img) in images.iter().enumerate() {
            let d = img.tensor.dims();
            let img_h = d[1];
            let img_w = d[2];
            let data = img.tensor.clone().into_data();
            let flat: Vec<f32> = data.iter::<f32>().collect();

            // Combine accumulated homography with translation
            let h_final = multiply_3x3(&t, &accumulated[idx]);

            // Invert for backward mapping
            let h_inv = invert_3x3(&h_final).ok_or_else(|| {
                IrisError::InvalidParameter(format!(
                    "Singular homography for image pair {}",
                    idx
                ))
            })?;

            for dy in 0..canvas_h {
                for dx in 0..canvas_w {
                    let denom = h_inv[2][0] * dx as f64 + h_inv[2][1] * dy as f64 + h_inv[2][2];
                    if denom.abs() < 1e-10 {
                        continue;
                    }
                    let sx =
                        (h_inv[0][0] * dx as f64 + h_inv[0][1] * dy as f64 + h_inv[0][2]) / denom;
                    let sy =
                        (h_inv[1][0] * dx as f64 + h_inv[1][1] * dy as f64 + h_inv[1][2]) / denom;

                    let sx_r = sx.round() as isize;
                    let sy_r = sy.round() as isize;

                    if sx_r >= 0 && sx_r < img_w as isize && sy_r >= 0 && sy_r < img_h as isize {
                        let src_x = sx_r as usize;
                        let src_y = sy_r as usize;

                        // Distance-based weight: prefer pixels closer to image center
                        let cx = src_x as f64 - img_w as f64 / 2.0;
                        let cy = src_y as f64 - img_h as f64 / 2.0;
                        let max_r = (img_w as f64 + img_h as f64) / 4.0;
                        let dist = (cx * cx + cy * cy).sqrt() / max_r;
                        let w = (1.0 - dist).max(0.0) as f32;

                        let ci = dy * canvas_w + dx;
                        for ch in 0..3 {
                            let src_idx = ch * img_h * img_w + src_y * img_w + src_x;
                            out_vals[ch * canvas_h * canvas_w + ci] += flat[src_idx] * w;
                        }
                        weight_canvas[ci] += w;
                    }
                }
            }
        }

        // Normalize by accumulated weights
        for ci in 0..canvas_h * canvas_w {
            if weight_canvas[ci] > 1e-10 {
                for ch in 0..3 {
                    let idx = ch * canvas_h * canvas_w + ci;
                    out_vals[idx] = (out_vals[idx] / weight_canvas[ci]).clamp(0.0, 1.0);
                }
            }
        }

        let device = images[0].tensor.device();
        let data = burn::tensor::TensorData::new(out_vals, [3, canvas_h, canvas_w]);
        let tensor = burn::tensor::Tensor::<B, 3>::from_data(data, &device);
        Ok(Image::new(tensor))
    }
}

/// Compute homography from img1 to img2 using ORB features + BFMatcher + DLT + RANSAC.
fn compute_homography<B: Backend>(
    img1: &Image<B>,
    img2: &Image<B>,
) -> Result<[[f64; 3]; 3]> {
    let detector = FeatureDetector::new(FeatureType::ORB).with_max_features(500);

    let kps1 = detector.detect(img1)?;
    let kps2 = detector.detect(img2)?;

    if kps1.len() < 4 || kps2.len() < 4 {
        return Ok([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    }

    let desc1 = detector.compute(img1, &kps1)?;
    let desc2 = detector.compute(img2, &kps2)?;

    let matcher = BFMatcher;
    let matches = matcher.match_descriptors(&desc1, &desc2)?;

    // Filter matches with Lowe's ratio test (approximate with distance threshold)
    let median_dist = {
        let mut dists: Vec<f32> = matches.iter().map(|m| m.distance).collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        dists[dists.len() / 2]
    };
    let threshold = median_dist * 1.5;

    let good_matches: Vec<_> = matches
        .iter()
        .filter(|m| m.distance <= threshold)
        .collect();

    if good_matches.len() < 4 {
        return Err(IrisError::InvalidParameter(
            "Not enough good matches for homography estimation".into(),
        ));
    }

    // Collect matched point pairs
    let pts1: Vec<(f64, f64)> = good_matches
        .iter()
        .map(|m| {
            let kp = &kps1[m.query_idx];
            (kp.pt.x, kp.pt.y)
        })
        .collect();
    let pts2: Vec<(f64, f64)> = good_matches
        .iter()
        .map(|m| {
            let kp = &kps2[m.train_idx];
            (kp.pt.x, kp.pt.y)
        })
        .collect();

    // RANSAC homography estimation
    let h = ransac_homography(&pts1, &pts2, 1000, 5.0)?;
    Ok(h)
}

/// RANSAC-based homography estimation using DLT.
fn ransac_homography(
    src: &[(f64, f64)],
    dst: &[(f64, f64)],
    max_iterations: usize,
    inlier_threshold: f64,
) -> Result<[[f64; 3]; 3]> {
    let n = src.len();
    if n < 4 {
        return Err(IrisError::InvalidParameter(
            "Need at least 4 point pairs for homography".into(),
        ));
    }

    let mut best_h = [[0.0f64; 3]; 3];
    let mut best_inliers = 0;

    // Deterministic pseudo-random
    let mut seed: u64 = 0xDEAD_BEEF_CAFE_BABE;

    for _ in 0..max_iterations {
        // Pick 4 random distinct points
        let mut indices = [0usize; 4];
        let mut used = vec![false; n];
        let mut valid = true;
        for i in 0..4 {
            loop {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let idx = ((seed >> 33) as usize) % n;
                if !used[idx] {
                    used[idx] = true;
                    indices[i] = idx;
                    break;
                }
            }
            if indices[i] >= n {
                valid = false;
                break;
            }
        }
        if !valid {
            continue;
        }

        let sample_src: Vec<(f64, f64)> = indices.iter().map(|&i| src[i]).collect();
        let sample_dst: Vec<(f64, f64)> = indices.iter().map(|&i| dst[i]).collect();

        if let Ok(h_candidate) = compute_homography_dlt(&sample_src, &sample_dst) {
            // Count inliers
            let mut inlier_count = 0;
            for (s, d) in src.iter().zip(dst.iter()) {
                let projected = apply_homography(&h_candidate, s.0, s.1);
                let dx = projected.0 - d.0;
                let dy = projected.1 - d.1;
                if (dx * dx + dy * dy).sqrt() < inlier_threshold {
                    inlier_count += 1;
                }
            }

            if inlier_count > best_inliers {
                best_inliers = inlier_count;
                best_h = h_candidate;
            }
        }
    }

    if best_inliers == 0 {
        return Err(IrisError::Generic(
            "RANSAC homography failed to find any inliers".into(),
        ));
    }

    // Refine with all inliers
    let inlier_src: Vec<(f64, f64)> = src
        .iter()
        .zip(dst.iter())
        .filter_map(|(s, d)| {
            let projected = apply_homography(&best_h, s.0, s.1);
            let dx = projected.0 - d.0;
            let dy = projected.1 - d.1;
            if (dx * dx + dy * dy).sqrt() < inlier_threshold {
                Some(*s)
            } else {
                None
            }
        })
        .collect();
    let inlier_dst: Vec<(f64, f64)> = src
        .iter()
        .zip(dst.iter())
        .filter_map(|(s, d)| {
            let projected = apply_homography(&best_h, s.0, s.1);
            let dx = projected.0 - d.0;
            let dy = projected.1 - d.1;
            if (dx * dx + dy * dy).sqrt() < inlier_threshold {
                Some(*d)
            } else {
                None
            }
        })
        .collect();

    if inlier_src.len() >= 4 {
        compute_homography_dlt(&inlier_src, &inlier_dst)
    } else {
        Ok(best_h)
    }
}

/// Direct Linear Transform (DLT) homography computation from 4+ point correspondences.
/// Solves Ah = 0 via Gaussian elimination on the 2n×9 matrix A.
fn compute_homography_dlt(
    src: &[(f64, f64)],
    dst: &[(f64, f64)],
) -> Result<[[f64; 3]; 3]> {
    let n = src.len();
    if n < 4 {
        return Err(IrisError::InvalidParameter(
            "DLT requires at least 4 point pairs".into(),
        ));
    }

    // Normalize points (translate centroid to origin, scale to mean distance sqrt(2))
    let (src_norm, t_src) = normalize_points(src);
    let (dst_norm, t_dst) = normalize_points(dst);

    // Build 2n x 9 matrix A
    let mut a = vec![0.0f64; 2 * n * 9];
    for i in 0..n {
        let (x, y) = src_norm[i];
        let (xp, yp) = dst_norm[i];
        // Row 2i
        a[2 * i * 9] = -x;
        a[(2 * i) * 9 + 1] = -y;
        a[(2 * i) * 9 + 2] = -1.0;
        a[(2 * i) * 9 + 6] = xp * x;
        a[(2 * i) * 9 + 7] = xp * y;
        a[(2 * i) * 9 + 8] = xp;
        // Row 2i+1
        a[(2 * i + 1) * 9 + 3] = -x;
        a[(2 * i + 1) * 9 + 4] = -y;
        a[(2 * i + 1) * 9 + 5] = -1.0;
        a[(2 * i + 1) * 9 + 6] = yp * x;
        a[(2 * i + 1) * 9 + 7] = yp * y;
        a[(2 * i + 1) * 9 + 8] = yp;
    }

    // Find null space of A (2n x 9) via Gaussian elimination on A^T A (9 x 9)
    let ata = multiply_at_a(&a, 2 * n, 9);

    // Use Gaussian elimination with partial pivoting on the 9x9 system
    // to find the null space vector
    let h_vec = null_space_via_elimination(&ata, 9)?;

    let mut h_norm = [
        [h_vec[0], h_vec[1], h_vec[2]],
        [h_vec[3], h_vec[4], h_vec[5]],
        [h_vec[6], h_vec[7], h_vec[8]],
    ];

    // Denormalize: H = T_dst^{-1} * H_norm * T_src
    let t_dst_inv = invert_3x3(&t_dst).ok_or_else(|| {
        IrisError::InvalidParameter("Singular destination normalization transform".into())
    })?;
    let h_denorm = multiply_3x3(&t_dst_inv, &h_norm);
    h_norm = multiply_3x3(&h_denorm, &t_src);

    Ok(h_norm)
}

/// Find the null space vector of a symmetric n×n matrix via Gaussian elimination
/// with partial pivoting. The matrix is assumed to have a 1D null space.
fn null_space_via_elimination(m: &[f64], n: usize) -> Result<Vec<f64>> {
    // Work on a copy
    let mut mat = vec![0.0f64; n * n];
    mat.copy_from_slice(m);

    // Forward elimination with partial pivoting to get upper triangular form
    for col in 0..n {
        // Find pivot
        let mut max_val = mat[col * n + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if mat[row * n + col].abs() > max_val {
                max_val = mat[row * n + col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            continue;
        }
        // Swap rows
        for k in 0..n {
            mat.swap(col * n + k, max_row * n + k);
        }

        // Eliminate below
        let pivot = mat[col * n + col];
        for row in (col + 1)..n {
            let factor = mat[row * n + col] / pivot;
            for k in col..n {
                mat[row * n + k] -= factor * mat[col * n + k];
            }
        }
    }

    // Now mat is upper triangular. Find the column with the smallest pivot
    // (the near-zero eigenvalue).
    let mut free_col = 0;
    let mut min_pivot = f64::MAX;
    for col in 0..n {
        let piv = mat[col * n + col].abs();
        if piv < min_pivot {
            min_pivot = piv;
            free_col = col;
        }
    }

    // Back-substitute to find the null space vector
    // Set v[free_col] = 1, solve for the rest
    let mut v = vec![0.0f64; n];
    v[free_col] = 1.0;

    // Process rows from bottom to top (excluding the free_col row)
    for row in (0..n).rev() {
        if row == free_col {
            continue;
        }
        // mat[row][row] * v[row] + sum_{j>row} mat[row][j] * v[j] = 0
        let mut sum = 0.0;
        for j in (row + 1)..n {
            sum += mat[row * n + j] * v[j];
        }
        if mat[row * n + row].abs() > 1e-15 {
            v[row] = -sum / mat[row * n + row];
        }
    }

    // Normalize
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-15 {
        for x in &mut v {
            *x /= norm;
        }
    }

    Ok(v)
}

/// Normalize point set: translate centroid to origin, scale to mean distance sqrt(2).
fn normalize_points(points: &[(f64, f64)]) -> (Vec<(f64, f64)>, [[f64; 3]; 3]) {
    let n = points.len() as f64;
    let cx = points.iter().map(|p| p.0).sum::<f64>() / n;
    let cy = points.iter().map(|p| p.1).sum::<f64>() / n;

    let mean_dist = points
        .iter()
        .map(|p| ((p.0 - cx).powi(2) + (p.1 - cy).powi(2)).sqrt())
        .sum::<f64>()
        / n;

    let scale = if mean_dist > 1e-10 {
        std::f64::consts::SQRT_2 / mean_dist
    } else {
        1.0
    };

    let normalized: Vec<(f64, f64)> = points
        .iter()
        .map(|p| ((p.0 - cx) * scale, (p.1 - cy) * scale))
        .collect();

    let t = [
        [scale, 0.0, -cx * scale],
        [0.0, scale, -cy * scale],
        [0.0, 0.0, 1.0],
    ];

    (normalized, t)
}

/// Multiply A^T * A to get a 9x9 symmetric matrix.
fn multiply_at_a(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut ata = vec![0.0f64; cols * cols];
    for i in 0..cols {
        for j in 0..cols {
            let mut sum = 0.0f64;
            for k in 0..rows {
                sum += a[k * cols + i] * a[k * cols + j];
            }
            ata[i * cols + j] = sum;
        }
    }
    ata
}

/// Find the null space of a symmetric 9x9 matrix using Jacobi eigenvalue
/// decomposition. Returns the eigenvector corresponding to the smallest
/// eigenvalue.
#[allow(dead_code)]
fn null_space_jacobi(m: &[f64]) -> Vec<f64> {
    let n = 9;
    let mut a = m.to_vec();
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    for _ in 0..200 {
        // Find largest off-diagonal element
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[i * n + j].abs() > max_val {
                    max_val = a[i * n + j].abs();
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < 1e-12 {
            break;
        }

        let diff = a[p * n + p] - a[q * n + q];
        let theta = if diff.abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * ((2.0 * a[p * n + q]) / diff).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Save rows p and q
        let mut row_p = [0.0f64; 9];
        let mut row_q = [0.0f64; 9];
        for j in 0..n {
            row_p[j] = a[p * n + j];
            row_q[j] = a[q * n + j];
        }

        // Update columns p and q for all rows (except p and q)
        for i in 0..n {
            if i == p || i == q {
                continue;
            }
            let aip = a[i * n + p];
            let aiq = a[i * n + q];
            a[i * n + p] = c * aip + s * aiq;
            a[i * n + q] = -s * aip + c * aiq;
        }

        // Update rows p and q
        for j in 0..n {
            a[p * n + j] = c * row_p[j] + s * row_q[j];
            a[q * n + j] = -s * row_p[j] + c * row_q[j];
        }
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;

        // Update eigenvectors: V' = V * G
        for i in 0..n {
            let vip = v[i * n + p];
            let viq = v[i * n + q];
            v[i * n + p] = c * vip + s * viq;
            v[i * n + q] = -s * vip + c * viq;
        }
    }

    // Find eigenvector with smallest eigenvalue
    let mut min_idx = 0;
    let mut min_val = a[0];
    for i in 1..n {
        if a[i * n + i] < min_val {
            min_val = a[i * n + i];
            min_idx = i;
        }
    }

    (0..n).map(|i| v[i * n + min_idx]).collect()
}

/// Solve a 9x9 linear system using Gaussian elimination with partial pivoting.
#[allow(dead_code)]
fn solve_9x9(a: &[f64], b: &[f64]) -> Option<Vec<f64>> {
    let n = 9;
    let mut aug = vec![vec![0.0f64; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i * n + j];
        }
        aug[i][n] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return None;
        }
        aug.swap(col, max_row);

        // Eliminate below
        for row in (col + 1)..n {
            let factor = aug[row][col] / aug[col][col];
            for k in col..=n {
                aug[row][k] -= factor * aug[col][k];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        if aug[i][i].abs() < 1e-15 {
            return None;
        }
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    Some(x)
}

/// Multiply two 3x3 matrices.
fn multiply_3x3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut result = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

/// Invert a 3x3 matrix using cofactor expansion.
fn invert_3x3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if det.abs() < 1e-12 {
        return None;
    }

    let inv_det = 1.0 / det;
    let inv = [
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
    Some(inv)
}

/// Apply a 3x3 homography to a 2D point.
fn apply_homography(h: &[[f64; 3]; 3], x: f64, y: f64) -> (f64, f64) {
    let denom = h[2][0] * x + h[2][1] * y + h[2][2];
    if denom.abs() < 1e-10 {
        return (x, y);
    }
    let wx = (h[0][0] * x + h[0][1] * y + h[0][2]) / denom;
    let wy = (h[1][0] * x + h[1][1] * y + h[1][2]) / denom;
    (wx, wy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{TestBackend, test_device};
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn test_stitching_identical_images() {
        let device = test_device();

        // Two identical images — homography should be near identity
        let flat_data = vec![0.5f32; 3 * 16 * 16];
        let img = Image::new(
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [3, 16, 16]), &device),
        );

        let stitcher = Stitcher;
        let stitched = stitcher.stitch(&[img.clone(), img]).unwrap();
        assert_eq!(stitched.shape()[0], 3);
        assert!(stitched.shape()[1] > 0);
        assert!(stitched.shape()[2] > 0);
    }

    #[test]
    fn test_stitching_single_image() {
        let device = test_device();
        let flat_data = vec![0.3f32; 3 * 8 * 8];
        let img = Image::new(
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [3, 8, 8]), &device),
        );

        let stitcher = Stitcher;
        let result = stitcher.stitch(std::slice::from_ref(&img)).unwrap();
        assert_eq!(result.shape(), [3, 8, 8]);
    }

    #[test]
    fn test_stitching_empty_input() {
        let stitcher = Stitcher;
        let empty: Vec<Image<TestBackend>> = vec![];
        assert!(stitcher.stitch(&empty).is_err());
    }

    #[test]
    fn test_homography_identity() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let mut h = compute_homography_dlt(&pts, &pts).unwrap();

        // Normalize by h[2][2] (homographies are scale-invariant)
        let s = h[2][2];
        if s.abs() > 1e-10 {
            for row in &mut h {
                for val in row.iter_mut() {
                    *val /= s;
                }
            }
        }

        // Should approximate identity
        assert!((h[0][0] - 1.0).abs() < 0.01, "h[0][0] = {}", h[0][0]);
        assert!((h[1][1] - 1.0).abs() < 0.01, "h[1][1] = {}", h[1][1]);
        assert!((h[2][2] - 1.0).abs() < 0.01, "h[2][2] = {}", h[2][2]);
        assert!(h[0][1].abs() < 0.01);
        assert!(h[0][2].abs() < 0.01);
    }

    #[test]
    fn test_invert_3x3() {
        let m = [
            [2.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ];
        let inv = invert_3x3(&m).unwrap();
        let product = multiply_3x3(&m, &inv);

        // Product should be identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product[i][j] - expected).abs() < 1e-10,
                    "({}, {}) = {} expected {}",
                    i,
                    j,
                    product[i][j],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_solve_9x9() {
        // Solve a simple system: 3x=6 → x=2
        let mut a = vec![0.0f64; 81];
        for i in 0..9 {
            a[i * 9 + i] = 1.0;
        }
        let b = vec![1.0f64; 9];
        let x = solve_9x9(&a, &b).unwrap();
        for v in &x {
            assert!((v - 1.0).abs() < 1e-10);
        }
    }
}
