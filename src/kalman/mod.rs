#![allow(non_snake_case)]

use crate::error::{IrisError, Result};

/// A discrete linear Kalman filter for tracking state estimates through noisy measurements.
///
/// The state evolves as:  x(k) = F * x(k-1) + B * u(k) + w(k),  w ~ N(0, Q)
/// Measurements are:      z(k) = H * x(k) + v(k),                 v ~ N(0, R)
pub struct KalmanFilter {
    /// State dimension.
    state_dim: usize,
    /// Measurement dimension.
    measure_dim: usize,
    /// State transition matrix [state_dim x state_dim].
    F: Vec<Vec<f64>>,
    /// Measurement matrix [measure_dim x state_dim].
    H: Vec<Vec<f64>>,
    /// Process noise covariance [state_dim x state_dim].
    Q: Vec<Vec<f64>>,
    /// Measurement noise covariance [measure_dim x measure_dim].
    R: Vec<Vec<f64>>,
    /// Current state estimate [state_dim x 1].
    x: Vec<f64>,
    /// Current error covariance [state_dim x state_dim].
    P: Vec<Vec<f64>>,
}

impl KalmanFilter {
    /// Creates a new Kalman filter with the given state and measurement dimensions.
    ///
    /// All matrices are initialized to sensible defaults:
    /// - `F` = identity
    /// - `H` = identity (top-left sub-block) or zero-padded
    /// - `Q`, `R` = small diagonal noise
    /// - `x` = zero vector
    /// - `P` = identity (high initial uncertainty)
    pub fn new(state_dim: usize, measure_dim: usize) -> Self {
        let F = Self::eye(state_dim);
        let H = Self::default_h(state_dim, measure_dim);
        let Q = Self::scaled_eye(state_dim, 1e-2);
        let R = Self::scaled_eye(measure_dim, 1e-1);
        let x = vec![0.0; state_dim];
        let P = Self::eye(state_dim);

        Self {
            state_dim,
            measure_dim,
            F,
            H,
            Q,
            R,
            x,
            P,
        }
    }

    /// Sets the state transition matrix `F`.
    pub fn set_transition(&mut self, f: Vec<Vec<f64>>) -> Result<()> {
        if f.len() != self.state_dim || f.iter().any(|row| row.len() != self.state_dim) {
            return Err(IrisError::InvalidParameter(format!(
                "F must be [{0} x {0}]",
                self.state_dim
            )));
        }
        self.F = f;
        Ok(())
    }

    /// Sets the measurement matrix `H`.
    pub fn set_measurement(&mut self, h: Vec<Vec<f64>>) -> Result<()> {
        if h.len() != self.measure_dim || h.iter().any(|row| row.len() != self.state_dim) {
            return Err(IrisError::InvalidParameter(format!(
                "H must be [{0} x {1}]",
                self.measure_dim, self.state_dim
            )));
        }
        self.H = h;
        Ok(())
    }

    /// Sets the process noise covariance `Q`.
    pub fn set_process_noise(&mut self, q: Vec<Vec<f64>>) -> Result<()> {
        if q.len() != self.state_dim || q.iter().any(|row| row.len() != self.state_dim) {
            return Err(IrisError::InvalidParameter(format!(
                "Q must be [{0} x {0}]",
                self.state_dim
            )));
        }
        self.Q = q;
        Ok(())
    }

    /// Sets the measurement noise covariance `R`.
    pub fn set_measurement_noise(&mut self, r: Vec<Vec<f64>>) -> Result<()> {
        if r.len() != self.measure_dim || r.iter().any(|row| row.len() != self.measure_dim) {
            return Err(IrisError::InvalidParameter(format!(
                "R must be [{0} x {0}]",
                self.measure_dim
            )));
        }
        self.R = r;
        Ok(())
    }

    /// Sets the initial state vector.
    pub fn set_initial_state(&mut self, x: Vec<f64>) -> Result<()> {
        if x.len() != self.state_dim {
            return Err(IrisError::InvalidParameter(format!(
                "State vector must have length {0}",
                self.state_dim
            )));
        }
        self.x = x;
        Ok(())
    }

    /// Prediction step: x̂ = F * x̂,  P = F * P * F^T + Q.
    pub fn predict(&mut self) {
        // x = F * x
        self.x = mat_vec_mul(&self.F, &self.x);

        // P = F * P * F^T + Q
        let fp = mat_mul(&self.F, &self.P);
        let f_t = transpose(&self.F);
        let fpft = mat_mul(&fp, &f_t);
        self.P = mat_add(&fpft, &self.Q);
    }

    /// Update step incorporating a measurement z.
    ///
    /// Computes:
    /// - y   = z - H * x̂               (innovation)
    /// - S   = H * P * H^T + R         (innovation covariance)
    /// - K   = P * H^T * S^{-1}        (Kalman gain)
    /// - x̂   = x̂ + K * y              (updated state)
    /// - P   = (I - K * H) * P         (updated covariance)
    pub fn update(&mut self, z: &[f64]) -> Result<()> {
        if z.len() != self.measure_dim {
            return Err(IrisError::InvalidParameter(format!(
                "Measurement vector must have length {0}",
                self.measure_dim
            )));
        }

        // y = z - H * x
        let hx = mat_vec_mul(&self.H, &self.x);
        let y: Vec<f64> = z.iter().zip(hx.iter()).map(|(zi, hxi)| zi - hxi).collect();

        // S = H * P * H^T + R
        let hp = mat_mul(&self.H, &self.P);
        let ht = transpose(&self.H);
        let hpht = mat_mul(&hp, &ht);
        let s = mat_add(&hpht, &self.R);

        // K = P * H^T * S^{-1}
        let s_inv = inverse(&s)?;
        let pht = mat_mul(&self.P, &ht);
        let k = mat_mul(&pht, &s_inv);

        // x = x + K * y
        let ky = mat_vec_mul(&k, &y);
        self.x = vec_add(&self.x, &ky);

        // P = (I - K * H) * P
        let kh = mat_mul(&k, &self.H);
        let i_kh = mat_sub(&Self::eye(self.state_dim), &kh);
        self.P = mat_mul(&i_kh, &self.P);

        Ok(())
    }

    /// Returns the current state estimate.
    pub fn state(&self) -> &[f64] {
        &self.x
    }

    /// Returns the current error covariance matrix (flattened row-major).
    pub fn covariance_flat(&self) -> Vec<f64> {
        self.P.iter().flat_map(|row| row.iter().copied()).collect()
    }

    /// Returns a reference to the covariance matrix.
    pub fn covariance(&self) -> &[Vec<f64>] {
        &self.P
    }

    // ── Matrix helpers ──────────────────────────────────────────────

    fn eye(n: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect()
    }

    fn scaled_eye(n: usize, scale: f64) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| (0..n).map(|j| if i == j { scale } else { 0.0 }).collect())
            .collect()
    }

    fn default_h(state_dim: usize, measure_dim: usize) -> Vec<Vec<f64>> {
        let mut h = vec![vec![0.0; state_dim]; measure_dim];
        let m = measure_dim.min(state_dim);
        for i in 0..m {
            h[i][i] = 1.0;
        }
        h
    }
}

// ── Linear algebra functions for small matrices (Vec<Vec<f64>>) ────

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = b[0].len();
    let k = b.len();
    let mut result = vec![vec![0.0; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i][p] * b[p][j];
            }
            result[i][j] = sum;
        }
    }
    result
}

fn mat_vec_mul(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter()
        .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

fn mat_add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b.iter())
        .map(|(ra, rb)| ra.iter().zip(rb.iter()).map(|(ai, bi)| ai + bi).collect())
        .collect()
}

fn mat_sub(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b.iter())
        .map(|(ra, rb)| ra.iter().zip(rb.iter()).map(|(ai, bi)| ai - bi).collect())
        .collect()
}

fn transpose(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if m.is_empty() {
        return vec![];
    }
    let rows = m.len();
    let cols = m[0].len();
    (0..cols)
        .map(|j| (0..rows).map(|i| m[i][j]).collect())
        .collect()
}

fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai + bi).collect()
}

/// Invert a small square matrix via Gauss-Jordan elimination with partial pivoting.
fn inverse(m: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = m.len();
    if n == 0 || m.iter().any(|row| row.len() != n) {
        return Err(IrisError::InvalidParameter(
            "Matrix must be square and non-empty".into(),
        ));
    }

    // Build augmented matrix [M | I]
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = m[i].clone();
            row.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            row
        })
        .collect();

    for col in 0..n {
        // Partial pivoting: find row with largest absolute value in this column
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return Err(IrisError::Tensor(
                "Matrix is singular and cannot be inverted".into(),
            ));
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for j in 0..(2 * n) {
            aug[col][j] /= pivot;
        }

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in 0..(2 * n) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Extract inverse from right half
    let inv: Vec<Vec<f64>> = (0..n).map(|i| aug[i][n..(2 * n)].to_vec()).collect();

    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kalman_identity_tracking() {
        // 1D constant-position model: state = [position, velocity], measurement = [position]
        let mut kf = KalmanFilter::new(2, 1);

        // Set F for constant velocity: pos' = pos + vel, vel' = vel
        kf.set_transition(vec![vec![1.0, 1.0], vec![0.0, 1.0]])
            .unwrap();
        kf.set_measurement(vec![vec![1.0, 0.0]]).unwrap();
        kf.set_process_noise(vec![vec![1e-4, 0.0], vec![0.0, 1e-4]])
            .unwrap();
        kf.set_measurement_noise(vec![vec![0.1]]).unwrap();

        // True position is 0, moving at velocity 1.0 per step
        kf.set_initial_state(vec![0.0, 0.0]).unwrap();

        let mut last_pos = 0.0f64;
        for step in 1..=10 {
            kf.predict();

            // Noisy measurement of true position = step
            let true_pos = step as f64;
            let measurement = [true_pos + 0.05]; // small noise
            kf.update(&measurement).unwrap();

            let state = kf.state();
            // State should track near true position
            assert!(
                (state[0] - true_pos).abs() < 2.0,
                "Step {step}: expected ~{true_pos}, got {}",
                state[0]
            );
            last_pos = state[0];
        }
        // After 10 steps, last position estimate should be close to 10
        assert!((last_pos - 10.0).abs() < 2.0);
    }

    #[test]
    fn test_kalman_predict_only() {
        let mut kf = KalmanFilter::new(2, 1);
        kf.set_transition(vec![vec![1.0, 1.0], vec![0.0, 1.0]])
            .unwrap();
        kf.set_initial_state(vec![0.0, 1.0]).unwrap();

        kf.predict();
        let s = kf.state();
        // pos = 0 + 1 = 1, vel = 1
        assert!((s[0] - 1.0).abs() < 1e-10);
        assert!((s[1] - 1.0).abs() < 1e-10);

        kf.predict();
        let s = kf.state();
        // pos = 1 + 1 = 2, vel = 1
        assert!((s[0] - 2.0).abs() < 1e-10);
        assert!((s[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kalman_covariance_dims() {
        let kf = KalmanFilter::new(3, 2);
        let cov = kf.covariance_flat();
        assert_eq!(cov.len(), 9); // 3x3
    }

    #[test]
    fn test_inverse_2x2() {
        let m = vec![vec![4.0, 7.0], vec![2.0, 6.0]];
        let inv = inverse(&m).unwrap();
        // Check M * M^{-1} ≈ I
        let product = mat_mul(&m, &inv);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product[i][j] - expected).abs() < 1e-10,
                    "({i},{j}): got {}",
                    product[i][j]
                );
            }
        }
    }

    #[test]
    fn test_inverse_singular() {
        let m = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert!(inverse(&m).is_err());
    }

    #[test]
    fn test_measurement_mismatch() {
        let mut kf = KalmanFilter::new(2, 1);
        assert!(kf.update(&[1.0, 2.0]).is_err());
    }
}
