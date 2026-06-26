---
title: "Kalman Filter Reference"
description: "API reference for Iris Kalman filter — discrete Kalman filter for state estimation and prediction."
keywords: ["Kalman filter", "state estimation", "prediction", "tracking"]
canonical: "https://muhammad-fiaz.github.io/iris-cv/api/kalman"
---

# Kalman Filter Reference

Implements a discrete Kalman filter for linear state estimation and prediction.

::: note
This module is under active development. API signatures may change between versions.
:::

## KalmanFilter

```rust
pub struct KalmanFilter {
    state_dim: usize,
    measure_dim: usize,
    F: Vec<Vec<f64>>,   // State transition matrix
    H: Vec<Vec<f64>>,   // Observation matrix
    Q: Vec<Vec<f64>>,   // Process noise covariance
    R: Vec<Vec<f64>>,   // Measurement noise covariance
    x: Vec<f64>,        // State vector
    P: Vec<Vec<f64>>,   // State covariance matrix
}

impl KalmanFilter {
    pub fn new(state_dim: usize, measure_dim: usize) -> Self;
    pub fn set_transition(&mut self, f: Vec<Vec<f64>>) -> Result<()>;
    pub fn set_measurement(&mut self, h: Vec<Vec<f64>>) -> Result<()>;
    pub fn set_process_noise(&mut self, q: Vec<Vec<f64>>) -> Result<()>;
    pub fn set_measurement_noise(&mut self, r: Vec<Vec<f64>>) -> Result<()>;
    pub fn set_initial_state(&mut self, x: Vec<f64>) -> Result<()>;
    pub fn predict(&mut self);
    pub fn update(&mut self, z: &[f64]) -> Result<()>;
    pub fn state(&self) -> &[f64];
    pub fn covariance(&self) -> &[Vec<f64>];
    pub fn covariance_flat(&self) -> Vec<f64>;
}
```

### Example

```rust
use iris::prelude::*;

// 1D position tracking: state = [position, velocity], measurement = [position]
let mut kf = KalmanFilter::new(2, 1);

// State transition: position += velocity * dt
kf.set_transition(vec![vec![1.0, 1.0], vec![0.0, 1.0]])?;

// Observation: we only measure position
kf.set_measurement(vec![vec![1.0, 0.0]])?;

// Process and measurement noise
kf.set_process_noise(vec![vec![0.1, 0.0], vec![0.0, 0.1]])?;
kf.set_measurement_noise(vec![vec![1.0]])?;

// Initial state
kf.set_initial_state(vec![0.0, 1.0])?;

// Predict and update loop
kf.predict();
kf.update(&[1.0])?;
println!("Estimated state: {:?}", kf.state());
```
