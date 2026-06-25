---
title: "Kalman Filter Reference"
description: "API reference for Iris Kalman filter — discrete linear Kalman filter for state estimation and tracking."
keywords: ["kalman filter", "state estimation", "tracking", "prediction", "measurement"]
canonical: "https://muhammad-fiaz.github.io/iris/api/kalman"
---

# Kalman Filter Reference

Discrete linear Kalman filter for state estimation and object tracking.

::: note
This module is under active development. API signatures may change between versions.
:::

## KalmanFilter

```rust
pub struct KalmanFilter {
    state: Vec<f64>,
    covariance: Vec<f64>,
    transition: Vec<f64>,
    observation: Vec<f64>,
    process_noise: Vec<f64>,
    measurement_noise: Vec<f64>,
}

impl KalmanFilter {
    pub fn new(
        dim_state: usize,
        dim_measurement: usize,
    ) -> Self;

    pub fn predict(&mut self) -> &Vec<f64>;
    pub fn update(&mut self, measurement: &[f64]) -> &Vec<f64>;
    pub fn state(&self) -> &[f64];
    pub fn covariance(&self) -> &[f64];
    pub fn set_transition(&mut self, matrix: Vec<f64>);
    pub fn set_observation(&mut self, matrix: Vec<f64>);
    pub fn set_process_noise(&mut self, matrix: Vec<f64>);
    pub fn set_measurement_noise(&mut self, matrix: Vec<f64>);
}
```

### Constructor

#### `new(dim_state, dim_measurement)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `dim_state` | `usize` | Dimension of the state vector. |
| `dim_measurement` | `usize` | Dimension of the measurement vector. |

Initializes all matrices to identity or zero. Call the setters to configure transition and noise models.

### Methods

#### `predict()`

Advances the state estimate forward using the transition model.

**Returns:** Reference to the predicted state vector.

#### `update(measurement)`

Incorporates a new measurement to correct the predicted state.

| Parameter | Type | Description |
|-----------|------|-------------|
| `measurement` | `&[f64]` | Observed measurement vector. |

**Returns:** Reference to the updated (a posteriori) state vector.

#### `state()`

Returns the current state estimate as `&[f64]`.

#### `covariance()`

Returns the current error covariance matrix as a flat `&[f64]`.

## Example

```rust
use iris::prelude::*;

// Track a 2D point: state = [x, y, vx, vy], measurement = [x, y]
let mut kf = KalmanFilter::new(4, 2);

// Configure linear motion model (dt = 1.0)
kf.set_transition(vec![
    1.0, 0.0, 1.0, 0.0,
    0.0, 1.0, 0.0, 1.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
]);

// Observe positions
for measurement in &[[1.0, 2.0], [2.1, 3.9], [3.0, 6.2], [4.1, 7.8]] {
    kf.predict();
    kf.update(measurement);
    let s = kf.state();
    println!("Estimated: ({:.2}, {:.2})", s[0], s[1]);
}
```
