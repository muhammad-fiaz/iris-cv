# Edge Detection & Gradients

Gradients and edge maps highlight structural changes and borders within images. Observers provides standard first and second-order image derivative approximations.

## Gradient Operators

### Sobel
Calculates first-order derivatives along the X and Y axes.

```rust
// Compute Sobel derivatives
let (sobel_x, sobel_y) = image.sobel(3)?;
```

### Scharr
Provides more accurate derivative estimations compared to the Sobel operator.

```rust
// Compute Scharr derivatives
let (scharr_x, scharr_y) = image.scharr()?;
```

### Laplacian
Calculates the second-order derivative (sum of second spatial derivatives), emphasizing high-frequency regions.

```rust
// Compute Laplacian gradient
let laplacian = image.laplacian(3)?;
```

## Canny Edge Detection

Canny edge detection is a multi-stage process that reduces noise, computes gradients, performs non-maximum suppression, and uses hysteresis thresholding.

```rust
// Run Canny edge detector with low/high threshold ratio
let edges = image.canny(0.1, 0.3)?;
```
