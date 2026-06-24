# Compute & Filters Reference

Details image filters, edge extraction, thresholding, and morphological operation signatures.

## Filters

```rust
impl<B: Backend> Image<B> {
    pub fn box_blur(self, kernel_size: usize) -> Result<Self>;
    pub fn gaussian_blur(self, kernel_size: usize, sigma: f64) -> Result<Self>;
    pub fn median_blur(self, kernel_size: usize) -> Result<Self>;
    pub fn bilateral_filter(self, d: usize, sigma_color: f32, sigma_space: f64) -> Result<Self>;
    pub fn sep_filter_2d(&self, kernel_x: &[f32], kernel_y: &[f32]) -> Result<Self>;
}
```

## Gradients & Edge Detection

```rust
impl<B: Backend> Image<B> {
    pub fn sobel(&self, kernel_size: usize) -> Result<(Self, Self)>;
    pub fn scharr(&self) -> Result<(Self, Self)>;
    pub fn laplacian(&self, kernel_size: usize) -> Result<Self>;
    pub fn canny(&self, low_threshold: f32, high_threshold: f32) -> Result<Self>;
}
```

## Thresholding

```rust
impl<B: Backend> Image<B> {
    pub fn threshold(&self, thresh: f32, maxval: f32, thresh_type: ThresholdType) -> Result<Self>;
    pub fn threshold_otsu(&self, maxval: f32) -> Result<Self>;
}
```

## Morphology

```rust
impl<B: Backend> Image<B> {
    pub fn dilate(self, kernel_size: usize) -> Result<Self>;
    pub fn erode(self, kernel_size: usize) -> Result<Self>;
    pub fn morph_open(self, kernel_size: usize) -> Result<Self>;
    pub fn morph_close(self, kernel_size: usize) -> Result<Self>;
    pub fn morphology_ex(&self, op: MorphOp, kernel_size: usize) -> Result<Self>;
}
```
