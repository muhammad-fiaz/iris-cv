# Core Module Reference

The `core` module contains common algebraic representations, basic geometries, and utility structures.

## Geometries

### `Point<T>`
A structure representing 2D pixel or spatial coordinates.

```rust
pub struct Point<T> {
    pub x: T,
    pub y: T,
}

impl<T> Point<T> {
    pub fn new(x: T, y: T) -> Self;
}
```

### `Rect<T>`
A structure defining a 2D straight rectangle bounds.

```rust
pub struct Rect<T> {
    pub x: T,
    pub y: T,
    pub width: T,
    pub height: T,
}

impl<T> Rect<T> {
    pub fn new(x: T, y: T, width: T, height: T) -> Self;
}
```

### `Size<T>`
A structure defining X/Y spatial dimensions.

```rust
pub struct Size<T> {
    pub width: T,
    pub height: T,
}

impl<T> Size<T> {
    pub fn new(width: T, height: T) -> Self;
}
```

### `Scalar`
A structure representing a 4-element double array (typically used for colors).

```rust
pub struct Scalar(pub [f64; 4]);

impl Scalar {
    pub fn new(v0: f64, v1: f64, v2: f64, v3: f64) -> Self;
    pub fn all(val: f64) -> Self;
}
```

## Matrix Representation

### `Mat<B, const D: usize>`
Wraps a multi-dimensional Burn `Tensor` to expose standard shape query methods.

```rust
pub struct Mat<B: Backend, const D: usize> {
    pub tensor: Tensor<B, D>,
}

impl<B: Backend, const D: usize> Mat<B, D> {
    pub fn new(tensor: Tensor<B, D>) -> Self;
    pub fn shape(&self) -> Vec<usize>;
    pub fn total(&self) -> usize;
}
```
