/// Represents a 4-element scalar value (commonly used for colors or bounds).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scalar(pub [f64; 4]);

impl Scalar {
    /// Creates a new Scalar with the given values.
    pub fn new(v0: f64, v1: f64, v2: f64, v3: f64) -> Self {
        Scalar([v0, v1, v2, v3])
    }

    /// Creates a new Scalar with all values set to the same value.
    pub fn all(val: f64) -> Self {
        Scalar([val, val, val, val])
    }
}

impl Default for Scalar {
    fn default() -> Self {
        Scalar([0.0, 0.0, 0.0, 0.0])
    }
}

/// Represents a 2D point with coordinates of type T.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Point<T> {
    pub x: T,
    pub y: T,
}

impl<T> Point<T> {
    /// Creates a new Point.
    pub fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

/// Represents a 2D rectangle with coordinates, width, and height of type T.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Rect<T> {
    pub x: T,
    pub y: T,
    pub width: T,
    pub height: T,
}

impl<T> Rect<T> {
    /// Creates a new Rect.
    pub fn new(x: T, y: T, width: T, height: T) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

/// Represents a 2D size (width and height) of type T.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Size<T> {
    pub width: T,
    pub height: T,
}

impl<T> Size<T> {
    /// Creates a new Size.
    pub fn new(width: T, height: T) -> Self {
        Self { width, height }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_types() {
        let p = Point::new(10, 20);
        assert_eq!(p.x, 10);
        assert_eq!(p.y, 20);

        let r = Rect::new(1, 2, 3, 4);
        assert_eq!(r.x, 1);
        assert_eq!(r.y, 2);
        assert_eq!(r.width, 3);
        assert_eq!(r.height, 4);

        let s = Size::new(100, 200);
        assert_eq!(s.width, 100);
        assert_eq!(s.height, 200);

        let col = Scalar::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(col.0[0], 1.0);
        assert_eq!(col.0[3], 4.0);

        let col_all = Scalar::all(5.0);
        assert_eq!(col_all.0, [5.0, 5.0, 5.0, 5.0]);
        
        let col_def = Scalar::default();
        assert_eq!(col_def.0, [0.0, 0.0, 0.0, 0.0]);
    }
}

