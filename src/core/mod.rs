pub mod mat;
pub mod rng;
pub mod types;
pub mod utils;

pub use mat::Mat;
pub use rng::Rng;
pub use types::{Point, Rect, Scalar, Size};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng() {
        let mut rng = Rng::new(42);
        let val1 = rng.next_u32();
        let val2 = rng.next_u32();
        assert_ne!(val1, val2);

        let f = rng.next_f32();
        assert!((0.0..1.0).contains(&f));
    }

    #[test]
    fn test_types() {
        let p = Point::new(10, 20);
        assert_eq!(p.x, 10);
        assert_eq!(p.y, 20);

        let r = Rect::new(0, 0, 100, 200);
        assert_eq!(r.width, 100);
        assert_eq!(r.height, 200);

        let s = Size::new(800, 600);
        assert_eq!(s.width, 800);
        assert_eq!(s.height, 600);

        let sc = Scalar::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(sc.0[0], 1.0);
        assert_eq!(sc.0[3], 4.0);
    }
}
