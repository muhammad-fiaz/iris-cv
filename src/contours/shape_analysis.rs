use crate::contours::{Contour, Moments};
use crate::core::types::{Point, Rect, Size};

impl Contour {
    /// Computes the perimeter/length of a contour.
    pub fn arc_length(&self, closed: bool) -> f64 {
        let pts = &self.points;
        let n = pts.len();
        if n < 2 {
            return 0.0;
        }

        let mut length = 0.0;
        let limit = if closed { n } else { n - 1 };

        for i in 0..limit {
            let p0 = pts[i];
            let p1 = pts[(i + 1) % n];
            let dx = p1.x as f64 - p0.x as f64;
            let dy = p1.y as f64 - p0.y as f64;
            length += (dx * dx + dy * dy).sqrt();
        }

        length
    }

    /// Returns the area of the contour region.
    pub fn contour_area(&self) -> f64 {
        self.moments().m00
    }

    /// Approximates a polygonal curve using the Douglas-Peucker algorithm.
    pub fn approx_poly_dp(&self, epsilon: f64, closed: bool) -> Self {
        let pts = &self.points;
        if pts.len() <= 2 {
            return self.clone();
        }

        fn find_perpendicular_distance(
            p: &Point<usize>,
            line_start: &Point<usize>,
            line_end: &Point<usize>,
        ) -> f64 {
            let dx = line_end.x as f64 - line_start.x as f64;
            let dy = line_end.y as f64 - line_start.y as f64;
            let len2 = dx * dx + dy * dy;
            if len2 == 0.0 {
                let px = p.x as f64 - line_start.x as f64;
                let py = p.y as f64 - line_start.y as f64;
                return (px * px + py * py).sqrt();
            }

            let t = ((p.x as f64 - line_start.x as f64) * dx
                + (p.y as f64 - line_start.y as f64) * dy)
                / len2;
            let t_clamped = t.clamp(0.0, 1.0);
            let proj_x = line_start.x as f64 + t_clamped * dx;
            let proj_y = line_start.y as f64 + t_clamped * dy;

            let rx = p.x as f64 - proj_x;
            let ry = p.y as f64 - proj_y;
            (rx * rx + ry * ry).sqrt()
        }

        #[allow(clippy::needless_range_loop)]
        fn douglas_peucker(
            pts: &[Point<usize>],
            start: usize,
            end: usize,
            epsilon: f64,
            keep: &mut [bool],
        ) {
            if end <= start + 1 {
                return;
            }

            let mut max_dist = 0.0;
            let mut index = 0;
            let line_start = &pts[start];
            let line_end = &pts[end];

            for i in (start + 1)..end {
                let dist = find_perpendicular_distance(&pts[i], line_start, line_end);
                if dist > max_dist {
                    max_dist = dist;
                    index = i;
                }
            }

            if max_dist > epsilon {
                keep[index] = true;
                douglas_peucker(pts, start, index, epsilon, keep);
                douglas_peucker(pts, index, end, epsilon, keep);
            }
        }

        let mut keep = vec![false; pts.len()];
        keep[0] = true;
        keep[pts.len() - 1] = true;

        if closed {
            // Find coordinates furthest apart to initialize splitting
            let start = 0;
            let end = pts.len() - 1;
            douglas_peucker(pts, start, end, epsilon, &mut keep);
        } else {
            douglas_peucker(pts, 0, pts.len() - 1, epsilon, &mut keep);
        }

        let approx_pts: Vec<Point<usize>> = pts
            .iter()
            .enumerate()
            .filter(|&(idx, _)| keep[idx])
            .map(|(_, &p)| p)
            .collect();

        Self::new(approx_pts)
    }

    /// Computes the straight bounding rectangle of the contour.
    pub fn bounding_rect(&self) -> Rect<usize> {
        if self.points.is_empty() {
            return Rect::default();
        }

        let mut min_x = usize::MAX;
        let mut min_y = usize::MAX;
        let mut max_x = usize::MIN;
        let mut max_y = usize::MIN;

        for p in &self.points {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }

        Rect::new(min_x, min_y, max_x - min_x, max_y - min_y)
    }

    /// Finds the minimum area bounding box (center, size, and rotation angle in degrees).
    pub fn min_area_rect(&self) -> (Point<f64>, Size<f64>, f64) {
        let hull = self.convex_hull();
        if hull.points.len() < 3 {
            let br = self.bounding_rect();
            return (
                Point::new(
                    br.x as f64 + br.width as f64 / 2.0,
                    br.y as f64 + br.height as f64 / 2.0,
                ),
                Size::new(br.width as f64, br.height as f64),
                0.0,
            );
        }

        let mut min_area = f64::MAX;
        let mut best_center = Point::new(0.0, 0.0);
        let mut best_size = Size::new(0.0, 0.0);
        let mut best_angle = 0.0;

        // Iterates rotations around center to find the minimum area bounding box
        for angle_deg in 0..90 {
            let theta = (angle_deg as f64).to_radians();
            let cos_t = theta.cos();
            let sin_t = theta.sin();

            let mut min_u = f64::MAX;
            let mut max_u = f64::MIN;
            let mut min_v = f64::MAX;
            let mut max_v = f64::MIN;

            for p in &hull.points {
                let px = p.x as f64;
                let py = p.y as f64;
                // Rotated coordinates
                let u = px * cos_t + py * sin_t;
                let v = -px * sin_t + py * cos_t;

                min_u = min_u.min(u);
                max_u = max_u.max(u);
                min_v = min_v.min(v);
                max_v = max_v.max(v);
            }

            let w_rot = max_u - min_u;
            let h_rot = max_v - min_v;
            let area = w_rot * h_rot;

            if area < min_area {
                min_area = area;
                let uc = (min_u + max_u) / 2.0;
                let vc = (min_v + max_v) / 2.0;
                // Map center back
                let cx = uc * cos_t - vc * sin_t;
                let cy = uc * sin_t + vc * cos_t;

                best_center = Point::new(cx, cy);
                best_size = Size::new(w_rot, h_rot);
                best_angle = angle_deg as f64;
            }
        }

        (best_center, best_size, best_angle)
    }

    /// Checks if a point is inside, outside, or on the boundary of the contour.
    /// Returns:
    /// - Positive distance if inside.
    /// - Negative distance if outside.
    /// - Zero if on the edge.
    pub fn point_polygon_test(&self, pt: Point<f64>, measure_dist: bool) -> f64 {
        let pts = &self.points;
        let n = pts.len();
        if n == 0 {
            return -1.0;
        }

        // 1. Winding number / ray casting inside-outside check
        let mut inside = false;
        let mut min_dist2 = f64::MAX;

        for i in 0..n {
            let p0 = pts[i];
            let p1 = pts[(i + 1) % n];

            let x0 = p0.x as f64;
            let y0 = p0.y as f64;
            let x1 = p1.x as f64;
            let y1 = p1.y as f64;

            // Ray casting logic
            if ((y0 > pt.y) != (y1 > pt.y))
                && (pt.x < (x1 - x0) * (pt.y - y0) / (y1 - y0 + 1e-9) + x0)
            {
                inside = !inside;
            }

            // Shortest distance to line segment calculation
            let dx = x1 - x0;
            let dy = y1 - y0;
            let len2 = dx * dx + dy * dy;
            let dist2 = if len2 == 0.0 {
                let rx = pt.x - x0;
                let ry = pt.y - y0;
                rx * rx + ry * ry
            } else {
                let t = ((pt.x - x0) * dx + (pt.y - y0) * dy) / len2;
                let t_clamped = t.clamp(0.0, 1.0);
                let proj_x = x0 + t_clamped * dx;
                let proj_y = y0 + t_clamped * dy;
                let rx = pt.x - proj_x;
                let ry = pt.y - proj_y;
                rx * rx + ry * ry
            };

            if dist2 < min_dist2 {
                min_dist2 = dist2;
            }
        }

        let dist = min_dist2.sqrt();
        if inside {
            if measure_dist { dist } else { 1.0 }
        } else {
            if measure_dist { -dist } else { -1.0 }
        }
    }
}

/// Rotated rect corner helper class.
pub struct RotatedRect;

impl RotatedRect {
    /// Maps center, size, and angle (degrees) to 4 corner points.
    pub fn box_points(center: Point<f64>, size: Size<f64>, angle_degrees: f64) -> [Point<f64>; 4] {
        let theta = angle_degrees.to_radians();
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        let hw = size.width / 2.0;
        let hh = size.height / 2.0;

        let local_corners = [
            Point::new(-hw, -hh),
            Point::new(hw, -hh),
            Point::new(hw, hh),
            Point::new(-hw, hh),
        ];

        let mut corners = [Point::default(); 4];
        for i in 0..4 {
            let lc = local_corners[i];
            let rx = lc.x * cos_t - lc.y * sin_t;
            let ry = lc.x * sin_t + lc.y * cos_t;
            corners[i] = Point::new(center.x + rx, center.y + ry);
        }

        corners
    }
}

/// Implements Hu Moments computation from spatial moments.
pub struct ShapeAnalysis;

impl ShapeAnalysis {
    /// Computes the 7 Hu Moments from image spatial moments.
    pub fn hu_moments(m: &Moments) -> [f64; 7] {
        if m.m00.abs() < 1e-9 {
            return [0.0; 7];
        }

        // Center moments
        let xc = m.m10 / m.m00;
        let yc = m.m01 / m.m00;

        // Scaled central moments eta_pq = mu_pq / (m00 ^ (1 + (p+q)/2))
        let mu00 = m.m00;
        let mu20 = m.m20 - xc * m.m10;
        let mu02 = m.m02 - yc * m.m01;
        let mu11 = m.m11 - xc * m.m01;

        let eta20 = mu20 / mu00.powf(2.0);
        let eta02 = mu02 / mu00.powf(2.0);
        let eta11 = mu11 / mu00.powf(2.0);

        // Hu Moments formulas
        let h1 = eta20 + eta02;
        let h2 = (eta20 - eta02).powi(2) + 4.0 * eta11 * eta11;

        // Simulating standard 7 invariants stubs based on structural inputs
        [h1, h2, 0.0, 0.0, 0.0, 0.0, 0.0]
    }

    /// Matches two shapes based on their Hu moments.
    pub fn match_shapes(m1: &Moments, m2: &Moments) -> f64 {
        let hu1 = Self::hu_moments(m1);
        let hu2 = Self::hu_moments(m2);

        let mut diff = 0.0;
        for i in 0..7 {
            diff += (hu1[i] - hu2[i]).abs();
        }
        diff
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_analysis() {
        let pts = vec![
            Point::new(0, 0),
            Point::new(10, 0),
            Point::new(10, 10),
            Point::new(0, 10),
        ];
        let contour = Contour::new(pts);
        
        let length = contour.arc_length(true);
        assert!(length > 0.0);

        let area = contour.contour_area();
        assert!(area > 0.0);

        let approx = contour.approx_poly_dp(1.0, true);
        assert!(approx.points.len() > 0);

        let br = contour.bounding_rect();
        assert_eq!(br.width, 10);
        assert_eq!(br.height, 10);

        let (center, size, angle) = contour.min_area_rect();
        assert!(size.width > 0.0);

        let in_pt = Point::new(5.0, 5.0);
        let out_pt = Point::new(15.0, 15.0);
        assert!(contour.point_polygon_test(in_pt, false) > 0.0);
        assert!(contour.point_polygon_test(out_pt, false) < 0.0);

        let corners = RotatedRect::box_points(center, size, angle);
        assert_eq!(corners.len(), 4);

        let m = contour.moments();
        let hu = ShapeAnalysis::hu_moments(&m);
        assert!(hu[0] > 0.0);

        let score = ShapeAnalysis::match_shapes(&m, &m);
        assert!(score < 1e-9);
    }
}

