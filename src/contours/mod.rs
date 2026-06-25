pub mod shape_analysis;

pub use shape_analysis::{RotatedRect, ShapeAnalysis};

use crate::core::types::Point;
use crate::error::Result;
use crate::image::Image;
use burn::tensor::backend::Backend;

type HierarchyEntry = [i32; 4];
type ContourResult = (Vec<Vec<Point<usize>>>, Vec<HierarchyEntry>);

/// Represents a contour outline as a list of points.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Contour {
    pub points: Vec<Point<usize>>,
}

/// Image moments representing spatial distribution.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Moments {
    pub m00: f64,
    pub m10: f64,
    pub m01: f64,
    pub m20: f64,
    pub m02: f64,
    pub m11: f64,
    pub m30: f64,
    pub m03: f64,
    pub m21: f64,
    pub m12: f64,
}

/// A convexity defect: the deepest point between two contour points and its convex hull.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ConvexityDefect {
    pub start: Point<f64>,
    pub end: Point<f64>,
    pub far_point: Point<f64>,
    pub depth: f64,
}

/// Contour retrieval modes for hierarchy-based contour detection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetrievalMode {
    /// Retrieves only the extreme outer contours.
    External,
    /// Retrieves all contours and creates a flat list.
    List,
    /// Retrieves all contours and organizes them into two-level hierarchies
    /// (external and holes).
    CComp,
    /// Retrieves all contours and builds a full hierarchy tree.
    Tree,
    /// Flood fill retrieval mode.
    FloodFill,
}

impl Moments {
    /// Computes the centroid (center of mass) from the moments.
    #[must_use]
    pub fn centroid(&self) -> Option<Point<f64>> {
        if self.m00.abs() < 1e-9 {
            None
        } else {
            Some(Point::new(self.m10 / self.m00, self.m01 / self.m00))
        }
    }
}

impl Contour {
    /// Creates a new Contour with given points.
    #[must_use]
    pub fn new(points: Vec<Point<usize>>) -> Self {
        Self { points }
    }

    /// Finds convexity defects between a contour and its convex hull.
    ///
    /// A convexity defect is a region where the contour deviates inward from
    /// its convex hull. For each defect we record the start point (hull vertex),
    /// end point (next hull vertex), the farthest contour point from the hull
    /// edge, and the perpendicular depth of that point.
    pub fn convexity_defects(contour: &[Point<f64>], hull: &[Point<f64>]) -> Vec<ConvexityDefect> {
        if contour.len() < 3 || hull.len() < 3 {
            return Vec::new();
        }

        let mut defects = Vec::new();

        for hi in 0..hull.len() {
            let a = hull[hi];
            let b = hull[(hi + 1) % hull.len()];

            let abx = b.x - a.x;
            let aby = b.y - a.y;
            let ab_len2 = abx * abx + aby * aby;
            if ab_len2 < 1e-12 {
                continue;
            }

            let mut max_depth = 0.0;
            let mut far_point = a;

            for &p in contour {
                let apx = p.x - a.x;
                let apy = p.y - a.y;

                // Project P onto line AB, clamped to segment
                let t = ((apx * abx + apy * aby) / ab_len2).clamp(0.0, 1.0);
                let proj_x = a.x + t * abx;
                let proj_y = a.y + t * aby;

                let dx = p.x - proj_x;
                let dy = p.y - proj_y;
                let depth = (dx * dx + dy * dy).sqrt();

                if depth > max_depth {
                    max_depth = depth;
                    far_point = p;
                }
            }

            if max_depth > 1e-6 {
                defects.push(ConvexityDefect {
                    start: a,
                    end: b,
                    far_point,
                    depth: max_depth,
                });
            }
        }

        defects
    }

    /// Computes the convex hull of the contour using Andrew's Monotone Chain algorithm.
    #[must_use]
    pub fn convex_hull(&self) -> Self {
        let mut pts = self.points.clone();
        if pts.len() <= 3 {
            return Self::new(pts);
        }

        // Sort points lexicographically by x, then y
        pts.sort_by(|a, b| {
            if a.x == b.x {
                a.y.cmp(&b.y)
            } else {
                a.x.cmp(&b.x)
            }
        });

        fn cross(o: &Point<usize>, a: &Point<usize>, b: &Point<usize>) -> isize {
            (a.x as isize - o.x as isize) * (b.y as isize - o.y as isize)
                - (a.y as isize - o.y as isize) * (b.x as isize - o.x as isize)
        }

        let mut lower = Vec::new();
        for p in &pts {
            while lower.len() >= 2
                && cross(&lower[lower.len() - 2], &lower[lower.len() - 1], p) <= 0
            {
                lower.pop();
            }
            lower.push(*p);
        }

        let mut upper = Vec::new();
        for p in pts.iter().rev() {
            while upper.len() >= 2
                && cross(&upper[upper.len() - 2], &upper[upper.len() - 1], p) <= 0
            {
                upper.pop();
            }
            upper.push(*p);
        }

        lower.pop();
        upper.pop();
        lower.extend(upper);

        Self::new(lower)
    }

    /// Computes the polygon moments using the Shoelace formula and Green's Theorem.
    #[must_use]
    pub fn moments(&self) -> Moments {
        let pts = &self.points;
        let n = pts.len();
        if n < 3 {
            return Moments::default();
        }

        let mut m00 = 0.0;
        let mut m10 = 0.0;
        let mut m01 = 0.0;
        let mut m20 = 0.0;
        let mut m02 = 0.0;
        let mut m11 = 0.0;
        let mut m30 = 0.0;
        let mut m03 = 0.0;
        let mut m21 = 0.0;
        let mut m12 = 0.0;

        for i in 0..n {
            let p0 = pts[i];
            let p1 = pts[(i + 1) % n];

            let xi = p0.x as f64;
            let yi = p0.y as f64;
            let xi1 = p1.x as f64;
            let yi1 = p1.y as f64;

            let cross = xi * yi1 - xi1 * yi;
            m00 += cross;
            m10 += (xi + xi1) * cross;
            m01 += (yi + yi1) * cross;
            m20 += (xi * xi + xi * xi1 + xi1 * xi1) * cross;
            m02 += (yi * yi + yi * yi1 + yi1 * yi1) * cross;
            m11 += (2.0 * xi * yi + xi * yi1 + xi1 * yi + 2.0 * xi1 * yi1) * cross;
            m30 += (xi * xi * xi + xi * xi * xi1 + xi * xi1 * xi1 + xi1 * xi1 * xi1) * cross;
            m03 += (yi * yi * yi + yi * yi * yi1 + yi * yi1 * yi1 + yi1 * yi1 * yi1) * cross;
            m21 += (2.0 * xi * xi * yi + xi * xi * yi1 + 2.0 * xi * xi1 * yi
                + xi * xi1 * yi1
                + xi1 * xi1 * yi
                + 2.0 * xi1 * xi1 * yi1)
                * cross;
            m12 += (2.0 * yi * yi * xi + yi * yi * xi1 + 2.0 * yi * yi1 * xi
                + yi * yi1 * xi1
                + yi1 * yi1 * xi
                + 2.0 * yi1 * yi1 * xi1)
                * cross;
        }

        m00 /= 2.0;
        m10 /= 6.0;
        m01 /= 6.0;
        m20 /= 12.0;
        m02 /= 12.0;
        m11 /= 24.0;
        m30 /= 20.0;
        m03 /= 20.0;
        m21 /= 60.0;
        m12 /= 60.0;

        Moments {
            m00: m00.abs(),
            m10: m10.abs(),
            m01: m01.abs(),
            m20: m20.abs(),
            m02: m02.abs(),
            m11: m11.abs(),
            m30: m30.abs(),
            m03: m03.abs(),
            m21: m21.abs(),
            m12: m12.abs(),
        }
    }
}

impl<B: Backend> Image<B> {
    /// Scans a binary image (grayscale, thresholded) to find contours (connected components).
    /// Uses a basic boundary-following algorithm to find contiguous shapes.
    pub fn find_contours(&self) -> Result<Vec<Contour>> {
        let gray = self.grayscale()?;
        let dims = gray.tensor.dims();
        let h = dims[1];
        let w = dims[2];

        let tensor_data = gray.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

        let mut visited = vec![false; h * w];
        let mut contours = Vec::new();

        // 8-neighbor offsets
        let dx = [1, 1, 0, -1, -1, -1, 0, 1];
        let dy = [0, 1, 1, 1, 0, -1, -1, -1];

        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                let idx = y * w + x;
                if flat_vals[idx] > 0.5 && !visited[idx] {
                    // Start of a boundary
                    let mut pts = Vec::new();
                    let mut cx = x;
                    let mut cy = y;
                    let mut dir = 0;

                    pts.push(Point::new(cx, cy));
                    visited[idx] = true;

                    // Trace boundary
                    let mut loop_count = 0;
                    loop {
                        let mut found = false;
                        for i in 0..8 {
                            let ndir = (dir + i) % 8;
                            let nx = cx as isize + dx[ndir];
                            let ny = cy as isize + dy[ndir];

                            if nx >= 0 && nx < w as isize && ny >= 0 && ny < h as isize {
                                let nidx = (ny as usize) * w + (nx as usize);
                                if flat_vals[nidx] > 0.5 {
                                    cx = nx as usize;
                                    cy = ny as usize;
                                    visited[nidx] = true;
                                    pts.push(Point::new(cx, cy));
                                    dir = (ndir + 5) % 8; // Backtrack direction
                                    found = true;
                                    break;
                                }
                            }
                        }

                        if !found || (cx == x && cy == y) || loop_count > 10000 {
                            break;
                        }
                        loop_count += 1;
                    }

                    if pts.len() >= 3 {
                        contours.push(Contour::new(pts));
                    }
                }
            }
        }

        Ok(contours)
    }

    /// Finds contours in a binary image with hierarchy information.
    ///
    /// Returns a tuple of `(contours, hierarchy)` where each hierarchy entry
    /// is `[next, prev, child, parent]` using -1 as a sentinel for "none".
    /// The hierarchy encodes the nesting relationship between external contours
    /// and holes.
    pub fn find_contours_with_hierarchy(
        &self,
        mode: RetrievalMode,
    ) -> Result<ContourResult> {
        let gray = self.grayscale()?;
        let dims = gray.tensor.dims();
        let h = dims[1];
        let w = dims[2];

        let tensor_data = gray.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

        // Build a binary mask
        let mut binary = vec![false; h * w];
        for i in 0..(h * w) {
            binary[i] = flat_vals[i] > 0.5;
        }

        // Label connected components with 4-connectivity flood fill
        let mut labels = vec![0i32; h * w];
        let mut label_count: i32 = 0;

        let dx4 = [1, 0, -1, 0];
        let dy4 = [0, 1, 0, -1];

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if binary[idx] && labels[idx] == 0 {
                    label_count += 1;
                    // BFS flood fill
                    let mut stack = vec![(x, y)];
                    labels[idx] = label_count;
                    while let Some((cx, cy)) = stack.pop() {
                        for d in 0..4 {
                            let nx = cx as isize + dx4[d];
                            let ny = cy as isize + dy4[d];
                            if nx >= 0 && nx < w as isize && ny >= 0 && ny < h as isize {
                                let nidx = ny as usize * w + nx as usize;
                                if binary[nidx] && labels[nidx] == 0 {
                                    labels[nidx] = label_count;
                                    stack.push((nx as usize, ny as usize));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Extract contours per label using boundary tracing
        let dx8 = [1, 1, 0, -1, -1, -1, 0, 1];
        let dy8 = [0, 1, 1, 1, 0, -1, -1, -1];

        let mut all_contours: Vec<Vec<Point<usize>>> = Vec::new();
        let mut visited = vec![false; h * w];

        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                let idx = y * w + x;
                if !binary[idx] || visited[idx] {
                    continue;
                }

                // Check if this is a boundary pixel (has a background neighbor)
                let mut is_boundary = false;
                for d in 0..8 {
                    let nx = x as isize + dx8[d];
                    let ny = y as isize + dy8[d];
                    if nx >= 0
                        && nx < w as isize
                        && ny >= 0
                        && ny < h as isize
                        && !binary[ny as usize * w + nx as usize]
                    {
                        is_boundary = true;
                        break;
                    }
                }
                if !is_boundary {
                    visited[idx] = true;
                    continue;
                }

                // Trace the boundary
                let mut pts = Vec::new();
                let mut cx = x;
                let mut cy = y;
                let mut dir = 0;

                pts.push(Point::new(cx, cy));
                visited[idx] = true;

                let mut loop_count = 0;
                loop {
                    let mut found = false;
                    for i in 0..8 {
                        let ndir = (dir + i) % 8;
                        let nx = cx as isize + dx8[ndir];
                        let ny = cy as isize + dy8[ndir];

                        if nx >= 0 && nx < w as isize && ny >= 0 && ny < h as isize {
                            let nidx = ny as usize * w + nx as usize;
                            if binary[nidx] {
                                cx = nx as usize;
                                cy = ny as usize;
                                visited[nidx] = true;
                                pts.push(Point::new(cx, cy));
                                dir = (ndir + 5) % 8;
                                found = true;
                                break;
                            }
                        }
                    }

                    if !found || (cx == x && cy == y) || loop_count > 10000 {
                        break;
                    }
                    loop_count += 1;
                }

                if pts.len() >= 3 {
                    all_contours.push(pts);
                }
            }
        }

        // Build hierarchy based on point-in-polygon inclusion testing
        let n = all_contours.len();

        // Determine parent-child by containment:
        // For each contour, find the smallest enclosing contour.
        let mut parent_map: Vec<Option<usize>> = vec![None; n];

        for i in 0..n {
            let ci = &all_contours[i];
            let center_i = {
                let sx: f64 = ci.iter().map(|p| p.x as f64).sum::<f64>() / ci.len() as f64;
                let sy: f64 = ci.iter().map(|p| p.y as f64).sum::<f64>() / ci.len() as f64;
                Point::new(sx, sy)
            };

            let mut best_parent: Option<usize> = None;
            let mut best_area = f64::MAX;

            for j in 0..n {
                if i == j {
                    continue;
                }
                // Check if center of contour i is inside contour j
                let cj = &all_contours[j];
                if Self::point_inside_polygon(center_i, cj) {
                    let area = Self::polygon_area(cj);
                    if area < best_area {
                        best_area = area;
                        best_parent = Some(j);
                    }
                }
            }
            parent_map[i] = best_parent;
        }

        // Apply RetrievalMode filter
        let include = |idx: usize, mode: RetrievalMode| -> bool {
            match mode {
                RetrievalMode::External => parent_map[idx].is_none(),
                RetrievalMode::List | RetrievalMode::CComp | RetrievalMode::Tree => true,
                RetrievalMode::FloodFill => true,
            }
        };

        // Build filtered index mapping
        let filtered_indices: Vec<usize> = (0..n).filter(|&i| include(i, mode)).collect();
        let mut index_map: Vec<i32> = vec![-1; n];
        for (new_idx, &old_idx) in filtered_indices.iter().enumerate() {
            index_map[old_idx] = new_idx as i32;
        }

        let filtered_n = filtered_indices.len();
        let mut filtered_hierarchy = vec![[-1i32; 4]; filtered_n];

        for (new_idx, &old_idx) in filtered_indices.iter().enumerate() {
            // next sibling
            let next_sibling = {
                let mut found = -1i32;
                if let Some(parent) = parent_map[old_idx] {
                    for k in (old_idx + 1)..n {
                        if parent_map[k] == Some(parent) && index_map[k] >= 0 {
                            found = index_map[k];
                            break;
                        }
                    }
                }
                found
            };

            // prev sibling
            let prev_sibling = {
                let mut found = -1i32;
                if let Some(parent) = parent_map[old_idx] {
                    for k in (0..old_idx).rev() {
                        if parent_map[k] == Some(parent) && index_map[k] >= 0 {
                            found = index_map[k];
                            break;
                        }
                    }
                }
                found
            };

            // first child
            let first_child = {
                let mut found = -1i32;
                for k in 0..n {
                    if parent_map[k] == Some(old_idx) && index_map[k] >= 0 {
                        found = index_map[k];
                        break;
                    }
                }
                found
            };

            let parent = parent_map[old_idx]
                .map(|p| index_map[p])
                .unwrap_or(-1);

            filtered_hierarchy[new_idx] = [next_sibling, prev_sibling, first_child, parent];
        }

        let contours_out: Vec<Vec<Point<usize>>> =
            filtered_indices.into_iter().map(|i| all_contours[i].clone()).collect();

        Ok((contours_out, filtered_hierarchy))
    }

    /// Ray-casting point-in-polygon test.
    fn point_inside_polygon(point: Point<f64>, polygon: &[Point<usize>]) -> bool {
        let n = polygon.len();
        if n < 3 {
            return false;
        }
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let yi = polygon[i].y as f64;
            let yj = polygon[j].y as f64;
            let xi = polygon[i].x as f64;
            let xj = polygon[j].x as f64;

            if ((yi > point.y) != (yj > point.y))
                && (point.x < (xj - xi) * (point.y - yi) / (yj - yi + 1e-9) + xi)
            {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    /// Signed area of a polygon using the Shoelace formula.
    fn polygon_area(polygon: &[Point<usize>]) -> f64 {
        let n = polygon.len();
        if n < 3 {
            return 0.0;
        }
        let mut area = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            area += polygon[i].x as f64 * polygon[j].y as f64;
            area -= polygon[j].x as f64 * polygon[i].y as f64;
        }
        area.abs() / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{TestBackend, test_device};
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn test_contours_and_moments() {
        let pts = vec![
            Point::new(0, 0),
            Point::new(10, 0),
            Point::new(10, 10),
            Point::new(0, 10),
        ];
        let contour = Contour::new(pts);
        let hull = contour.convex_hull();
        assert_eq!(hull.points.len(), 4);

        let m = contour.moments();
        assert!(m.m00 > 0.0);
        let centroid = m.centroid().unwrap();
        assert!(centroid.x > 0.0);

        let device = test_device();
        // Create an image with a single pixel set to 1.0 (binary mask)
        let mut flat_data = vec![0.0f32; 10 * 10];
        // Set a 3x3 block to 1.0
        for y in 2..5 {
            for x in 2..5 {
                flat_data[y * 10 + x] = 1.0;
            }
        }
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [1, 10, 10]), &device);
        let img = Image::new(tensor);
        let found = img.find_contours().unwrap();
        assert!(!found.is_empty());
    }

    #[test]
    fn test_convexity_defects() {
        // Create a concave polygon: a square with a notch cut into one side.
        // Points go clockwise: bottom-left -> bottom-right -> notch inward -> notch back -> top-right -> top-left
        let contour = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 5.0),
            Point::new(7.0, 3.0),  // notch inward
            Point::new(5.0, 5.0),  // notch bottom
            Point::new(5.0, 10.0),
            Point::new(0.0, 10.0),
        ];

        // Convex hull of this shape
        let hull = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 5.0),
            Point::new(5.0, 10.0),
            Point::new(0.0, 10.0),
        ];

        let defects = Contour::convexity_defects(&contour, &hull);
        // There should be at least one defect where the contour goes inward
        assert!(!defects.is_empty());
        for d in &defects {
            assert!(d.depth > 0.0);
        }
    }

    #[test]
    fn test_find_contours_with_hierarchy() {
        let device = test_device();
        // Create image with two separate blobs
        let mut flat_data = vec![0.0f32; 20 * 20];
        // Blob 1: 3x3 at (2,2)
        for y in 2..5 {
            for x in 2..5 {
                flat_data[y * 20 + x] = 1.0;
            }
        }
        // Blob 2: 3x3 at (12,12)
        for y in 12..15 {
            for x in 12..15 {
                flat_data[y * 20 + x] = 1.0;
            }
        }
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [1, 20, 20]), &device);
        let img = Image::new(tensor);

        let (contours, hierarchy) = img.find_contours_with_hierarchy(RetrievalMode::External).unwrap();
        assert!(!contours.is_empty());
        assert_eq!(contours.len(), hierarchy.len());
        // Hierarchy entries should have 4 elements each
        for h in &hierarchy {
            assert_eq!(h.len(), 4);
        }
    }
}
