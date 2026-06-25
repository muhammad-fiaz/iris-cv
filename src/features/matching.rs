use crate::core::types::{Point, Scalar};
use crate::error::{IrisError, Result};
use crate::features::KeyPoint;
use crate::image::Image;
use burn::tensor::{Tensor, TensorData, backend::Backend};

/// Represents a descriptor match between two keypoints.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DMatch {
    pub query_idx: usize,
    pub train_idx: usize,
    pub img_idx: usize,
    pub distance: f32,
}

pub struct BFMatcher;

impl BFMatcher {
    /// Performs brute-force matching between query descriptors [N1, D] and train descriptors [N2, D].
    pub fn match_descriptors<B: Backend>(
        &self,
        query: &Tensor<B, 2>,
        train: &Tensor<B, 2>,
    ) -> Result<Vec<DMatch>> {
        let q_dims = query.dims();
        let t_dims = train.dims();
        let n1 = q_dims[0];
        let _n2 = t_dims[0];

        // Query shape: [N1, 1, D], Train shape: [1, N2, D]
        let q_unsqueezed = query.clone().unsqueeze_dim::<3>(1);
        let t_unsqueezed = train.clone().unsqueeze_dim::<3>(0);

        let diff = q_unsqueezed.sub(t_unsqueezed);
        let squared_diff = diff.powf_scalar(2.0);
        let dists = squared_diff.sum_dim(2).squeeze::<2>().sqrt(); // [N1, N2]

        let min_dists = dists.clone().min_dim(1).squeeze::<1>(); // [N1]
        let argmins = dists.argmin(1).squeeze::<1>(); // [N1]

        let min_dists_data = min_dists.into_data();
        let argmins_data = argmins.into_data();

        let min_dists_vec: Vec<f32> = min_dists_data.iter::<f32>().collect();
        let argmins_vec: Vec<i32> = argmins_data.iter::<i32>().collect();

        let mut matches = Vec::new();
        for i in 0..n1 {
            matches.push(DMatch {
                query_idx: i,
                train_idx: argmins_vec[i] as usize,
                img_idx: 0,
                distance: min_dists_vec[i],
            });
        }
        Ok(matches)
    }
}

// ---------------------------------------------------------------------------
// FLANN-based KD-tree matcher
// ---------------------------------------------------------------------------

/// A single node in the KD-tree used by the FLANN matcher.
struct KdNode {
    /// Index of the feature vector in the train set.
    idx: usize,
    /// Split axis (0 .. D-1).
    axis: usize,
    /// Threshold value stored at this node (the coordinate along `axis`).
    threshold: f32,
    /// Optional left child index in the nodes vec (None for leaves).
    left: Option<usize>,
    /// Optional right child index in the nodes vec.
    right: Option<usize>,
}

/// FLANN-inspired approximate nearest-neighbour matcher built on a set of
/// KD-trees.
///
/// Each tree is built by recursively partitioning the descriptor space along
/// alternating axes. At query time the trees are searched in parallel with a
/// limited number of checks per tree to trade accuracy for speed.
pub struct FlannMatcher {
    /// Number of nearest neighbours to return per query descriptor.
    k: usize,
    /// Number of KD-trees in the forest.
    trees: usize,
    /// Maximum number of leaf-node checks before the search is terminated
    /// (the *checks* parameter in FLANN).
    checks: usize,
}

impl Default for FlannMatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl FlannMatcher {
    /// Creates a FLANN matcher with reasonable defaults (`k=2`, `trees=5`,
    /// `checks=32`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            k: 2,
            trees: 5,
            checks: 32,
        }
    }

    /// Sets the number of nearest neighbours.
    #[must_use]
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Sets the number of trees in the forest.
    #[must_use]
    pub fn with_trees(mut self, trees: usize) -> Self {
        self.trees = trees;
        self
    }

    /// Sets the maximum number of leaf checks during search.
    #[must_use]
    pub fn with_checks(mut self, checks: usize) -> Self {
        self.checks = checks;
        self
    }

    // -- internal helpers ---------------------------------------------------

    /// Recursively build a KD-tree over the slice of (index, feature) pairs.
    fn build_kd_tree(
        items: &mut [(usize, Vec<f32>)],
        depth: usize,
        nodes: &mut Vec<KdNode>,
    ) -> Option<usize> {
        if items.is_empty() {
            return None;
        }
        if items.len() == 1 {
            let idx = items[0].0;
            let axis = depth % items[0].1.len();
            let threshold = items[0].1[axis];
            let node_idx = nodes.len();
            nodes.push(KdNode {
                idx,
                axis,
                threshold,
                left: None,
                right: None,
            });
            return Some(node_idx);
        }

        let dim = items[0].1.len();
        let axis = depth % dim;

        // Median-of-three pivot selection
        let mid = items.len() / 2;
        // Simple nth_element via partial sort
        items.select_nth_unstable_by(mid, |a, b| {
            a.1[axis]
                .partial_cmp(&b.1[axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let threshold = items[mid].1[axis];
        let split_idx = mid;

        let node = KdNode {
            idx: items[split_idx].0,
            axis,
            threshold,
            left: None,
            right: None,
        };
        let node_pos = nodes.len();
        nodes.push(node);

        let (left_items, right_items) = items.split_at_mut(split_idx);
        let (right_items, _) = right_items.split_at_mut(1); // skip the pivot

        let left_child = Self::build_kd_tree(left_items, depth + 1, nodes);
        let right_child = Self::build_kd_tree(right_items, depth + 1, nodes);

        nodes[node_pos].left = left_child;
        nodes[node_pos].right = right_child;

        Some(node_pos)
    }

    /// Search the KD-tree for the single nearest neighbour of `query`.
    /// Returns (train_idx, distance).
    fn search_nn(
        nodes: &[KdNode],
        train: &[Vec<f32>],
        query: &[f32],
        root: Option<usize>,
        checks_remaining: &mut usize,
    ) -> (usize, f32) {
        let mut best_idx = 0usize;
        let mut best_dist = f32::MAX;
        Self::search_nn_recursive(
            nodes,
            train,
            query,
            root,
            checks_remaining,
            &mut best_idx,
            &mut best_dist,
        );
        (best_idx, best_dist)
    }

    fn search_nn_recursive(
        nodes: &[KdNode],
        train: &[Vec<f32>],
        query: &[f32],
        node_idx: Option<usize>,
        checks: &mut usize,
        best_idx: &mut usize,
        best_dist: &mut f32,
    ) {
        let idx = match node_idx {
            Some(i) => i,
            None => return,
        };
        if *checks == 0 {
            return;
        }

        let node = &nodes[idx];

        // Leaf or we still have budget: evaluate this node
        let dist = euclidean_dist_sq(query, &train[node.idx]);
        if dist < *best_dist {
            *best_dist = dist;
            *best_idx = node.idx;
        }
        *checks = checks.saturating_sub(1);

        let diff = query[node.axis] - node.threshold;
        let (near, far) = if diff <= 0.0 {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        Self::search_nn_recursive(nodes, train, query, near, checks, best_idx, best_dist);

        // Only visit the far subtree if the splitting hyperplane is closer
        // than the current best distance.
        if diff * diff < *best_dist {
            Self::search_nn_recursive(nodes, train, query, far, checks, best_idx, best_dist);
        }
    }

    // -- public API ---------------------------------------------------------

    /// Finds approximate nearest-neighbour matches between two descriptor
    /// matrices.
    ///
    /// `desc1` is the query matrix of shape `[N1, D]` and `desc2` is the
    /// train matrix of shape `[N2, D]`. The function returns one
    /// [`DMatch`] per query descriptor.
    pub fn find_matches<B: Backend>(
        &self,
        desc1: &Tensor<B, 2>,
        desc2: &Tensor<B, 2>,
    ) -> Result<Vec<DMatch>> {
        let q_dims = desc1.dims();
        let t_dims = desc2.dims();
        let n1 = q_dims[0];
        let n2 = t_dims[0];
        let dim = q_dims[1];

        if dim != t_dims[1] {
            return Err(IrisError::DimensionMismatch {
                expected: vec![n1, dim],
                actual: vec![n2, t_dims[1]],
            });
        }

        let q_data = desc1.clone().into_data();
        let t_data = desc2.clone().into_data();
        let q_flat: Vec<f32> = q_data.iter::<f32>().collect();
        let t_flat: Vec<f32> = t_data.iter::<f32>().collect();

        // Materialise into Vec<Vec<f32>> for indexed access
        let query_vecs: Vec<Vec<f32>> = (0..n1)
            .map(|i| q_flat[i * dim..(i + 1) * dim].to_vec())
            .collect();
        let train_vecs: Vec<Vec<f32>> = (0..n2)
            .map(|i| t_flat[i * dim..(i + 1) * dim].to_vec())
            .collect();

        // Build one KD-tree per tree in the forest, each over a random
        // permutation of the training set to improve diversity.
        let mut forest: Vec<Vec<KdNode>> = Vec::with_capacity(self.trees);
        let mut forest_roots: Vec<Option<usize>> = Vec::with_capacity(self.trees);

        for t in 0..self.trees {
            // Deterministic pseudo-random permutation via LCG seeded by tree index
            let mut indices: Vec<usize> = (0..n2).collect();
            let mut seed = (t as u32).wrapping_mul(1103515245).wrapping_add(12345);
            for i in (1..n2).rev() {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let j = (seed as usize) % (i + 1);
                indices.swap(i, j);
            }

            let mut items: Vec<(usize, Vec<f32>)> = indices
                .into_iter()
                .map(|i| (i, train_vecs[i].clone()))
                .collect();
            let mut nodes = Vec::new();
            let root = Self::build_kd_tree(&mut items, 0, &mut nodes);
            forest.push(nodes);
            forest_roots.push(root);
        }

        // Query each tree and aggregate distances via average
        let mut matches = Vec::with_capacity(n1);

        for qi in 0..n1 {
            // Accumulate distances from all trees
            let mut dist_acc = vec![0.0f32; n2];
            let mut count = vec![0u32; n2];

            for t in 0..self.trees {
                let mut checks = self.checks;
                let (best_train_idx, best_dist) = Self::search_nn(
                    &forest[t],
                    &train_vecs,
                    &query_vecs[qi],
                    forest_roots[t],
                    &mut checks,
                );
                dist_acc[best_train_idx] += best_dist;
                count[best_train_idx] += 1;
            }

            // For each train point, compute average distance across the trees
            // that visited it. Points never visited get MAX distance.
            let mut best_idx = 0usize;
            let mut best_dist = f32::MAX;
            for ti in 0..n2 {
                if count[ti] > 0 {
                    let avg = dist_acc[ti] / count[ti] as f32;
                    if avg < best_dist {
                        best_dist = avg;
                        best_idx = ti;
                    }
                }
            }

            // Fallback: brute-force nearest if the forest was too shallow
            if best_dist.is_infinite() || best_dist >= f32::MAX {
                for ti in 0..n2 {
                    let d = euclidean_dist_sq(&query_vecs[qi], &train_vecs[ti]);
                    if d < best_dist {
                        best_dist = d;
                        best_idx = ti;
                    }
                }
            }

            matches.push(DMatch {
                query_idx: qi,
                train_idx: best_idx,
                img_idx: 0,
                distance: best_dist.sqrt(),
            });
        }

        Ok(matches)
    }
}

/// Squared Euclidean distance between two equal-length slices.
fn euclidean_dist_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Helper to draw descriptor matching matches between two images.
pub struct MatchDrawer;

impl MatchDrawer {
    /// Combines two images horizontally and draws matching lines between corresponding keypoints.
    pub fn draw_matches<B: Backend>(
        img1: &Image<B>,
        kps1: &[KeyPoint],
        img2: &Image<B>,
        kps2: &[KeyPoint],
        matches: &[DMatch],
    ) -> Result<Image<B>> {
        let h1 = img1.height();
        let w1 = img1.width();
        let h2 = img2.height();
        let w2 = img2.width();

        let out_h = h1.max(h2);
        let out_w = w1 + w2;
        let c = img1.channels();

        let device = img1.tensor.device();

        // 2. Crop/copy img1 and img2 onto out_flat directly
        // Simple pixel copy on CPU for canvas assembly
        let data1 = img1.tensor.clone().into_data();
        let data2 = img2.tensor.clone().into_data();
        let flat1: Vec<f32> = data1.iter::<f32>().collect();
        let flat2: Vec<f32> = data2.iter::<f32>().collect();
        let mut out_flat = vec![0.0f32; c * out_h * out_w];

        for ch in 0..c {
            for y in 0..h1 {
                for x in 0..w1 {
                    out_flat[ch * out_h * out_w + y * out_w + x] = flat1[ch * h1 * w1 + y * w1 + x];
                }
            }
            for y in 0..h2 {
                for x in 0..w2 {
                    out_flat[ch * out_h * out_w + y * out_w + (x + w1)] =
                        flat2[ch * h2 * w2 + y * w2 + x];
                }
            }
        }

        let assembled_tensor =
            Tensor::<B, 3>::from_data(TensorData::new(out_flat, [c, out_h, out_w]), &device);
        let mut assembled = Image::new(assembled_tensor);

        // 3. Draw matching lines
        for m in matches {
            let kp1 = &kps1[m.query_idx];
            let kp2 = &kps2[m.train_idx];

            let p1 = Point::new(kp1.pt.x as usize, kp1.pt.y as usize);
            let p2 = Point::new(kp2.pt.x as usize + w1, kp2.pt.y as usize);

            assembled = assembled.draw_line(p1, p2, Scalar::new(0.0, 1.0, 0.0, 0.0))?;
        }

        Ok(assembled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{TestBackend, test_device};

    #[test]
    fn test_descriptor_matching() {
        let device = test_device();
        let query = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![1.0f32, 0.0, 0.0, 1.0], [2, 2]),
            &device,
        );
        let train = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![1.0f32, 0.0, 0.0, 1.0], [2, 2]),
            &device,
        );

        let matcher = BFMatcher;
        let matches = matcher.match_descriptors(&query, &train).unwrap();
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].query_idx, 0);
        assert_eq!(matches[0].train_idx, 0);
        assert_eq!(matches[1].query_idx, 1);
        assert_eq!(matches[1].train_idx, 1);

        let flat_data = vec![0.5f32; 3 * 8 * 8];
        let t1 = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(flat_data.clone(), [3, 8, 8]),
            &device,
        );
        let t2 =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [3, 8, 8]), &device);
        let img1 = Image::new(t1);
        let img2 = Image::new(t2);

        let kps1 = vec![KeyPoint::new(1.0, 1.0, 2.0), KeyPoint::new(2.0, 2.0, 2.0)];
        let kps2 = vec![KeyPoint::new(1.0, 1.0, 2.0), KeyPoint::new(2.0, 2.0, 2.0)];

        let drawn = MatchDrawer::draw_matches(&img1, &kps1, &img2, &kps2, &matches).unwrap();
        assert_eq!(drawn.shape(), [3, 8, 16]);
    }

    #[test]
    fn test_flann_matcher_known_pairs() {
        let device = test_device();
        // Two groups of descriptors that are clearly separable:
        //   group A around [1, 0, 0, 0]
        //   group B around [0, 1, 0, 0]
        let query = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![1.01, 0.01, 0.0, 0.0, 0.01, 0.99, 0.0, 0.0], [2, 4]),
            &device,
        );
        let train = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(
                vec![
                    1.0, 0.0, 0.0, 0.0, // best match for query row 0
                    0.0, 1.0, 0.0, 0.0, // best match for query row 1
                    1.05, 0.05, 0.0, 0.0,
                ],
                [3, 4],
            ),
            &device,
        );

        let matcher = FlannMatcher::new();
        let matches = matcher.find_matches(&query, &train).unwrap();
        assert_eq!(matches.len(), 2);

        // Each query descriptor should match the train descriptor closest to it
        assert_eq!(matches[0].query_idx, 0);
        assert_eq!(matches[0].train_idx, 0); // [1,0,0,0] ↔ [1,0,0,0]
        assert_eq!(matches[1].query_idx, 1);
        assert_eq!(matches[1].train_idx, 1); // [1,0,0,0] ↔ [0,1,0,0]
    }
}
