use crate::core::types::{Point, Scalar};
use crate::error::Result;
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
    use crate::test_helpers::{test_device, TestBackend};

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
        let t1 =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data.clone(), [3, 8, 8]), &device);
        let t2 = Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [3, 8, 8]), &device);
        let img1 = Image::new(t1);
        let img2 = Image::new(t2);

        let kps1 = vec![KeyPoint::new(1.0, 1.0, 2.0), KeyPoint::new(2.0, 2.0, 2.0)];
        let kps2 = vec![KeyPoint::new(1.0, 1.0, 2.0), KeyPoint::new(2.0, 2.0, 2.0)];

        let drawn = MatchDrawer::draw_matches(&img1, &kps1, &img2, &kps2, &matches).unwrap();
        assert_eq!(drawn.shape(), [3, 8, 16]);
    }
}
