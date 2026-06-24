use crate::error::{IrisError, Result};
use burn::tensor::{Int, Tensor, TensorData, backend::Backend};

/// Classical Machine Learning algorithms powered by Burn.
pub struct KMeans<B: Backend> {
    pub k: usize,
    pub max_iter: usize,
    pub centroids: Option<Tensor<B, 2>>,
}

impl<B: Backend> KMeans<B> {
    /// Creates a new `KMeans` model.
    #[must_use]
    pub fn new(k: usize, max_iter: usize) -> Self {
        Self {
            k,
            max_iter,
            centroids: None,
        }
    }

    /// Fits K-Means on the input data of shape [N, D] (N samples, D features).
    pub fn fit(&mut self, data: &Tensor<B, 2>) -> Result<()> {
        let dims = data.dims();
        let n = dims[0];
        let d = dims[1];

        if n < self.k {
            return Err(IrisError::InvalidParameter(
                "Number of data points must be >= K".into(),
            ));
        }

        let device = &data.device();

        // 1. Initialize centroids by picking the first K points
        let initial_centroids = data.clone().slice([0..self.k, 0..d]);
        let mut centroids = initial_centroids;

        // 2. Iterate
        for _ in 0..self.max_iter {
            // Compute Euclidean distances between all N points and K centroids
            // points shape: [N, 1, D], centroids shape: [1, K, D]
            let p_unsqueezed = data.clone().unsqueeze_dim::<3>(1); // [N, 1, D]
            let c_unsqueezed = centroids.clone().unsqueeze_dim::<3>(0); // [1, K, D]

            let diff = p_unsqueezed.sub(c_unsqueezed); // [N, K, D]
            let squared_diff = diff.powf_scalar(2.0);
            let dists = squared_diff.sum_dim(2).squeeze::<2>(); // [N, K]

            // Find closest centroid for each point
            let assignments = dists.argmin(1).squeeze::<1>(); // [N]

            // Update centroids
            // For a pure tensor implementation, we can group and mean, or do it on CPU for exactness
            let assignments_data = assignments.into_data();
            let assignments_vec: Vec<i32> = assignments_data.iter::<i32>().collect();
            let data_data = data.clone().into_data();
            let flat_data: Vec<f32> = data_data.iter::<f32>().collect();

            let mut new_centroids_data = vec![0.0f32; self.k * d];
            let mut counts = vec![0.0f32; self.k];

            for i in 0..n {
                let cluster = assignments_vec[i] as usize;
                counts[cluster] += 1.0;
                for j in 0..d {
                    new_centroids_data[cluster * d + j] += flat_data[i * d + j];
                }
            }

            for k in 0..self.k {
                let count = counts[k].max(1.0);
                for j in 0..d {
                    new_centroids_data[k * d + j] /= count;
                }
            }

            centroids =
                Tensor::<B, 2>::from_data(TensorData::new(new_centroids_data, [self.k, d]), device);
        }

        self.centroids = Some(centroids);
        Ok(())
    }

    /// Predicts closest cluster assignments for inputs of shape [N, D].
    pub fn predict(&self, data: &Tensor<B, 2>) -> Result<Tensor<B, 1, Int>> {
        let centroids = self.centroids.as_ref().ok_or_else(|| {
            IrisError::Generic(
                "K-Means centroids are not initialized. Fit the model first.".into(),
            )
        })?;

        // points shape: [N, 1, D], centroids shape: [1, K, D]
        let p_unsqueezed = data.clone().unsqueeze_dim::<3>(1); // [N, 1, D]
        let c_unsqueezed = centroids.clone().unsqueeze_dim::<3>(0); // [1, K, D]

        let diff = p_unsqueezed.sub(c_unsqueezed); // [N, K, D]
        let squared_diff = diff.powf_scalar(2.0);
        let dists = squared_diff.sum_dim(2).squeeze::<2>(); // [N, K]

        let assignments = dists.argmin(1).squeeze::<1>(); // [N]
        Ok(assignments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::Wgpu;

    #[test]
    fn test_kmeans_clustering() {
        let device = Default::default();
        let data = Tensor::<Wgpu, 2>::from_data(
            TensorData::new(vec![1.0f32, 1.0, 1.1, 1.1, 10.0, 10.0, 10.2, 10.2], [4, 2]),
            &device,
        );

        let mut km = KMeans::new(2, 5);
        km.fit(&data).unwrap();

        assert!(km.centroids.is_some());
        let assignments = km.predict(&data).unwrap();
        assert_eq!(assignments.dims(), [4]);
    }
}
