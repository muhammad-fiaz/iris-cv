use burn::tensor::{Tensor, backend::Backend};

/// A multi-dimensional dense array representation.
/// It wraps a Burn Tensor and exposes convenient operations.
#[derive(Clone, Debug)]
pub struct Mat<B: Backend, const D: usize> {
    /// The underlying Burn Tensor.
    pub tensor: Tensor<B, D>,
}

impl<B: Backend, const D: usize> Mat<B, D> {
    /// Creates a new Mat from a Burn Tensor.
    pub fn new(tensor: Tensor<B, D>) -> Self {
        Self { tensor }
    }

    /// Returns the shape/dimensions of the Mat.
    pub fn shape(&self) -> Vec<usize> {
        self.tensor.dims().to_vec()
    }

    /// Returns the total number of elements in the Mat.
    pub fn total(&self) -> usize {
        self.tensor.shape().num_elements()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{test_device, TestBackend};
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn test_mat_operations() {
        let device = test_device();
        let tensor = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]),
            &device,
        );
        let mat = Mat::new(tensor);
        assert_eq!(mat.shape(), vec![2, 2]);
        assert_eq!(mat.total(), 4);
    }
}
