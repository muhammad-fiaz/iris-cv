use burn::tensor::backend::Backend;

/// Utility helpers for configuring and running Burn operations.
pub struct BurnUtils;

impl BurnUtils {
    #[must_use]
    pub fn backend_name<B: Backend>() -> String {
        let device = Default::default();
        B::name(&device)
    }

    /// Selects the best device available for this execution.
    #[must_use]
    pub fn best_device<B: Backend>() -> B::Device {
        // Defaults to WGPU or CPU device based on the backend used.
        Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::TestBackend;

    #[test]
    fn test_burn_utils() {
        let name = BurnUtils::backend_name::<TestBackend>();
        assert!(!name.is_empty());
        let _device = BurnUtils::best_device::<TestBackend>();
    }
}
