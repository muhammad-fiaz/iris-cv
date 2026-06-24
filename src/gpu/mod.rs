/// Custom GPU pipeline/kernel compilation abstractions (e.g. for WebGPU or CUDA).
pub struct GpuContext {
    pub enabled: bool,
}

impl GpuContext {
    #[must_use]
    pub fn new() -> Self {
        Self { enabled: true }
    }

    /// Queries if acceleration is supported by the active device.
    #[must_use]
    pub fn is_accelerated(&self) -> bool {
        self.enabled
    }
}

impl Default for GpuContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_context() {
        let ctx = GpuContext::default();
        assert!(ctx.is_accelerated());
    }
}
