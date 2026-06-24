/// Common utility functions for the Iris library.
pub struct Utils;

impl Utils {
    /// Formats a time duration in milliseconds.
    #[must_use]
    pub fn format_duration_ms(ms: f64) -> String {
        format!("{ms:.2} ms")
    }

    /// Linear interpolation between two float values.
    #[must_use]
    pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utils() {
        assert_eq!(Utils::format_duration_ms(12.345), "12.35 ms");
        assert_eq!(Utils::lerp(0.0, 10.0, 0.5), 5.0);
    }
}
