/// A pure Rust pseudo-random number generator implementing Xorshift128+.
/// This provides standard random number generation functionality without external dependencies.
#[derive(Clone, Debug)]
pub struct Rng {
    state: [u64; 2],
}

impl Rng {
    /// Creates a new RNG with a given seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut rng = Self {
            state: [seed, seed ^ 0x72636467_75736867],
        };
        // Warm up the generator
        for _ in 0..10 {
            rng.next_u64();
        }
        rng
    }

    /// Generates the next random 64-bit unsigned integer.
    pub fn next_u64(&mut self) -> u64 {
        let mut s1 = self.state[0];
        let s0 = self.state[1];
        let result = s1.wrapping_add(s0);
        self.state[0] = s0;
        s1 ^= s1 << 23; // a
        self.state[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5); // b, c
        result
    }

    /// Generates the next random 32-bit unsigned integer.
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Generates the next random 32-bit float in the range [0.0, 1.0).
    pub fn next_f32(&mut self) -> f32 {
        let val = self.next_u32();
        (val as f32) / (u32::MAX as f32)
    }

    /// Generates the next random 64-bit float in the range [0.0, 1.0).
    pub fn next_f64(&mut self) -> f64 {
        let val = self.next_u64();
        (val as f64) / (u64::MAX as f64)
    }
}

impl Default for Rng {
    fn default() -> Self {
        Self::new(123456789)
    }
}
