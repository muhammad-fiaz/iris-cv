# Examples

The Observers repository includes several complete examples showcasing key capabilities of the library. You can run any example directly from the cloned repository.

## Running Examples

```bash
# Clone the repository
git clone https://github.com/muhammad-fiaz/observers.git
cd observers

# Run the Canny Edge Detection demo
cargo run --example canny

# Run the Contours and shapes demo
cargo run --example contours

# Run the QR Code reader demo
cargo run --example qr_detection

# Run the Face recognition demo
cargo run --example face_recognition

# Run the GUI window manager demo (requires wgpu backend)
cargo run --example gui_windows
```

## Available Examples

| Example | Command | Description |
|---|---|---|
| **Canny Edges** | `cargo run --example canny` | Reads an input image, computes gradients, runs non-maximum suppression, and saves edge maps. |
| **Contours** | `cargo run --example contours` | Finds connected components and traces outer boundaries, outputting polygon coordinates. |
| **GUI Windows** | `cargo run --example gui_windows` | Demonstrates the modern direct `Gui` API, registering callbacks, drawing trackbars, and rendering frames. |
| **QR Detection** | `cargo run --example qr_detection` | Scans frames to detect finder patterns, locating and decoding QR code payloads. |
| **Safetensors** | `cargo run --example safetensors_loading` | Compiles weight loading structures to fetch and execute neural nets using `.safetensors`. |
