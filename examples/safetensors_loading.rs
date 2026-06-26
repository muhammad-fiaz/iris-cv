// Demonstrates weight loading from binary and safetensors files,
// and Non-Maximum Suppression (NMS) for bounding box filtering.
// Uses mock data (no image file needed).

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = WgpuDevice::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // 1. Create and load mock binary weights
    let weight_data = vec![0.0f32; 100 * 100];
    let bin_path = "mock_weights.bin";

    let mut bin_bytes = Vec::new();
    for &val in &weight_data {
        bin_bytes.extend_from_slice(&val.to_ne_bytes());
    }
    std::fs::write(bin_path, &bin_bytes).unwrap();
    println!("Generated mock bin weights file: '{}'", bin_path);

    let loaded_tensor = WeightLoader::load_bin::<Backend>(bin_path, &device, [100, 100])?;
    println!("Loaded bin tensor shape: {:?}", loaded_tensor.dims());

    let _ = std::fs::remove_file(bin_path);

    // 2. Safetensors loading (if feature enabled)
    #[cfg(feature = "safetensors")]
    {
        use std::collections::BTreeMap;

        println!("Safetensors feature enabled. Generating mock safetensors...");
        let sf_path = "mock_weights.safetensors";

        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                weight_data.as_ptr() as *const u8,
                weight_data.len() * std::mem::size_of::<f32>(),
            )
        };

        let mut data = BTreeMap::new();
        let view =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![100, 100], bytes)
                .unwrap();
        data.insert("weight_1".to_string(), view);

        safetensors::tensor::serialize_to_file(&data, None, std::path::Path::new(sf_path)).unwrap();

        let loaded_map = WeightLoader::load_safetensors::<Backend>(sf_path, &device)?;
        println!("Loaded safetensors keys: {:?}", loaded_map.keys());
        if let Some(t) = loaded_map.get("weight_1") {
            println!("Loaded tensor 'weight_1' shape: {:?}", t.dims());
        }

        let _ = std::fs::remove_file(sf_path);
    }

    #[cfg(not(feature = "safetensors"))]
    {
        println!("Safetensors feature disabled. Skipping safetensors demo.");
    }

    // 3. Non-Maximum Suppression
    let bboxes = vec![
        Rect::new(10, 10, 50, 50),
        Rect::new(12, 12, 48, 48),
        Rect::new(100, 100, 40, 40),
    ];
    let scores = vec![0.9, 0.75, 0.82];

    let keep_indices = nms_boxes(&bboxes, &scores, 0.5, 0.4);
    println!("NMS kept indices: {:?}", keep_indices);
    assert_eq!(keep_indices, vec![0, 2]);

    println!("Safetensors loading and NMS example completed.");
    Ok(())
}
