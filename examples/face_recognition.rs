use burn::backend::wgpu::Wgpu;
use observers::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = Default::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // 1. Instantiate detector and recognizer using the new weight loader APIs
    let detector = FaceDetector::<Backend>::default();
    let recognizer = FaceRecognizer::from_onnx("facenet_mock.onnx", &device)?;

    // 2. Generate mock face images
    let w = 250;
    let h = 250;
    let flat_data1 = vec![0.3f32; 3 * h * w];
    let tensor_data1 = TensorData::new(flat_data1, [3, h, w]);
    let img1 = Image::new(Tensor::<Backend, 3>::from_data(tensor_data1, &device));

    let flat_data2 = vec![0.32f32; 3 * h * w];
    let tensor_data2 = TensorData::new(flat_data2, [3, h, w]);
    let img2 = Image::new(Tensor::<Backend, 3>::from_data(tensor_data2, &device));

    // 3. Extract embeddings
    // (In real scenarios, detect first, crop face, then extract)
    let faces = detector.detect(&img1)?;
    println!("Detected {} face(s) in image 1", faces.len());

    let emb1 = recognizer.extract_embedding(&img1)?;
    let emb2 = recognizer.extract_embedding(&img2)?;

    println!("Embedding 1 shape: {:?}", emb1.dims());
    println!("Embedding 2 shape: {:?}", emb2.dims());

    // 4. Compute cosine similarity
    let similarity = recognizer.compute_similarity(&emb1, &emb2)?;
    println!("Face similarity score: {:.4}", similarity);

    Ok(())
}
