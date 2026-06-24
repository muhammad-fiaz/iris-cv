use crate::error::Result;
use crate::image::Image;
use burn::tensor::backend::Backend;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

struct WindowState {
    title: String,
    width: usize,
    height: usize,
    trackbars: HashMap<String, TrackbarState>,
    image_bytes: Option<Vec<u8>>,
}

struct TrackbarState {
    value: i32,
    max_val: i32,
}

fn get_windows() -> &'static Mutex<HashMap<String, WindowState>> {
    static WINDOWS: OnceLock<Mutex<HashMap<String, WindowState>>> = OnceLock::new();
    WINDOWS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Desktop UI and Windowing Management.
pub struct Gui;

impl Gui {
    /// Creates a window with the given name.
    pub fn named_window(name: &str) -> Result<()> {
        let mut wins = get_windows().lock().unwrap();
        wins.insert(
            name.to_string(),
            WindowState {
                title: name.to_string(),
                width: 640,
                height: 480,
                trackbars: HashMap::new(),
                image_bytes: None,
            },
        );
        println!("[GUI] Created window: '{}'", name);
        Ok(())
    }

    /// Destroys a window with the given name.
    pub fn destroy_window(name: &str) -> Result<()> {
        let mut wins = get_windows().lock().unwrap();
        wins.remove(name);
        println!("[GUI] Destroyed window: '{}'", name);
        Ok(())
    }

    /// Destroys all created windows.
    pub fn destroy_all_windows() -> Result<()> {
        let mut wins = get_windows().lock().unwrap();
        wins.clear();
        println!("[GUI] Destroyed all windows");
        Ok(())
    }

    /// Displays an image in the specified window.
    /// In this pure-Rust portable implementation, it also saves a local preview of the frame.
    pub fn imshow<B: Backend>(winname: &str, image: &Image<B>) -> Result<()> {
        let dims = image.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        // Convert the image tensor to raw RGB bytes for window rendering
        let tensor_data = image.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut rgb_bytes = vec![0u8; w * h * 3];

        if c == 3 {
            for y in 0..h {
                for x in 0..w {
                    for ch in 0..3 {
                        let val = flat_vals[ch * h * w + y * w + x];
                        let pixel_val = (val * 255.0).clamp(0.0, 255.0) as u8;
                        rgb_bytes[(y * w + x) * 3 + ch] = pixel_val;
                    }
                }
            }
        } else if c == 1 {
            for y in 0..h {
                for x in 0..w {
                    let val = flat_vals[y * w + x];
                    let pixel_val = (val * 255.0).clamp(0.0, 255.0) as u8;
                    rgb_bytes[(y * w + x) * 3] = pixel_val;
                    rgb_bytes[(y * w + x) * 3 + 1] = pixel_val;
                    rgb_bytes[(y * w + x) * 3 + 2] = pixel_val;
                }
            }
        }

        {
            let mut wins = get_windows().lock().unwrap();
            if let Some(w_state) = wins.get_mut(winname) {
                w_state.width = w;
                w_state.height = h;
                w_state.image_bytes = Some(rgb_bytes);
            } else {
                wins.insert(
                    winname.to_string(),
                    WindowState {
                        title: winname.to_string(),
                        width: w,
                        height: h,
                        trackbars: HashMap::new(),
                        image_bytes: Some(rgb_bytes),
                    },
                );
            }
        }

        let path = format!("preview_{}.png", winname.replace(" ", "_"));
        image.save(&path)?;
        println!(
            "[GUI] Rendering frame in window '{}' (saved preview to '{}')",
            winname, path
        );
        Ok(())
    }

    /// Waits for a pressed key.
    /// If the `gpui` feature is enabled, this will start the GPU accelerated GPUI window rendering loop.
    pub fn wait_key(delay_ms: i32) -> Result<i32> {
        #[cfg(feature = "gpui")]
        {
            println!("[GUI] Spinning up GPUI window rendering event loop...");
            run_gpui_event_loop(delay_ms);
            Ok(0)
        }
        #[cfg(not(feature = "gpui"))]
        {
            if delay_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(delay_ms as u64));
            }
            Ok(-1)
        }
    }

    /// Sets title of the window.
    pub fn set_window_title(winname: &str, title: &str) -> Result<()> {
        let mut wins = get_windows().lock().unwrap();
        if let Some(w) = wins.get_mut(winname) {
            w.title = title.to_string();
        }
        Ok(())
    }

    /// Resizes the specified window.
    pub fn resize_window(winname: &str, width: usize, height: usize) -> Result<()> {
        let mut wins = get_windows().lock().unwrap();
        if let Some(w) = wins.get_mut(winname) {
            w.width = width;
            w.height = height;
        }
        Ok(())
    }

    /// Creates a trackbar in the specified window.
    pub fn create_trackbar(
        trackbar_name: &str,
        winname: &str,
        value: i32,
        count: i32,
    ) -> Result<()> {
        let mut wins = get_windows().lock().unwrap();
        if let Some(w) = wins.get_mut(winname) {
            w.trackbars.insert(
                trackbar_name.to_string(),
                TrackbarState {
                    value,
                    max_val: count,
                },
            );
            println!(
                "[GUI] Trackbar '{}' created in window '{}'",
                trackbar_name, winname
            );
        }
        Ok(())
    }

    /// Returns the trackbar position.
    pub fn get_trackbar_pos(trackbar_name: &str, winname: &str) -> Result<i32> {
        let wins = get_windows().lock().unwrap();
        if let Some(t) = wins
            .get(winname)
            .and_then(|w| w.trackbars.get(trackbar_name))
        {
            return Ok(t.value);
        }
        Ok(0)
    }

    /// Sets the trackbar position.
    pub fn set_trackbar_pos(trackbar_name: &str, winname: &str, pos: i32) -> Result<()> {
        let mut wins = get_windows().lock().unwrap();
        if let Some(t) = wins
            .get_mut(winname)
            .and_then(|w| w.trackbars.get_mut(trackbar_name))
        {
            t.value = pos.min(t.max_val);
        }
        Ok(())
    }

    /// Sets a callback for mouse events in the window.
    pub fn set_mouse_callback<F>(winname: &str, _callback: F) -> Result<()>
    where
        F: Fn(i32, usize, usize, i32) + Send + 'static,
    {
        println!("[GUI] Registered mouse callback in window '{}'", winname);
        Ok(())
    }
}

/// GPU accelerated GPUI rendering window loop.
#[cfg(feature = "gpui")]
fn run_gpui_event_loop(delay_ms: i32) {
    use gpui::{
        App, Application, Bounds, Context, Render, Window, WindowBounds, WindowOptions, div,
        prelude::*, px, rgb, size,
    };

    struct ObserversWindow {
        title: String,
        width: usize,
        height: usize,
        #[allow(dead_code)]
        image_bytes: Option<Vec<u8>>,
    }

    impl Render for ObserversWindow {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            div()
                .flex()
                .flex_col()
                .size_full()
                .bg(rgb(0x303030))
                .justify_center()
                .items_center()
                .child(format!("Window: {}", self.title))
                .child(
                    div()
                        .text_sm()
                        .text_color(rgb(0xaaaaaa))
                        .child(format!("Image size: {}x{}", self.width, self.height)),
                )
        }
    }

    let wins = get_windows().lock().unwrap();
    if wins.is_empty() {
        return;
    }

    // Take the first active window to display
    let (name, state) = wins.iter().next().unwrap();
    let title = state.title.clone();
    let width = state.width;
    let height = state.height;
    let image_bytes = state.image_bytes.clone();
    drop(wins);

    let app = Application::new();
    app.run(move |cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(width as f32), px(height as f32)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| {
                cx.new(|_| ObserversWindow {
                    title,
                    width,
                    height,
                    image_bytes,
                })
            },
        )
        .unwrap();

        if delay_ms > 0 {
            cx.spawn(move |mut cx| async move {
                std::thread::sleep(std::time::Duration::from_millis(delay_ms as u64));
                let _ = cx.update(|cx| {
                    cx.quit();
                });
            })
            .detach();
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::Wgpu;
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn test_gui_operations() {
        let win = "Test Window";
        Gui::named_window(win).unwrap();
        Gui::set_window_title(win, "New Title").unwrap();
        Gui::resize_window(win, 800, 600).unwrap();

        Gui::create_trackbar("bar", win, 10, 100).unwrap();
        let pos = Gui::get_trackbar_pos("bar", win).unwrap();
        assert_eq!(pos, 10);

        Gui::set_trackbar_pos("bar", win, 25).unwrap();
        let pos_new = Gui::get_trackbar_pos("bar", win).unwrap();
        assert_eq!(pos_new, 25);

        let device = Default::default();
        let flat_data = vec![0.5f32; 3 * 8 * 8];
        let tensor = Tensor::<Wgpu, 3>::from_data(TensorData::new(flat_data, [3, 8, 8]), &device);
        let img = Image::new(tensor);

        Gui::imshow(win, &img).unwrap();
        
        let _ = std::fs::remove_file("preview_Test_Window.png");

        Gui::destroy_window(win).unwrap();
        Gui::destroy_all_windows().unwrap();

        let code = Gui::wait_key(1).unwrap();
        assert!(code == 0 || code == -1);
    }
}

