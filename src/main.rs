// ██╗    ██╗ ██████╗ ███████╗██╗         ████████╗ ██████╗ ██╗   ██╗
// ██║    ██║██╔════╝ ██╔════╝██║         ╚══██╔══╝██╔═══██╗╚██╗ ██╔╝
// ██║ █╗ ██║██║  ███╗███████╗██║            ██║   ██║   ██║ ╚████╔╝
// ██║███╗██║██║   ██║╚════██║██║            ██║   ██║   ██║  ╚██╔╝
// ╚███╔███╔╝╚██████╔╝███████║███████╗       ██║   ╚██████╔╝   ██║
//  ╚══╝╚══╝  ╚═════╝ ╚══════╝╚══════╝       ╚═╝    ╚═════╝    ╚═╝

// High-performance shader renderer using wgpu
// Live preview with winit or render to video file
// Inspired by ShaderToy

// Copyright 2025 Servus Altissimi (Pseudonym)

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use anyhow::{Result, anyhow};
use clap::Parser;
use std::borrow::Cow;
use std::fs;
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{EventLoop, ActiveEventLoop},
    window::{Window, WindowId},
};

const FRAMES_AHEAD: usize = 3; // don't touch

// CL arguments for config
#[derive(Parser, Debug)]
#[command(author, version, about = "WGSL Toy, preview or render shaders.", long_about = None)]
struct Args {
    #[arg(help = "Path to WGSL shader file")]
    shader: String,

    #[arg(short, long, help = "Render to video instead of live preview")]
    render: bool,

    #[arg(short, long, help = "Duration in seconds (required for --render)")]
    duration: Option<u32>,

    #[arg(short, long, default_value = "output.mp4")]
    output: String,

    #[arg(short = 'W', long, default_value_t = 1920)] // to be consistent
    width: u32,

    #[arg(short = 'H', long, default_value_t = 1080)]
    height: u32,

    #[arg(short, long, default_value_t = 60)]
    fps: u32,

    #[arg(long, default_value_t = 18)]
    crf: u32,

    #[arg(long, default_value = "ultrafast")]
    preset: String,

    #[arg(short, long, default_value_t = false)]
    verbose: bool,
}

// GPU vertex data: fullscreen quad covering NDC [-1,1].
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    resolution: [f32; 2],
    time: f32,
    _padding: f32, // 16-byte alignment padding
}

const VERTICES: &[Vertex] = &[
    Vertex { position: [-1.0, -1.0] },
    Vertex { position: [1.0, -1.0] },
    Vertex { position: [-1.0, 1.0] },
    Vertex { position: [1.0, -1.0] },
    Vertex { position: [1.0, 1.0] },
    Vertex { position: [-1.0, 1.0] },
];

struct FrameBuffer {
    output_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

struct Renderer {
    instance: Arc<wgpu::Instance>, // Shared WGPU instance to avoid device ID conflicts between offscreen and window rendering
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    width: u32,
    height: u32,
    verbose: bool,
    // Store padding info to avoid recalculating
    padded_bytes_per_row: u32,
    unpadded_bytes_per_row: u32,
    surface_format: wgpu::TextureFormat, // Store format for consistency
}

// Steps:
// 1. Create WGPU instance and select high-performance GPU adapter
// 2. Request logical device + queue from the adapter
// 3. Create shader module from WGSL source
// 4. Define uniform layout and pipeline layout
// 5. Create render pipeline (vertex + fragment stages)
// 6. Initialize fullscreen quad vertex buffer
impl Renderer {
    async fn new(shader_source: &str, width: u32, height: u32, verbose: bool) -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|e| anyhow!("Failed to find any GPU adapter: {:?}", e))?;

        if verbose {
            println!("[GPU] Using adapter: {:?}", adapter.get_info());
        }

        let (device, queue): (wgpu::Device, wgpu::Queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let instance = Arc::new(instance); // Wrap instance in Arc for sharing

        let surface_format = wgpu::TextureFormat::Bgra8UnormSrgb;

        if verbose {
            println!("[INIT] Using surface format: {:?}", surface_format);
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_source)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x2,
                    }],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Calculate padding once during initialization
        let unpadded_bytes_per_row = 4 * width;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = (unpadded_bytes_per_row + align - 1) / align * align;

        if verbose {
            println!("[INIT] Renderer initialized");
            println!("[INIT] Unpadded bytes per row: {}", unpadded_bytes_per_row);
            println!("[INIT] Padded bytes per row: {}", padded_bytes_per_row);
        }

        Ok(Self {
            instance,
            device,
            queue,
            pipeline,
            vertex_buffer,
            bind_group_layout,
            width,
            height,
            verbose,
            padded_bytes_per_row,
            unpadded_bytes_per_row,
            surface_format,
        })
    }

    fn create_offscreen_buffers(&self) -> (wgpu::Texture, wgpu::TextureView, Vec<FrameBuffer>) {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Render Texture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let output_buffer_size = (self.padded_bytes_per_row * self.height) as wgpu::BufferAddress;

        let frame_buffers: Vec<FrameBuffer> = (0..FRAMES_AHEAD)
            .map(|i| {
                let uniform_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("Uniform Buffer {}", i)),
                    size: std::mem::size_of::<Uniforms>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("Bind Group {}", i)),
                    layout: &self.bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    }],
                });

                let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("Output Buffer {}", i)),
                    size: output_buffer_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });

                FrameBuffer {
                    output_buffer,
                    uniform_buffer,
                    bind_group,
                }
            })
            .collect();

        if self.verbose {
            println!("[INIT] Created {} framebuffers for pipelining", FRAMES_AHEAD);
        }

        (texture, texture_view, frame_buffers)
    }

    fn submit_frame_offscreen(
        &self,
        frame: u32,
        fps: u32,
        buffer_idx: usize,
        texture: &wgpu::Texture,
        texture_view: &wgpu::TextureView,
        frame_buffers: &[FrameBuffer],
    ) {
        let time = frame as f32 / fps as f32;

        let uniforms = Uniforms {
            resolution: [self.width as f32, self.height as f32],
            time,
            _padding: 0.0,
        };

        let fb = &frame_buffers[buffer_idx];
        self.queue.write_buffer(&fb.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &fb.bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..6, 0..1);
        }

        // Use pre-calculated padded bytes per row
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &fb.output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(Some(encoder.finish()));
    }

    fn read_frame(&self, buffer_idx: usize, frame_buffers: &[FrameBuffer]) -> Vec<u8> {
        let fb = &frame_buffers[buffer_idx];
        let buffer_slice = fb.output_buffer.slice(..);

        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();

        // Remove padding from the buffer data only if present
        let result = if self.padded_bytes_per_row != self.unpadded_bytes_per_row {
            let mut unpadded = Vec::with_capacity((self.unpadded_bytes_per_row * self.height) as usize);
            for row in 0..self.height {
                let row_start = (row * self.padded_bytes_per_row) as usize;
                let row_end = row_start + self.unpadded_bytes_per_row as usize;
                unpadded.extend_from_slice(&data[row_start..row_end]);
            }
            unpadded
        } else {
            data.to_vec()
        };

        drop(data);
        fb.output_buffer.unmap();

        result
    }

    async fn render_video(&self, args: &Args) -> Result<()> {
        let duration = args.duration.ok_or_else(|| anyhow!("--duration required for --render mode"))?;
        let total_frames = args.fps * duration;

        println!("{}", "=".repeat(64));
        println!("Starting FFMpeg encoding");
        println!("{}\n", "=".repeat(64));

        // I know there's a crate for this
        let mut ffmpeg = Command::new("ffmpeg")
            .args(&[
                "-y",
                "-f", "rawvideo",
                "-pix_fmt", "rgba",
                "-s", &format!("{}x{}", args.width, args.height),
                "-r", &format!("{}", args.fps),
                "-i", "-",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", &format!("{}", args.crf),
                "-preset", &args.preset,
                &args.output,
            ])
            .stdin(Stdio::piped())
            .stderr(if args.verbose { Stdio::inherit() } else { Stdio::null() })
            .spawn()
            .map_err(|e| anyhow!("Failed to start FFMpeg: {}. Make sure FFMpeg is installed and in $PATH.", e))?;

        let mut stdin = ffmpeg.stdin.take().ok_or_else(|| anyhow!("Failed to open FFMpeg stdin"))?;

        let (texture, texture_view, frame_buffers) = self.create_offscreen_buffers();

        // Pre-submit first batch
        for frame in 0..FRAMES_AHEAD.min(total_frames as usize) {
            self.submit_frame_offscreen(frame as u32, args.fps, frame % FRAMES_AHEAD, &texture, &texture_view, &frame_buffers);
        }

        for frame in 0..total_frames {
            let buffer_idx = frame as usize % FRAMES_AHEAD;
            let pixels = self.read_frame(buffer_idx, &frame_buffers);

            // Submit next frame, while encoding current one
            let next_frame = frame + FRAMES_AHEAD as u32;
            if next_frame < total_frames {
                self.submit_frame_offscreen(next_frame, args.fps, buffer_idx, &texture, &texture_view, &frame_buffers);
            }

            stdin.write_all(&pixels)?;

            // Progress bar, TODO implement this design in other projects
            if frame % 60 == 0 || frame == total_frames - 1 {
                let progress = (frame as f32 / total_frames as f32) * 100.0;
                let bar_width = 50;

                let filled = (progress / 100.0 * bar_width as f32) as usize;
                let bar: String = "█".repeat(filled) + &"░".repeat(bar_width - filled);

                print!("\r[{}] {:.1}% ({}/{})", bar, progress, frame + 1, total_frames);
                std::io::stdout().flush()?;
            }
        }

        drop(stdin);
        let status = ffmpeg.wait()?;

        if !status.success() {
            return Err(anyhow!("FFMpeg encoding failed"));
        }

        println!("\n\n{}", "=".repeat(64));
        println!("Video successfully saved");
        println!("{}", "=".repeat(64));
        println!("Output: {}", args.output);
        println!("Resolution: {}x{}", args.width, args.height);
        println!("Duration: {} seconds @ {} fps", duration, args.fps);
        println!("Total frames: {}\n", total_frames);

        Ok(())
    }

    async fn run_window(self, args: Args) -> Result<()> {
        let event_loop = EventLoop::new()?;

        println!("{}", "=".repeat(64));
        println!("Live preview window opened");
        println!("{}", "=".repeat(64));
        println!("Press ESC or close window to exit the previes\n");

        let mut app = App::new(self, args);
        event_loop.run_app(&mut app)?;

        Ok(())
    }
}

struct App {
    renderer: Renderer,
    args: Args,
    window: Option<Arc<Window>>,
    surface: Option<wgpu::Surface<'static>>,
    surface_config: Option<wgpu::SurfaceConfiguration>, // Store config to enable reconfiguration on resize
    uniform_buffer: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,
    start_time: Option<std::time::Instant>,
    current_width: u32,
    current_height: u32,
}

impl App {
    fn new(renderer: Renderer, args: Args) -> Self {
        Self {
            current_width: args.width,
            current_height: args.height,
            renderer,
            args,
            window: None,
            surface: None,
            surface_config: None,
            uniform_buffer: None,
            bind_group: None,
            start_time: None,
        }
    }

    // Helper method to reconfigure surface when window is resized
    fn resize(&mut self, new_width: u32, new_height: u32) {
        if new_width == 0 || new_height == 0 {
            return;
        }

        self.current_width = new_width;
        self.current_height = new_height;

        if let (Some(surface), Some(config)) = (&self.surface, &mut self.surface_config) {
            config.width = new_width;
            config.height = new_height;
            surface.configure(&self.renderer.device, config);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window_attributes = winit::window::Window::default_attributes()
            .with_title("WGSL Renderer")
            .with_inner_size(winit::dpi::PhysicalSize::new(self.args.width, self.args.height));

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        let surface = self.renderer.instance.create_surface(window.clone()).unwrap();

        let adapter = pollster::block_on(self.renderer.instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })).unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: self.args.width,
            height: self.args.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&self.renderer.device, &config);

        let uniform_buffer = self.renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Window Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.renderer.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Window Bind Group"),
            layout: &self.renderer.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        self.window = Some(window);
        self.surface = Some(surface);
        self.surface_config = Some(config);
        self.uniform_buffer = Some(uniform_buffer);
        self.bind_group = Some(bind_group);
        self.start_time = Some(std::time::Instant::now());
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event: winit::event::KeyEvent {
                    physical_key: winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape),
                    ..
                },
                ..
            } => {
                event_loop.exit();
            }

            WindowEvent::Resized(physical_size) => {
                self.resize(physical_size.width, physical_size.height);
            }

            WindowEvent::RedrawRequested => {
                let time = self.start_time.unwrap().elapsed().as_secs_f32();

                // Use current window dimensions for uniform resolution
                let uniforms = Uniforms {
                    resolution: [self.current_width as f32, self.current_height as f32],
                    time,
                    _padding: 0.0,
                };

                self.renderer.queue.write_buffer(self.uniform_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&[uniforms]));

                let output = match self.surface.as_ref().unwrap().get_current_texture() {
                    Ok(output) => output,
                    Err(e) => {
                        eprintln!("Surface error: {:?}", e);
                        return;
                    }
                };

                let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = self.renderer.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Render Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });

                    render_pass.set_pipeline(&self.renderer.pipeline);
                    render_pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
                    render_pass.set_vertex_buffer(0, self.renderer.vertex_buffer.slice(..));
                    render_pass.draw(0..6, 0..1);
                }

                self.renderer.queue.submit(Some(encoder.finish()));
                output.present();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    if args.render && args.duration.is_none() {
        eprintln!("Error: --duration required when using --render\n");
        eprintln!("Example: program shader.wgsl --render --duration 10");
        std::process::exit(1);
    }

    let shader_source = fs::read_to_string(&args.shader)
        .map_err(|e| anyhow!("Failed to read shader file '{}': {}", args.shader, e))?;

    println!("\n{}", "=".repeat(64));
    println!("   WGSL Shader Renderer");
    println!("{}", "=".repeat(64));
    println!("\nShader: {}", args.shader);

    if args.render {
        println!("Mode: Render to video");
        println!("Output: {}", args.output);
        println!("Duration: {} seconds, {} fps", args.duration.unwrap(), args.fps);
        println!("CRF: {} (preset: {})", args.crf, args.preset);
    } else {
        println!("Mode: Live preview");
    }

    println!("Resolution: {}*{}\n", args.width, args.height);

    let renderer = Renderer::new(&shader_source, args.width, args.height, args.verbose).await?;

    if args.render {
        renderer.render_video(&args).await?;
    } else {
        renderer.run_window(args).await?;
    }

    Ok(())
}
