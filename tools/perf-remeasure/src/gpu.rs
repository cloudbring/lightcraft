//! Headless wgpu driver for the GPU claims (artcraft-algorithms#190: C04 median r=24, C07 box blur r=128).
//!
//! `lightcraft-gpu` keeps `box_pass` and `median_hv` private, so they cannot be called from a paired
//! harness. This module re-creates exactly what they do, from vendored sources:
//! - OLD shaders: `vendor/old_blur.wgsl` = crates/gpu/src/wgsl/blur.wgsl at 06c1b9ae0775464ce83ef0a9588be770337600f1
//!   (merge-base with origin/main). OLD dispatch (render.rs there): `box_h` for every horizontal pass,
//!   workgroups groups2(w,h,[64,4]); median `median_h`/`median_v`, groups2(w,h,[16,16]), params [w,h,1,r].
//! - NEW box: `vendor/new28_blur.wgsl` = blur.wgsl at 63b2f24add7ab02ae52078d3be5257e1ff0ce12a (algo/28-gpu-box-blur-scan).
//!   NEW dispatch (render.rs `box_pass` at 63b2f24a, identical at f9d3651c): horizontal with r in 32..=255 runs
//!   `box_h_scan`, groups2(w,h,[256,1]); otherwise `box_h`.
//! - NEW median: `vendor/new29_blur.wgsl` + generated selection network (`vendor/median_network.rs`) at
//!   f9d3651c18faeb756748939673c94bdafb4c3d86 (algo/29-median-filter); kernel `median_h_{r}`/`median_v_{r}` for r<=24,
//!   as `render.rs::median_hv` picks them.
//! Module assembly follows ctx.rs `module_source` (bindings P, src, dst, then `pu`), minus the generated constants,
//! which blur.wgsl does not use. Device: default adapter, high-performance preference, adapter limits (as `Gpu::new`).
use std::collections::HashMap;
use wgpu::util::DeviceExt;

pub struct Dev {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
}

pub struct Kern {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

pub struct Set(pub HashMap<String, Kern>);

pub fn dev() -> Dev {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).expect("no GPU adapter");
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("perf-remeasure"), required_limits: adapter.limits(), ..Default::default() })).expect("device");
    Dev { device, queue, info }
}

fn entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

const HEADER: &str = "@group(0) @binding(0) var<storage, read> P: array<u32>;\n@group(0) @binding(1) var<storage, read> src: array<f32>;\n@group(0) @binding(2) var<storage, read_write> dst: array<f32>;\nfn pu(i: u32) -> u32 { return P[i]; }\nfn pf(i: u32) -> f32 { return bitcast<f32>(P[i]); }\n";

/// One shader module (`blur_src` + `extra`) and the pipelines for `entries`.
pub fn build(d: &Dev, blur_src: &str, extra: &str, entries: &[String], into: &mut Set) {
    let src = format!("{HEADER}{blur_src}{extra}");
    let module = d.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(src.into()) });
    let layout = d.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[entry(0, true), entry(1, true), entry(2, false)] });
    let pl = d.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
    for e in entries {
        let pipeline = d.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(e), layout: Some(&pl), module: &module, entry_point: Some(e), compilation_options: Default::default(), cache: None });
        into.0.insert(e.clone(), Kern { pipeline, layout: layout.clone() });
    }
}

pub fn groups2(w: usize, h: usize, wg: [usize; 2]) -> [u32; 3] {
    [w.div_ceil(wg[0]).max(1) as u32, h.div_ceil(wg[1]).max(1) as u32, 1]
}

pub fn storage(d: &Dev, words: usize) -> wgpu::Buffer {
    d.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (words.max(1) * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
}

pub fn upload(d: &Dev, data: &[f32]) -> wgpu::Buffer {
    d.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST })
}

/// Record kernel `name` (params `p`, `src` -> `dst`) into `enc`, like `Gpu::run`.
pub fn run(d: &Dev, s: &Set, enc: &mut wgpu::CommandEncoder, name: &str, p: &[u32], src: &wgpu::Buffer, dst: &wgpu::Buffer, groups: [u32; 3]) {
    let k = s.0.get(name).unwrap_or_else(|| panic!("no kernel {name}"));
    let params = d.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(p), usage: wgpu::BufferUsages::STORAGE });
    let bg = d.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &k.layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: src.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: dst.as_entire_binding() },
        ],
    });
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(name), timestamp_writes: None });
    pass.set_pipeline(&k.pipeline);
    pass.set_bind_group(0, &bg, &[]);
    pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
}

/// Submit and wait for completion (the timed region of every GPU measurement).
pub fn submit_wait(d: &Dev, enc: wgpu::CommandEncoder) {
    d.queue.submit([enc.finish()]);
    d.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).expect("poll");
}

pub fn read(d: &Dev, b: &wgpu::Buffer, words: usize) -> Vec<f32> {
    let size = (words * 4) as u64;
    let staging = d.device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let mut enc = d.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(b, 0, &staging, 0, size);
    d.queue.submit([enc.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    d.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).expect("poll");
    rx.recv().unwrap().expect("map");
    let v = bytemuck::cast_slice::<u8, f32>(&staging.slice(..).get_mapped_range().expect("range")).to_vec();
    staging.unmap();
    v
}
