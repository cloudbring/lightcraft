//! Paired remeasurement of lightcraft speedup claims (artcraft-algorithms#190).
//! Usage: perf-remeasure <median|box|gaussian> <out_dir> <photos_dir>
//! Env: OLD_REV, NEW_REV (recorded), REPS (default 20). Gaussian is run with RAYON_NUM_THREADS=1 and unset.
mod common;
mod gpu;
#[path = "../vendor/median_network.rs"]
mod median_network;
use common::*;
use craft_bench::{report::Revs, Config, Paired};
use serde_json::json;

fn cfg() -> Config {
    let reps = std::env::var("REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    Config { repetitions: reps, revs: Revs { old: std::env::var("OLD_REV").ok(), new: std::env::var("NEW_REV").ok() }, ..Config::default() }
}

const OLD_BLUR: &str = include_str!("../vendor/old_blur.wgsl");
const NEW28_BLUR: &str = include_str!("../vendor/new28_blur.wgsl");
const NEW29_BLUR: &str = include_str!("../vendor/new29_blur.wgsl");

// ----------------------------------------------------------------- inputs

fn luma_planes(photos: &str) -> Vec<(String, usize, usize, Vec<f32>)> {
    load_photos(photos).into_iter().map(|(n, w, h, px)| (n, w, h, px.iter().map(|p| 0.2126 * srgb_to_linear(p[0]) + 0.7152 * srgb_to_linear(p[1]) + 0.0722 * srgb_to_linear(p[2])).collect())).collect()
}

/// Shelf-pack the real photo luma planes (cycling through them) into one w x h plane.
fn mosaic(ph: &[(String, usize, usize, Vec<f32>)], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    let (mut x, mut y, mut row_h, mut k) = (0usize, 0usize, 0usize, 0usize);
    while y < h {
        let (_, pw, ph_, d) = &ph[k % ph.len()];
        k += 1;
        if x + pw > w { x = 0; y += row_h; row_h = 0; if y >= h { break; } }
        if *pw > w { continue; }
        for yy in 0..(*ph_).min(h - y) { out[(y + yy) * w + x..(y + yy) * w + x + pw].copy_from_slice(&d[yy * pw..(yy + 1) * pw]); }
        x += pw;
        row_h = row_h.max(*ph_);
    }
    out
}

fn synthetic(w: usize, h: usize) -> Vec<f32> {
    (0..w * h).map(|i| { let (x, y) = (i % w, i / w); ((x * 37 + y * 73) % 256) as f32 / 255.0 }).collect()
}

struct Case {
    w: usize,
    h: usize,
    src: wgpu::Buffer,
    tmp_o: wgpu::Buffer,
    out_o: wgpu::Buffer,
    tmp_n: wgpu::Buffer,
    out_n: wgpu::Buffer,
}

fn cases(d: &gpu::Dev, photos: &str) -> Vec<(String, Case)> {
    let ph = luma_planes(photos);
    let mut sets: Vec<(String, usize, usize, Vec<f32>)> = vec![("synthetic 6000x4000".into(), 6000, 4000, synthetic(6000, 4000)), ("real-photo mosaic 6000x4000 (52 JPEGs)".into(), 6000, 4000, mosaic(&ph, 6000, 4000))];
    let mut big: Vec<_> = ph.iter().collect();
    big.sort_by_key(|p| std::cmp::Reverse(p.1 * p.2));
    for (n, w, h, p) in big.into_iter().take(3) { sets.push((format!("real photo {n} native {w}x{h}"), *w, *h, p.clone())); }
    sets.into_iter().map(|(n, w, h, data)| {
        let c = Case { w, h, src: gpu::upload(d, &data), tmp_o: gpu::storage(d, w * h), out_o: gpu::storage(d, w * h), tmp_n: gpu::storage(d, w * h), out_n: gpu::storage(d, w * h) };
        (n, c)
    }).collect()
}

// ----------------------------------------------------------------- GPU claims

fn encoder(d: &gpu::Dev) -> wgpu::CommandEncoder { d.device.create_command_encoder(&Default::default()) }

fn median_old(d: &gpu::Dev, s: &gpu::Set, c: &Case, r: usize) {
    let p = [c.w as u32, c.h as u32, 1, r as u32];
    let mut e = encoder(d);
    gpu::run(d, s, &mut e, "median_h", &p, &c.src, &c.tmp_o, gpu::groups2(c.w, c.h, [16, 16]));
    gpu::run(d, s, &mut e, "median_v", &p, &c.tmp_o, &c.out_o, gpu::groups2(c.w, c.h, [16, 16]));
    gpu::submit_wait(d, e);
}
fn median_new(d: &gpu::Dev, s: &gpu::Set, c: &Case, r: usize) {
    let p = [c.w as u32, c.h as u32, 1, r as u32];
    let (h, v) = (format!("median_h_{r}"), format!("median_v_{r}"));
    let mut e = encoder(d);
    gpu::run(d, s, &mut e, &h, &p, &c.src, &c.tmp_n, gpu::groups2(c.w, c.h, [16, 16]));
    gpu::run(d, s, &mut e, &v, &p, &c.tmp_n, &c.out_n, gpu::groups2(c.w, c.h, [16, 16]));
    gpu::submit_wait(d, e);
}
fn box_old(d: &gpu::Dev, s: &gpu::Set, c: &Case, r: usize) {
    let chunk = (2 * r + 1).clamp(16, 128);
    let p = [c.w as u32, c.h as u32, 1, r as u32, chunk as u32];
    let mut e = encoder(d);
    gpu::run(d, s, &mut e, "box_h", &p, &c.src, &c.out_o, gpu::groups2(c.w, c.h, [64, 4]));
    gpu::submit_wait(d, e);
}
fn box_new(d: &gpu::Dev, s: &gpu::Set, c: &Case, r: usize) {
    let chunk = (2 * r + 1).clamp(16, 128);
    let p = [c.w as u32, c.h as u32, 1, r as u32, chunk as u32];
    let (k, g) = if (32..=255).contains(&r) { ("box_h_scan", gpu::groups2(c.w, c.h, [256, 1])) } else { ("box_h", gpu::groups2(c.w, c.h, [64, 4])) };
    let mut e = encoder(d);
    gpu::run(d, s, &mut e, k, &p, &c.src, &c.out_n, g);
    gpu::submit_wait(d, e);
}

fn run_gpu(out: &str, photos: &str, what: &str) {
    let d = gpu::dev();
    println!("GPU: {} ({:?}, {:?})", d.info.name, d.info.backend, d.info.device_type);
    let mut old = gpu::Set(Default::default());
    gpu::build(&d, OLD_BLUR, "", &["box_h", "median_h", "median_v"].map(String::from), &mut old);
    let mut new = gpu::Set(Default::default());
    let gpu_meta = json!({"adapter": d.info.name, "backend": format!("{:?}", d.info.backend), "device_type": format!("{:?}", d.info.device_type), "driver": d.info.driver, "driver_info": d.info.driver_info});
    if what == "median" {
        for r in 1..=24usize {
            gpu::build(&d, NEW29_BLUR, &median_network::median_network_source(r), &[format!("median_h_{r}"), format!("median_v_{r}")], &mut new);
        }
    } else {
        gpu::build(&d, NEW28_BLUR, "", &["box_h", "box_h_scan"].map(String::from), &mut new);
    }
    let cs = cases(&d, photos);
    let radii: &[usize] = if what == "median" { &[8, 24] } else { &[32, 128] };
    for &r in radii {
        // Out-of-band equality check (readback), then timing.
        let mut ok = Vec::new();
        let mut dropped = Vec::new();
        let mut max_diff = 0.0f64;
        for (name, c) in &cs {
            let n = c.w * c.h;
            let diff = if what == "median" {
                median_old(&d, &old, c, r); median_new(&d, &new, c, r);
                let (a, b) = (gpu::read(&d, &c.out_o, n), gpu::read(&d, &c.out_n, n));
                a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count() as f64
            } else {
                box_old(&d, &old, c, r); box_new(&d, &new, c, r);
                let (a, b) = (gpu::read(&d, &c.out_o, n), gpu::read(&d, &c.out_n, n));
                a.iter().zip(&b).map(|(x, y)| (x - y).abs() as f64).fold(0.0, f64::max)
            };
            max_diff = max_diff.max(diff);
            let pass = if what == "median" { diff == 0.0 } else { diff <= 1e-4 };
            if pass { ok.push(name.clone()) } else { dropped.push(json!({"input": name, "reason": format!("diff {diff} (median: count of differing pixels; box: max abs)")})); }
        }
        println!("r={r}: equality max_diff {max_diff}, dropped {}", dropped.len());
        let items: Vec<(String, &Case)> = cs.iter().filter(|(n, _)| ok.contains(n)).map(|(n, c)| (n.clone(), c)).collect();
        let (label, claim, tol) = if what == "median" { (format!("C04-gpu-median-r{r}"), "C04 GPU median r=24 11.6x (6000x4000, 374.5 ms -> 32.2 ms)", "bit-identical (total order); count of differing pixels") } else { (format!("C07-gpu-box-h-r{r}"), "C07 GPU box blur r=128 6.8x (horizontal pass, 6000x4000, 1 channel, 59.4 ms -> 8.8 ms)", "max abs diff <= 1e-4 (documented CPU parity error 1.2e-5)") };
        let rep = Paired::new(format!("lightcraft GPU {what} r={r} on {} ({:?})", d.info.name, d.info.backend))
            .old(|c: &&Case| if what == "median" { median_old(&d, &old, c, r) } else { box_old(&d, &old, c, r) })
            .new(|c: &&Case| if what == "median" { median_new(&d, &new, c, r) } else { box_new(&d, &new, c, r) })
            .inputs(items).check(|_, _| true).run(&cfg()).expect("run");
        write(out, &label, &rep, json!({"claim": claim, "gpu": gpu_meta, "radius": r, "equality": tol, "equality_checked": "out-of-band by readback before timing (craft-bench check closure is a no-op for () outputs)", "max_diff_over_inputs": max_diff, "excluded": dropped,
            "timed_region": "encode + submit + wait for completion; buffers preallocated; parameters buffer created per dispatch like Gpu::run", "vendored": "shaders and dispatch rules: see src/gpu.rs header"}));
    }
}

// ----------------------------------------------------------------- CPU Gaussian (C11)

fn reference_gaussian(w: usize, h: usize, data: &[f32], sigma: f64) -> Vec<f64> {
    let rad = (sigma * 5.0).ceil() as i64;
    let k: Vec<f64> = (-rad..=rad).map(|i| (-(i * i) as f64 / (2.0 * sigma * sigma)).exp()).collect();
    let s: f64 = k.iter().sum();
    let k: Vec<f64> = k.iter().map(|v| v / s).collect();
    let mut tmp = vec![0.0f64; w * h];
    for y in 0..h { for x in 0..w { let mut a = 0.0; for (j, kv) in k.iter().enumerate() { let xx = (x as i64 + j as i64 - rad).clamp(0, w as i64 - 1) as usize; a += kv * data[y * w + xx] as f64; } tmp[y * w + x] = a; } }
    let mut out = vec![0.0f64; w * h];
    for y in 0..h { for x in 0..w { let mut a = 0.0; for (j, kv) in k.iter().enumerate() { let yy = (y as i64 + j as i64 - rad).clamp(0, h as i64 - 1) as usize; a += kv * tmp[yy * w + x]; } out[y * w + x] = a; } }
    out
}

fn run_gaussian(out: &str, photos: &str) {
    let ph = luma_planes(photos);
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into());
    let sets: Vec<(String, Vec<f32>)> = vec![("synthetic 6000x4000".into(), synthetic(6000, 4000)), ("real-photo mosaic 6000x4000 (52 JPEGs)".into(), mosaic(&ph, 6000, 4000))];
    let (w, h) = (6000usize, 4000usize);
    let mk_old = |d: &[f32]| old_raster::Plane::from_fn(w, h, |x, y| d[y * w + x]);
    let mk_new = |d: &[f32]| new_raster::Plane::from_fn(w, h, |x, y| d[y * w + x]);
    // Equivalence: (1) the new crate's legacy `gaussian_box` is bit-identical to OLD `gaussian`;
    // (2) new `gaussian` differs from OLD by design (variance-exact; #176): its error against an f64 direct
    // convolution on a 512x512 crop is recorded next to the old one.
    let mut equality = Vec::new();
    for (name, d) in &sets {
        let (o, n) = (mk_old(d), mk_new(d));
        for sigma in [1.0f32, 8.0, 64.0] {
            let a = old_raster::blur::gaussian(&o, sigma);
            let b = new_raster::blur::gaussian_box(&n, sigma);
            let same = a.data.iter().zip(&b.data).all(|(x, y)| x.to_bits() == y.to_bits());
            equality.push(json!({"input": name, "sigma": sigma, "new_gaussian_box_bit_identical_to_old_gaussian": same}));
            assert!(same, "legacy gaussian_box differs from OLD gaussian at sigma {sigma}");
        }
    }
    let mut accuracy = Vec::new();
    {
        let (cw, ch) = (512usize, 512usize);
        let (_, d) = &sets[1];
        let crop: Vec<f32> = (0..cw * ch).map(|i| d[(i / cw) * w + (i % cw)]).collect();
        let o = old_raster::Plane::from_fn(cw, ch, |x, y| crop[y * cw + x]);
        let n = new_raster::Plane::from_fn(cw, ch, |x, y| crop[y * cw + x]);
        for sigma in [1.0f32, 8.0, 64.0] {
            let rf = reference_gaussian(cw, ch, &crop, sigma as f64);
            let ro = old_raster::blur::gaussian(&o, sigma);
            let rn = new_raster::blur::gaussian(&n, sigma);
            let err = |v: &[f32]| { let (mut mx, mut sum) = (0.0f64, 0.0f64); for (a, b) in v.iter().zip(&rf) { let e = (*a as f64 - b).abs(); mx = mx.max(e); sum += e; } (mx, sum / v.len() as f64) };
            let (eo, en) = (err(&ro.data), err(&rn.data));
            accuracy.push(json!({"sigma": sigma, "crop": "512x512 of real mosaic", "old_boxes": {"max_abs_err": eo.0, "mean_abs_err": eo.1}, "new_recursive": {"max_abs_err": en.0, "mean_abs_err": en.1}}));
        }
    }
    for sigma in [1.0f32, 8.0, 64.0] {
        let items: Vec<(String, (old_raster::Plane, new_raster::Plane))> = sets.iter().map(|(n, d)| (n.clone(), (mk_old(d), mk_new(d)))).collect();
        let rep = Paired::new(format!("lightcraft CPU gaussian sigma={sigma}, 6000x4000 plane, rayon threads {threads}"))
            .old(|i: &(old_raster::Plane, new_raster::Plane)| old_raster::blur::gaussian(&i.0, sigma).data)
            .new(|i: &(old_raster::Plane, new_raster::Plane)| new_raster::blur::gaussian(&i.1, sigma).data)
            .inputs(items).check(|_, _| true).run(&cfg()).expect("run");
        let t = if threads == "1" { "1thread" } else { "alltreads" };
        write(out, &format!("C11-gaussian-sigma{sigma}-{t}"), &rep, json!({"claim": "C11 Gaussian 1.4-1.6x at sigma>=8 (24 MP, single thread; boxes -> recursive)", "rayon_threads": threads, "equality": "NOT bit-equal by design: new gaussian is variance-exact (#176). Checked instead: new gaussian_box == old gaussian bitwise (all inputs/sigmas), and accuracy vs an f64 direct convolution (below)", "legacy_equality": equality, "accuracy_vs_f64_reference": accuracy, "output_checked_in_craft_bench": "no-op closure; checks are out-of-band"}));
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (what, out, photos) = (a[1].as_str(), a[2].as_str(), a.get(3).map_or("", |s| s));
    match what {
        "median" | "box" => run_gpu(out, photos, what),
        "gaussian" => run_gaussian(out, photos),
        _ => eprintln!("median|box|gaussian"),
    }
}
