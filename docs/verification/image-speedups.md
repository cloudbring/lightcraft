# Image speedup remeasurement: lightcraft (artcraft-algorithms#190, claims C04, C07, C11)

Measured 2026-11-04 (evening, US Eastern) on Apple M6 (Metal, integrated GPU), 12 logical cores, 32 GiB, macOS (Darwin 27.0.0),
rustc 1.99.0, release profile as in the lightcraft workspace. Harness: `tools/perf-remeasure/` (standalone workspace; craft-bench at
cloudbring/craft-libs rev 94c6d76e5bc2c936d3db5ce5e19d70213e8390c0). Raw per-run JSON, fingerprint, revs: `docs/verification/data/*.json`.

## Method

- OLD = storytold/lightcraft `06c1b9ae0775464ce83ef0a9588be770337600f1` (merge-base of all source branches with origin/main).
- NEW: C04 = `f9d3651c18faeb756748939673c94bdafb4c3d86` (tip of algo/29-median-filter; the ticket's 11.6x was measured at 4d3c434e, f9d3651c adds the
  selection network for every r <= 24). C07 = `63b2f24add7ab02ae52078d3be5257e1ff0ce12a` (algo/28-gpu-box-blur-scan). C11 = `39aa4d256939c3703f1e22c0f2a715a4c98a189f`
  (algo/176-vectorised-recurrence, "use this branch"). The worktree sits on f9d3651c; for C11 `crates/raster` was temporarily checked out from the
  176 branch (`git checkout 39aa4d25 -- crates/raster`) and restored afterwards, so repeat that checkout to rebuild C11.
- **GPU claims (C04, C07) do not call lightcraft-gpu.** `box_pass` and `median_hv` are private there, so the harness drives the same WGSL kernels on the
  same Metal device through its own wgpu code (`src/gpu.rs`), with the shaders and dispatch rules vendored with citations: OLD `blur.wgsl` at 06c1b9ae,
  NEW `blur.wgsl` at 63b2f24a (box) and f9d3651c (median) plus the vendored selection-network generator from `ctx.rs@f9d3651c`. This reproduces the
  kernels and dispatch choice, not the app's buffer pool or error scopes. Timed region: encode, submit, wait for completion; buffers are preallocated;
  the parameter buffer is created per dispatch as `Gpu::run` does. The app's own `bench_median_29`/`bench_kernels` ignored tests were not used.
- C11 uses the public `lightcraft_raster::blur::gaussian` of OLD (renamed git dep) and NEW (path), 6000x4000 f32 plane, run once with
  `RAYON_NUM_THREADS=1` (as the claim) and once with all cores.
- Paired ABBA, 20 timed runs per side per input, 2 warm-ups, 2000 bootstrap resamples. Speedup = old/new; CI = bootstrap 95% CI of the median ratio
  (geometric mean over inputs for the aggregate rows in the table; per-input rows below). CPU-time columns are process CPU time and are meaningless
  for GPU work (the CPU waits).
- Load: 3 to 4.5 of 12 cores during the runs (other processes on this Mac), under the guard limit of 6.0 but not idle. One earlier attempt hit load 9 and was
  discarded (marked contended, rerun).

## Inputs

- Real: the 52 distinct JPEGs extracted from the office corpus (`word/media/*.jpeg`, >= 40 KB, sha256-deduplicated), converted to linear luma. They are small
  (largest 2048x1365), so the 24 MP inputs are a **mosaic** of those photos shelf-packed to 6000x4000 (the claim's size), plus the three largest photos at native size.
  Lightcraft's raw corpus is not on this machine, so no camera raws.
- Synthetic (as the original claims): `((x*37+y*73)%256)/255` at 6000x4000.
- Outputs: median bit-identical on every input (0 differing pixels); box blur max abs difference 3.6e-6 (limit 1e-4, the ticket quoted 1.2e-5 vs CPU); read back from the GPU and compared
  out-of-band before timing (the craft-bench check closure is a no-op for unit outputs). Gaussian: NEW `gaussian_box` is bit-identical to OLD `gaussian` on all
  6 input/sigma pairs (legacy behaviour preserved); NEW `gaussian` differs from OLD **by design** (variance-exact, #176). Against an f64 direct convolution on a 512x512
  crop the max abs error is 0.0009/0.029/0.011 (new) vs 0.135/0.121/0.080 (old) at sigma 1/8/64, but the mean abs error at sigma 8 is higher for the new filter
  (0.0023 vs 0.0013) and at sigma 64 slightly lower (0.0019 vs 0.0026), so accuracy is better at the extremes and mixed at sigma 8 on this crop.

## Results

| run | inputs | wall speedup (median ratio old/new) [95% CI] | CPU-time speedup [95% CI] | load before/after (limit) | publishable |
|---|---|---|---|---|---|
| C04-gpu-median-r24 | 5 | 13.43x [13.34, 13.48] | 1.08x [0.92, 1.20] | 3.80 / 3.39 (6.0) | yes |
| C04-gpu-median-r8 | 5 | 7.01x [6.97, 7.08] | 1.10x [1.04, 1.19] | 4.05 / 3.80 (6.0) | yes |
| C07-gpu-box-h-r128 | 5 | 5.89x [5.87, 5.93] | 1.00x [0.92, 1.08] | 3.20 / 3.20 (6.0) | yes |
| C07-gpu-box-h-r32 | 5 | 1.62x [1.61, 1.64] | 1.00x [0.95, 1.06] | 3.39 / 3.20 (6.0) | yes |
| C11-gaussian-sigma1-1thread | 2 | 0.94x [0.94, 0.95] | 0.94x [0.94, 0.95] | 3.30 / 3.19 (6.0) | yes |
| C11-gaussian-sigma1-alltreads | 2 | 1.13x [1.10, 1.15] | 0.94x [0.93, 0.95] | 3.24 / 3.24 (6.0) | yes |
| C11-gaussian-sigma64-1thread | 2 | 1.50x [1.48, 1.51] | 1.50x [1.49, 1.50] | 3.10 / 3.24 (6.0) | yes |
| C11-gaussian-sigma64-alltreads | 2 | 1.54x [1.52, 1.58] | 1.74x [1.73, 1.76] | 3.24 / 4.50 (6.0) | yes |
| C11-gaussian-sigma8-1thread | 2 | 1.30x [1.29, 1.30] | 1.30x [1.29, 1.30] | 3.19 / 3.10 (6.0) | yes |
| C11-gaussian-sigma8-alltreads | 2 | 1.19x [1.17, 1.21] | 1.29x [1.28, 1.30] | 3.24 / 3.24 (6.0) | yes |

Per input (wall, median old -> new, speedup [95% CI]):

- C04 median r=24: synthetic 6000x4000 322.3 -> 22.7 ms, 14.21x [14.17, 14.24]; real mosaic 321.4 -> 22.7 ms, 14.18x [14.01, 14.22]; native photos 13.2x, 13.2x, 12.5x.
- C04 median r=8: 8.29x [8.25, 8.33] synthetic, 8.27x mosaic.
- C07 box r=128 (horizontal, 1 channel): synthetic 59.0 -> 8.8 ms, 6.73x [6.70, 6.74]; mosaic 6.72x [6.70, 6.74]; native photos 5.9x, 5.7x, 4.7x. r=32: 1.75x.
- C11 single thread: sigma 1 0.94x [0.94, 0.95] (32 -> 34 ms); sigma 8 1.30x [1.29, 1.30] (92 -> 71 ms); sigma 64 1.50x [1.49, 1.50] (106-107 -> 71 ms). All cores: 1.13x, 1.19x, 1.54x.

## Verdicts

| claim | claimed | measured wall speedup [95% CI] | verdict |
|---|---|---|---|
| C04 GPU median r=24 | 11.6x (374.5 -> 32.2 ms) | synthetic 14.21x [14.17, 14.24]; real mosaic 14.18x [14.01, 14.22]; native photos 12.5x to 13.2x | **reproduced** (higher than claimed: the CI excludes 11.6x on the high side; this OLD is the merge-base float-compare shader, the ticket's baseline was the slower total-order variant, and absolute times differ by machine state). Output bit-identical. |
| C07 GPU box blur r=128 | 6.8x (59.4 -> 8.8 ms) | synthetic 6.73x [6.70, 6.74]; real mosaic 6.72x [6.70, 6.74]; native photos 4.7x to 5.9x | **reproduced** (59.0 -> 8.8 ms matches the ticket's times; smaller images gain less). |
| C11 Gaussian 1.4-1.6x at sigma>=8 (status page says 1.5x) | 1.4-1.6x, single thread, 24 MP | sigma 8: 1.30x [1.29, 1.30]; sigma 64: 1.50x [1.49, 1.50]; sigma 1: 0.94x. All cores: 1.19x and 1.54x | **partially**: sigma 64 reproduces; sigma 8 is 1.3x, below the claimed range (the ticket's own pass-2 numbers give 97/70.5 = 1.38x at sigma 8). The result is a different, variance-exact filter, not an identical one. |

## Caveats

- Real inputs are small photos and a mosaic of them; kernel time here does not depend on pixel content (neither kernel branches on data), so the real and synthetic rows agree.
- GPU timing is wall clock of a synchronous submit on an integrated GPU shared with the window server and other apps.
- The GPU reproduction re-implements the dispatch in the harness (see Method), it does not exercise lightcraft-gpu's own code path.
