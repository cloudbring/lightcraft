# Recursive Gaussian CPU filtering

Issue #176 replaces the under-width three-box CPU blur for **process V3**. V1/V2
keep their previous output. This is a numerical-width correction, not a claim of
Lightroom rendering parity. Measurements below are from November 3, 2026.

## Filter and boundaries

`crates/raster/src/blur.rs::gaussian` retains the public generic image API. Built-in
`f32`, RGB and RGBA floating pixels use f64 states throughout both line passes.
Rows and transposed columns are independent rayon jobs; 32-pixel transpose tiles
avoid long strided reads. Below sigma 2.5, the existing variance-exact sampled
kernel helper is used, with f64 accumulation and clamp-edge samples.

For sigma at least 2.5, use Young, van Vliet and van Ginkel, *Recursive Gabor
Filtering*, IEEE Transactions on Signal Processing 50(11), 2798–2805 (2002):
the real pole parameter is 1.16680, the complex pair parameters 1.10783 and
1.40586. The refined polynomial initializes q:

```text
q = -0.2568 + 0.5784 sigma + 0.0561 sigma²       (sigma < 3.556)
q = 2.5091 + 0.9804 (sigma - 3.556)             (otherwise)
```

The polynomial alone is not variance-exact. Refine it with five Newton steps on
the pole family's exact two-sided second moment, `A q² + B q = sigma²`. With
`m0 = 1.16680`, `m1 = 1.10783`, `m2 = 1.40586`, and `N = m1² + m2²`:

```text
A = 2 [1/m0² + 2 (m1² - m2²)/N²]
B = 2 [1/m0 + 2 m1/N]
```

This calibrates width without replacing the third-order pole family. The causal
recurrence is `u[n] = gain x[n] + a1 u[n-1] + a2 u[n-2] + a3 u[n-3]`, followed
by its anticausal counterpart. Calculate gain from the pole factors instead of
subtracting nearly cancelling feedback coefficients. Center the forward state
about the left endpoint and the reverse state about the right endpoint; a
constant line then has exactly zero residual states.

The reverse pass starts from the three terminal forward states using equations
(3) and (14) of Triggs and Sdika, *Boundary Conditions for Young–van Vliet
Recursive Filtering*, IEEE Transactions on Signal Processing 54(6), 2365–2367 (2006),
doi:10.1109/TSP.2006.871980. The initialization models the causal continuation
under a clamped original signal, rather than clamping the intermediate forward
output. Short-line tests compare it against an explicitly extended signal.

Nonpositive/NaN sigma and empty images return a copy. Huge/infinite sigma is
bounded by `MAX_SIGMA` and twice the longer image dimension, as for legacy
rendering. External custom `Pixel` implementations keep their source-compatible
legacy default; they can override `Pixel::gaussian_line` using `Gaussian::filter`
and channel conversions to opt into the f64 implementation.

## Accuracy

The impulse is centered with a margin of at least 32 sigma (64 pixels minimum).
Variance is the normalized second moment; shape error compares the output to a
normalized sampled Gaussian. All nine variance errors are below 0.5%.

| Sigma | Measured variance | Relative variance error % | Maximum absolute shape error |
|---:|---:|---:|---:|
| 0.5 | 0.250000000 | 0.0000001 | 0.033370653 |
| 1 | 1.000000007 | 0.0000007 | 0.000727700 |
| 2 | 4.000000086 | 0.0000021 | 0.001081133 |
| 2.5 | 6.249999968 | 0.0000005 | 0.023720975 |
| 4 | 15.999999881 | 0.0000007 | 0.011704820 |
| 8 | 64.000000238 | 0.0000004 | 0.005003588 |
| 16 | 256.000000258 | 0.0000001 | 0.002319387 |
| 32 | 1024.000001054 | 0.0000001 | 0.001112587 |
| 64 | 4095.999999101 | < 0.0000001 | 0.000543994 |

**Acceptance exceptions:** the requested shape bound of 0.01 is not met at
sigma 2.5 and 4. The tests record the measured values and assert 0.023722 and
0.011706 respectively, as authorized by the ticket's fixed acceptance exception
rule. Sigma at least 8 meets 0.01; the bound does not apply below 2.5.

Maximum constant-image error is **0**, including every border pixel, single-pixel
lines, positive/negative constants and RGB channels. DC gain error is at most
**1.58e-8**. Symmetry error is at most **2.78e-17**; row and column impulse
outputs are identical. Explicitly clamped nonconstant lines of lengths
1, 2, 3, 7 and 31 agree with padded-reference filtering within 1e-6, including
effective-sigma capping on tiny images.

## Versions and fixtures

All settings-aware CPU render paths select V3's filter explicitly, including
mask/spot overlays and heal spots. The pre-NR cache family bit invalidates linear
images and spatial planes across V2/V3 switches, including draft reuse.
Unversioned guided/mask/heal helper APIs retain their legacy defaults so GPU
reference calculations remain the box implementation. Automatic upright,
straighten and merge-feature detection retain their old box preprocessing
independently of process. Changing the tuned Harris/MOPS windows otherwise moved
panorama rotation errors to 0.108–0.114 degrees, above the existing 0.1-degree bound;
the geometry assertions remain unchanged.

No raster/pipeline hash fixtures encoded the old width. The only changed pixel
fixture is `visualize::tests::sharpening_previews_show_the_mask_and_detail`:
V3's SharpenMask edge is `[102, 102, 102, 255]`, formerly `[255; 4]`. An explicit
V2 assertion retains `[255; 4]`. Process serialization tests cover V2/V3 round
trips, and `process_switch_rebuilds_recursive_blur_stages` checks cached versus
fresh renders through V1/V2/V3 with NR, spatial effects, subject masking and heal.
The engine fixture
`media::tests::old_relative_white_balance_on_a_kelvin_camera_keeps_its_look_and_learns_the_as_shot_white`
now explicitly selects V2, matching the older catalog it represents instead of
following the latest default. Its pixel-equality assertion is unchanged. On V3,
the two white-balance arithmetic paths differed by at most 1 LSB in 128 of
16,384 channels; no legacy fixture values were rewritten to accept that difference.

GPU shaders and scan kernels are unchanged. See [gpu-pipeline.md](gpu-pipeline.md)
for process-specific parity bounds and the pending Nehab recursive WGSL port.

## One-thread performance

6000 × 4000 (24 MP) scalar plane, release build, `RAYON_NUM_THREADS=1`, minimum
of nine runs per case. Wall and process CPU time are both measured; allocation
and transposes are included. Supplied pre-change baseline: 34.4 / 97.1 / 109.6 ms.

| Sigma | Supplied before ms | Legacy rerun wall ms | V3 wall ms | V3 process CPU ms |
|---:|---:|---:|---:|---:|
| 1 | 34.4 | 34.3 | 79.2 | 79.2 |
| 8 | 97.1 | 96.8 | 244.8 | 244.7 |
| 64 | 109.6 | 106.4 | 236.3 | 236.2 |

This is a **2.16–2.52× slowdown** versus the supplied baseline, not a speedup.
The f64 forward/reverse recurrence and transposes cost more than vectorized boxes;
radius-independent runtime and corrected variance are the tradeoff. The 24 MP
RGB sigma-4 result is 352.4 ms wall / 350.3 ms process CPU.

```sh
RAYON_NUM_THREADS=1 CARGO_TARGET_DIR="$PWD/target-176" \
  cargo run --release -p lightcraft-raster --example blur_bench
CARGO_TARGET_DIR="$PWD/target-176" \
  cargo test -p lightcraft-raster -p lightcraft-pipeline
```

## Validation

Native and wasm checks use the rustup stable toolchain (rustc/cargo 1.99.0),
with `CARGO_TARGET_DIR` set to this worktree's `target-176` throughout.
Full-workspace CI also sets `CRAFT_FONTS_DIR` to the existing craft-fonts checkout:
without it, the unrelated Settings-dialog integration test fails on a missing
Georgian glyph. `CARGO_INCREMENTAL=0` avoids exhausting the shared disk with
embedded-font incremental artifacts; the first font-enabled attempt filled it.

- `cargo test -p lightcraft-raster -p lightcraft-pipeline`: 260 passed,
  0 failed, 3 ignored; empty doctest suites also pass.
- `cargo xtask ci`: the workspace test phase passes with 5,601 passed,
  0 failed and 43 ignored (including doctests); fmt, workspace clippy and HEIF
  gates also pass. The full command stops at parity because its build-artifact
  directory disappeared during the subsequent CLI build, not on a failed
  assertion. Standalone parity, layers, assets (42 attributed files) and wasm
  reruns all pass. CI records 152 runtime-skipped checks for unavailable corpus
  files, models, Cyrillic fonts and GPU-memory exhaustion, separately from ignored
  tests; these are not claims of verification.
- `cargo clippy --workspace --all-targets --keep-going -- -D warnings`: passes.
- `cargo fmt --all -- --check` and `git diff --check`: pass.
- `cargo xtask layers`: passes for 33 crates.
- `cargo xtask wasm`: passes; eight existing UI warnings are not errors.
- `cargo test -p lightcraft-merge --test it`: 21 passed, 0 failed; the
  panorama geometry bounds are unchanged.
- A 1200 × 800 headless demo snapshot with Clarity +30, Texture +25,
  Dehaze +15 and luminance NR 20 passes all five control requests; the
  CPU render settles with no renders in flight and the screenshot is inspected.

The worktree-local build directory and temporary logs/screenshots are removed
after validation; no media fixtures or build artifacts are committed.
