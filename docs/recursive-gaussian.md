# Recursive Gaussian CPU filtering

Issue #176 replaces the under-width three-box CPU blur for **process V3**. V1/V2
keep their previous output. This is a numerical-width correction, not a claim of
Lightroom rendering parity. Measurements below are from November 3, 2026.

## Filter and boundaries

`crates/raster/src/blur.rs::gaussian` retains the public generic image API. Built-in
`f32`, RGB and RGBA floating pixels use f64 states throughout both line passes.
Eight independent rows are interleaved in each worker's forward and reverse
recurrences. The column pass walks eight adjacent columns together, using safe,
disjoint mutable row slices assigned to worker stripes; neither native pass
transposes the image. The f64 forward buffer is reused between blocks within
each worker job, with no per-line allocations. Release aarch64 assembly contains
packed NEON `fmul.2d`, `fadd.2d` and f32-to-f64 conversions, not scalar-only
dependency chains. No SIMD dependency or unsafe code is added.

Below sigma 2.5, the existing variance-exact sampled kernel helper is used, with
f64 accumulation and clamp-edge samples. Both axes share a 64-row tiled working
buffer, avoiding a full-image intermediate; row scratch is reused per worker.
The seven-tap horizontal window is unrolled while retaining tap summation order.
Custom pixels still use their existing line hook and transpose-based fallback.

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
and all working buffers are included. Supplied pre-change box baseline:
34.4 / 97.1 / 109.6 ms. Second-pass measurements use an Apple M6 and the same
rustup release toolchain and benchmark; the scalar V3 rerun is before these edits.

| Sigma | Pass 1 reported ms | Scalar V3 rerun ms | Boxes rerun ms | Interleaved V3 wall ms | Interleaved V3 CPU ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 79.2 | 68.1 | 30.9 | 33.6 | 33.6 |
| 8 | 244.8 | 246.4 | 87.9 | 70.5 | 70.4 |
| 64 | 236.3 | 251.9 | 101.6 | 70.2 | 70.2 |

All requested absolute budgets (35 / 100 / 110 ms) pass. The interleaved
implementation is 2.03–3.59× faster than the scalar V3 rerun and faster than
boxes at sigma 8 and 64. At sigma 1, the same-run boxes remain 2.7 ms faster;
the variance-exact filter is below the supplied box baseline and requested cap.
The 24 MP RGB sigma-4 rerun improves from 380.5 ms wall / 375.1 ms CPU to
160.0 ms wall / 160.0 ms CPU.

The retained scalar line implementation provides a pass-1 reference through a
custom test pixel, exercising the original two-transpose image path. Native
plane, RGB and RGBA results are **bit-identical (maximum delta 0)** across
sampled and recursive sigmas, HDR values, signed zero, tiny images, odd/tail
dimensions, 4097-pixel lines, 64-row FIR tile boundaries and 1/2/4/16 worker
stripes. The pass-1 variance, DC, border, symmetry and shape bounds are unchanged.

```sh
export PATH="$(dirname "$(rustup which cargo)"):$PATH"
RAYON_NUM_THREADS=1 CARGO_TARGET_DIR="$PWD/target-176b" \
  cargo run --release -p lightcraft-raster --example blur_bench
CARGO_TARGET_DIR="$PWD/target-176b" \
  cargo test -p lightcraft-raster -p lightcraft-pipeline
```

## Pass-2 validation

The rustup stable toolchain, `CARGO_INCREMENTAL=0` and worktree-local
`CARGO_TARGET_DIR=target-176b` are used throughout; the target directory is
removed after validation. No processing version, coefficient, fixture or
acceptance threshold changes in this pass.

- Raster and pipeline tests: 263 passed, 0 failed, 3 ignored.
- Raster without the parallel feature: 36 passed, 0 failed.
- Raster/pipeline clippy `--all-targets -- -D warnings`, fmt and diff checks pass.
- Release accuracy tests: 6 passed, 0 failed; all original bounds are unchanged.
- Assembly inspection confirms packed f64 recurrence independently of the debug
  test build.
- `cargo xtask ci`: all eight gates pass (fmt, workspace clippy, HEIF, tests,
  parity, layers, assets and wasm), with 5,604 workspace tests passed, 0 failed
  and 43 ignored. Its 152 runtime-skipped checks (141 corpus, 8 model, 2 fonts,
  1 GPU) are not claims of verification. The wasm build retains eight existing
  UI warnings; no new warnings are introduced.

## Shared craft-blur migration (#2)

The consumer pins `cloudbring/craft-libs` branch `algo/2-craft-blur` at
`75bb40cf0c06b7e046874f389376acbfd86915ab` in `Cargo.lock`. The workspace
dependency has a TODO to repoint to the upstream revision when the crate lands.
No processing version, cache key, fixture or #176 acceptance test is changed.

V3 uses `craft_blur::Gaussian` with `Quality::Recursive` and `Edge::Clamp`,
except for **2.5 <= sigma < 8**. The local sampled-kernel implementation and
unused #33 helper are removed. Native floating pixels expose their existing
storage as safe flat slices, without a channel conversion or second image copy.
Recursive rows and disjoint column stripes retain per-worker reusable scratch;
sampled output bands use the crate's tiled API with source halos and reusable
per-worker scratch, retaining parallelism without duplicating convolution code.

### Compatibility decisions and upstream blocker

`BoxApprox` is **not** a bit-identical replacement for V1/V2. Its f64 running
sums differ from LightCraft's f32 running sums, even at sigma 1: on a seeded
257 x 131 signed random plane, 30,681 of 33,667 output samples differ in bits.
`gaussian_box`, its radii, horizontal passes and re-primed vertical bands remain
unchanged, including for the GPU and existing non-versioned callers.

The supplied crate's **public Recursive quality is not fully bit-identical to
39aa4d25 either**: it selects the sampled kernel below sigma 8 to enforce its
0.01 shape-error contract, whereas V3 selects recurrence from sigma 2.5.
At sigma 4 all 33,667 samples in that random plane differ. Its raw recursive
constructor and pass execution are private, so the current public API cannot
express V3's existing response in this interval. The original coefficients,
scalar line hook and vectorized recurrence are retained exclusively for this
compatibility interval in `RecursiveCompatibility` and `blur/compatibility.rs`.
This also avoids regressing the interval to a slower scalar/transpose path.

**Unfinished:** removing that last recurrence and making V3 a wholly thin
wrapper requires an upstream compatibility constructor/executor that retains
the original 2.5 transition without weakening the public quality contract.
Changing V3 output or a saved processing version to hide this mismatch is not
part of this migration. The new tests explicitly demonstrate both incompatible
shared paths, alongside frozen baseline hashes for random planes.

### Identity and timing evidence

An isolated reference package was built from the original raster sources at
`39aa4d256939c3703f1e22c0f2a715a4c98a189f`, under the disposable `target-2`
directory. Direct `f32::to_bits` comparisons cover 13 dimensions (empty planes,
singletons, thin/tail dimensions and 4097-pixel lines), 20 sigmas including both
transitions, NaN/infinity, signed random values, and plane/RGB/RGBA pixels.
Both Gaussian and legacy boxes match: **1,560 cases / 57,494,080 samples /
zero differing bits**, independently with one and four Rayon workers.

After CI completed and its debug artifacts were removed to relieve disk
pressure, the same-binary, alternating, best-of-nine 24 MP single-worker release
run, including allocations, measured:

| Sigma | Original 39aa4d25 ms | Shared consumer ms | Change |
|---:|---:|---:|---:|
| 1 | 33.45 | 34.30 | +2.55% |
| 8 | 70.38 | 69.93 | -0.65% |
| 64 | 70.50 | 69.93 | -0.81% |

This run meets both the 5% same-run regression budget and the historical
33.6 / 70.5 / 70.2 ms targets within 5%. An immediate repeat measured reference
53.42 / 96.07 / 101.07 ms and consumer 54.57 / 99.43 / 97.63 ms, also within
the same-run budget (+2.15% / +3.50% / -3.40%), but not the historical absolute
times. The shared host remains sensitive to concurrent workloads.

Earlier runs during concurrent builds included reference 51.10 / 155.52 /
137.93 ms and consumer 71.75 / 219.98 / 179.10 ms. Those loaded runs exceeded
the regression budget; the post-cleanup acceptance measurements are not a
claim of a stable 5% bound under arbitrary concurrent host workloads.

The blur module files remove **59 net lines** (651 down to 592 across the
blur module and its helper), excluding added integration tests/documentation.
The unchanged #176 numerical/fixture tests and three new compatibility tests
pass: raster/pipeline **265 passed / 0 failed / 3 ignored**; raster without
default features **38 / 0**; release #176 accuracy tests **6 / 0**. Workspace
clippy with `--all-targets -- -D warnings`, fmt, layers (33 crates), assets
(42 attributed files) and wasm pass. Full `cargo xtask ci` also passes with
`CRAFT_FONTS_DIR` set to the existing craft-fonts checkout: **5,606 workspace
tests passed / 0 failed / 43 ignored**, plus 152 separately recorded runtime
skips (141 corpus, 8 model, 2 fonts, 1 GPU), not claims of verification. The
initial CI attempt without craft-fonts hit the existing missing Georgian glyph
failure in the Settings dialog; the font-enabled rerun passes that test.
Wasm retains the eight existing UI warnings. All checks use the rustup toolchain
and worktree-local `target-2` directory; the reference sources, executables,
build artifacts and generated `target/ci` reports are removed afterward.

## Pass-1 validation

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
