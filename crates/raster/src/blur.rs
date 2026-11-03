//! Gaussian blur via craft-blur's bit-identical V3 RecursiveRaw policy at every sigma.
//! [`gaussian_box`] preserves the legacy process and the GPU's three-box approximation.

use crate::resample::Pixel;
use crate::{Image, par_rows};

use craft_blur::{Cancel, Edge, Quality, Scratch, Shape};

/// The largest blur sigma (pixels) the box radii are made for. A box wider than any image (the
/// encoders' limit is 65535 px) blurs to the same flat result, and a larger sigma made the box
/// width's square overflow (from about 23 170 px: a panic where overflow is checked, a garbage
/// radius of billions of pixels, i.e. a hung render, where it isn't).
pub const MAX_SIGMA: f32 = 65_536.0;

/// Box radii approximating a Gaussian of `sigma` with 3 passes (Kovesi / Wells). A sigma that is
/// not a number counts as 0 (no blur); one above [`MAX_SIGMA`] (infinity too) as `MAX_SIGMA`.
pub fn box_radii(sigma: f32) -> [usize; 3] {
    let sigma = if sigma.is_nan() { 0.0 } else { sigma.clamp(0.0, MAX_SIGMA) };
    let n = 3.0f32;
    let w_ideal = (12.0 * sigma * sigma / n + 1.0).sqrt();
    let mut wl = w_ideal.floor() as i32;
    if wl % 2 == 0 {
        wl -= 1;
    }
    let wu = wl + 2;
    // (the square in i64: it passes i32 from a box 46 341 px wide)
    let m_ideal = (12.0 * sigma * sigma - n * (i64::from(wl) * i64::from(wl)) as f32 - 4.0 * n * wl as f32 - 3.0 * n) / (-4.0 * wl as f32 - 4.0);
    let m = m_ideal.round() as i32;
    std::array::from_fn(|i| (((if (i as i32) < m { wl } else { wu }) - 1) / 2).max(0) as usize)
}

/// Standard deviation (px) of the blur [`gaussian_box`] actually makes for `sigma`: its boxes are
/// whole pixels wide, so small radii snap to a few steps (0 below about 0.58 px, then √(2/3) px for one
/// 3-px box per axis, …).
pub fn box_sigma(sigma: f32) -> f32 {
    if sigma.is_nan() || sigma <= 0.3 {
        return 0.0;
    }
    box_radii(sigma).iter().map(|&r| (((2 * r + 1) * (2 * r + 1) - 1) as f32) / 12.0).sum::<f32>().sqrt()
}

/// One horizontal box pass over `row` (clamped edges), using `tmp` as scratch.
#[inline]
fn box_row<T: Pixel>(row: &mut [T], tmp: &mut Vec<T>, r: usize) {
    let w = row.len();
    if r == 0 || w == 0 {
        return;
    }
    tmp.clear();
    tmp.extend_from_slice(row);
    let s = &tmp[..];
    let last = w - 1;
    let inv = 1.0 / (2 * r + 1) as f32;
    let mut acc = T::zero();
    for i in 0..=2 * r {
        acc = acc.madd(s[i.saturating_sub(r).min(last)], 1.0);
    }
    for x in 0..w {
        row[x] = T::zero().madd(acc, inv);
        acc = acc.madd(s[(x + r + 1).min(last)], 1.0).madd(s[x.saturating_sub(r)], -1.0);
    }
}

/// Rows per parallel band in the vertical pass (each band re-primes its running sums).
pub const BAND: usize = 32;

/// One vertical box pass `src → dst` (clamped edges): a running sum per column, vectorized across
/// the row, parallel over horizontal bands. Cache-friendly — no transposes.
fn box_v<T: Pixel>(src: &Image<T>, dst: &mut Image<T>, r: usize) {
    let (w, h) = (src.width, src.height);
    let inv = 1.0 / (2 * r + 1) as f32;
    let last = h - 1;
    let row = |y: usize| &src.data[y * w..(y + 1) * w];
    par_rows(&mut dst.data, w * BAND, |band, out| {
        let y0 = band * BAND;
        let mut acc = vec![T::zero(); w];
        for i in 0..=2 * r {
            let yy = (y0 + i).saturating_sub(r).min(last);
            for (a, v) in acc.iter_mut().zip(row(yy)) {
                *a = a.madd(*v, 1.0);
            }
        }
        for (k, o) in out.chunks_mut(w).enumerate() {
            let y = y0 + k;
            for (d, a) in o.iter_mut().zip(&acc) {
                *d = T::zero().madd(*a, inv);
            }
            let (add, sub) = (row((y + r + 1).min(last)), row(y.saturating_sub(r)));
            for ((a, p), m) in acc.iter_mut().zip(add).zip(sub) {
                *a = a.madd(*p, 1.0).madd(*m, -1.0);
            }
        }
    });
}

/// The box radii of [`gaussian_box`] on a `w × h` image (the GPU renderer's twin uses them too), or
/// `None` when the blur leaves the image as it is: `sigma` at most 0.3 or not a number, or an
/// empty image. A sigma wider than twice the image's long side counts as that: a crop a hair
/// wide shown to fit asked the tone and dehaze stages for sigmas of millions of pixels (infinite
/// for a crop of no size), which blur no flatter than that but cost boxes that wide on every row.
pub fn radii_for(sigma: f32, w: usize, h: usize) -> Option<[usize; 3]> {
    if sigma.is_nan() || sigma <= 0.3 || w == 0 || h == 0 {
        return None;
    }
    let cap = (w.max(h) as f32 * 2.0).max(1.0);
    Some(box_radii(sigma.min(cap)))
}

/// Blur with Gaussian `sigma` (pixels). `sigma <= 0.3` returns a copy.
pub fn gaussian_box<T: Pixel>(img: &Image<T>, sigma: f32) -> Image<T> {
    let Some(radii) = radii_for(sigma, img.width, img.height) else {
        return img.clone();
    };
    let mut a = img.clone();
    par_rows(&mut a.data, img.width, |_, row| {
        let mut tmp = Vec::with_capacity(row.len());
        for r in radii {
            box_row(row, &mut tmp, r);
        }
    });
    let mut b = Image::<T>::new(img.width, img.height);
    for r in radii {
        if r == 0 {
            continue;
        }
        box_v(&a, &mut b, r);
        std::mem::swap(&mut a, &mut b);
    }
    a
}

/// Precomputed Gaussian preserving the V3 response across the entire sigma range.
pub struct Gaussian {
    sigma: f32,
    shared: craft_blur::Gaussian,
}

impl Gaussian {
    fn new(sigma: f32) -> Result<Self, craft_blur::Error> {
        Ok(Self { sigma, shared: craft_blur::Gaussian::new(sigma, Quality::RecursiveRaw)? })
    }

    /// Compatibility implementation for custom pixel types that retain legacy box filtering.
    pub fn filter_pixels<T: Pixel>(&self, line: &mut [T]) {
        let mut scratch = Vec::new();
        for radius in box_radii(self.sigma) {
            box_row(line, &mut scratch, radius);
        }
    }

    /// Filter one line, converting a custom pixel to and from fixed-size channels.
    pub fn filter<const CHANNELS: usize, T: Copy>(
        &self,
        line: &mut [T],
        widen: impl Fn(T) -> [f64; CHANNELS],
        narrow: impl Fn([f64; CHANNELS]) -> T,
    ) {
        let Ok(shape) = Shape::new(line.len(), 1, CHANNELS) else { return };
        let mut pixels: Vec<[f32; CHANNELS]> = line.iter().map(|&pixel| widen(pixel).map(|value| value as f32)).collect();
        if self.shared.filter_rows(pixels.as_flattened_mut(), shape, Edge::Clamp, &mut Scratch::default(), Cancel::Never).is_ok() {
            for (pixel, value) in line.iter_mut().zip(pixels) {
                *pixel = narrow(value.map(f64::from));
            }
        }
    }

    pub(crate) fn filter_packed<const CHANNELS: usize, T: Pixel>(
        &self,
        image: &Image<T>,
        flatten: impl Fn(&mut [T]) -> &mut [f32] + Sync,
    ) -> Option<Image<T>> {
        let filter = &self.shared;
        let shape = Shape::new(image.width, image.height, CHANNELS).ok()?;
        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() > 1
            && let craft_blur::Separable::Kernel(kernel) = filter.separable()
        {
            use rayon::prelude::*;
            const ROWS: usize = 64;
            let half = kernel.len() / 2;
            let mut output = Image::new(image.width, image.height);
            output
                .data
                .par_chunks_mut(image.width.saturating_mul(ROWS))
                .enumerate()
                .try_for_each_init(
                    || (Vec::new(), Scratch::default()),
                    |(tile, scratch), (band, pixels)| -> Result<(), craft_blur::Error> {
                        let row = band * ROWS;
                        let start = row.saturating_sub(half);
                        let end = row.saturating_add(pixels.len() / image.width).saturating_add(half).min(image.height);
                        tile.clear();
                        tile.extend_from_slice(&image.data[start * image.width..end * image.width]);
                        let tile_shape = Shape::new(image.width, end - start, CHANNELS)?;
                        filter.apply(flatten(tile), tile_shape, Edge::Clamp, scratch, Cancel::Never)?;
                        let offset = (row - start) * image.width;
                        pixels.copy_from_slice(&tile[offset..offset + pixels.len()]);
                        Ok(())
                    },
                )
                .ok()?;
            return Some(output);
        }
        let mut output = image.clone();
        apply_shared(filter, flatten(&mut output.data), shape).ok()?;
        Some(output)
    }
}

fn apply_shared(filter: &craft_blur::Gaussian, pixels: &mut [f32], shape: Shape) -> Result<(), craft_blur::Error> {
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && matches!(filter.separable(), craft_blur::Separable::Recursive(_)) {
        use rayon::prelude::*;
        let row_samples = shape.width() * shape.channels();
        pixels.par_chunks_mut(row_samples * craft_blur::LINE_BATCH).try_for_each_init(Scratch::default, |scratch, batch| {
            let batch_shape = Shape::new(shape.width(), batch.len() / row_samples, shape.channels())?;
            filter.filter_rows(batch, batch_shape, Edge::Clamp, scratch, Cancel::Never)
        })?;
        let stripe_width = shape.width().div_ceil(rayon::current_num_threads()).div_ceil(craft_blur::LINE_BATCH) * craft_blur::LINE_BATCH;
        let mut stripes: Vec<Vec<&mut [f32]>> = (0..shape.width().div_ceil(stripe_width)).map(|_| Vec::with_capacity(shape.height())).collect();
        for row in pixels.chunks_mut(row_samples) {
            for (stripe, samples) in stripes.iter_mut().zip(row.chunks_mut(stripe_width * shape.channels())) {
                stripe.push(samples);
            }
        }
        return stripes.par_iter_mut().try_for_each_init(Scratch::default, |scratch, stripe| {
            filter.filter_columns(stripe, shape.channels(), Edge::Clamp, scratch, Cancel::Never)
        });
    }
    filter.apply(pixels, shape, Edge::Clamp, &mut Scratch::default(), Cancel::Never)
}

fn transpose<T: Pixel>(source: &Image<T>, destination: &mut Image<T>) {
    const TILE: usize = 32;
    let width = destination.width;
    par_rows(&mut destination.data, width.saturating_mul(TILE), |band, rows| {
        for start in (0..width).step_by(TILE) {
            let end = (start + TILE).min(width);
            for (offset, row) in rows.chunks_mut(width).enumerate() {
                let column = band * TILE + offset;
                for (index, pixel) in row.iter_mut().enumerate().take(end).skip(start) {
                    *pixel = source.data[index * source.width + column];
                }
            }
        }
    });
}

/// Blur built-in floating pixels with Gaussian `sigma` (pixels), clamping outside the edges.
/// Custom pixels keep legacy filtering unless they override [`Pixel::gaussian_line`].
/// Non-positive/NaN sigmas and empty images return a copy; huge/infinite sigmas are bounded.
pub fn gaussian<T: Pixel>(img: &Image<T>, sigma: f32) -> Image<T> {
    if sigma.is_nan() || sigma <= 0.0 || img.width == 0 || img.height == 0 {
        return img.clone();
    }
    let sigma = sigma.min(MAX_SIGMA).min((img.width.max(img.height) as f32 * 2.0).max(1.0));
    let Ok(filter) = Gaussian::new(sigma) else { return img.clone() };
    if let Some(output) = T::gaussian_image(img, &filter) {
        return output;
    }
    let mut horizontal = img.clone();
    par_rows(&mut horizontal.data, img.width, |_, row| T::gaussian_line(row, &filter));
    let mut transposed = Image::<T>::new(img.height, img.width);
    transpose(&horizontal, &mut transposed);
    par_rows(&mut transposed.data, img.height, |_, line| T::gaussian_line(line, &filter));
    transpose(&transposed, &mut horizontal);
    horizontal
}

#[cfg(test)]
#[path = "blur_tests.rs"]
mod accuracy_tests;

#[cfg(test)]
mod tests {
    /// The box radii of the formula as it was, for the check below.
    fn box_radii_before(sigma: f32) -> [usize; 3] {
        let n = 3.0f32;
        let w_ideal = (12.0 * sigma * sigma / n + 1.0).sqrt();
        let mut wl = w_ideal.floor() as i32;
        if wl % 2 == 0 {
            wl -= 1;
        }
        let wu = wl + 2;
        let m_ideal = (12.0 * sigma * sigma - n * (wl * wl) as f32 - 4.0 * n * wl as f32 - 3.0 * n) / (-4.0 * wl as f32 - 4.0);
        let m = m_ideal.round() as i32;
        std::array::from_fn(|i| (((if (i as i32) < m { wl } else { wu }) - 1) / 2).max(0) as usize)
    }

    /// A blur sigma from a degenerate edit or a deep zoom (found by a UI campaign: a render
    /// worker's blur with a huge sigma) squared the box width past i32, which panics in a debug
    /// build and gave a radius of billions of pixels in a release one. Huge, infinite and NaN
    /// sigmas give bounded radii, and a blur with them is quick and stays finite; every sigma that
    /// worked before keeps exactly its radii.
    #[test]
    fn huge_and_non_finite_sigmas_give_bounded_radii() {
        for sigma in [26_800.0, 46_341.0, 1e5, 1e9, f32::MAX, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, -5.0] {
            let r = box_radii(sigma);
            assert!(r.iter().all(|&r| r <= MAX_SIGMA as usize * 2), "{sigma}: {r:?}");
        }
        assert_eq!(box_radii(f32::NAN), [0, 0, 0]);
        assert_eq!(box_radii(f32::NEG_INFINITY), [0, 0, 0]);
        assert_eq!(box_radii(1e9), box_radii(MAX_SIGMA));
        assert_eq!(box_radii(f32::INFINITY), box_radii(MAX_SIGMA));
        // on an image, the blur is at most twice its long side; NaN and tiny sigmas don't blur
        assert_eq!(radii_for(3.1e7, 1400, 900), Some(box_radii(2800.0)));
        assert_eq!(radii_for(f32::INFINITY, 1400, 900), Some(box_radii(2800.0)));
        assert_eq!(radii_for(40.0, 1400, 900), Some(box_radii(40.0)), "an ordinary sigma is as asked");
        assert_eq!(radii_for(f32::NAN, 1400, 900), None);
        assert_eq!(radii_for(0.2, 1400, 900), None);
        assert_eq!(radii_for(5.0, 0, 900), None);
        let p = crate::Plane { width: 4, height: 3, data: (0..12).map(|i| i as f32).collect() };
        let t = std::time::Instant::now();
        for sigma in [1e9, f32::INFINITY, f32::NAN] {
            let b = gaussian(&p, sigma);
            assert!(b.data.iter().all(|v| v.is_finite()), "{sigma}");
        }
        assert!(t.elapsed().as_secs_f64() < 30.0, "a huge sigma's blur is bounded");
        let mut s = 0.0f32;
        while s < 23_000.0 {
            assert_eq!(box_radii(s), box_radii_before(s), "sigma {s}");
            s = s * 1.013 + 0.07;
        }
    }

    use super::*;

    #[test]
    fn preserves_constant_and_mass() {
        let img = Image::<f32>::filled(50, 30, 0.7);
        let b = gaussian(&img, 4.0);
        assert!(b.data.iter().all(|v| (v - 0.7).abs() < 1e-5));
        let mut imp = Image::<f32>::new(101, 101);
        imp.set(50, 50, 1.0);
        let b = gaussian(&imp, 5.0);
        let sum: f32 = b.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "{sum}");
        // peak moved to the centre and symmetric
        assert!((b.get(45, 50) - b.get(55, 50)).abs() < 1e-6);
        assert!((b.get(50, 45) - b.get(50, 55)).abs() < 1e-6);
    }

    #[test]
    fn box_sigma_is_the_blur_made() {
        assert_eq!(box_sigma(0.3), 0.0);
        assert_eq!(box_sigma(0.5), 0.0, "no box wider than 1 px");
        assert!((box_sigma(1.0) - (2.0f32 / 3.0).sqrt()).abs() < 1e-6);
        for sigma in [1.0f32, 2.5, 8.0] {
            let mut imp = Image::<f32>::new(201, 1);
            imp.set(100, 0, 1.0);
            let b = gaussian_box(&imp, sigma);
            let var: f32 = b.data.iter().enumerate().map(|(i, v)| v * ((i as f32 - 100.0).powi(2))).sum();
            assert!((var.sqrt() - box_sigma(sigma)).abs() < 1e-3, "{sigma}: {} vs {}", var.sqrt(), box_sigma(sigma));
        }
    }

    #[test]
    fn variance_close_to_sigma_squared() {
        let mut imp = Image::<f32>::new(201, 1);
        imp.set(100, 0, 1.0);
        let sigma = 8.0;
        let b = gaussian(&imp, sigma);
        let var: f32 = b.data.iter().enumerate().map(|(i, v)| v * ((i as f32 - 100.0).powi(2))).sum();
        assert!((var.sqrt() - sigma).abs() / sigma < 0.1, "{}", var.sqrt());
    }
}
