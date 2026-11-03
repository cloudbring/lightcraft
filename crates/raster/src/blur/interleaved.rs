use super::Gaussian;
use crate::{Image, resample::Pixel};

const LANES: usize = 8;
type Block<const CHANNELS: usize> = [[f64; LANES]; CHANNELS];

fn batches<T: Send, S: Send + Default>(data: &mut [T], length: usize, run: impl Fn(&mut S, usize, &mut [T]) + Sync + Send) {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        data.par_chunks_mut(length).enumerate().for_each_init(S::default, |scratch, (index, batch)| run(scratch, index, batch));
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch = S::default();
        for (index, batch) in data.chunks_mut(length).enumerate() {
            run(&mut scratch, index, batch);
        }
    }
}

impl Gaussian {
    /// Filter built-in floating pixels with eight interleaved f64 recurrence chains and
    /// per-worker scratch. Columns traverse row-major blocks without an image transpose.
    pub fn filter_image<const CHANNELS: usize, T: Pixel>(
        &self,
        image: &Image<T>,
        widen: impl Fn(T) -> [f64; CHANNELS] + Copy + Send + Sync,
        narrow: impl Fn([f64; CHANNELS]) -> T + Copy + Send + Sync,
    ) -> Image<T> {
        if image.width == 0 || image.height == 0 {
            return image.clone();
        }
        if let Some(kernel) = &self.kernel {
            return Sampled { kernel, widen, narrow }.filter_image(image);
        }
        let mut output = image.clone();
        batches(&mut output.data, image.width.saturating_mul(LANES), |forward: &mut Vec<Block<CHANNELS>>, _, pixels| {
            let mut rows = pixels.chunks_mut(image.width);
            let mut lines: [&mut [T]; LANES] = std::array::from_fn(|_| rows.next().unwrap_or(&mut []));
            let count = lines.iter().take_while(|line| !line.is_empty()).count();
            self.recursive::<true, CHANNELS, T>(&mut lines[..count], forward, widen, narrow);
        });
        #[cfg(feature = "parallel")]
        let workers = rayon::current_num_threads();
        #[cfg(not(feature = "parallel"))]
        let workers = 1;
        let stripe_width = image.width.div_ceil(workers).div_ceil(LANES) * LANES;
        let mut stripes: Vec<Vec<&mut [T]>> = (0..image.width.div_ceil(stripe_width)).map(|_| Vec::with_capacity(image.height)).collect();
        for row in output.data.chunks_mut(image.width) {
            for (stripe, pixels) in stripes.iter_mut().zip(row.chunks_mut(stripe_width)) {
                stripe.push(pixels);
            }
        }
        batches(&mut stripes, 1, |forward: &mut Vec<Block<CHANNELS>>, _, stripes| {
            for stripe in stripes {
                self.recursive::<false, CHANNELS, T>(stripe, forward, widen, narrow);
            }
        });
        output
    }

    fn recursive<const ROWS: bool, const CHANNELS: usize, T: Copy>(
        &self,
        lines: &mut [&mut [T]],
        forward: &mut Vec<Block<CHANNELS>>,
        widen: impl Fn(T) -> [f64; CHANNELS],
        narrow: impl Fn([f64; CHANNELS]) -> T,
    ) {
        let length = if ROWS { lines[0].len() } else { lines.len() };
        let width = if ROWS { lines.len() } else { lines[0].len() };
        forward.resize(length, [[0.0; LANES]; CHANNELS]);
        for start in (0..width).step_by(LANES) {
            let count = (width - start).min(LANES);
            let read = |position: usize| {
                let mut block = [[0.0; LANES]; CHANNELS];
                for lane in 0..count {
                    let pixel = if ROWS { lines[start + lane][position] } else { lines[position][start + lane] };
                    let channels = widen(pixel);
                    for channel in 0..CHANNELS {
                        block[channel][lane] = channels[channel];
                    }
                }
                block
            };
            let left = read(0);
            let right = read(length - 1);
            let mut state = [[[0.0; LANES]; CHANNELS]; 3];
            for (position, output) in forward.iter_mut().enumerate() {
                let input = read(position);
                let value = std::array::from_fn(|channel| {
                    std::array::from_fn(|lane| {
                        self.gain * (input[channel][lane] - left[channel][lane])
                            + self.feedback[0] * state[0][channel][lane]
                            + self.feedback[1] * state[1][channel][lane]
                            + self.feedback[2] * state[2][channel][lane]
                    })
                });
                state = [value, state[0], state[1]];
                *output = value;
            }
            let mut future: [Block<CHANNELS>; 3] = self.boundary.map(|row| {
                std::array::from_fn(|channel| {
                    std::array::from_fn(|lane| {
                        row.iter()
                            .zip(&state)
                            .map(|(coefficient, past)| coefficient * (past[channel][lane] + left[channel][lane] - right[channel][lane]))
                            .sum::<f64>()
                    })
                })
            });
            for position in (0..length).rev() {
                if position != length - 1 {
                    let value = std::array::from_fn(|channel| {
                        std::array::from_fn(|lane| {
                            self.gain * (forward[position][channel][lane] + left[channel][lane] - right[channel][lane])
                                + self.feedback[0] * future[0][channel][lane]
                                + self.feedback[1] * future[1][channel][lane]
                                + self.feedback[2] * future[2][channel][lane]
                        })
                    });
                    future = [value, future[0], future[1]];
                }
                for lane in 0..count {
                    let pixel = narrow(std::array::from_fn(|channel| future[0][channel][lane] + right[channel][lane]));
                    if ROWS {
                        lines[start + lane][position] = pixel;
                    } else {
                        lines[position][start + lane] = pixel;
                    }
                }
            }
        }
    }
}

struct Sampled<'a, Widen, Narrow> {
    kernel: &'a [f32],
    widen: Widen,
    narrow: Narrow,
}

impl<Widen, Narrow> Sampled<'_, Widen, Narrow> {
    fn filter_image<const CHANNELS: usize, T: Pixel>(&self, source: &Image<T>) -> Image<T>
    where
        Widen: Fn(T) -> [f64; CHANNELS] + Copy + Send + Sync,
        Narrow: Fn([f64; CHANNELS]) -> T + Copy + Send + Sync,
    {
        const ROWS: usize = 64;
        let mut output = Image::new(source.width, source.height);
        batches(&mut output.data, source.width.saturating_mul(ROWS), |scratch: &mut (Vec<T>, (Vec<T>, Vec<[f64; CHANNELS]>)), band, pixels| {
            let half = self.kernel.len() / 2;
            let height = pixels.len() / source.width + 2 * half;
            let mut horizontal = Image { width: source.width, height, data: std::mem::take(&mut scratch.0) };
            horizontal.data.resize(source.width * height, T::default());
            for (offset, row) in horizontal.data.chunks_mut(source.width).enumerate() {
                let source_row = (band * ROWS + offset).saturating_sub(half).min(source.height - 1);
                self.row::<true, CHANNELS, T>(source, source_row, row, &mut scratch.1);
            }
            for (offset, row) in pixels.chunks_mut(source.width).enumerate() {
                self.row::<false, CHANNELS, T>(&horizontal, offset + half, row, &mut scratch.1);
            }
            scratch.0 = horizontal.data;
        });
        output
    }

    fn row<const ROWS: bool, const CHANNELS: usize, T: Pixel>(
        &self,
        source: &Image<T>,
        row: usize,
        pixels: &mut [T],
        scratch: &mut (Vec<T>, Vec<[f64; CHANNELS]>),
    ) where
        Widen: Fn(T) -> [f64; CHANNELS] + Copy + Send + Sync,
        Narrow: Fn([f64; CHANNELS]) -> T + Copy + Send + Sync,
    {
        let Self { kernel, widen, narrow } = self;
        let (padded, values) = scratch;
        let half = kernel.len() / 2;
        if ROWS {
            let input = &source.data[row * source.width..(row + 1) * source.width];
            padded.resize(source.width + 2 * half, input[0]);
            padded[..half].fill(input[0]);
            padded[half..half + source.width].copy_from_slice(input);
            padded[half + source.width..].fill(input[source.width - 1]);
            if let Ok(weights) = <&[f32; 7]>::try_from(*kernel) {
                for (window, pixel) in padded.array_windows::<7>().zip(pixels) {
                    let mut value = [0.0; CHANNELS];
                    for tap in 0..7 {
                        let input = widen(window[tap]);
                        for channel in 0..CHANNELS {
                            value[channel] += input[channel] * f64::from(weights[tap]);
                        }
                    }
                    *pixel = narrow(value);
                }
                return;
            }
        }
        values.resize(pixels.len(), [0.0; CHANNELS]);
        for (tap, &weight) in kernel.iter().enumerate() {
            let input = if ROWS {
                &padded[tap..tap + pixels.len()]
            } else {
                let source_row = row.saturating_add(tap).saturating_sub(half).min(source.height - 1);
                &source.data[source_row * source.width..(source_row + 1) * source.width]
            };
            if tap == 0 {
                for (value, &pixel) in values.iter_mut().zip(input) {
                    *value = widen(pixel).map(|channel| 0.0 + channel * f64::from(weight));
                }
            } else {
                for (value, &pixel) in values.iter_mut().zip(input) {
                    let input = widen(pixel);
                    for channel in 0..CHANNELS {
                        value[channel] += input[channel] * f64::from(weight);
                    }
                }
            }
        }
        for (pixel, &value) in pixels.iter_mut().zip(values.iter()) {
            *pixel = narrow(value);
        }
    }
}
