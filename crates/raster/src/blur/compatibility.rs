use super::RecursiveCompatibility;
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

impl RecursiveCompatibility {
    /// Filter built-in floating pixels with eight interleaved f64 recurrence chains and
    /// per-worker scratch. Columns traverse row-major blocks without an image transpose.
    pub(super) fn filter_image<const CHANNELS: usize, T: Pixel>(
        &self,
        image: &Image<T>,
        widen: impl Fn(T) -> [f64; CHANNELS] + Copy + Send + Sync,
        narrow: impl Fn([f64; CHANNELS]) -> T + Copy + Send + Sync,
    ) -> Image<T> {
        if image.width == 0 || image.height == 0 {
            return image.clone();
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
