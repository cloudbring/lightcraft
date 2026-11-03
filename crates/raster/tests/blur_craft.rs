use craft_blur::{Cancel, Edge, Gaussian, Quality, Scratch, Shape};
use lightcraft_raster::{
    Plane,
    blur::{gaussian, gaussian_box},
};

const SIGMAS: [f32; 27] = [
    0.0,
    0.01,
    0.3,
    0.5,
    0.7,
    1.0,
    1.3,
    2.0,
    2.4,
    2.4999998,
    2.5,
    2.5000002,
    3.0,
    3.5559998,
    3.556,
    4.0,
    5.0,
    6.0,
    7.0,
    7.9,
    7.9999995,
    8.0,
    16.0,
    64.0,
    128.0,
    f32::INFINITY,
    f32::NAN,
];

fn hash(values: &[f32]) -> u64 {
    values.iter().fold(0xcbf29ce484222325_u64, |hash, value| (hash ^ u64::from(value.to_bits())).wrapping_mul(0x100000001b3))
}

#[test]
fn random_planes_keep_39aa4d25_v3_and_legacy_bits() {
    let expected = [
        [0x784ca68cbec7b8c8, 0x784ca68cbec7b8c8],
        [0x959aa09ed0f1aa7e, 0xe0b9428831e3ef74],
        [0x9e9b3fcf7af39560, 0xed1f1003c5d1e44a],
        [0xda304521da17e93d, 0x29a1200eeef853e9],
        [0xc79aa92dae4ee17a, 0x0e21b80256fdd3c2],
        [0xebfdffd84789bf62, 0x8fb5129821861427],
        [0xff0d92da6e18b2f6, 0xe31a01208a2fc190],
        [0x918afed3c7d8bff7, 0x6645d942db9eb34e],
    ];
    let mut state = 0x176_39aa4d25_u64;
    for ((width, height), expected) in [(1, 1), (1, 37), (37, 1), (2, 3), (7, 9), (17, 15), (65, 97), (257, 131)].into_iter().zip(expected) {
        let input = Plane {
            width,
            height,
            data: (0..width * height)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 40) as f32 / 65536.0 - 128.0
                })
                .collect(),
        };
        let mut actual = [0xcbf29ce484222325_u64; 2];
        for sigma in SIGMAS {
            for (index, output) in [gaussian(&input, sigma), gaussian_box(&input, sigma)].iter().enumerate() {
                actual[index] = (actual[index] ^ hash(&output.data)).wrapping_mul(0x100000001b3);
            }
        }
        assert_eq!(actual, expected, "{width}x{height}");
    }
}

fn shared(input: &Plane, sigma: f32, quality: Quality) -> Plane {
    let mut output = input.clone();
    Gaussian::new(sigma, quality)
        .expect("valid filter")
        .apply(&mut output.data, Shape::new(input.width, input.height, 1).expect("valid shape"), Edge::Clamp, &mut Scratch::default(), Cancel::Never)
        .expect("valid image");
    output
}

#[test]
fn shared_box_is_not_a_bit_identical_legacy_replacement() {
    let input = Plane::from_fn(37, 19, |column, row| ((column * 127 + row * 37 + 19) % 257) as f32 / 17.0 - 3.25);
    let expected = gaussian_box(&input, 1.0);
    let actual = shared(&input, 1.0, Quality::BoxApprox { passes: 3 });
    assert_ne!(hash(&actual.data), hash(&expected.data));
}

#[test]
fn recursive_raw_preserves_v3_in_shared_sampled_guard_band() {
    let input = Plane::from_fn(129, 1, |column, _| f32::from(column == 64));
    for sigma in SIGMAS.into_iter().filter(|sigma| (2.5..8.0).contains(sigma)) {
        assert_ne!(hash(&gaussian(&input, sigma).data), hash(&shared(&input, sigma, Quality::Recursive).data));
    }
    for sigma in [1.0, 8.0, 64.0] {
        assert_eq!(hash(&gaussian(&input, sigma).data), hash(&shared(&input, sigma, Quality::Recursive).data));
    }
}

#[test]
fn shared_recursive_raw_is_bit_identical_across_sigma_range() {
    for (width, height) in [(1, 1), (1, 37), (37, 1), (2, 3), (7, 9), (17, 15), (65, 97), (257, 131)] {
        let input = Plane::from_fn(width, height, |column, row| ((column * 127 + row * 37 + 19) % 257) as f32 / 17.0 - 3.25);
        for sigma in SIGMAS.into_iter().filter(|sigma| *sigma > 0.0 && sigma.is_finite()) {
            let effective_sigma = sigma.min((width.max(height) as f32 * 2.0).max(1.0));
            let expected = shared(&input, effective_sigma, Quality::RecursiveRaw);
            let actual = gaussian(&input, sigma);
            for (index, (actual, expected)) in actual.data.iter().zip(&expected.data).enumerate() {
                assert_eq!(actual.to_bits(), expected.to_bits(), "{width}x{height} sigma={sigma} sample={index}");
            }
        }
    }
}
