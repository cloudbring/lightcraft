use craft_blur::{Cancel, Edge, Gaussian, Quality, Scratch, Shape};
use lightcraft_raster::{
    Plane,
    blur::{gaussian, gaussian_box},
};

fn hash(values: &[f32]) -> u64 {
    values.iter().fold(0xcbf29ce484222325_u64, |hash, value| (hash ^ u64::from(value.to_bits())).wrapping_mul(0x100000001b3))
}

#[test]
fn random_planes_keep_39aa4d25_v3_and_legacy_bits() {
    let expected = [
        [0xc7378d25e4f3a150, 0xc7378d25e4f3a150],
        [0x578288f15f20ca49, 0x1e2659ccecff566a],
        [0x84dc10a99925e81d, 0x24cad63bb0e79c99],
        [0xe6ccb9b1415d3120, 0xaf20c49f92c3173f],
        [0x320aa26f39916508, 0x1fe4a28492f3ab10],
        [0x82e01581a94de9b6, 0x819de3817b37d962],
        [0x609bcf3477ae1651, 0x7be9ebe334385d00],
        [0xf37497383f31fcc0, 0x394ec5ff14373bf5],
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
        for sigma in [0.0, 0.01, 0.3, 0.5, 0.7, 1.0, 1.3, 2.0, 2.4, 2.5, 3.0, 4.0, 7.9, 8.0, 16.0, 64.0, 128.0, f32::INFINITY, f32::NAN] {
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
fn v3_keeps_recurrence_in_shared_sampled_guard_band() {
    let input = Plane::from_fn(129, 1, |column, _| f32::from(column == 64));
    for sigma in [2.5, 4.0, 7.9] {
        assert_ne!(hash(&gaussian(&input, sigma).data), hash(&shared(&input, sigma, Quality::Recursive).data));
    }
    for sigma in [1.0, 8.0, 64.0] {
        assert_eq!(hash(&gaussian(&input, sigma).data), hash(&shared(&input, sigma, Quality::Recursive).data));
    }
}
