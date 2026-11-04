//! VENDORED, unmodified except for `pub`: `median_comparators` and `median_network_source` from
//! cloudbring/lightcraft crates/gpu/src/ctx.rs at f9d3651c18faeb756748939673c94bdafb4c3d86 (branch algo/29-median-filter,
//! artcraft-algorithms#29). They are private in lightcraft-gpu, so the paired harness cannot call them.

pub fn median_comparators(radius: usize) -> Vec<(usize, usize, bool, bool)> {
    fn merge(start: usize, len: usize, stride: usize, pairs: &mut Vec<(usize, usize)>) {
        let step = stride * 2;
        if step < len {
            merge(start, len, step, pairs);
            merge(start + stride, len, step, pairs);
            for index in (start + stride..start + len - stride).step_by(step) {
                pairs.push((index, index + stride));
            }
        } else {
            pairs.push((start, start + stride));
        }
    }
    fn sort(start: usize, len: usize, pairs: &mut Vec<(usize, usize)>) {
        if len > 1 {
            sort(start, len / 2, pairs);
            sort(start + len / 2, len / 2, pairs);
            merge(start, len, 1, pairs);
        }
    }
    let len = (2 * radius + 1).next_power_of_two();
    let mut pairs = Vec::new();
    sort(0, len, &mut pairs);
    let mut needed = vec![false; len];
    needed[radius] = true;
    let mut result = Vec::new();
    for (low, high) in pairs.into_iter().rev() {
        let (need_low, need_high) = (needed[low], needed[high]);
        if need_low || need_high {
            result.push((low, high, need_low, need_high));
            needed[low] = true;
            needed[high] = true;
        }
    }
    result.reverse();
    result
}

pub fn median_network_source(radius: usize) -> String {
    use std::fmt::Write;
    let count = 2 * radius + 1;
    let len = count.next_power_of_two();
    let width = 16 + 2 * radius;
    let mut source = format!("\nvar<workgroup> median_tile_{radius}: array<u32, {}>;\n", width * 16);
    for vertical in [false, true] {
        let axis = if vertical { "v" } else { "h" };
        let (along, across, limit, other_limit) = if vertical { ("g.y", "g.x", "h", "w") } else { ("g.x", "g.y", "w", "h") };
        let (local_along, local_across) = if vertical { ("l.y", "l.x") } else { ("l.x", "l.y") };
        let input = if vertical { "position * w + other" } else { "other * w + position" };
        let _ = write!(
            source,
            "\n@compute @workgroup_size(16,16) fn median_{axis}_{radius}(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {{\nlet w = pu(0u); let h = pu(1u);\nfor (var k = {local_along}; k < {width}u; k += 16u) {{\nlet position = u32(clamp(i32({along} - {local_along} + k) - {radius}, 0, i32({limit}) - 1));\nlet other = min({across}, {other_limit} - 1u);\nmedian_tile_{radius}[{local_across} * {width}u + k] = median_key(src[{input}]);\n}}\nworkgroupBarrier();\nif (g.x >= w || g.y >= h) {{ return; }}\nvar v: array<u32, {len}>;\n"
        );
        for index in 0..len {
            if index < count {
                let _ = writeln!(source, "v[{index}] = median_tile_{radius}[{local_across} * {width}u + {local_along} + {index}u];");
            } else {
                let _ = writeln!(source, "v[{index}] = 0xffffffffu;");
            }
        }
        for (index, (low, high, need_low, need_high)) in median_comparators(radius).into_iter().enumerate() {
            let _ = writeln!(source, "let a{index} = v[{low}]; let b{index} = v[{high}];");
            if need_low {
                let _ = writeln!(source, "v[{low}] = min(a{index}, b{index});");
            }
            if need_high {
                let _ = writeln!(source, "v[{high}] = max(a{index}, b{index});");
            }
        }
        let _ = writeln!(
            source,
            "let key = v[{radius}]; let bits = key ^ select(0xffffffffu, 0x80000000u, (key & 0x80000000u) != 0u); dst[g.y * w + g.x] = bitcast<f32>(bits); }}"
        );
    }
    source
}

