//! #188: OLD (inline FNV-1a writer / fold) vs NEW (`craft_hash::Fnv64`) fingerprints.
//!
//! OLD, from lightcraft @ 06c1b9ae0775464ce83ef0a9588be770337600f1 (merge-base of
//! `algo/37-craft-hash` with `origin/main`):
//!   * `crates/catalog/src/local.rs`: the private `Fnv` writer and `Photo::fingerprint` (vendored in
//!     `tests/old/mod.rs`),
//!   * `crates/develop/src/lib.rs`: `DevelopSettings::hash64` (the byte fold, in `tests/old/mod.rs`).
//!
//! NEW is the public `Photo::fingerprint` and `DevelopSettings::hash64` of this tree.

mod old;

use lightcraft_catalog::{Flag, Photo, PhotoId, Source};
use lightcraft_develop::DevelopSettings;
use proptest::prelude::*;
use serde_json::Value;

fn photo(id: u64, name: &str, w: u32, h: u32, rating: u8, flag_i: u8, captured: Option<String>) -> Photo {
    let mut p = Photo::new(PhotoId(id), Source::File { path: format!("/p/{name}") }, name, "JPG", w, h, "2026-09-20T10:00:00");
    p.rating = rating % 6;
    p.flag = match flag_i % 3 {
        0 => Flag::None,
        1 => Flag::Pick,
        _ => Flag::Reject,
    };
    p.captured = captured;
    p
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 500, failure_persistence: None, ..ProptestConfig::default() })]

    /// `Photo::fingerprint` equals the old streaming-FNV fingerprint, with and without a baseline.
    #[test]
    fn photo_fingerprint_matches(
        id in any::<u64>(), name in "[ -~é日本]{0,40}", w in 0u32..20000, h in 0u32..20000, rating in 0u8..9, flag in 0u8..3,
        captured in prop::option::of("[0-9T:-]{0,19}"), baseline in any::<bool>(),
    ) {
        let mut p = photo(id, &name, w, h, rating, flag, captured);
        prop_assert_eq!(p.fingerprint(), old::fingerprint(&p));
        if baseline {
            p.set_local_baseline();
            prop_assert!(p.local_baseline.is_some());
            // The baseline itself is excluded from the hash, in both.
            prop_assert_eq!(p.fingerprint(), old::fingerprint(&p));
            let mut q = p.clone();
            q.local_baseline = None;
            prop_assert_eq!(p.fingerprint(), q.fingerprint());
        }
    }

    /// `DevelopSettings::hash64` on settings with every numeric leaf nudged at random.
    #[test]
    fn develop_hash64_matches(nudges in prop::collection::vec((any::<prop::sample::Index>(), -3.0f64..3.0), 0..12)) {
        let mut v = serde_json::to_value(DevelopSettings::default()).unwrap();
        let mut leaves = vec![];
        collect_numbers(&mut v, &mut vec![], &mut leaves);
        for (ix, d) in &nudges {
            if leaves.is_empty() {
                break;
            }
            let path = &leaves[ix.index(leaves.len())];
            if let Some(n) = at(&mut v, path).as_f64() {
                *at(&mut v, path) = serde_json::json!(n + d);
            }
        }
        // Settings that don't deserialise (out of range) are skipped: there is nothing to hash.
        if let Ok(s) = serde_json::from_value::<DevelopSettings>(v) {
            prop_assert_eq!(s.hash64(), old::hash64(&s));
        }
    }
}

fn collect_numbers(v: &mut Value, path: &mut Vec<String>, out: &mut Vec<Vec<String>>) {
    match v {
        Value::Number(_) => out.push(path.clone()),
        Value::Object(m) => {
            for (k, c) in m.iter_mut() {
                path.push(k.clone());
                collect_numbers(c, path, out);
                path.pop();
            }
        }
        Value::Array(a) => {
            for (i, c) in a.iter_mut().enumerate() {
                path.push(i.to_string());
                collect_numbers(c, path, out);
                path.pop();
            }
        }
        _ => {}
    }
}

fn at<'a>(v: &'a mut Value, path: &[String]) -> &'a mut Value {
    let mut cur = v;
    for k in path {
        cur = match cur {
            Value::Object(m) => m.get_mut(k).unwrap(),
            Value::Array(a) => a.get_mut(k.parse::<usize>().unwrap()).unwrap(),
            _ => unreachable!(),
        };
    }
    cur
}

#[test]
fn default_settings_hash64_matches() {
    let s = DevelopSettings::default();
    assert_eq!(s.hash64(), old::hash64(&s));
}
