//! OLD fingerprint code, vendored for the #188 old-vs-new equivalence test.
//!
//! Source: https://github.com/storytold/lightcraft @ 06c1b9ae0775464ce83ef0a9588be770337600f1
//! (merge-base of cloudbring/lightcraft `algo/37-craft-hash` with `origin/main`):
//!   * `crates/catalog/src/local.rs`: struct `Fnv`, its `Write` impl, and the body of
//!     `Photo::fingerprint` (verbatim; the method became the free function `fingerprint`)
//!   * `crates/develop/src/lib.rs`: the body of `DevelopSettings::hash64` (verbatim)

use lightcraft_catalog::Photo;
use lightcraft_develop::DevelopSettings;

/// 64-bit FNV-1a over everything written to it (stable across builds and platforms, unlike
/// `std`'s hasher), so a photo is hashed as it serialises, without a buffer.
struct Fnv(u64);

impl std::io::Write for Fnv {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        for b in buf {
            self.0 ^= u64::from(*b);
            self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
        }
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub fn fingerprint(this: &Photo) -> u64 {
    let mut h = Fnv(0xcbf2_9ce4_8422_2325);
    let r = if this.local_baseline.is_some() {
        let mut p = this.clone();
        p.local_baseline = None;
        serde_json::to_writer(&mut h, &p)
    } else {
        serde_json::to_writer(&mut h, this)
    };
    // (serialising a photo can't fail; if it ever did, no baseline would match: kept)
    if r.is_err() { 0 } else { h.0 }
}

pub fn hash64(this: &DevelopSettings) -> u64 {
    let s = serde_json::to_string(this).unwrap_or_default();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}
