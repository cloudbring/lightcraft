//! OLD CRC-32 helpers, vendored verbatim for the #188 old-vs-new equivalence test.
//!
//! Source: https://github.com/storytold/lightcraft @ 06c1b9ae0775464ce83ef0a9588be770337600f1
//! (merge-base of cloudbring/lightcraft `algo/37-craft-hash` with `origin/main`):
//! `crates/denoise-core/src/archive.rs` fn `crc32` and `crates/denoise-core/src/product.rs` fn `crc`
//! (two copies of the same bitwise-table CRC-32, kept separate here as they were there).
//! `hash` is `crates/denoise-core/src/hash.rs` (see its header).

pub mod hash;

/// CRC-32 (IEEE) of `bytes`. (archive.rs)
pub fn archive_crc32(bytes: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, t) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *t = c;
    }
    !bytes.iter().fold(!0u32, |c, &b| table[((c ^ u32::from(b)) & 0xff) as usize] ^ (c >> 8))
}

pub fn product_crc(bytes: &[u8]) -> u32 {
    // CRC-32 (IEEE), bitwise: strips are small and this runs once per strip
    let mut table = [0u32; 256];
    for (i, t) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *t = c;
    }
    !bytes.iter().fold(!0u32, |c, &b| table[((c ^ u32::from(b)) & 0xff) as usize] ^ (c >> 8))
}
