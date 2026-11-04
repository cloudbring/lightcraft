//! #188: OLD (inline CRC-32 / SHA-256 / FNV) vs NEW (`craft-hash`) digests.
//!
//! OLD is vendored in `tests/old/` (lightcraft @ 06c1b9ae0775464ce83ef0a9588be770337600f1). NEW is
//! the public `denoise_core::hash` module and the `craft_hash` functions that `archive.rs`
//! (`crc32`) and `product.rs` (`crc`) now call. Inputs: random bytes at every length class and
//! chunking, plus a deterministic sample of the corpus files (`ARTCRAFT_CORPUS`, default
//! /Users/emm/ArtCraft/corpus; the corpus tests say so and pass when it is absent).

mod old;

use lightcraft_denoise_core::hash;
use proptest::prelude::*;
use std::io::Cursor;
use std::path::PathBuf;

/// Up to `max` corpus files (each at most 6 MB), spread evenly over the sorted file list.
fn corpus_sample(max: usize) -> Vec<(PathBuf, Vec<u8>)> {
    let root = PathBuf::from(std::env::var("ARTCRAFT_CORPUS").unwrap_or_else(|_| "/Users/emm/ArtCraft/corpus".into()));
    let mut files = vec![];
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if e.metadata().is_ok_and(|m| m.len() > 0 && m.len() <= 6 << 20) {
                files.push(p);
            }
        }
    }
    files.sort();
    if files.is_empty() {
        eprintln!("SKIP corpus: no files under {}", root.display());
        return vec![];
    }
    let step = (files.len() / max).max(1);
    files.into_iter().step_by(step).take(max).filter_map(|p| std::fs::read(&p).ok().map(|b| (p, b))).collect()
}

fn old_fnv1a64(bytes: &[u8]) -> u64 {
    // develop/src/lib.rs `hash64` and catalog/src/local.rs `Fnv`: the same fold.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn check_all(bytes: &[u8]) {
    assert_eq!(craft_hash::crc32(bytes), old::archive_crc32(bytes), "crc32 (archive.rs)");
    assert_eq!(craft_hash::crc32(bytes), old::product_crc(bytes), "crc (product.rs)");
    assert_eq!(craft_hash::fnv1a64(bytes), old_fnv1a64(bytes), "fnv1a64");
    assert_eq!(hash::sha256_hex(bytes), old::hash::sha256_hex(bytes), "sha256_hex");
    assert_eq!(hash::sha256_reader(Cursor::new(bytes)).unwrap(), old::hash::sha256_reader(Cursor::new(bytes)).unwrap(), "sha256_reader");
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 400, failure_persistence: None, ..ProptestConfig::default() })]

    #[test]
    fn random_bytes_digest_identically(bytes in prop::collection::vec(any::<u8>(), 0..70_000)) {
        check_all(&bytes);
    }

    /// Boundaries around SHA-256 blocks (64) and the 1 MiB read buffer.
    #[test]
    fn block_boundary_lengths(len in prop_oneof![0usize..200, 63usize..66, (1usize << 20) - 3..(1 << 20) + 3], fill in any::<u8>()) {
        let bytes: Vec<u8> = (0..len).map(|i| fill.wrapping_add(i as u8).wrapping_mul(31)).collect();
        check_all(&bytes);
    }

    /// A reader that returns short reads must not change the digest.
    #[test]
    fn short_reads_digest_identically(bytes in prop::collection::vec(any::<u8>(), 0..5000), chunk in 1usize..97) {
        struct Chunked<'a>(&'a [u8], usize);
        impl std::io::Read for Chunked<'_> {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                let n = self.0.len().min(self.1).min(out.len());
                out[..n].copy_from_slice(&self.0[..n]);
                self.0 = &self.0[n..];
                Ok(n)
            }
        }
        prop_assert_eq!(hash::sha256_reader(Chunked(&bytes, chunk)).unwrap(), old::hash::sha256_reader(Chunked(&bytes, chunk)).unwrap());
    }
}

#[test]
fn corpus_files_digest_identically() {
    let files = corpus_sample(80);
    let mut total = 0u64;
    for (path, bytes) in &files {
        check_all(bytes);
        assert_eq!(hash::sha256_file(path).unwrap(), old::hash::sha256_file(path).unwrap(), "{}", path.display());
        total += bytes.len() as u64;
    }
    eprintln!("corpus: {} files, {} bytes digested identically", files.len(), total);
}
