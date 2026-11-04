//! OLD code, vendored verbatim for the #188 old-vs-new equivalence test.
//!
//! Source: https://github.com/storytold/lightcraft @ 06c1b9ae0775464ce83ef0a9588be770337600f1
//! (merge-base of cloudbring/lightcraft `algo/37-craft-hash` with `origin/main`), file `crates/denoise-core/src/hash.rs`.
//! Unchanged apart from dropping the file's own #[cfg(test)] module and the module doc line.

use std::io::{self, Read};
use std::path::Path;

use sha2::{Digest, Sha256};

/// Lowercase hex SHA-256 of everything `r` yields.
pub fn sha256_reader<R: Read>(mut r: R) -> io::Result<String> {
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = r.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(buf.get(..n).unwrap_or(&[]));
    }
    Ok(hex(&h.finalize()))
}

/// Lowercase hex SHA-256 of a file.
pub fn sha256_file(path: &Path) -> io::Result<String> {
    sha256_reader(io::BufReader::new(std::fs::File::open(path)?))
}

/// Lowercase hex SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
