//! SHA-256 helpers for model files.

pub use craft_hash::{sha256_file, sha256_hex, sha256_reader};

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    /// Oracle: faces/src/hash.rs at 06c1b9ae before ticket #37.
    fn matches_previous_hash_helpers() {
        for bytes in [b"".as_slice(), b"abc", &[42; 4097]] {
            let old: String = Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect();
            assert_eq!(sha256_hex(bytes), old);
            assert_eq!(sha256_reader(bytes).unwrap(), old);
        }
    }
}
