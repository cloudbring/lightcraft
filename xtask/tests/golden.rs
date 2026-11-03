use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);

struct Tree(PathBuf);

impl Tree {
    fn new() -> Self {
        let tree =
            Self(std::env::temp_dir().join(format!("lightcraft-xtask-golden-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed))));
        std::fs::create_dir(&tree.0).unwrap();
        tree.write("Cargo.toml", include_bytes!("golden/version.input.toml"));
        tree.write("member/Cargo.toml", b"[package]\nname = \"golden\"\nversion.workspace = true\nedition.workspace = true\n");
        tree.write("member/src/lib.rs", b"#[test]\nfn sample() {}\n// #[test] ignored\n");
        tree
    }

    fn write(&self, name: &str, bytes: &[u8]) {
        let path = self.0.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    fn command(&self, args: &[&str]) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
        command.args(args).current_dir(&self.0).env("CARGO_MANIFEST_DIR", self.0.join("member"));
        command
    }

    fn run(&self, args: &[&str], expected: &[u8]) {
        assert_output(self.command(args).output().unwrap(), expected);
    }
}

impl Drop for Tree {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn assert_output(output: Output, expected: &[u8]) {
    assert!(output.status.success(), "stdout: {}\nstderr: {}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    assert_eq!(output.stdout, expected);
}

#[test]
fn version_stdout_manifest_and_lock_match_legacy() {
    let tree = Tree::new();
    tree.write("CHANGELOG.md", b"# Changelog\n\n## [Unreleased]\nKeep this unchanged.\n");
    tree.run(&["version"], include_bytes!("golden/version.stdout"));
    tree.run(&["version", "set", "1.2.3-rc.1"], include_bytes!("golden/version-set.stdout"));
    assert_eq!(std::fs::read(tree.0.join("Cargo.toml")).unwrap(), include_bytes!("golden/version.updated.toml"));
    assert_eq!(std::fs::read(tree.0.join("Cargo.lock")).unwrap(), include_bytes!("golden/version.updated.lock"));
    assert_eq!(std::fs::read(tree.0.join("CHANGELOG.md")).unwrap(), b"# Changelog\n\n## [Unreleased]\nKeep this unchanged.\n");
    let crlf = include_str!("golden/version.input.toml").replace('\n', "\r\n");
    assert_eq!(craft_xtask::version::replace(&crlf, "1.2.3-rc.1").unwrap(), include_str!("golden/version.updated.toml").replace('\n', "\r\n"));
}

#[test]
fn stats_stdout_and_exclusions_match_legacy() {
    let tree = Tree::new();
    for directory in ["target", "fuzz", ".hidden"] {
        tree.write(&format!("member/{directory}/ignored.rs"), b"#[test]\nfn ignored() {}\n");
    }
    tree.run(&["stats"], include_bytes!("golden/stats.stdout"));
}

#[test]
fn ico_stdout_and_bytes_match_legacy() {
    let tree = Tree::new();
    for size in [16_u32, 32, 48, 256] {
        let mut png = b"\x89PNG\r\n\x1a\n\0\0\0\x0dIHDR".to_vec();
        png.extend_from_slice(&size.to_be_bytes());
        png.extend_from_slice(&size.to_be_bytes());
        png.extend_from_slice(&[8, 6, 0, 0, 0, 1, 2, 3, 4]);
        tree.write(&format!("{size}.png"), &png);
    }
    tree.run(&["ico", "out.ico", "16.png", "32.png", "48.png", "256.png"], include_bytes!("golden/ico.stdout"));
    let actual: String = std::fs::read(tree.0.join("out.ico")).unwrap().iter().map(|byte| format!("{byte:02x}")).collect();
    assert_eq!(actual, include_str!("golden/ico.hex").trim_end());
}

#[test]
fn current_tree_layers_report_matches_legacy() {
    let output = Command::new(env!("CARGO_BIN_EXE_xtask")).arg("layers").env("CARGO_MANIFEST_DIR", env!("CARGO_MANIFEST_DIR")).output().unwrap();
    assert_output(output, include_bytes!("golden/layers.stdout"));
}

#[test]
fn current_tree_stats_match_except_changed_xtask_and_total() {
    let output = Command::new(env!("CARGO_BIN_EXE_xtask")).arg("stats").env("CARGO_MANIFEST_DIR", env!("CARGO_MANIFEST_DIR")).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let unchanged =
        |report: &str| report.lines().filter(|line| !line.starts_with("xtask ") && !line.starts_with("TOTAL ")).collect::<Vec<_>>().join("\n");
    assert_eq!(unchanged(&String::from_utf8(output.stdout).unwrap()), unchanged(include_str!("golden/stats-repo.stdout")));
}

#[cfg(unix)]
fn parity_tree() -> Tree {
    use std::os::unix::fs::PermissionsExt;
    let tree = Tree::new();
    tree.write("docs/parity-checklist.md", include_bytes!("golden/parity.input.md"));
    tree.write("docs/gaps.md", b"## Ranked\n\n| 1 | example |\n");
    tree.write("crates/engine/src/import.rs", b"pub fn import() {}\n");
    tree.write("crates/ui-egui/src/menus.rs", b"pub const UI_COMMANDS: &[UiCommand] = &[\n    (\"view.detail\", \"Detail\", None, \"View\"),\n];\n");
    tree.write("registry.sh", include_bytes!("golden/registry.sh"));
    std::fs::set_permissions(tree.0.join("registry.sh"), std::fs::Permissions::from_mode(0o700)).unwrap();
    tree
}

#[cfg(unix)]
#[test]
fn parity_report_and_write_match_legacy_with_fixture_registries() {
    let tree = parity_tree();
    let output = tree.command(&["parity"]).env("CARGO", tree.0.join("registry.sh")).output().unwrap();
    assert_output(output, include_bytes!("golden/parity-run.stdout"));
    assert_eq!(std::fs::read(tree.0.join("docs/parity-checklist.md")).unwrap(), include_bytes!("golden/parity.input.md"));
    let output = tree.command(&["parity", "--write"]).env("CARGO", tree.0.join("registry.sh")).output().unwrap();
    assert_output(output, include_bytes!("golden/parity-write.stdout"));
    assert_eq!(std::fs::read(tree.0.join("docs/parity-checklist.md")).unwrap(), include_bytes!("golden/parity.updated.md"));
}

#[cfg(unix)]
#[test]
fn parity_still_rejects_missing_evidence() {
    let tree = parity_tree();
    std::fs::remove_file(tree.0.join("crates/engine/src/import.rs")).unwrap();
    let output = tree.command(&["parity"]).env("CARGO", tree.0.join("registry.sh")).output().unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("missing path"));
}

#[test]
fn streaming_checksum_matches_standard_vector() {
    let tree = Tree::new();
    tree.write("hash-input", b"abc");
    assert_eq!(craft_xtask::sha256_hex(&tree.0.join("hash-input")).unwrap(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    assert!(craft_xtask::sha256_hex(&tree.0.join("missing")).is_err());
}

#[test]
fn version_validation_retains_legacy_inputs() {
    for version in ["0.1.0", "1.20.300", "1.0.0-rc.1", "1.0.0-alpha", "1.0.0-x-y.2"] {
        assert!(craft_xtask::version::validate(version).is_ok(), "{version}");
    }
    for version in ["1.0", "1.0.0.0", "v1.0.0", "01.0.0", "1.0.0-", "1.0.0-a..b", "1.0.0+meta", "1.a.0", ""] {
        assert!(craft_xtask::version::validate(version).is_err(), "{version}");
    }
}

#[test]
fn ico_rejects_empty_invalid_and_oversized_inputs() {
    assert!(craft_xtask::ico::pack(&[]).is_err());
    assert!(craft_xtask::ico::pack(&[b"GIF89a".to_vec()]).is_err());
    let mut png = b"\x89PNG\r\n\x1a\n\0\0\0\x0dIHDR".to_vec();
    png.extend_from_slice(&512_u32.to_be_bytes());
    png.extend_from_slice(&512_u32.to_be_bytes());
    assert!(craft_xtask::ico::pack(&[png]).is_err());
}

#[test]
fn help_keeps_local_and_shared_tasks() {
    let output = Command::new(env!("CARGO_BIN_EXE_xtask")).arg("--help").output().unwrap();
    assert!(output.status.success());
    let help = String::from_utf8(output.stdout).unwrap();
    for task in ["assets", "bench", "ci", "corpus", "raw-frame", "wasm", "web", "parity", "layers"] {
        assert!(help.contains(task), "{task}");
    }
    assert!(help.contains(craft_xtask::version::cli_help()));
    assert!(help.contains(craft_xtask::stats::cli_help()));
}
