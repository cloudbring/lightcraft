# LightCraft xtask

`cargo xtask --help` lists the app's native development tasks.

Ticket #70 uses `craft-xtask` from `cloudbring/craft-libs`'s `algo/integration`
branch, locked to `3c4d2f7`. It remains an xtask-only dependency, not an app
runtime dependency. The dependency must be repointed after the upstream merge.

## Shared tasks

- **version:** default shared configuration, rooted at this workspace. Includes
  the original parser, staging path and offline/online lockfile-refresh policy.
- **stats:** default exclusions (`target`, `fuzz`, dot directories), with symlink
  traversal enabled to match LightCraft. Shared traversal adds cycle detection
  and finite traversal budgets, and counts aliased directories only once.
- **ico:** caller-relative positional paths and arbitrary ordered square PNG
  sizes from 1 through 256, exactly as before. LightCraft had no ICNS command;
  none is added by this migration.
- **layers:** `src/task_policy.rs` holds the original table, intra-layer edges
  and UI restrictions, then formats `craft_xtask::layers::Report` from
  `layers::compute` / `layers::check_graph`.
- **SHA-256:** corpus checksums use `craft_xtask::sha256_hex`; xtask no longer
  depends on sha2 directly.

## Shared API limitations

Shared `layers::render` only prints `layers: ok` on success; `src/task_policy.rs`
therefore formats the table and legacy diagnostics over `layers::Report`.

**Parity stays local and unchanged.** Shared `parity::compute` implements the
4-column `D/P/M` weighted checklist format, not LightCraft's 6-column `✅/🟡/⬜/🚫`
section summary, `--write` flag, or command/control/path/gap evidence validation.
The shared `SURVEY.md` explicitly identifies LightCraft parity as app-owned.

Assets, bench, ci/ci_lock/ci_quick/ci_report, corpus downloading, raw_samples,
wasm and web remain local. No application or CI workflow changes are included.

Successful report/edited-file goldens are byte-exact. Shared version adds `tag`
and no longer echoes lockfile-update subprocess commands to stderr. Shared I/O
errors and icon usage/validation diagnostics can differ.

## Scoped verification

Set `CARGO_TARGET_DIR` to your isolated target directory for every command:

```sh
cargo build -p xtask
cargo test -p xtask
cargo clippy -p xtask --all-targets -- -D warnings
cargo fmt -p xtask --check
cargo xtask layers
```

Do not run full `cargo xtask parity` under an xtask-only build restriction:
it runs `lightcraft-cli` to obtain command and control registries. The golden
suite tests the complete parity task with explicit fixture registries instead,
including non-writing checks, `--write`, and rejected missing evidence.
See `tests/golden/README.md` for capture provenance and fixture limits.
