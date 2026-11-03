# Legacy task goldens

Captured from LightCraft commit `06c1b9ae0775464ce83ef0a9588be770337600f1`
before replacing its task implementations, on 2026-11-03. Only authorized
first-party LightCraft code was read. No oracle implementation is vendored.
All fixtures are text and smaller than 200 KB; ICO bytes are hex-encoded
synthetic headers, not copied artwork.

- `layers.stdout`: unmodified `cargo xtask layers` output on that repo tree.
- `stats-repo.stdout`: unmodified stats report on that tree. The regression
  test compares everything except xtask's row and TOTAL, which necessarily
  change as xtask source and tests are migrated.
- `stats.stdout`: original stats on a temporary workspace whose single package
  `golden` has the three source lines constructed in `Tree::new`.
- `version*`: original version read/set on that same temporary workspace,
  including a real offline Cargo lockfile refresh, with no application build.
  The original parser includes a trailing comment in the printed old version
  and drops it when replacing the line; these fixtures intentionally preserve
  that behavior rather than silently fixing it during extraction.
- `ico*`: original ICO task on synthetic 16/32/48/256-pixel IHDR inputs in a
  temporary directory, preserving input order, stdout and every output byte.
- `parity*`: original summary and complete parity run on the input checklist,
  with the menu source, referenced path and gaps table constructed by
  `parity_tree`. `registry.sh` supplies explicit synthetic CLI registries;
  these are not claimed to be the current app's actual registries. The script
  rejects all other Cargo commands and cannot build the app. Full-task tests
  using this shell shim are Unix-only; original parity unit tests run everywhere.

Capture used a temporary Rust driver including the original modules, linked
against the already-built xtask serde_json dependency. Its root and Cargo
working directory selected only temporary fixture trees. No checkout, stash,
app build or source dependency copy was added to the repository.

To refresh intentionally changed goldens, run the corresponding baseline
task on the same inputs and inspect the byte diff; do not bless new output
without a baseline. A changed current workspace graph/source legitimately
requires refreshing the two repo-wide reports.
