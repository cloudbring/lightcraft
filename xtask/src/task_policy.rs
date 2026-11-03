//! LightCraft layer policy and legacy report formatting over `craft_xtask::layers::Report`.

use craft_xtask::layers::{Args, Config, DepKind, ExternalRule, Report, Violation, compute};
#[cfg(test)]
pub use craft_xtask::layers::{Crate, Dep};
use std::fmt::Write;

/// Where a workspace crate sits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// Regular layered crate.
    Layer(u8),
    /// Layer 0, and additionally may depend on no workspace crate at all.
    Standalone,
    /// Test tooling: may depend on anything up to L5 (it sits at L6 for rule
    /// purposes); other crates may use it only as a dev-dependency.
    Testkit,
    /// Binaries and build tooling: exempt from the rules.
    Exempt,
}

/// The layering table. Names are package names without the `lightcraft-`
/// prefix.
pub const TABLE: &[(&str, Class)] = &[
    ("geom", Class::Layer(0)),
    ("color", Class::Layer(0)),
    ("raster", Class::Layer(0)),
    ("tiff", Class::Layer(0)),
    ("sysmem", Class::Layer(0)),
    ("fetch", Class::Layer(0)),
    ("denoise-core", Class::Layer(1)),
    ("denoise", Class::Layer(1)),
    ("heif", Class::Layer(0)),
    ("raw", Class::Layer(1)),
    ("codecs", Class::Layer(1)),
    ("meta", Class::Layer(1)),
    ("develop", Class::Layer(1)),
    ("lensdb", Class::Layer(2)),
    ("inpaint", Class::Layer(1)),
    ("faces", Class::Layer(1)),
    ("geocode", Class::Layer(1)),
    ("scenes", Class::Layer(1)),
    ("slideshow", Class::Layer(1)),
    ("pipeline", Class::Layer(2)),
    ("gpu", Class::Layer(3)),
    ("catalog", Class::Layer(3)),
    ("preview", Class::Layer(3)),
    ("export", Class::Layer(3)),
    ("merge", Class::Layer(3)),
    ("segment", Class::Layer(3)),
    ("engine", Class::Layer(4)),
    ("ui-egui", Class::Layer(5)),
    ("mcp", Class::Layer(5)),
    ("testkit", Class::Testkit),
    // L6 apps and tooling
    ("lightcraft", Class::Exempt),
    ("cli", Class::Exempt),
    ("web", Class::Exempt),
    ("xtask", Class::Exempt),
];

/// Explicit orderings *within* a layer (earlier may be used by later).
/// L0: `raster` builds on `color` and `geom`; `color` uses `geom` for matrices.
/// L1: `raw` and `codecs` read metadata through `meta`; `develop` uses `meta` for XMP.
pub const INTRA_LAYER_ORDER: &[&[&str]] = &[
    &["geom", "color", "raster"],
    &["tiff", "raster"],
    &["denoise-core", "denoise"],
    &["meta", "raw"],
    &["meta", "codecs"],
    &["meta", "develop"],
    &["develop", "scenes"],
    &["codecs", "raw"],
];

pub fn short_name(pkg: &str) -> &str {
    pkg.strip_prefix("lightcraft-").unwrap_or(pkg)
}

pub fn classify(pkg: &str) -> Option<Class> {
    let s = short_name(pkg);
    TABLE.iter().find(|(n, _)| *n == s).map(|(_, c)| *c)
}

pub fn describe(class: Option<Class>) -> String {
    match class {
        Some(Class::Layer(l)) => format!("L{l}"),
        Some(Class::Standalone) => "L0 standalone".into(),
        Some(Class::Testkit) => "testkit".into(),
        Some(Class::Exempt) => "exempt".into(),
        None => "UNREGISTERED".into(),
    }
}

pub fn config() -> Config {
    let full = |name: &str| if matches!(name, "lightcraft" | "xtask") { name.to_owned() } else { format!("lightcraft-{name}") };
    let pick = |target| TABLE.iter().filter(|(_, c)| *c == target).map(|(n, _)| full(n)).collect();
    Config {
        root: crate::root(),
        layers: (0..=5)
            .map(|l| {
                let names = TABLE
                    .iter()
                    .filter(|(_, c)| {
                        matches!(c, Class::Layer(v) if *v == l) || (*c == Class::Standalone && l == 0) || (*c == Class::Testkit && l == 5)
                    })
                    .map(|(n, _)| full(n))
                    .collect();
                (format!("L{l}"), names)
            })
            .collect(),
        exempt: pick(Class::Exempt),
        standalone: pick(Class::Standalone),
        testkits: pick(Class::Testkit),
        allowed_edges: INTRA_LAYER_ORDER
            .iter()
            .flat_map(|chain| chain.iter().enumerate().flat_map(|(i, from)| chain[..i].iter().map(move |to| (full(from), full(to)))))
            .collect(),
        external_rules: vec![ExternalRule {
            dependencies: ["craft-ui", "egui", "eframe", "winit", "egui_kittest", "rfd", "bevy*"].map(str::to_owned).to_vec(),
            allowed_from: vec!["lightcraft-ui-egui".into(), "lightcraft-mcp".into(), "lightcraft-testkit".into()],
            allowed_dev_from: vec![],
        }],
        ..Config::default()
    }
}

fn diagnostic(v: &Violation) -> Option<String> {
    let level = |name| match classify(name) {
        Some(Class::Layer(value)) => value,
        Some(Class::Standalone) => 0,
        Some(Class::Testkit) => 5,
        _ => 7,
    };
    let kind = v.kind.map(|k| format!("{k:?}")).unwrap_or_default();
    Some(match v.rule.as_str() {
        "unregistered workspace crate" => {
            format!("{}: unknown workspace crate; register it in xtask/src/task_policy.rs TABLE (see plan/architecture.md §3)", v.from)
        }
        "testkit is dev-only" => format!("{}: lightcraft-testkit may only be a dev-dependency", v.from),
        "standalone dependency" => format!("{}: standalone crate must not depend on workspace crate {}", v.from, v.to),
        "forbidden external dependency" => {
            format!("{} (L{}) depends on UI crate `{}`; UI toolkits are only allowed in L5+", v.from, level(&v.from), v.to)
        }
        "disallowed layer edge" => {
            classify(&v.to)?;
            format!("{} (L{}) -> {} (L{}) [{kind}]: may only depend on strictly lower layers", v.from, level(&v.from), v.to, level(&v.to))
        }
        _ => v.to_string(),
    })
}

pub fn render(report: &Report) -> (String, usize) {
    let mut violations: Vec<_> = report.violations.iter().filter_map(diagnostic).collect();
    violations.sort();
    violations.dedup();
    let mut output = format!("Dependency layering (plan/architecture.md §3)\n\n{:<28} {:<14} workspace deps\n", "crate", "layer");
    for krate in &report.crates {
        let deps: Vec<_> = krate
            .deps
            .iter()
            .filter(|dep| dep.workspace)
            .map(|dep| {
                let kind = match dep.kind {
                    DepKind::Normal => "",
                    DepKind::Dev => " (dev)",
                    DepKind::Build => " (build)",
                };
                format!("{}{kind}", short_name(&dep.name))
            })
            .collect();
        let _ = writeln!(output, "{:<28} {:<14} {}", krate.name, describe(classify(&krate.name)), deps.join(", "));
    }
    let _ = if violations.is_empty() {
        writeln!(output, "\nOK: {} crates, no layering violations.", report.crates.len())
    } else {
        writeln!(output, "\n{} violation(s):\n  - {}", violations.len(), violations.join("\n  - "))
    };
    (output, violations.len())
}

#[cfg(test)]
pub fn report(crates: &[Crate]) -> Result<(String, usize), String> {
    Ok(render(&craft_xtask::layers::check_graph(&config(), crates)?))
}

pub(crate) fn cmd_layers() -> Result<(), String> {
    let (output, violations) = render(&compute(&config(), &Args::default())?);
    print!("{output}");
    if violations == 0 { Ok(()) } else { Err(format!("{violations} layering violation(s)")) }
}

#[cfg(test)]
#[path = "../tests/support/policy.rs"]
mod tests;
