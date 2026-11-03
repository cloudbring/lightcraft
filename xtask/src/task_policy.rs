//! LightCraft layer policy and legacy report formatting over the shared checker.

use super::metadata;
use craft_xtask::layers::{Config, ExternalRule};
pub use craft_xtask::layers::{Crate, Dep, DepKind};
use serde_json::Value;
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

pub fn from_metadata(meta: &Value) -> Result<Vec<Crate>, String> {
    let pkgs = meta["packages"].as_array().ok_or("metadata: no packages array")?;
    let members: Vec<&str> = pkgs.iter().filter_map(|p| p["name"].as_str()).collect();
    let mut out = Vec::new();
    for p in pkgs {
        let name = p["name"].as_str().ok_or("package without name")?.to_owned();
        let mut deps = Vec::new();
        for d in p["dependencies"].as_array().into_iter().flatten() {
            let dname = d["name"].as_str().unwrap_or_default().to_owned();
            let kind = match d["kind"].as_str() {
                Some("dev") => DepKind::Dev,
                Some("build") => DepKind::Build,
                _ => DepKind::Normal,
            };
            let workspace = members.contains(&dname.as_str()) || d["path"].is_string();
            deps.push(Dep { name: dname, kind, workspace });
        }
        out.push(Crate { name, deps });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
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
    let mut config = Config { root: crate::root(), ..Config::default() };
    config.standalone = TABLE.iter().filter(|(_, class)| *class == Class::Standalone).map(|(name, _)| full(name)).collect();
    for layer in 0..=5 {
        let names = TABLE
            .iter()
            .filter(|(_, class)| match class {
                Class::Layer(value) => *value == layer,
                Class::Standalone => layer == 0,
                Class::Testkit => layer == 5,
                Class::Exempt => false,
            })
            .map(|(name, _)| full(name))
            .collect();
        config.layers.push((format!("L{layer}"), names));
    }
    for (name, class) in TABLE {
        match class {
            Class::Exempt => config.exempt.push(full(name)),
            Class::Standalone => {}
            Class::Testkit => config.testkits.push(full(name)),
            Class::Layer(_) => {}
        }
    }
    for chain in INTRA_LAYER_ORDER {
        for (index, from) in chain.iter().enumerate() {
            for to in chain.iter().take(index) {
                config.allowed_edges.push((full(from), full(to)));
            }
        }
    }
    config.external_rules.push(ExternalRule {
        dependencies: ["craft-ui", "egui", "eframe", "winit", "egui_kittest", "rfd", "bevy*"].map(str::to_owned).to_vec(),
        allowed_from: vec!["lightcraft-ui-egui".into(), "lightcraft-mcp".into(), "lightcraft-testkit".into()],
        allowed_dev_from: vec![],
    });
    config
}

fn diagnostic(message: String) -> Option<String> {
    if let Some(name) = message.strip_suffix(": unregistered workspace crate") {
        return Some(format!("{name}: unknown workspace crate; register it in xtask/src/task_policy.rs TABLE (see plan/architecture.md §3)"));
    }
    let Some((source, rest)) = message.split_once(" -> ") else { return Some(message) };
    let Some((target, rule)) = rest.split_once(": ") else { return Some(message) };
    let (target, kind) = target.split_once(" [").map_or((target, ""), |(target, kind)| (target, kind.trim_end_matches(']')));
    let level = |name| match classify(name) {
        Some(Class::Layer(value)) => value,
        Some(Class::Standalone) => 0,
        Some(Class::Testkit) => 5,
        _ => 7,
    };
    Some(match rule {
        "testkit is dev-only" => format!("{source}: lightcraft-testkit may only be a dev-dependency"),
        "standalone dependency" => format!("{source}: standalone crate must not depend on workspace crate {target}"),
        "forbidden external dependency" => {
            format!("{source} (L{}) depends on UI crate `{target}`; UI toolkits are only allowed in L5+", level(source))
        }
        "disallowed layer edge" => {
            classify(target)?;
            format!("{source} (L{}) -> {target} (L{}) [{kind}]: may only depend on strictly lower layers", level(source), level(target))
        }
        _ => message,
    })
}

pub fn report(crates: &[Crate]) -> Result<(String, usize), String> {
    let mut violations: Vec<_> = craft_xtask::layers::check(&config(), crates)?.into_iter().filter_map(diagnostic).collect();
    violations.sort();
    violations.dedup();
    let mut output = String::from("Dependency layering (plan/architecture.md §3)\n\n");
    writeln!(output, "{:<28} {:<14} workspace deps", "crate", "layer").map_err(|error| error.to_string())?;
    for krate in crates {
        let dependencies: Vec<_> = krate
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
        writeln!(output, "{:<28} {:<14} {}", krate.name, describe(classify(&krate.name)), dependencies.join(", "))
            .map_err(|error| error.to_string())?;
    }
    output.push('\n');
    if violations.is_empty() {
        writeln!(output, "OK: {} crates, no layering violations.", crates.len()).map_err(|error| error.to_string())?;
    } else {
        writeln!(output, "{} violation(s):", violations.len()).map_err(|error| error.to_string())?;
        for violation in &violations {
            writeln!(output, "  - {violation}").map_err(|error| error.to_string())?;
        }
    }
    Ok((output, violations.len()))
}

pub(crate) fn cmd_layers() -> Result<(), String> {
    let (output, violations) = report(&from_metadata(&metadata()?)?)?;
    print!("{output}");
    if violations == 0 { Ok(()) } else { Err(format!("{violations} layering violation(s)")) }
}

#[cfg(test)]
#[path = "../tests/support/policy.rs"]
mod tests;
