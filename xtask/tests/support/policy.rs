use super::{Class, Crate, Dep, DepKind, config, report};

fn edge(source: &str, target: &str, kind: DepKind, workspace: bool) -> Crate {
    Crate { name: source.into(), deps: vec![Dep { name: target.into(), kind, workspace }] }
}

#[test]
fn layer_policy_preserves_direction_and_testkit_edges() {
    for (source, target, kind, allowed) in [
        ("lightcraft-engine", "lightcraft-develop", DepKind::Normal, true),
        ("lightcraft-develop", "lightcraft-engine", DepKind::Normal, false),
        ("lightcraft-catalog", "lightcraft-preview", DepKind::Normal, false),
        ("lightcraft-raster", "lightcraft-color", DepKind::Normal, true),
        ("lightcraft-color", "lightcraft-raster", DepKind::Normal, false),
        ("lightcraft-pipeline", "lightcraft-testkit", DepKind::Dev, true),
        ("lightcraft-pipeline", "lightcraft-testkit", DepKind::Normal, false),
        ("lightcraft-pipeline", "lightcraft-testkit", DepKind::Build, false),
        ("lightcraft-engine", "lightcraft-engine", DepKind::Dev, true),
    ] {
        let graph = [edge(source, target, kind, true)];
        let violations = craft_xtask::layers::check(&config(), &graph).unwrap();
        assert_eq!(violations.is_empty(), allowed, "{source} -> {target}: {violations:?}");
    }
}

#[test]
fn ui_policy_and_app_exemptions() {
    for dependency in ["craft-ui", "egui", "eframe", "winit", "egui_kittest", "rfd", "bevy_render"] {
        for (source, allowed) in [("lightcraft-engine", false), ("lightcraft-mcp", true), ("lightcraft-testkit", true)] {
            for kind in [DepKind::Normal, DepKind::Dev, DepKind::Build] {
                assert_eq!(craft_xtask::layers::check(&config(), &[edge(source, dependency, kind, false)]).unwrap().is_empty(), allowed);
            }
        }
    }
    for app in ["lightcraft", "lightcraft-cli", "lightcraft-web", "xtask"] {
        assert!(craft_xtask::layers::check(&config(), &[edge(app, "lightcraft-ui-egui", DepKind::Normal, true)]).unwrap().is_empty());
    }
}

#[test]
fn diagnostics_keep_legacy_wording_and_counts() {
    let (text, count) = report(&[
        edge("lightcraft-develop", "lightcraft-engine", DepKind::Normal, true),
        edge("lightcraft-engine", "egui", DepKind::Dev, false),
        edge("lightcraft-pipeline", "lightcraft-testkit", DepKind::Build, true),
    ])
    .unwrap();
    assert_eq!(count, 3);
    assert!(text.contains("lightcraft-develop (L1) -> lightcraft-engine (L4) [Normal]: may only depend on strictly lower layers"));
    assert!(text.contains("lightcraft-engine (L4) depends on UI crate `egui`; UI toolkits are only allowed in L5+"));
    assert!(text.contains("lightcraft-pipeline: lightcraft-testkit may only be a dev-dependency"));
    let (_, count) = report(&[Crate { name: "lightcraft-mystery".into(), deps: vec![] }]).unwrap();
    assert_eq!(count, 1);
    assert_eq!(super::describe(Some(Class::Standalone)), "L0 standalone");
}
