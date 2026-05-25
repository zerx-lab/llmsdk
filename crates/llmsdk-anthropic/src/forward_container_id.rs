//! Multi-step container-id forwarding helper.
//!
//! Mirrors `forward-anthropic-container-id-from-last-step.ts`. Use it in a
//! `prepareStep` hook (or equivalent) to reuse a container across iterations
//! of the same conversation.
// Rust guideline compliant 2026-02-21

use llmsdk_provider::shared::{ProviderMetadata, ProviderOptions};
use serde_json::{Map as JsonMap, Value as JsonValue, json};

/// Walk `steps` from the most recent backwards and return a
/// [`ProviderOptions`] payload that pins the Anthropic `container.id`
/// found in the latest step that carried one.
///
/// Returns `None` when no step exposed a `container.id`.
///
/// # Examples
///
/// ```
/// use llmsdk_anthropic::forward_anthropic_container_id_from_last_step;
/// use std::collections::HashMap;
///
/// let mut anthropic_meta = serde_json::Map::new();
/// anthropic_meta.insert(
///     "container".into(),
///     serde_json::json!({"id": "cnt_abc"}),
/// );
/// let mut step_meta = HashMap::new();
/// step_meta.insert("anthropic".to_owned(), anthropic_meta);
/// let steps = vec![step_meta];
///
/// let forwarded = forward_anthropic_container_id_from_last_step(&steps)
///     .expect("found container id");
/// assert!(forwarded.contains_key("anthropic"));
/// ```
#[must_use]
pub fn forward_anthropic_container_id_from_last_step(
    steps: &[ProviderMetadata],
) -> Option<ProviderOptions> {
    for step in steps.iter().rev() {
        let Some(anthropic) = step.get("anthropic") else {
            continue;
        };
        let Some(container) = anthropic.get("container") else {
            continue;
        };
        let Some(id) = container.get("id").and_then(JsonValue::as_str) else {
            continue;
        };
        let mut anthropic_opts = JsonMap::new();
        anthropic_opts.insert("container".into(), json!({ "id": id }));
        let mut out = ProviderOptions::new();
        out.insert("anthropic".into(), anthropic_opts);
        return Some(out);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn step(anthropic: serde_json::Value) -> ProviderMetadata {
        let mut map = HashMap::new();
        if let serde_json::Value::Object(obj) = anthropic {
            map.insert("anthropic".to_owned(), obj);
        }
        map
    }

    #[test]
    fn returns_none_when_no_container_id() {
        assert!(forward_anthropic_container_id_from_last_step(&[]).is_none());
        assert!(forward_anthropic_container_id_from_last_step(&[step(json!({}))]).is_none());
    }

    #[test]
    fn picks_latest_step_with_container_id() {
        let steps = vec![
            step(json!({"container": {"id": "older"}})),
            step(json!({})),
            step(json!({"container": {"id": "latest"}})),
        ];
        let out = forward_anthropic_container_id_from_last_step(&steps).expect("found");
        assert_eq!(
            out["anthropic"]["container"]["id"],
            serde_json::Value::String("latest".into())
        );
    }
}
