//! Anthropic-on-Bedrock model handle.
// Rust guideline compliant 2026-05-25

use std::sync::Arc;

use llmsdk_anthropic::internal::{AnthropicMessagesModel, Inner as AnthropicInner};
use llmsdk_provider::ProviderError;
use serde_json::Value;

use crate::config::Inner as BedrockInner;
use crate::sigv4_auth::AnthropicAuthAdapter;

/// Anthropic-on-Bedrock model handle.
///
/// Public re-export of [`AnthropicMessagesModel`] with a Bedrock-flavored
/// [`Inner`](AnthropicInner) baked in.
pub type AmazonBedrockAnthropicModel = AnthropicMessagesModel;

impl AmazonBedrockAnthropicModelExt for AmazonBedrockAnthropicModel {}

/// Internal extension trait used to expose the cross-crate constructor as a
/// named factory (`AmazonBedrockAnthropicModel::new`) — type aliases cannot
/// carry inherent methods of their own.
pub(crate) trait AmazonBedrockAnthropicModelExt {
    /// Construct an Anthropic-on-Bedrock model handle from the parent
    /// Bedrock provider state + a Bedrock Anthropic model id (e.g.
    /// `"anthropic.claude-3-5-sonnet-20241022-v2:0"`).
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError`] when the underlying
    /// [`AnthropicInner`](llmsdk_anthropic::internal::Inner) builder fails
    /// to materialize (e.g. HTTP client init).
    #[allow(
        clippy::new_ret_no_self,
        reason = "type aliases cannot define inherent `new`; this is the closest user-facing factory"
    )]
    fn new(
        bedrock: Arc<BedrockInner>,
        model_id: String,
    ) -> Result<AmazonBedrockAnthropicModel, ProviderError> {
        let auth_adapter = Arc::new(AnthropicAuthAdapter {
            auth: bedrock.auth.clone(),
        });
        let base_url = bedrock.runtime_base_url.clone();
        // Bedrock rejects `output_config.format` for `claude-opus-4-7`
        // (incl. regional variants like `us.anthropic.claude-opus-4-7`).
        // Mirrors ai-sdk `supportsNativeStructuredOutput: !modelId.includes('claude-opus-4-7')`
        // in `amazon-bedrock-anthropic-provider.ts:336`.
        let strips_structured_output = model_id.contains("claude-opus-4-7");

        let inner = AnthropicInner::builder()
            .base_url(base_url)
            .provider_name("bedrock.anthropic.messages")
            .http_client(bedrock.http.clone())
            .request_auth(auth_adapter)
            .endpoint(|base, model_id, is_streaming| {
                let suffix = if is_streaming {
                    "invoke-with-response-stream"
                } else {
                    "invoke"
                };
                format!("{base}/model/{}/{suffix}", encode_path_segment(model_id))
            })
            .body_transform(move |body: &mut Value| {
                let Some(obj) = body.as_object_mut() else {
                    return;
                };
                // Bedrock injects model + version via URL / body knobs.
                obj.remove("model");
                obj.remove("stream");
                obj.insert(
                    "anthropic_version".to_owned(),
                    Value::String("bedrock-2023-05-31".to_owned()),
                );
                if strips_structured_output {
                    strip_output_config_format(obj);
                }
                apply_bedrock_tool_upgrades(obj);
            })
            .build()?;

        Ok(AnthropicMessagesModel::new(Arc::new(inner), model_id))
    }
}

/// Re-export of the chat path-segment encoder so the URL hook can call it
/// without depending on the chat module type.
fn encode_path_segment(input: &str) -> String {
    crate::chat::encode_path_segment(input)
}

/// Remove `output_config.format` from the request body, deleting the whole
/// `output_config` key when it becomes empty.
///
/// Used to satisfy Bedrock's rejection of `output_config.format` for
/// `claude-opus-4-7` (and regional variants like `us.anthropic.claude-opus-4-7`).
fn strip_output_config_format(obj: &mut serde_json::Map<String, Value>) {
    if let Some(Value::Object(oc)) = obj.get_mut("output_config") {
        oc.remove("format");
        if oc.is_empty() {
            obj.remove("output_config");
        }
    }
}

/// Upgrade legacy Anthropic tool versions to the variants Bedrock accepts,
/// rename tools that picked up a new name (`text_editor_20250728` →
/// `str_replace_based_edit_tool`), and collect the required
/// `anthropic_beta` tokens into the request body.
///
/// Mirrors `BEDROCK_TOOL_VERSION_MAP` / `BEDROCK_TOOL_NAME_MAP` /
/// `BEDROCK_TOOL_BETA_MAP` from
/// `amazon-bedrock-anthropic-provider.ts`.
fn apply_bedrock_tool_upgrades(obj: &mut serde_json::Map<String, Value>) {
    let Some(Value::Array(tools)) = obj.get_mut("tools") else {
        return;
    };
    let mut required_betas: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

    for tool in tools.iter_mut() {
        let Some(tool_obj) = tool.as_object_mut() else {
            continue;
        };
        let original_type = tool_obj
            .get("type")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        let Some(original_type) = original_type else {
            continue;
        };

        let upgraded_type = bedrock_tool_version_map(&original_type);
        if let Some(new_type) = upgraded_type {
            tool_obj.insert("type".to_owned(), Value::String(new_type.to_owned()));
            if let Some(new_name) = bedrock_tool_name_map(new_type) {
                tool_obj.insert("name".to_owned(), Value::String(new_name.to_owned()));
            }
            if let Some(beta) = bedrock_tool_beta_map(new_type) {
                required_betas.insert(beta.to_owned());
            }
            continue;
        }

        // Even when no version upgrade is needed, surface the required beta
        // and apply any name override (parity with upstream's else branch).
        if let Some(beta) = bedrock_tool_beta_map(&original_type) {
            required_betas.insert(beta.to_owned());
        }
        if let Some(new_name) = bedrock_tool_name_map(&original_type) {
            tool_obj.insert("name".to_owned(), Value::String(new_name.to_owned()));
        }
    }

    if required_betas.is_empty() {
        return;
    }

    // Merge with any existing `anthropic_beta` array a caller may have set.
    let mut merged: std::collections::BTreeSet<String> = required_betas;
    if let Some(Value::Array(existing)) = obj.get("anthropic_beta") {
        for v in existing {
            if let Some(s) = v.as_str() {
                merged.insert(s.to_owned());
            }
        }
    }
    let array: Vec<Value> = merged.into_iter().map(Value::String).collect();
    obj.insert("anthropic_beta".to_owned(), Value::Array(array));
}

fn bedrock_tool_version_map(type_id: &str) -> Option<&'static str> {
    match type_id {
        "bash_20241022" => Some("bash_20250124"),
        "text_editor_20241022" => Some("text_editor_20250728"),
        "computer_20241022" => Some("computer_20250124"),
        _ => None,
    }
}

fn bedrock_tool_name_map(type_id: &str) -> Option<&'static str> {
    match type_id {
        "text_editor_20250728" => Some("str_replace_based_edit_tool"),
        _ => None,
    }
}

fn bedrock_tool_beta_map(type_id: &str) -> Option<&'static str> {
    match type_id {
        "bash_20250124"
        | "text_editor_20250124"
        | "text_editor_20250429"
        | "text_editor_20250728"
        | "computer_20250124" => Some("computer-use-2025-01-24"),
        "bash_20241022" | "text_editor_20241022" | "computer_20241022" => {
            Some("computer-use-2024-10-22")
        }
        "tool_search_tool_regex_20251119" | "tool_search_tool_bm25_20251119" => {
            Some("tool-search-tool-2025-10-19")
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn run(input: Value) -> Value {
        let Value::Object(mut obj) = input else {
            panic!("test input must be object");
        };
        apply_bedrock_tool_upgrades(&mut obj);
        Value::Object(obj)
    }

    #[test]
    fn upgrades_bash_20241022_and_adds_beta() {
        let out = run(json!({
            "tools": [{"type": "bash_20241022", "name": "bash"}]
        }));
        assert_eq!(
            out["tools"][0]["type"].as_str(),
            Some("bash_20250124"),
            "type upgraded"
        );
        assert!(
            out["anthropic_beta"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v == "computer-use-2025-01-24")
        );
    }

    #[test]
    fn upgrades_text_editor_renames_tool() {
        let out = run(json!({
            "tools": [{"type": "text_editor_20241022", "name": "str_replace_editor"}]
        }));
        assert_eq!(
            out["tools"][0]["type"].as_str(),
            Some("text_editor_20250728")
        );
        assert_eq!(
            out["tools"][0]["name"].as_str(),
            Some("str_replace_based_edit_tool")
        );
    }

    #[test]
    fn preserves_already_current_versions_but_adds_beta() {
        let out = run(json!({
            "tools": [{"type": "computer_20250124", "name": "computer"}]
        }));
        assert_eq!(out["tools"][0]["type"].as_str(), Some("computer_20250124"));
        assert!(
            out["anthropic_beta"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v == "computer-use-2025-01-24")
        );
    }

    #[test]
    fn strip_output_config_format_drops_format_key() {
        let mut obj = json!({
            "output_config": {
                "format": {"type": "json_schema"},
                "effort": "high"
            }
        })
        .as_object()
        .cloned()
        .unwrap();
        strip_output_config_format(&mut obj);
        assert!(obj.get("output_config").unwrap().get("format").is_none());
        assert_eq!(
            obj.get("output_config").unwrap().get("effort"),
            Some(&Value::String("high".into()))
        );
    }

    #[test]
    fn strip_output_config_format_removes_empty_output_config() {
        let mut obj = json!({
            "output_config": {"format": {"type": "json_schema"}},
            "messages": []
        })
        .as_object()
        .cloned()
        .unwrap();
        strip_output_config_format(&mut obj);
        assert!(obj.get("output_config").is_none());
        assert!(obj.get("messages").is_some());
    }

    #[test]
    fn ignores_non_mapped_tools() {
        let out = run(json!({
            "tools": [{"type": "web_search_20250305", "name": "web_search"}]
        }));
        assert!(out.get("anthropic_beta").is_none());
        assert_eq!(
            out["tools"][0]["type"].as_str(),
            Some("web_search_20250305")
        );
    }

    #[test]
    fn merges_with_existing_anthropic_beta() {
        let out = run(json!({
            "anthropic_beta": ["preexisting-flag"],
            "tools": [{"type": "bash_20241022", "name": "bash"}]
        }));
        let arr = out["anthropic_beta"].as_array().unwrap();
        assert!(arr.iter().any(|v| v == "preexisting-flag"));
        assert!(arr.iter().any(|v| v == "computer-use-2025-01-24"));
    }
}
