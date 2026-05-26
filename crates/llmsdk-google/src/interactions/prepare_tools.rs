//! Maps llmsdk `Tool` definitions to the Gemini Interactions API `tools[]`
//! and `tool_choice` request fields.
//!
//! Mirrors `@ai-sdk/google/src/interactions/prepare-google-interactions-tools.ts`.
//! The Interactions surface supports the same Google provider-defined tool ids
//! used by the `:generateContent` path, plus a few that are Interactions-only
//! (`computer_use`, `mcp_server`, `retrieval`).
// Rust guideline compliant 2026-05-25

use llmsdk_provider::language_model::{Tool, ToolChoice};
use llmsdk_provider::shared::Warning;
use serde_json::{Map as JsonMap, Value as JsonValue, json};

/// Result of [`prepare_tools`]: typed wire entries + the resolved tool-choice
/// payload + any warnings (e.g. unsupported tool kinds were dropped).
pub(crate) struct PreparedTools {
    pub tools: Option<Vec<JsonValue>>,
    pub tool_choice: Option<JsonValue>,
    pub warnings: Vec<Warning>,
}

/// Translate the llmsdk tool list + tool-choice into Interactions wire shape.
///
/// `None` for both `tools` and `tool_choice` in the result means "do not emit
/// the field" — the Interactions API rejects `tool_choice` when no function
/// tool is in the list (mirrors upstream prepare-google-interactions-tools.ts
/// `hasFunctionTool` guard).
pub(crate) fn prepare_tools(
    tools: Option<&[Tool]>,
    tool_choice: Option<&ToolChoice>,
) -> PreparedTools {
    let mut warnings = Vec::new();
    let Some(tools) = tools.filter(|t| !t.is_empty()) else {
        return PreparedTools {
            tools: None,
            tool_choice: None,
            warnings,
        };
    };

    let mut wire: Vec<JsonValue> = Vec::new();
    for tool in tools {
        match tool {
            Tool::Function(f) => {
                let mut entry = JsonMap::new();
                entry.insert("type".into(), JsonValue::String("function".into()));
                entry.insert("name".into(), JsonValue::String(f.name.clone()));
                entry.insert(
                    "description".into(),
                    JsonValue::String(f.description.clone().unwrap_or_default()),
                );
                entry.insert(
                    "parameters".into(),
                    serde_json::to_value(&f.input_schema).unwrap_or(JsonValue::Null),
                );
                wire.push(JsonValue::Object(entry));
            }
            Tool::Provider(p) => match prepare_provider_tool(&p.id, p.args.as_ref()) {
                Ok(entry) => wire.push(entry),
                Err(message) => warnings.push(Warning::Other { message }),
            },
        }
    }

    // Drop tool_choice unless at least one function tool was emitted (the
    // Interactions API rejects `tool_choice` otherwise — see upstream
    // prepare-google-interactions-tools.ts:215).
    let has_function = wire
        .iter()
        .any(|t| t.get("type").and_then(JsonValue::as_str) == Some("function"));

    let tool_choice = if has_function {
        tool_choice.and_then(map_tool_choice)
    } else {
        None
    };

    PreparedTools {
        tools: (!wire.is_empty()).then_some(wire),
        tool_choice,
        warnings,
    }
}

fn prepare_provider_tool(
    id: &str,
    args: Option<&JsonMap<String, JsonValue>>,
) -> Result<JsonValue, String> {
    let empty = JsonMap::new();
    let args_map: &JsonMap<String, JsonValue> = args.unwrap_or(&empty);

    match id {
        "google.google_search" => {
            let mut search_types: Vec<&'static str> = Vec::new();
            if let Some(obj) = args_map.get("searchTypes").and_then(JsonValue::as_object) {
                if obj.contains_key("webSearch") {
                    search_types.push("web_search");
                }
                if obj.contains_key("imageSearch") {
                    search_types.push("image_search");
                }
            }
            let mut entry = JsonMap::new();
            entry.insert("type".into(), JsonValue::String("google_search".into()));
            if !search_types.is_empty() {
                entry.insert(
                    "search_types".into(),
                    JsonValue::Array(
                        search_types
                            .into_iter()
                            .map(|s| JsonValue::String(s.into()))
                            .collect(),
                    ),
                );
            }
            Ok(JsonValue::Object(entry))
        }
        "google.code_execution" => Ok(json!({"type": "code_execution"})),
        "google.url_context" => Ok(json!({"type": "url_context"})),
        "google.file_search" => {
            let mut entry = JsonMap::new();
            entry.insert("type".into(), JsonValue::String("file_search".into()));
            if let Some(v) = args_map.get("fileSearchStoreNames") {
                entry.insert("file_search_store_names".into(), v.clone());
            }
            if let Some(v) = args_map.get("topK") {
                entry.insert("top_k".into(), v.clone());
            }
            if let Some(v) = args_map.get("metadataFilter") {
                entry.insert("metadata_filter".into(), v.clone());
            }
            Ok(JsonValue::Object(entry))
        }
        "google.google_maps" => {
            let mut entry = JsonMap::new();
            entry.insert("type".into(), JsonValue::String("google_maps".into()));
            if let Some(v) = args_map.get("latitude") {
                entry.insert("latitude".into(), v.clone());
            }
            if let Some(v) = args_map.get("longitude") {
                entry.insert("longitude".into(), v.clone());
            }
            if let Some(v) = args_map.get("enableWidget") {
                entry.insert("enable_widget".into(), v.clone());
            }
            Ok(JsonValue::Object(entry))
        }
        "google.computer_use" => {
            let mut entry = JsonMap::new();
            entry.insert("type".into(), JsonValue::String("computer_use".into()));
            let environment = args_map
                .get("environment")
                .and_then(JsonValue::as_str)
                .unwrap_or("browser");
            entry.insert(
                "environment".into(),
                JsonValue::String(environment.to_owned()),
            );
            if let Some(v) = args_map.get("excludedPredefinedFunctions") {
                entry.insert("excludedPredefinedFunctions".into(), v.clone());
            }
            Ok(JsonValue::Object(entry))
        }
        "google.mcp_server" => {
            let mut entry = JsonMap::new();
            entry.insert("type".into(), JsonValue::String("mcp_server".into()));
            if let Some(v) = args_map.get("name") {
                entry.insert("name".into(), v.clone());
            }
            if let Some(v) = args_map.get("url") {
                entry.insert("url".into(), v.clone());
            }
            if let Some(v) = args_map.get("headers") {
                entry.insert("headers".into(), v.clone());
            }
            if let Some(v) = args_map.get("allowedTools") {
                entry.insert("allowed_tools".into(), v.clone());
            }
            Ok(JsonValue::Object(entry))
        }
        "google.retrieval" => {
            let mut entry = JsonMap::new();
            entry.insert("type".into(), JsonValue::String("retrieval".into()));
            let retrieval_types = args_map.get("retrievalTypes").cloned().unwrap_or_else(|| {
                JsonValue::Array(vec![JsonValue::String("vertex_ai_search".into())])
            });
            entry.insert("retrieval_types".into(), retrieval_types);
            if let Some(v) = args_map.get("vertexAiSearchConfig") {
                entry.insert("vertex_ai_search_config".into(), v.clone());
            }
            Ok(JsonValue::Object(entry))
        }
        other => Err(format!(
            "provider-defined tool {other} is not supported by google.interactions; tool dropped"
        )),
    }
}

fn map_tool_choice(choice: &ToolChoice) -> Option<JsonValue> {
    match choice {
        ToolChoice::Auto => Some(JsonValue::String("auto".into())),
        ToolChoice::Required => Some(JsonValue::String("any".into())),
        ToolChoice::None => Some(JsonValue::String("none".into())),
        ToolChoice::Tool { tool_name } => Some(json!({
            "allowed_tools": {
                "mode": "validated",
                "tools": [tool_name],
            }
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmsdk_provider::language_model::{FunctionTool, ProviderTool};
    use serde_json::{Map, json};

    fn provider_tool(id: &str, args: Option<JsonValue>) -> Tool {
        Tool::Provider(ProviderTool {
            id: id.into(),
            name: id.into(),
            args: args.and_then(|v| match v {
                JsonValue::Object(m) => Some(m),
                _ => None,
            }),
            provider_options: None,
        })
    }

    #[test]
    fn google_search_routes_search_types() {
        let tools = [provider_tool(
            "google.google_search",
            Some(json!({"searchTypes": {"webSearch": {}, "imageSearch": {}}})),
        )];
        let result = prepare_tools(Some(&tools), None);
        let tools_out = result.tools.expect("tools");
        assert_eq!(tools_out.len(), 1);
        assert_eq!(
            tools_out[0]
                .get("search_types")
                .and_then(JsonValue::as_array)
                .map(|a| a.len()),
            Some(2)
        );
    }

    #[test]
    fn code_execution_minimal() {
        let tools = [provider_tool("google.code_execution", None)];
        let result = prepare_tools(Some(&tools), None);
        assert_eq!(result.tools.unwrap()[0], json!({"type": "code_execution"}));
    }

    #[test]
    fn file_search_pass_through() {
        let tools = [provider_tool(
            "google.file_search",
            Some(json!({"topK": 3, "metadataFilter": "k=v", "fileSearchStoreNames": ["a"]})),
        )];
        let result = prepare_tools(Some(&tools), None);
        let t = &result.tools.unwrap()[0];
        assert_eq!(t.get("top_k").and_then(JsonValue::as_u64), Some(3));
        assert_eq!(
            t.get("metadata_filter").and_then(JsonValue::as_str),
            Some("k=v")
        );
        assert!(t.get("file_search_store_names").is_some());
    }

    #[test]
    fn computer_use_defaults_environment() {
        let tools = [provider_tool("google.computer_use", None)];
        let result = prepare_tools(Some(&tools), None);
        assert_eq!(
            result.tools.unwrap()[0]
                .get("environment")
                .and_then(JsonValue::as_str),
            Some("browser")
        );
    }

    #[test]
    fn mcp_server_routes_all_fields() {
        let tools = [provider_tool(
            "google.mcp_server",
            Some(json!({
                "name": "n",
                "url": "https://x",
                "headers": {"a": "b"},
                "allowedTools": [{"name": "t"}]
            })),
        )];
        let result = prepare_tools(Some(&tools), None);
        let t = &result.tools.unwrap()[0];
        assert_eq!(t.get("name").and_then(JsonValue::as_str), Some("n"));
        assert_eq!(t.get("url").and_then(JsonValue::as_str), Some("https://x"));
        assert!(t.get("allowed_tools").is_some());
    }

    #[test]
    fn retrieval_defaults_retrieval_types() {
        let tools = [provider_tool("google.retrieval", None)];
        let result = prepare_tools(Some(&tools), None);
        let t = &result.tools.unwrap()[0];
        assert_eq!(
            t.get("retrieval_types"),
            Some(&JsonValue::Array(vec![JsonValue::String(
                "vertex_ai_search".into()
            )]))
        );
    }

    #[test]
    fn unsupported_provider_tool_drops_with_warning() {
        let tools = [provider_tool("google.unknown_tool", None)];
        let result = prepare_tools(Some(&tools), None);
        assert!(result.tools.is_none());
        assert_eq!(result.warnings.len(), 1);
    }

    #[test]
    fn tool_choice_dropped_when_no_function_tool() {
        let tools = [provider_tool("google.code_execution", None)];
        let result = prepare_tools(Some(&tools), Some(&ToolChoice::Auto));
        assert!(result.tool_choice.is_none());
    }

    #[test]
    fn tool_choice_routed_when_function_tool_present() {
        let function = Tool::Function(FunctionTool {
            name: "f".into(),
            description: None,
            input_schema: serde_json::from_value(json!({"type": "object"})).unwrap(),
            input_examples: None,
            strict: None,
            provider_options: None,
        });
        let tools = [function];
        let result = prepare_tools(
            Some(&tools),
            Some(&ToolChoice::Tool {
                tool_name: "f".into(),
            }),
        );
        let _ = Map::<String, JsonValue>::new();
        let tc = result.tool_choice.expect("tool_choice");
        assert_eq!(
            tc.pointer("/allowed_tools/mode")
                .and_then(JsonValue::as_str),
            Some("validated")
        );
    }
}
