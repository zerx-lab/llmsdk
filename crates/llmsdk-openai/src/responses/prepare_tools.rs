//! Convert `CallOptions.tools` + `tool_choice` → wire `tools` + `tool_choice`.
//!
//! Mirrors `@ai-sdk/openai/src/responses/openai-responses-prepare-tools.ts`.
// Rust guideline compliant 2026-02-21

use std::collections::HashSet;

use llmsdk_provider::language_model::{FunctionTool, ProviderTool, Tool, ToolChoice};
use llmsdk_provider::shared::Warning;
use serde_json::{Value as JsonValue, json};

use super::options::{AllowedTools, AllowedToolsMode};
use super::tools::{
    apply_patch, code_interpreter, custom, file_search, ids, image_generation, mcp, shell,
    tool_search, web_search, web_search_preview,
};
use super::wire::request::{ToolChoiceMode, WireToolChoice};

/// Result of preparing the wire-shape tool block.
#[derive(Debug, Clone, Default)]
pub struct PreparedTools {
    pub tools: Option<Vec<JsonValue>>,
    pub tool_choice: Option<WireToolChoice>,
    pub warnings: Vec<Warning>,
    /// Logical name of the configured web_search* tool, if any (used by
    /// the response/stream parsers when emitting tool-call/result pairs).
    pub web_search_tool_name: Option<String>,
    /// True when the configured `shell` tool's environment runs server-side
    /// (`containerAuto` / `containerReference`), so the corresponding
    /// tool-call should carry `provider_executed: true`.
    pub is_shell_provider_executed: bool,
    /// Custom tool names declared via `openai.custom` (for `tool_choice: tool`
    /// disambiguation in `Selector(...)` against function / custom).
    pub custom_tool_names: HashSet<String>,
}

/// Route the call's tool list + choice to the wire shape.
pub fn prepare(
    tools: Option<&[Tool]>,
    tool_choice: Option<&ToolChoice>,
    allowed_tools: Option<&AllowedTools>,
) -> PreparedTools {
    let Some(tools) = tools.filter(|t| !t.is_empty()) else {
        return PreparedTools::default();
    };

    let mut prepared = PreparedTools::default();
    let mut wire_tools: Vec<JsonValue> = Vec::new();

    for tool in tools {
        match tool {
            Tool::Function(f) => wire_tools.push(serialize_function_tool(f)),
            Tool::Provider(p) => {
                if let Some(value) = route_provider_tool(p, &mut prepared) {
                    wire_tools.push(value);
                }
            }
        }
    }

    prepared.tools = Some(wire_tools);

    // `allowedTools` provider option overrides the per-call tool_choice.
    if let Some(allowed) = allowed_tools {
        prepared.tool_choice = Some(WireToolChoice::Selector(json!({
            "type": "allowed_tools",
            "mode": match allowed.mode.unwrap_or(AllowedToolsMode::Auto) {
                AllowedToolsMode::Auto => "auto",
                AllowedToolsMode::Required => "required",
            },
            "tools": allowed
                .tool_names
                .iter()
                .map(|name| json!({ "type": "function", "name": name }))
                .collect::<Vec<_>>(),
        })));
        return prepared;
    }

    prepared.tool_choice = match tool_choice {
        None => None,
        Some(ToolChoice::Auto) => Some(WireToolChoice::Mode(ToolChoiceMode::Auto)),
        Some(ToolChoice::None) => Some(WireToolChoice::Mode(ToolChoiceMode::None)),
        Some(ToolChoice::Required) => Some(WireToolChoice::Mode(ToolChoiceMode::Required)),
        Some(ToolChoice::Tool { tool_name }) => Some(map_tool_choice_tool(tool_name, &prepared)),
    };

    prepared
}

fn serialize_function_tool(tool: &FunctionTool) -> JsonValue {
    let mut obj = serde_json::Map::new();
    obj.insert("type".into(), json!("function"));
    obj.insert("name".into(), json!(tool.name));
    if let Some(desc) = &tool.description {
        obj.insert("description".into(), json!(desc));
    }
    obj.insert(
        "parameters".into(),
        serde_json::to_value(&tool.input_schema).unwrap_or(JsonValue::Null),
    );
    if let Some(strict) = tool.strict {
        obj.insert("strict".into(), json!(strict));
    }
    // OpenAI-specific `defer_loading` lives under provider_options.openai
    // (mirrors ai-sdk handling).
    if let Some(po) = &tool.provider_options
        && let Some(openai) = po.get("openai")
        && let Some(defer) = openai.get("deferLoading").and_then(JsonValue::as_bool)
    {
        obj.insert("defer_loading".into(), json!(defer));
    }
    JsonValue::Object(obj)
}

#[allow(
    clippy::too_many_lines,
    reason = "11 provider tools route through one switch"
)]
fn route_provider_tool(tool: &ProviderTool, prepared: &mut PreparedTools) -> Option<JsonValue> {
    let raw_args = tool
        .args
        .clone()
        .map(JsonValue::Object)
        .unwrap_or(json!({}));

    let push_invalid = |prepared: &mut PreparedTools, err: serde_json::Error| {
        prepared.warnings.push(Warning::UnsupportedTool {
            tool: tool.id.clone(),
            details: Some(format!("invalid args: {err}")),
        });
    };

    match tool.id.as_str() {
        ids::WEB_SEARCH => {
            let args: web_search::Args = match serde_json::from_value(raw_args) {
                Ok(v) => v,
                Err(e) => {
                    push_invalid(prepared, e);
                    return None;
                }
            };
            prepared.web_search_tool_name = Some(tool.name.clone());
            let mut obj = serde_json::Map::new();
            obj.insert("type".into(), json!("web_search"));
            if let Some(v) = args.external_web_access {
                obj.insert("external_web_access".into(), json!(v));
            }
            if let Some(f) = args.filters {
                if let Some(allowed) = f.allowed_domains {
                    obj.insert("filters".into(), json!({ "allowed_domains": allowed }));
                } else {
                    obj.insert("filters".into(), json!({}));
                }
            }
            if let Some(s) = args.search_context_size {
                obj.insert("search_context_size".into(), json!(s));
            }
            if let Some(loc) = args.user_location {
                obj.insert(
                    "user_location".into(),
                    serde_json::to_value(loc).unwrap_or(JsonValue::Null),
                );
            }
            Some(JsonValue::Object(obj))
        }
        ids::WEB_SEARCH_PREVIEW => {
            let args: web_search_preview::Args = match serde_json::from_value(raw_args) {
                Ok(v) => v,
                Err(e) => {
                    push_invalid(prepared, e);
                    return None;
                }
            };
            prepared.web_search_tool_name = Some(tool.name.clone());
            let mut obj = serde_json::Map::new();
            obj.insert("type".into(), json!("web_search_preview"));
            if let Some(s) = args.search_context_size {
                obj.insert("search_context_size".into(), json!(s));
            }
            if let Some(loc) = args.user_location {
                obj.insert(
                    "user_location".into(),
                    serde_json::to_value(loc).unwrap_or(JsonValue::Null),
                );
            }
            Some(JsonValue::Object(obj))
        }
        ids::FILE_SEARCH => {
            let args: file_search::Args = match serde_json::from_value(raw_args) {
                Ok(v) => v,
                Err(e) => {
                    push_invalid(prepared, e);
                    return None;
                }
            };
            let mut obj = serde_json::Map::new();
            obj.insert("type".into(), json!("file_search"));
            obj.insert("vector_store_ids".into(), json!(args.vector_store_ids));
            if let Some(n) = args.max_num_results {
                obj.insert("max_num_results".into(), json!(n));
            }
            if let Some(r) = args.ranking {
                let mut ro = serde_json::Map::new();
                if let Some(rk) = r.ranker {
                    ro.insert("ranker".into(), json!(rk));
                }
                if let Some(st) = r.score_threshold {
                    ro.insert("score_threshold".into(), json!(st));
                }
                obj.insert("ranking_options".into(), JsonValue::Object(ro));
            }
            if let Some(f) = args.filters {
                obj.insert("filters".into(), f);
            }
            Some(JsonValue::Object(obj))
        }
        ids::CODE_INTERPRETER => {
            let args: code_interpreter::Args = match serde_json::from_value(raw_args) {
                Ok(v) => v,
                Err(e) => {
                    push_invalid(prepared, e);
                    return None;
                }
            };
            let container = match args.container {
                None => json!({ "type": "auto", "file_ids": JsonValue::Null }),
                Some(code_interpreter::ContainerArg::Id(s)) => json!(s),
                Some(code_interpreter::ContainerArg::Auto(c)) => json!({
                    "type": "auto",
                    "file_ids": c.file_ids,
                }),
            };
            Some(json!({ "type": "code_interpreter", "container": container }))
        }
        ids::IMAGE_GENERATION => {
            let args: image_generation::Args = match serde_json::from_value(raw_args) {
                Ok(v) => v,
                Err(e) => {
                    push_invalid(prepared, e);
                    return None;
                }
            };
            // serde_json with skip_if=None drops all None fields, matching upstream.
            let mut value = serde_json::to_value(&args).unwrap_or(JsonValue::Null);
            if let JsonValue::Object(map) = &mut value {
                map.insert("type".into(), json!("image_generation"));
            }
            Some(value)
        }
        ids::LOCAL_SHELL => Some(json!({ "type": "local_shell" })),
        ids::SHELL => {
            let args: shell::Args = match serde_json::from_value(raw_args) {
                Ok(v) => v,
                Err(e) => {
                    push_invalid(prepared, e);
                    return None;
                }
            };
            if let Some(env) = &args.environment
                && !matches!(env, shell::Environment::Local { .. })
            {
                prepared.is_shell_provider_executed = true;
            }
            let mut obj = serde_json::Map::new();
            obj.insert("type".into(), json!("shell"));
            if let Some(env) = args.environment {
                obj.insert(
                    "environment".into(),
                    serde_json::to_value(env).unwrap_or(JsonValue::Null),
                );
            }
            Some(JsonValue::Object(obj))
        }
        ids::APPLY_PATCH => {
            let _args: apply_patch::Args = match serde_json::from_value(raw_args) {
                Ok(v) => v,
                Err(e) => {
                    push_invalid(prepared, e);
                    return None;
                }
            };
            Some(json!({ "type": "apply_patch" }))
        }
        ids::MCP => {
            let args: mcp::Args = match serde_json::from_value(raw_args) {
                Ok(v) => v,
                Err(e) => {
                    push_invalid(prepared, e);
                    return None;
                }
            };
            if args.server_url.is_none() && args.connector_id.is_none() {
                prepared.warnings.push(Warning::UnsupportedTool {
                    tool: tool.id.clone(),
                    details: Some("MCP tool requires serverUrl or connectorId".into()),
                });
                return None;
            }
            let mut obj = serde_json::Map::new();
            obj.insert("type".into(), json!("mcp"));
            obj.insert("server_label".into(), json!(args.server_label));
            if let Some(a) = args.allowed_tools {
                obj.insert(
                    "allowed_tools".into(),
                    serde_json::to_value(a).unwrap_or(JsonValue::Null),
                );
            }
            if let Some(auth) = args.authorization {
                obj.insert("authorization".into(), json!(auth));
            }
            if let Some(cid) = args.connector_id {
                obj.insert("connector_id".into(), json!(cid));
            }
            if let Some(h) = args.headers {
                obj.insert("headers".into(), json!(h));
            }
            obj.insert(
                "require_approval".into(),
                args.require_approval
                    .as_ref()
                    .map(|v| serde_json::to_value(v).unwrap_or(JsonValue::Null))
                    .unwrap_or_else(|| json!("never")),
            );
            if let Some(d) = args.server_description {
                obj.insert("server_description".into(), json!(d));
            }
            if let Some(u) = args.server_url {
                obj.insert("server_url".into(), json!(u));
            }
            Some(JsonValue::Object(obj))
        }
        ids::CUSTOM => {
            let args: custom::Args = match serde_json::from_value(raw_args) {
                Ok(v) => v,
                Err(e) => {
                    push_invalid(prepared, e);
                    return None;
                }
            };
            prepared.custom_tool_names.insert(tool.name.clone());
            let mut obj = serde_json::Map::new();
            obj.insert("type".into(), json!("custom"));
            obj.insert("name".into(), json!(tool.name));
            if let Some(d) = args.description {
                obj.insert("description".into(), json!(d));
            }
            if let Some(f) = args.format {
                obj.insert(
                    "format".into(),
                    serde_json::to_value(f).unwrap_or(JsonValue::Null),
                );
            }
            Some(JsonValue::Object(obj))
        }
        ids::TOOL_SEARCH => {
            let args: tool_search::Args = match serde_json::from_value(raw_args) {
                Ok(v) => v,
                Err(e) => {
                    push_invalid(prepared, e);
                    return None;
                }
            };
            let mut obj = serde_json::Map::new();
            obj.insert("type".into(), json!("tool_search"));
            if let Some(e) = args.execution {
                obj.insert("execution".into(), json!(e));
            }
            if let Some(d) = args.description {
                obj.insert("description".into(), json!(d));
            }
            if let Some(p) = args.parameters {
                obj.insert("parameters".into(), p);
            }
            Some(JsonValue::Object(obj))
        }
        other => {
            prepared.warnings.push(Warning::UnsupportedTool {
                tool: other.to_string(),
                details: Some(format!("unknown provider tool id {other}")),
            });
            None
        }
    }
}

/// `tool_choice: { type: "tool", toolName }` → wire `tool_choice` selector.
///
/// Provider-defined tools (file_search / web_search / etc.) need a `{ type }`
/// selector; function tools use `{ type: "function", name }`; custom tools use
/// `{ type: "custom", name }`.
fn map_tool_choice_tool(tool_name: &str, prepared: &PreparedTools) -> WireToolChoice {
    match tool_name {
        "code_interpreter" | "file_search" | "image_generation" | "web_search_preview"
        | "web_search" | "mcp" | "apply_patch" => {
            WireToolChoice::Selector(json!({ "type": tool_name }))
        }
        _ if prepared.custom_tool_names.contains(tool_name) => {
            WireToolChoice::Selector(json!({ "type": "custom", "name": tool_name }))
        }
        _ => WireToolChoice::Selector(json!({ "type": "function", "name": tool_name })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmsdk_provider::json::JsonSchema;
    use llmsdk_provider::shared::ProviderOptions;

    fn empty_schema() -> JsonSchema {
        serde_json::from_value(serde_json::json!({"type": "object"})).unwrap()
    }

    fn function_tool(name: &str) -> Tool {
        Tool::Function(FunctionTool {
            name: name.into(),
            description: Some("d".into()),
            input_schema: empty_schema(),
            input_examples: None,
            strict: Some(true),
            provider_options: None,
        })
    }

    fn provider_tool(id: &str, name: &str, args: serde_json::Value) -> Tool {
        Tool::Provider(ProviderTool {
            id: id.into(),
            name: name.into(),
            args: args.as_object().cloned(),
            provider_options: None,
        })
    }

    #[test]
    fn empty_tools_returns_none() {
        let p = prepare(None, None, None);
        assert!(p.tools.is_none());
        let p = prepare(Some(&[]), None, None);
        assert!(p.tools.is_none());
    }

    #[test]
    fn function_tool_includes_strict_and_parameters() {
        let p = prepare(Some(&[function_tool("weather")]), None, None);
        let tools = p.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["name"], "weather");
        assert_eq!(tools[0]["strict"], true);
        assert!(tools[0].get("parameters").is_some());
    }

    #[test]
    fn function_tool_picks_up_defer_loading() {
        let mut po = ProviderOptions::new();
        po.insert(
            "openai".into(),
            serde_json::json!({"deferLoading": true})
                .as_object()
                .unwrap()
                .clone(),
        );
        let t = Tool::Function(FunctionTool {
            name: "x".into(),
            description: None,
            input_schema: empty_schema(),
            input_examples: None,
            strict: None,
            provider_options: Some(po),
        });
        let p = prepare(Some(&[t]), None, None);
        assert_eq!(p.tools.unwrap()[0]["defer_loading"], true);
    }

    #[test]
    fn web_search_routes_with_search_context_size() {
        let p = prepare(
            Some(&[provider_tool(
                "openai.web_search",
                "web_search",
                serde_json::json!({"searchContextSize": "high"}),
            )]),
            None,
            None,
        );
        let tools = p.tools.unwrap();
        assert_eq!(tools[0]["type"], "web_search");
        assert_eq!(tools[0]["search_context_size"], "high");
        assert_eq!(p.web_search_tool_name.as_deref(), Some("web_search"));
    }

    #[test]
    fn file_search_translates_camel_to_snake() {
        let p = prepare(
            Some(&[provider_tool(
                "openai.file_search",
                "file_search",
                serde_json::json!({
                    "vectorStoreIds": ["vs1"],
                    "maxNumResults": 5,
                    "ranking": { "ranker": "auto", "scoreThreshold": 0.5 }
                }),
            )]),
            None,
            None,
        );
        let tools = p.tools.unwrap();
        assert_eq!(tools[0]["type"], "file_search");
        assert_eq!(tools[0]["vector_store_ids"][0], "vs1");
        assert_eq!(tools[0]["max_num_results"], 5);
        assert_eq!(tools[0]["ranking_options"]["score_threshold"], 0.5);
    }

    #[test]
    fn code_interpreter_default_container_is_auto() {
        let p = prepare(
            Some(&[provider_tool(
                "openai.code_interpreter",
                "code_interpreter",
                serde_json::json!({}),
            )]),
            None,
            None,
        );
        let tools = p.tools.unwrap();
        assert_eq!(tools[0]["container"]["type"], "auto");
    }

    #[test]
    fn code_interpreter_string_container() {
        let p = prepare(
            Some(&[provider_tool(
                "openai.code_interpreter",
                "code_interpreter",
                serde_json::json!({ "container": "cnt_abc" }),
            )]),
            None,
            None,
        );
        assert_eq!(p.tools.unwrap()[0]["container"], "cnt_abc");
    }

    #[test]
    fn mcp_requires_url_or_connector() {
        let p = prepare(
            Some(&[provider_tool(
                "openai.mcp",
                "mcp",
                serde_json::json!({ "serverLabel": "fs" }),
            )]),
            None,
            None,
        );
        // Tool is dropped + warning surfaced.
        assert!(p.tools.unwrap().is_empty());
        assert_eq!(p.warnings.len(), 1);
    }

    #[test]
    fn shell_container_auto_marks_provider_executed() {
        let p = prepare(
            Some(&[provider_tool(
                "openai.shell",
                "shell",
                serde_json::json!({
                    "environment": { "type": "containerAuto", "memoryLimit": "4g" }
                }),
            )]),
            None,
            None,
        );
        assert!(p.is_shell_provider_executed);
        let tools = p.tools.unwrap();
        assert_eq!(tools[0]["environment"]["type"], "containerAuto");
    }

    #[test]
    fn local_shell_takes_no_args() {
        let p = prepare(
            Some(&[provider_tool(
                "openai.local_shell",
                "local_shell",
                serde_json::json!({}),
            )]),
            None,
            None,
        );
        let tools = p.tools.unwrap();
        assert_eq!(tools[0], serde_json::json!({ "type": "local_shell" }));
    }

    #[test]
    fn custom_tool_records_name() {
        let p = prepare(
            Some(&[provider_tool(
                "openai.custom",
                "my_custom",
                serde_json::json!({ "description": "x" }),
            )]),
            Some(&ToolChoice::Tool {
                tool_name: "my_custom".into(),
            }),
            None,
        );
        assert!(p.custom_tool_names.contains("my_custom"));
        let WireToolChoice::Selector(sel) = p.tool_choice.unwrap() else {
            panic!("expected selector");
        };
        assert_eq!(sel["type"], "custom");
        assert_eq!(sel["name"], "my_custom");
    }

    #[test]
    fn tool_choice_required_maps_to_mode() {
        let p = prepare(
            Some(&[function_tool("x")]),
            Some(&ToolChoice::Required),
            None,
        );
        assert!(matches!(
            p.tool_choice,
            Some(WireToolChoice::Mode(ToolChoiceMode::Required))
        ));
    }

    #[test]
    fn tool_choice_tool_targets_provider_tool() {
        let p = prepare(
            Some(&[provider_tool(
                "openai.file_search",
                "file_search",
                serde_json::json!({ "vectorStoreIds": ["v"] }),
            )]),
            Some(&ToolChoice::Tool {
                tool_name: "file_search".into(),
            }),
            None,
        );
        let WireToolChoice::Selector(sel) = p.tool_choice.unwrap() else {
            panic!("selector");
        };
        assert_eq!(sel, serde_json::json!({ "type": "file_search" }));
    }

    #[test]
    fn allowed_tools_overrides_choice() {
        let p = prepare(
            Some(&[function_tool("a"), function_tool("b")]),
            Some(&ToolChoice::Required),
            Some(&AllowedTools {
                tool_names: vec!["a".into()],
                mode: Some(AllowedToolsMode::Required),
            }),
        );
        let WireToolChoice::Selector(sel) = p.tool_choice.unwrap() else {
            panic!("selector");
        };
        assert_eq!(sel["type"], "allowed_tools");
        assert_eq!(sel["mode"], "required");
        assert_eq!(sel["tools"][0]["name"], "a");
    }

    #[test]
    fn unknown_provider_tool_warns() {
        let p = prepare(
            Some(&[provider_tool(
                "openai.future_thing",
                "future_thing",
                serde_json::json!({}),
            )]),
            None,
            None,
        );
        assert!(p.tools.unwrap().is_empty());
        assert_eq!(p.warnings.len(), 1);
    }
}
