//! Convert llmsdk tool definitions / choices to xAI Responses wire format.
//!
//! Mirrors `xai-responses-prepare-tools.ts`. Recognises the seven xAI
//! provider-defined tool ids plus generic `function` tools; anything else
//! raises [`Warning::UnsupportedTool`].
// Rust guideline compliant 2026-05-25

use llmsdk_provider::language_model::{ProviderTool, Tool, ToolChoice};
use llmsdk_provider::shared::Warning;
use serde_json::{Value, json};

/// Resolved tool names for the provider-executed tools that the response
/// parser needs to route by id.
#[derive(Debug, Default)]
pub(crate) struct ResolvedToolNames {
    pub web_search: Option<String>,
    pub x_search: Option<String>,
    pub code_execution: Option<String>,
    pub mcp: Option<String>,
    pub file_search: Option<String>,
}

/// Output of [`prepare`].
#[derive(Debug)]
pub(crate) struct PreparedTools {
    pub tools: Option<Vec<Value>>,
    pub tool_choice: Option<Value>,
    pub warnings: Vec<Warning>,
    pub names: ResolvedToolNames,
}

/// Convert `tools` + `tool_choice` from a [`CallOptions`](llmsdk_provider::language_model::CallOptions)
/// to xAI Responses wire format.
pub(crate) fn prepare(tools: &[Tool], tool_choice: Option<&ToolChoice>) -> PreparedTools {
    let mut warnings: Vec<Warning> = Vec::new();
    let mut names = ResolvedToolNames::default();

    if tools.is_empty() {
        return PreparedTools {
            tools: None,
            tool_choice: None,
            warnings,
            names,
        };
    }

    let mut wire_tools: Vec<Value> = Vec::with_capacity(tools.len());
    for tool in tools {
        match tool {
            Tool::Function(f) => {
                let mut obj = serde_json::Map::new();
                obj.insert("type".into(), json!("function"));
                obj.insert("name".into(), json!(f.name.clone()));
                if let Some(d) = &f.description {
                    obj.insert("description".into(), json!(d));
                }
                obj.insert(
                    "parameters".into(),
                    remove_additional_properties_false(
                        serde_json::to_value(&f.input_schema).unwrap_or(Value::Null),
                    ),
                );
                if let Some(s) = f.strict {
                    obj.insert("strict".into(), json!(s));
                }
                wire_tools.push(Value::Object(obj));
            }
            Tool::Provider(p) => match p.id.as_str() {
                "xai.web_search" => {
                    names.web_search = Some(p.name.clone());
                    wire_tools.push(build_web_search(p));
                }
                "xai.x_search" => {
                    names.x_search = Some(p.name.clone());
                    wire_tools.push(build_x_search(p));
                }
                "xai.code_execution" => {
                    names.code_execution = Some(p.name.clone());
                    wire_tools.push(json!({ "type": "code_interpreter" }));
                }
                "xai.view_image" => {
                    wire_tools.push(json!({ "type": "view_image" }));
                }
                "xai.view_x_video" => {
                    wire_tools.push(json!({ "type": "view_x_video" }));
                }
                "xai.file_search" => {
                    names.file_search = Some(p.name.clone());
                    wire_tools.push(build_file_search(p));
                }
                "xai.mcp" => {
                    names.mcp = Some(p.name.clone());
                    wire_tools.push(build_mcp(p));
                }
                _ => warnings.push(Warning::UnsupportedTool {
                    tool: format!("provider-defined tool {}", p.name),
                    details: Some(format!("unknown xAI tool id `{}`", p.id)),
                }),
            },
        }
    }

    let wire_choice = resolve_tool_choice(tool_choice, tools, &mut warnings);

    PreparedTools {
        tools: (!wire_tools.is_empty()).then_some(wire_tools),
        tool_choice: wire_choice,
        warnings,
        names,
    }
}

fn resolve_tool_choice(
    tool_choice: Option<&ToolChoice>,
    tools: &[Tool],
    warnings: &mut Vec<Warning>,
) -> Option<Value> {
    let choice = tool_choice?;
    match choice {
        ToolChoice::Auto => Some(json!("auto")),
        ToolChoice::None => Some(json!("none")),
        ToolChoice::Required => Some(json!("required")),
        ToolChoice::Tool { tool_name } => {
            let selected = tools.iter().find(|t| {
                matches!(t, Tool::Function(f) if f.name == *tool_name)
                    || matches!(t, Tool::Provider(p) if p.name == *tool_name)
            })?;
            if matches!(selected, Tool::Provider(_)) {
                warnings.push(Warning::UnsupportedSetting {
                    setting: "toolChoice".into(),
                    details: Some(format!(
                        "toolChoice for server-side tool \"{tool_name}\" is not supported by xAI"
                    )),
                });
                return None;
            }
            Some(json!({ "type": "function", "name": tool_name }))
        }
    }
}

fn build_web_search(p: &ProviderTool) -> Value {
    let args = p.args.as_ref();
    let mut obj = serde_json::Map::new();
    obj.insert("type".into(), json!("web_search"));
    if let Some(v) = args.and_then(|a| a.get("allowedDomains")) {
        obj.insert("allowed_domains".into(), v.clone());
    }
    if let Some(v) = args.and_then(|a| a.get("excludedDomains")) {
        obj.insert("excluded_domains".into(), v.clone());
    }
    if let Some(v) = args.and_then(|a| a.get("enableImageUnderstanding")) {
        obj.insert("enable_image_understanding".into(), v.clone());
    }
    Value::Object(obj)
}

fn build_x_search(p: &ProviderTool) -> Value {
    let args = p.args.as_ref();
    let mut obj = serde_json::Map::new();
    obj.insert("type".into(), json!("x_search"));
    if let Some(v) = args.and_then(|a| a.get("allowedXHandles")) {
        obj.insert("allowed_x_handles".into(), v.clone());
    }
    if let Some(v) = args.and_then(|a| a.get("excludedXHandles")) {
        obj.insert("excluded_x_handles".into(), v.clone());
    }
    if let Some(v) = args.and_then(|a| a.get("fromDate")) {
        obj.insert("from_date".into(), v.clone());
    }
    if let Some(v) = args.and_then(|a| a.get("toDate")) {
        obj.insert("to_date".into(), v.clone());
    }
    if let Some(v) = args.and_then(|a| a.get("enableImageUnderstanding")) {
        obj.insert("enable_image_understanding".into(), v.clone());
    }
    if let Some(v) = args.and_then(|a| a.get("enableVideoUnderstanding")) {
        obj.insert("enable_video_understanding".into(), v.clone());
    }
    Value::Object(obj)
}

fn build_file_search(p: &ProviderTool) -> Value {
    let args = p.args.as_ref();
    let mut obj = serde_json::Map::new();
    obj.insert("type".into(), json!("file_search"));
    if let Some(v) = args.and_then(|a| a.get("vectorStoreIds")) {
        obj.insert("vector_store_ids".into(), v.clone());
    }
    if let Some(v) = args.and_then(|a| a.get("maxNumResults")) {
        obj.insert("max_num_results".into(), v.clone());
    }
    Value::Object(obj)
}

fn build_mcp(p: &ProviderTool) -> Value {
    let args = p.args.as_ref();
    let mut obj = serde_json::Map::new();
    obj.insert("type".into(), json!("mcp"));
    if let Some(v) = args.and_then(|a| a.get("serverUrl")) {
        obj.insert("server_url".into(), v.clone());
    }
    if let Some(v) = args.and_then(|a| a.get("serverLabel")) {
        obj.insert("server_label".into(), v.clone());
    }
    if let Some(v) = args.and_then(|a| a.get("serverDescription")) {
        obj.insert("server_description".into(), v.clone());
    }
    if let Some(v) = args.and_then(|a| a.get("allowedTools")) {
        obj.insert("allowed_tools".into(), v.clone());
    }
    if let Some(v) = args.and_then(|a| a.get("headers")) {
        obj.insert("headers".into(), v.clone());
    }
    if let Some(v) = args.and_then(|a| a.get("authorization")) {
        obj.insert("authorization".into(), v.clone());
    }
    Value::Object(obj)
}

/// Recursively strip `additionalProperties: false` from a JSON schema.
fn remove_additional_properties_false(value: Value) -> Value {
    match value {
        Value::Array(arr) => Value::Array(
            arr.into_iter()
                .map(remove_additional_properties_false)
                .collect(),
        ),
        Value::Object(mut map) => {
            if map.get("additionalProperties") == Some(&Value::Bool(false)) {
                map.remove("additionalProperties");
            }
            let mut new_map = serde_json::Map::new();
            for (k, v) in map {
                new_map.insert(k, remove_additional_properties_false(v));
            }
            Value::Object(new_map)
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmsdk_provider::language_model::FunctionTool;
    use serde_json::json;

    fn fn_tool(name: &str) -> Tool {
        Tool::Function(FunctionTool {
            name: name.into(),
            description: Some("weather".into()),
            input_schema: serde_json::from_value(json!({
                "type": "object",
                "properties": { "c": { "type": "string" } },
                "additionalProperties": false
            }))
            .unwrap(),
            input_examples: None,
            strict: None,
            provider_options: None,
        })
    }

    fn provider(id: &str, name: &str, args: Option<serde_json::Value>) -> Tool {
        Tool::Provider(ProviderTool {
            id: id.into(),
            name: name.into(),
            args: args.and_then(|v| v.as_object().cloned()),
            provider_options: None,
        })
    }

    #[test]
    fn empty_tools_returns_none() {
        let p = prepare(&[], None);
        assert!(p.tools.is_none());
        assert!(p.tool_choice.is_none());
    }

    #[test]
    fn function_tool_strips_additional_properties_false() {
        let p = prepare(&[fn_tool("f")], None);
        let tools = p.tools.unwrap();
        assert_eq!(tools[0]["type"], "function");
        assert!(tools[0]["parameters"].get("additionalProperties").is_none());
    }

    #[test]
    fn web_search_maps_camel_to_snake() {
        let p = prepare(
            &[provider(
                "xai.web_search",
                "web_search",
                Some(json!({
                    "allowedDomains": ["a.com"],
                    "excludedDomains": ["b.com"],
                    "enableImageUnderstanding": true
                })),
            )],
            None,
        );
        let tools = p.tools.unwrap();
        assert_eq!(tools[0]["type"], "web_search");
        assert_eq!(tools[0]["allowed_domains"][0], "a.com");
        assert_eq!(tools[0]["excluded_domains"][0], "b.com");
        assert_eq!(tools[0]["enable_image_understanding"], true);
        assert_eq!(p.names.web_search.as_deref(), Some("web_search"));
    }

    #[test]
    fn x_search_emits_all_fields() {
        let p = prepare(
            &[provider(
                "xai.x_search",
                "x_search",
                Some(json!({
                    "allowedXHandles": ["@a"],
                    "fromDate": "2020-01-01",
                    "enableVideoUnderstanding": true
                })),
            )],
            None,
        );
        let tools = p.tools.unwrap();
        assert_eq!(tools[0]["type"], "x_search");
        assert_eq!(tools[0]["from_date"], "2020-01-01");
        assert_eq!(tools[0]["enable_video_understanding"], true);
        assert_eq!(p.names.x_search.as_deref(), Some("x_search"));
    }

    #[test]
    fn code_execution_emits_code_interpreter_wire_type() {
        let p = prepare(
            &[provider("xai.code_execution", "code_execution", None)],
            None,
        );
        let tools = p.tools.unwrap();
        assert_eq!(tools[0]["type"], "code_interpreter");
        assert_eq!(p.names.code_execution.as_deref(), Some("code_execution"));
    }

    #[test]
    fn file_search_emits_vector_store_ids() {
        let p = prepare(
            &[provider(
                "xai.file_search",
                "fs",
                Some(json!({"vectorStoreIds": ["vs_1"], "maxNumResults": 5})),
            )],
            None,
        );
        let tools = p.tools.unwrap();
        assert_eq!(tools[0]["type"], "file_search");
        assert_eq!(tools[0]["vector_store_ids"][0], "vs_1");
        assert_eq!(tools[0]["max_num_results"], 5);
        assert_eq!(p.names.file_search.as_deref(), Some("fs"));
    }

    #[test]
    fn mcp_emits_server_url() {
        let p = prepare(
            &[provider(
                "xai.mcp",
                "mcp",
                Some(json!({"serverUrl": "https://x", "allowedTools": ["t"]})),
            )],
            None,
        );
        let tools = p.tools.unwrap();
        assert_eq!(tools[0]["type"], "mcp");
        assert_eq!(tools[0]["server_url"], "https://x");
        assert_eq!(tools[0]["allowed_tools"][0], "t");
    }

    #[test]
    fn unknown_provider_tool_warns() {
        let p = prepare(&[provider("xai.unknown", "u", None)], None);
        assert!(p.tools.is_none());
        assert_eq!(p.warnings.len(), 1);
    }

    #[test]
    fn tool_choice_auto_required_none() {
        for (choice, label) in [
            (ToolChoice::Auto, "auto"),
            (ToolChoice::None, "none"),
            (ToolChoice::Required, "required"),
        ] {
            let p = prepare(&[fn_tool("f")], Some(&choice));
            assert_eq!(p.tool_choice.unwrap(), json!(label));
        }
    }

    #[test]
    fn tool_choice_function_emits_function_obj() {
        let p = prepare(
            &[fn_tool("weather")],
            Some(&ToolChoice::Tool {
                tool_name: "weather".into(),
            }),
        );
        let v = p.tool_choice.unwrap();
        assert_eq!(v["type"], "function");
        assert_eq!(v["name"], "weather");
    }

    #[test]
    fn tool_choice_server_tool_warns_and_drops_choice() {
        let p = prepare(
            &[provider("xai.web_search", "web_search", None)],
            Some(&ToolChoice::Tool {
                tool_name: "web_search".into(),
            }),
        );
        assert!(p.tool_choice.is_none());
        assert_eq!(p.warnings.len(), 1);
    }
}
