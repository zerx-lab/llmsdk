//! Convert llmsdk tool definitions / choices to xAI wire format.
//!
//! Mirrors `xai-prepare-tools.ts` plus `remove-additional-properties.ts`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::language_model::{Tool, ToolChoice};
use llmsdk_provider::shared::Warning;
use serde_json::Value;

use super::wire::{
    WireFunctionDef, WireFunctionKind, WireTool, WireToolCallKind, WireToolChoice,
    WireToolChoiceFunction, WireToolChoiceSimple,
};

/// Prepared tool definitions and the resolved `tool_choice`.
pub(crate) struct PreparedTools {
    pub tools: Option<Vec<WireTool>>,
    pub tool_choice: Option<WireToolChoice>,
    pub warnings: Vec<Warning>,
}

/// Convert `tools` + `tool_choice` from a [`CallOptions`](llmsdk_provider::language_model::CallOptions)
/// to xAI wire format.
pub(crate) fn prepare(tools: &[Tool], tool_choice: Option<&ToolChoice>) -> PreparedTools {
    let mut warnings = Vec::new();
    let trimmed: Vec<&Tool> = tools.iter().collect();

    if trimmed.is_empty() {
        return PreparedTools {
            tools: None,
            tool_choice: None,
            warnings,
        };
    }

    let mut wire_tools = Vec::with_capacity(trimmed.len());
    for tool in trimmed {
        match tool {
            Tool::Function(f) => wire_tools.push(WireTool {
                kind: WireFunctionKind::Function,
                function: WireFunctionDef {
                    name: f.name.clone(),
                    description: f.description.clone(),
                    parameters: remove_additional_properties_false(
                        serde_json::to_value(&f.input_schema).unwrap_or(Value::Null),
                    ),
                    strict: f.strict,
                },
            }),
            Tool::Provider(p) => {
                warnings.push(Warning::Unsupported {
                    feature: format!("provider-defined feature {}", p.name),
                    details: Some(
                        "xAI chat completions does not accept provider-defined tools; \
                        use chat.tools.* on the responses endpoint instead"
                            .to_owned(),
                    ),
                });
            }
        }
    }

    if wire_tools.is_empty() {
        return PreparedTools {
            tools: None,
            tool_choice: None,
            warnings,
        };
    }

    let wire_choice = tool_choice.map(|c| match c {
        ToolChoice::Auto => WireToolChoice::Simple(WireToolChoiceSimple::Auto),
        ToolChoice::None => WireToolChoice::Simple(WireToolChoiceSimple::None),
        ToolChoice::Required => WireToolChoice::Simple(WireToolChoiceSimple::Required),
        ToolChoice::Tool { tool_name } => WireToolChoice::Tool {
            kind: WireToolCallKind::Function,
            function: WireToolChoiceFunction {
                name: tool_name.clone(),
            },
        },
    });

    PreparedTools {
        tools: Some(wire_tools),
        tool_choice: wire_choice,
        warnings,
    }
}

/// Recursively strip `additionalProperties: false` from a JSON schema.
///
/// xAI rejects schemas that explicitly forbid additional properties; the
/// safe transform is to drop the offending key entirely (xAI ignores all
/// additional properties anyway).
fn remove_additional_properties_false(value: Value) -> Value {
    match value {
        Value::Array(arr) => Value::Array(
            arr.into_iter()
                .map(remove_additional_properties_false)
                .collect(),
        ),
        Value::Object(mut map) => {
            if let Some(v) = map.get("additionalProperties")
                && v == &Value::Bool(false)
            {
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
    use llmsdk_provider::language_model::{FunctionTool, ProviderTool};
    use serde_json::json;

    fn fn_tool(name: &str) -> Tool {
        Tool::Function(FunctionTool {
            name: name.into(),
            description: None,
            input_schema: serde_json::from_value(json!({
                "type": "object",
                "properties": { "x": { "type": "string" } },
                "additionalProperties": false
            }))
            .unwrap(),
            input_examples: None,
            strict: None,
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
        assert_eq!(tools.len(), 1);
        let params = &tools[0].function.parameters;
        assert!(params.get("additionalProperties").is_none());
    }

    #[test]
    fn provider_tool_warns_and_drops() {
        let provider = Tool::Provider(ProviderTool {
            id: "xai.web_search".into(),
            name: "web_search".into(),
            args: None,
            provider_options: None,
        });
        let p = prepare(&[provider], None);
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
            let wire = serde_json::to_value(p.tool_choice.unwrap()).unwrap();
            assert_eq!(wire, json!(label));
        }
    }

    #[test]
    fn tool_choice_specific_tool() {
        let p = prepare(
            &[fn_tool("weather")],
            Some(&ToolChoice::Tool {
                tool_name: "weather".into(),
            }),
        );
        let wire = serde_json::to_value(p.tool_choice.unwrap()).unwrap();
        assert_eq!(wire["type"], "function");
        assert_eq!(wire["function"]["name"], "weather");
    }

    #[test]
    fn remove_additional_properties_recurses_into_nested() {
        let v = json!({
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": { "x": { "type": "string" } }
                }
            },
            "additionalProperties": false
        });
        let cleaned = remove_additional_properties_false(v);
        assert!(cleaned.get("additionalProperties").is_none());
        assert!(
            cleaned["properties"]["nested"]
                .get("additionalProperties")
                .is_none()
        );
    }

    #[test]
    fn remove_additional_properties_keeps_true_value() {
        let v = json!({"additionalProperties": true});
        let cleaned = remove_additional_properties_false(v);
        assert_eq!(cleaned["additionalProperties"], json!(true));
    }
}
