//! Convert llmsdk tool definitions / choices to Cohere wire format.
//!
//! Mirrors `cohere-prepare-tools.ts`. Cohere accepts only `function` tools;
//! `tool_choice` is `NONE` or `REQUIRED` (case-sensitive upper-case) — `auto`
//! is "absent" and `tool` becomes filtered tools + `REQUIRED`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::language_model::{Tool, ToolChoice};
use llmsdk_provider::shared::Warning;

use super::wire::{WireFunctionDef, WireFunctionKind, WireTool, WireToolChoice};

/// Prepared tool definitions and the resolved `tool_choice`.
pub(crate) struct PreparedTools {
    pub tools: Option<Vec<WireTool>>,
    pub tool_choice: Option<WireToolChoice>,
    pub warnings: Vec<Warning>,
}

/// Convert `tools` + `tool_choice` to Cohere wire format.
pub(crate) fn prepare(tools: &[Tool], tool_choice: Option<&ToolChoice>) -> PreparedTools {
    let mut warnings = Vec::new();

    if tools.is_empty() {
        return PreparedTools {
            tools: None,
            tool_choice: None,
            warnings,
        };
    }

    let mut wire_tools: Vec<WireTool> = Vec::with_capacity(tools.len());
    for tool in tools {
        match tool {
            Tool::Function(f) => wire_tools.push(WireTool {
                kind: WireFunctionKind::Function,
                function: WireFunctionDef {
                    name: f.name.clone(),
                    description: f.description.clone(),
                    parameters: serde_json::to_value(&f.input_schema)
                        .unwrap_or(serde_json::Value::Null),
                },
            }),
            Tool::Provider(p) => {
                warnings.push(Warning::UnsupportedTool {
                    tool: format!("provider-defined tool {}", p.id),
                    details: Some("Cohere chat does not accept provider-defined tools".to_owned()),
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

    let Some(choice) = tool_choice else {
        return PreparedTools {
            tools: Some(wire_tools),
            tool_choice: None,
            warnings,
        };
    };

    match choice {
        ToolChoice::Auto => PreparedTools {
            tools: Some(wire_tools),
            tool_choice: None,
            warnings,
        },
        ToolChoice::None => PreparedTools {
            tools: Some(wire_tools),
            tool_choice: Some(WireToolChoice::None),
            warnings,
        },
        ToolChoice::Required => PreparedTools {
            tools: Some(wire_tools),
            tool_choice: Some(WireToolChoice::Required),
            warnings,
        },
        ToolChoice::Tool { tool_name } => {
            wire_tools.retain(|t| t.function.name == *tool_name);
            PreparedTools {
                tools: Some(wire_tools),
                tool_choice: Some(WireToolChoice::Required),
                warnings,
            }
        }
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
                "properties": {"x": {"type": "string"}}
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
    fn provider_tool_warns_and_drops() {
        let provider = Tool::Provider(ProviderTool {
            id: "cohere.web_search".into(),
            name: "web_search".into(),
            args: None,
            provider_options: None,
        });
        let p = prepare(&[provider], None);
        assert!(p.tools.is_none());
        assert_eq!(p.warnings.len(), 1);
    }

    #[test]
    fn tool_choice_auto_is_absent() {
        let p = prepare(&[fn_tool("f")], Some(&ToolChoice::Auto));
        assert!(p.tool_choice.is_none());
    }

    #[test]
    fn tool_choice_none_maps_upper() {
        let p = prepare(&[fn_tool("f")], Some(&ToolChoice::None));
        let wire = serde_json::to_value(p.tool_choice.unwrap()).unwrap();
        assert_eq!(wire, json!("NONE"));
    }

    #[test]
    fn tool_choice_required_maps_upper() {
        let p = prepare(&[fn_tool("f")], Some(&ToolChoice::Required));
        let wire = serde_json::to_value(p.tool_choice.unwrap()).unwrap();
        assert_eq!(wire, json!("REQUIRED"));
    }

    #[test]
    fn tool_choice_specific_filters_and_requires() {
        let p = prepare(
            &[fn_tool("a"), fn_tool("b"), fn_tool("c")],
            Some(&ToolChoice::Tool {
                tool_name: "b".into(),
            }),
        );
        let tools = p.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "b");
        let wire = serde_json::to_value(p.tool_choice.unwrap()).unwrap();
        assert_eq!(wire, json!("REQUIRED"));
    }
}
