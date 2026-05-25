//! Convert llmsdk tool definitions / choices to Mistral wire format.
//!
//! Mirrors `mistral-prepare-tools.ts`. Mistral does not accept provider-
//! defined tools and spells `"required"` as `"any"`. The `tool` choice is
//! implemented by filtering the tools list and forcing `tool_choice = any`
//! (Mistral has no first-class single-tool selector).
// Rust guideline compliant 2026-05-25

use llmsdk_provider::language_model::{Tool, ToolChoice};
use llmsdk_provider::shared::Warning;

use super::wire::{
    WireFunctionDef, WireFunctionKind, WireTool, WireToolChoice, WireToolChoiceSimple,
};

/// Prepared tool definitions and the resolved `tool_choice`.
pub(crate) struct PreparedTools {
    pub tools: Option<Vec<WireTool>>,
    pub tool_choice: Option<WireToolChoice>,
    pub warnings: Vec<Warning>,
}

/// Convert `tools` + `tool_choice` from a [`CallOptions`](llmsdk_provider::language_model::CallOptions)
/// to Mistral wire format.
pub(crate) fn prepare(tools: &[Tool], tool_choice: Option<&ToolChoice>) -> PreparedTools {
    let mut warnings = Vec::new();

    if tools.is_empty() {
        return PreparedTools {
            tools: None,
            tool_choice: None,
            warnings,
        };
    }

    let mut wire_tools = Vec::with_capacity(tools.len());
    for tool in tools {
        match tool {
            Tool::Function(f) => wire_tools.push(WireTool {
                kind: WireFunctionKind::Function,
                function: WireFunctionDef {
                    name: f.name.clone(),
                    description: f.description.clone(),
                    parameters: serde_json::to_value(&f.input_schema)
                        .unwrap_or(serde_json::Value::Null),
                    strict: f.strict,
                },
            }),
            Tool::Provider(p) => {
                warnings.push(Warning::UnsupportedTool {
                    tool: format!("provider-defined tool {}", p.id),
                    details: Some(
                        "Mistral chat completions does not accept provider-defined tools"
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

    let (tools_out, choice_out) = match tool_choice {
        None => (Some(wire_tools), None),
        Some(ToolChoice::Auto) => (
            Some(wire_tools),
            Some(WireToolChoice::Simple(WireToolChoiceSimple::Auto)),
        ),
        Some(ToolChoice::None) => (
            Some(wire_tools),
            Some(WireToolChoice::Simple(WireToolChoiceSimple::None)),
        ),
        Some(ToolChoice::Required) => (
            Some(wire_tools),
            Some(WireToolChoice::Simple(WireToolChoiceSimple::Any)),
        ),
        Some(ToolChoice::Tool { tool_name }) => {
            // Mistral has no first-class "this tool" selector: filter the tool
            // list down to the chosen one and force `any` so the model is
            // required to call it.
            let filtered: Vec<WireTool> = wire_tools
                .into_iter()
                .filter(|t| t.function.name == *tool_name)
                .collect();
            (
                Some(filtered),
                Some(WireToolChoice::Simple(WireToolChoiceSimple::Any)),
            )
        }
    };

    PreparedTools {
        tools: tools_out,
        tool_choice: choice_out,
        warnings,
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
            description: Some("desc".into()),
            input_schema: serde_json::from_value(json!({
                "type": "object",
                "properties": { "x": { "type": "string" } }
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
    fn function_tool_pass_through() {
        let p = prepare(&[fn_tool("f")], None);
        assert_eq!(p.tools.unwrap().len(), 1);
        assert!(p.tool_choice.is_none());
        assert!(p.warnings.is_empty());
    }

    #[test]
    fn provider_tool_warns_and_drops() {
        let provider = Tool::Provider(ProviderTool {
            id: "mistral.code_execution".into(),
            name: "code_execution".into(),
            args: None,
            provider_options: None,
        });
        let p = prepare(&[provider], None);
        assert!(p.tools.is_none());
        assert_eq!(p.warnings.len(), 1);
    }

    #[test]
    fn required_maps_to_any() {
        let p = prepare(&[fn_tool("f")], Some(&ToolChoice::Required));
        let wire = serde_json::to_value(p.tool_choice.unwrap()).unwrap();
        assert_eq!(wire, json!("any"));
    }

    #[test]
    fn auto_and_none_pass_through() {
        for (choice, label) in [(ToolChoice::Auto, "auto"), (ToolChoice::None, "none")] {
            let p = prepare(&[fn_tool("f")], Some(&choice));
            let wire = serde_json::to_value(p.tool_choice.unwrap()).unwrap();
            assert_eq!(wire, json!(label));
        }
    }

    #[test]
    fn specific_tool_filters_and_forces_any() {
        let p = prepare(
            &[fn_tool("weather"), fn_tool("clock")],
            Some(&ToolChoice::Tool {
                tool_name: "clock".into(),
            }),
        );
        let tools = p.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "clock");
        let wire = serde_json::to_value(p.tool_choice.unwrap()).unwrap();
        assert_eq!(wire, json!("any"));
    }
}
