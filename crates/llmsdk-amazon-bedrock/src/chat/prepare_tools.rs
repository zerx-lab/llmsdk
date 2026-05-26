//! Convert llmsdk tool definitions into the Converse `toolConfig` payload.
//!
//! Mirrors `amazon-bedrock-prepare-tools.ts`. Function tools translate
//! directly into `toolSpec`. Provider-defined tools (`anthropic.*`) are
//! supported for Anthropic-on-Bedrock by emitting the same `toolSpec` shape;
//! the `anthropic.web_search_20250305` tool is explicitly filtered (Bedrock
//! does not host it) and surfaces an `UnsupportedTool` warning.
// Rust guideline compliant 2026-05-25

use std::collections::BTreeSet;

use llmsdk_provider::language_model::{Tool, ToolChoice};
use llmsdk_provider::shared::Warning;
use serde_json::{Map, Value};

use super::wire::{
    InputSchema, ToolChoiceTool, ToolChoiceWire, ToolConfig, ToolConfigEntry, ToolSpec,
};

/// Result of [`prepare_tools`].
pub(crate) struct PreparedTools {
    /// Final `toolConfig` (or `None` when no tools survived filtering).
    pub tool_config: Option<ToolConfig>,
    /// Warnings emitted for unsupported / filtered tools.
    pub warnings: Vec<Warning>,
    /// Extra fields merged into `additionalModelRequestFields` — currently
    /// only used to forward Anthropic `tool_choice` on Anthropic-on-Bedrock,
    /// matching ai-sdk's `additionalTools` return value.
    pub additional_tools: Option<Map<String, Value>>,
    /// Beta tokens collected from Anthropic provider-defined tools; the
    /// caller merges these into `additionalModelRequestFields.anthropic_beta`.
    pub betas: BTreeSet<String>,
}

/// Convert tool definitions + tool-choice into Bedrock's `toolConfig`.
pub(crate) fn prepare_tools(
    tools: Option<&[Tool]>,
    tool_choice: Option<&ToolChoice>,
    model_id: &str,
) -> PreparedTools {
    let mut warnings: Vec<Warning> = Vec::new();
    let mut betas: BTreeSet<String> = BTreeSet::new();
    let Some(tools) = tools else {
        return PreparedTools {
            tool_config: None,
            warnings,
            additional_tools: None,
            betas,
        };
    };
    if tools.is_empty() {
        return PreparedTools {
            tool_config: None,
            warnings,
            additional_tools: None,
            betas,
        };
    }

    let is_anthropic_model = model_id.contains("anthropic.");
    let mut specs: Vec<ToolConfigEntry> = Vec::with_capacity(tools.len());
    let mut has_anthropic_provider_tool = false;

    for tool in tools {
        match tool {
            Tool::Function(f) => {
                let strict = f.strict;
                let description = f
                    .description
                    .as_ref()
                    .filter(|d| !d.trim().is_empty())
                    .cloned();
                specs.push(ToolConfigEntry::Spec {
                    tool_spec: ToolSpec {
                        name: f.name.clone(),
                        description,
                        strict,
                        input_schema: InputSchema {
                            json: f.input_schema.clone().into(),
                        },
                    },
                });
            }
            Tool::Provider(p) => {
                // Web search is not available on Bedrock — filter it out and
                // warn, matching the upstream behavior.
                if p.id == "anthropic.web_search_20250305" {
                    warnings.push(Warning::UnsupportedTool {
                        tool: p.name.clone(),
                        details: Some(
                            "web_search_20250305 is not supported on Amazon Bedrock".to_owned(),
                        ),
                    });
                    continue;
                }
                if !is_anthropic_model {
                    warnings.push(Warning::UnsupportedTool {
                        tool: p.name.clone(),
                        details: Some(format!(
                            "provider-defined tool '{}' is only supported on Anthropic models on Bedrock",
                            p.id
                        )),
                    });
                    continue;
                }
                has_anthropic_provider_tool = true;
                for token in anthropic_provider_tool_betas(&p.id) {
                    betas.insert((*token).to_owned());
                }
                let schema = p
                    .args
                    .clone()
                    .map_or_else(|| Value::Object(Map::new()), Value::Object);
                specs.push(ToolConfigEntry::Spec {
                    tool_spec: ToolSpec {
                        name: p.name.clone(),
                        description: None,
                        strict: None,
                        input_schema: InputSchema { json: schema },
                    },
                });
            }
        }
    }

    if specs.is_empty() {
        return PreparedTools {
            tool_config: None,
            warnings,
            additional_tools: None,
            betas,
        };
    }

    // For Anthropic-on-Bedrock when at least one anthropic.* provider tool is
    // present, the tool_choice MUST be sent via additionalModelRequestFields
    // (Anthropic wire shape) rather than toolConfig.toolChoice (Bedrock shape).
    // Mirrors ai-sdk's amazon-bedrock-prepare-tools.ts behavior.
    let uses_anthropic_routing = has_anthropic_provider_tool && is_anthropic_model;
    let mut additional_tools: Option<Map<String, Value>> = None;
    let mut tool_choice_wire: Option<ToolChoiceWire> = None;

    if uses_anthropic_routing {
        if let Some(choice) = tool_choice {
            let anthropic_choice = match choice {
                ToolChoice::Auto => Some(serde_json::json!({ "type": "auto" })),
                ToolChoice::Required => Some(serde_json::json!({ "type": "any" })),
                ToolChoice::None => None,
                ToolChoice::Tool { tool_name } => Some(serde_json::json!({
                    "type": "tool",
                    "name": tool_name,
                })),
            };
            if let Some(value) = anthropic_choice {
                let mut map = Map::new();
                map.insert("tool_choice".to_owned(), value);
                additional_tools = Some(map);
            }
        }
    } else {
        tool_choice_wire = tool_choice.and_then(|choice| match choice {
            ToolChoice::Auto => Some(ToolChoiceWire::Auto { auto: Map::new() }),
            ToolChoice::Required => Some(ToolChoiceWire::Any { any: Map::new() }),
            ToolChoice::None => None,
            ToolChoice::Tool { tool_name } => Some(ToolChoiceWire::Tool {
                tool: ToolChoiceTool {
                    name: tool_name.clone(),
                },
            }),
        });
    }

    let tool_config = ToolConfig {
        tools: Some(specs),
        tool_choice: tool_choice_wire,
    };

    PreparedTools {
        tool_config: Some(tool_config),
        warnings,
        additional_tools,
        betas,
    }
}

/// Map an `anthropic.*` provider-tool id to the beta header tokens it
/// requires. Mirrors the `betas.add(...)` table in ai-sdk's
/// `anthropic-prepare-tools.ts`.
fn anthropic_provider_tool_betas(id: &str) -> &'static [&'static str] {
    match id {
        "anthropic.code_execution_20250522" => &["code-execution-2025-05-22"],
        "anthropic.code_execution_20250825" => &["code-execution-2025-08-25"],
        "anthropic.computer_20241022"
        | "anthropic.text_editor_20241022"
        | "anthropic.bash_20241022" => &["computer-use-2024-10-22"],
        "anthropic.computer_20250124"
        | "anthropic.text_editor_20250124"
        | "anthropic.text_editor_20250429"
        | "anthropic.bash_20250124" => &["computer-use-2025-01-24"],
        "anthropic.computer_20251124" => &["computer-use-2025-11-24"],
        "anthropic.memory_20250818" => &["context-management-2025-06-27"],
        "anthropic.web_fetch_20250910" => &["web-fetch-2025-09-10"],
        "anthropic.web_fetch_20260209" | "anthropic.web_search_20260209" => {
            &["code-execution-web-tools-2026-02-09"]
        }
        "anthropic.advisor_20260301" => &["advisor-tool-2026-03-01"],
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmsdk_provider::json::JsonSchema;
    use llmsdk_provider::language_model::{FunctionTool, ProviderTool};
    use serde_json::json;

    fn func(name: &str) -> Tool {
        let schema: JsonSchema = serde_json::from_value(json!({ "type": "object" })).unwrap();
        Tool::Function(FunctionTool {
            name: name.into(),
            description: Some("desc".into()),
            input_schema: schema,
            input_examples: None,
            strict: None,
            provider_options: None,
        })
    }

    #[test]
    fn empty_tools_yields_no_config() {
        let out = prepare_tools(Some(&[]), None, "anthropic.claude-3-haiku");
        assert!(out.tool_config.is_none());
        assert!(out.warnings.is_empty());
    }

    #[test]
    fn function_tool_emits_tool_spec() {
        let tools = vec![func("weather")];
        let out = prepare_tools(Some(&tools), None, "anthropic.claude-3-haiku");
        let cfg = out.tool_config.unwrap();
        assert_eq!(cfg.tools.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn web_search_provider_tool_is_filtered_with_warning() {
        let tools = vec![Tool::Provider(ProviderTool {
            id: "anthropic.web_search_20250305".into(),
            name: "web_search".into(),
            args: None,
            provider_options: None,
        })];
        let out = prepare_tools(Some(&tools), None, "anthropic.claude-3-haiku");
        assert!(out.tool_config.is_none());
        assert_eq!(out.warnings.len(), 1);
        assert!(matches!(out.warnings[0], Warning::UnsupportedTool { .. }));
    }

    #[test]
    fn tool_choice_auto_serializes_correctly() {
        let tools = vec![func("weather")];
        let out = prepare_tools(
            Some(&tools),
            Some(&ToolChoice::Auto),
            "anthropic.claude-3-haiku",
        );
        let cfg = out.tool_config.unwrap();
        let wire = serde_json::to_value(cfg.tool_choice).unwrap();
        assert!(wire["auto"].is_object());
    }

    #[test]
    fn tool_choice_required_becomes_any() {
        let tools = vec![func("weather")];
        let out = prepare_tools(
            Some(&tools),
            Some(&ToolChoice::Required),
            "anthropic.claude-3-haiku",
        );
        let cfg = out.tool_config.unwrap();
        let wire = serde_json::to_value(cfg.tool_choice).unwrap();
        assert!(wire["any"].is_object());
    }
}
