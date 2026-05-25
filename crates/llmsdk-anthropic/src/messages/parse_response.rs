//! Convert an `Anthropic` Messages response into a [`GenerateResult`].
//!
//! Mirrors the post-processing in `anthropic-language-model.ts`'s
//! `doGenerate`. M6 surfaces text and `tool_use` content; everything else
//! is dropped with no warning (deliberately silent — server tools are
//! out-of-scope rather than misconfigurations).
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;

use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    Content, GenerateResponse, GenerateResult, ReasoningPart, ResponseMetadata, TextPart,
    ToolCallPart, ToolResult, ToolResultOutput,
};
use llmsdk_provider::shared::{Headers, ProviderOptions, Warning};
use serde_json::{Map, Value as JsonValue};

use super::wire::{MessagesResponse, ResponseContent};
use super::{finish_reason, usage};

/// Parse a successful Messages response.
#[allow(
    clippy::too_many_lines,
    reason = "post-processing handles ~10 ResponseContent variants in one place"
)]
pub(crate) fn parse_response(
    response: MessagesResponse,
    headers: HashMap<String, String>,
    request_body: Option<serde_json::Value>,
    warnings: Vec<Warning>,
) -> Result<GenerateResult, ProviderError> {
    let mut content = Vec::new();
    for part in response.content {
        match part {
            ResponseContent::Text { text, citations } if !text.is_empty() => {
                let provider_options = citations.map(|c| {
                    let mut m = Map::new();
                    m.insert("citations".into(), c);
                    let mut po = ProviderOptions::new();
                    po.insert("anthropic".into(), m);
                    po
                });
                content.push(Content::Text(TextPart {
                    text,
                    provider_options,
                }));
            }
            ResponseContent::ToolUse {
                id,
                name,
                input,
                caller,
                dynamic,
            } => {
                let provider_options = if caller.is_some() || dynamic.is_some() {
                    let mut m = Map::new();
                    if let Some(c) = caller {
                        m.insert("caller".into(), c);
                    }
                    if let Some(d) = dynamic {
                        m.insert("dynamic".into(), JsonValue::Bool(d));
                    }
                    let mut po = ProviderOptions::new();
                    po.insert("anthropic".into(), m);
                    Some(po)
                } else {
                    None
                };
                content.push(Content::ToolCall(ToolCallPart {
                    tool_call_id: id,
                    tool_name: name,
                    input,
                    provider_executed: None,
                    dynamic: None,
                    provider_options,
                }));
            }
            ResponseContent::Compaction(v) => {
                // Compaction notices are informational — surface via
                // provider_metadata-style marker but no first-class block.
                let mut anthropic = Map::new();
                anthropic.insert("type".into(), JsonValue::String("compaction".into()));
                anthropic.insert("compaction".into(), v);
                let mut po = ProviderOptions::new();
                po.insert("anthropic".into(), anthropic);
                content.push(Content::Custom {
                    kind: "anthropic.compaction".into(),
                    provider_options: Some(po),
                });
            }
            ResponseContent::Thinking {
                thinking,
                signature,
            } => {
                content.push(Content::Reasoning(ReasoningPart {
                    text: thinking,
                    provider_options: thinking_provider_options(signature.as_deref(), None),
                }));
            }
            ResponseContent::RedactedThinking { data } => {
                // Redacted: empty text, opaque `redactedData` on metadata.
                content.push(Content::Reasoning(ReasoningPart {
                    text: String::new(),
                    provider_options: thinking_provider_options(None, Some(&data)),
                }));
            }
            ResponseContent::ServerToolUse { id, name, input } => {
                content.push(Content::ToolCall(ToolCallPart {
                    tool_call_id: id,
                    tool_name: name,
                    input,
                    provider_executed: Some(true),
                    dynamic: None,
                    provider_options: None,
                }));
            }
            ResponseContent::WebSearchToolResult(v)
            | ResponseContent::WebFetchToolResult(v)
            | ResponseContent::CodeExecutionToolResult(v)
            | ResponseContent::BashCodeExecutionToolResult(v)
            | ResponseContent::TextEditorCodeExecutionToolResult(v)
            | ResponseContent::McpToolUse(v)
            | ResponseContent::McpToolResult(v)
            | ResponseContent::ToolSearchToolResult(v)
            | ResponseContent::AdvisorToolResult(v) => {
                let (tool_call_id, tool_name) = extract_tool_call_id_and_name(&v);
                content.push(Content::ToolResult(ToolResult {
                    tool_call_id,
                    tool_name,
                    output: ToolResultOutput::Json {
                        value: v,
                        provider_options: None,
                    },
                    provider_metadata: None,
                }));
            }
            // Drop: empty text, anything we don't surface.
            ResponseContent::Text { .. } | ResponseContent::Other => {}
        }
    }

    if content.is_empty() {
        return Err(ProviderError::no_content_generated());
    }

    let finish = finish_reason::map(response.stop_reason.as_deref());
    let usage = usage::convert(&response.usage);

    let response_meta = GenerateResponse {
        metadata: ResponseMetadata {
            id: response.id,
            timestamp: None,
            model_id: response.model,
            headers: Some(headers_to_provider(headers)),
        },
        body: None,
    };

    Ok(GenerateResult {
        content,
        finish_reason: finish,
        usage,
        provider_metadata: None,
        request: request_body.map(|body| llmsdk_provider::shared::RequestInfo { body: Some(body) }),
        response: Some(response_meta),
        warnings,
    })
}

fn headers_to_provider(raw: HashMap<String, String>) -> Headers {
    raw.into_iter().map(|(k, v)| (k, Some(v))).collect()
}

/// Pluck `tool_use_id` and `name` out of a server-tool result payload,
/// falling back to empty strings when the upstream did not surface them.
fn extract_tool_call_id_and_name(v: &JsonValue) -> (String, String) {
    let tool_call_id = v
        .get("tool_use_id")
        .and_then(|x| x.as_str())
        .unwrap_or_default()
        .to_owned();
    let tool_name = v
        .get("name")
        .and_then(|x| x.as_str())
        .unwrap_or_default()
        .to_owned();
    (tool_call_id, tool_name)
}

/// Build a `provider_options` map for a [`ReasoningPart`] that carries the
/// `signature` (visible thinking) and/or `redactedData` (server-redacted).
fn thinking_provider_options(
    signature: Option<&str>,
    redacted_data: Option<&str>,
) -> Option<ProviderOptions> {
    if signature.is_none() && redacted_data.is_none() {
        return None;
    }
    let mut anthropic = Map::new();
    if let Some(sig) = signature {
        anthropic.insert("signature".to_owned(), JsonValue::String(sig.to_owned()));
    }
    if let Some(data) = redacted_data {
        anthropic.insert(
            "redactedData".to_owned(),
            JsonValue::String(data.to_owned()),
        );
    }
    let mut po = ProviderOptions::new();
    po.insert("anthropic".to_owned(), anthropic);
    Some(po)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::wire::ResponseUsage;
    use llmsdk_provider::language_model::FinishReasonKind;

    fn empty_headers() -> HashMap<String, String> {
        HashMap::new()
    }

    #[test]
    fn parses_plain_text() {
        let resp = MessagesResponse {
            id: Some("msg_1".into()),
            model: Some("claude-3-5-sonnet".into()),
            content: vec![ResponseContent::Text {
                text: "hello".into(),
                citations: None,
            }],
            stop_reason: Some("end_turn".into()),
            usage: ResponseUsage {
                input_tokens: 3,
                output_tokens: 2,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };
        let r = parse_response(resp, empty_headers(), None, vec![]).unwrap();
        assert_eq!(r.content.len(), 1);
        assert_eq!(r.finish_reason.unified, FinishReasonKind::Stop);
        assert_eq!(r.usage.input_tokens.total, Some(3));
    }

    #[test]
    fn parses_tool_use() {
        let resp = MessagesResponse {
            id: None,
            model: None,
            content: vec![ResponseContent::ToolUse {
                id: "tu_1".into(),
                name: "get_weather".into(),
                input: serde_json::json!({"city": "NYC"}),
                caller: None,
                dynamic: None,
            }],
            stop_reason: Some("tool_use".into()),
            usage: ResponseUsage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };
        let r = parse_response(resp, empty_headers(), None, vec![]).unwrap();
        if let Content::ToolCall(tc) = &r.content[0] {
            assert_eq!(tc.tool_call_id, "tu_1");
            assert_eq!(tc.input["city"], "NYC");
        } else {
            panic!("expected tool call");
        }
        assert_eq!(r.finish_reason.unified, FinishReasonKind::ToolCalls);
    }

    #[test]
    fn empty_content_yields_no_content_error() {
        let resp = MessagesResponse {
            id: None,
            model: None,
            content: vec![],
            stop_reason: None,
            usage: ResponseUsage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };
        let err = parse_response(resp, empty_headers(), None, vec![]).unwrap_err();
        assert!(format!("{err}").contains("no content"));
    }
}
