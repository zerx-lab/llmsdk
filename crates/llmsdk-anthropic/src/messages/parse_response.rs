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
    Content, GenerateResponse, GenerateResult, ResponseMetadata, TextPart, ToolCallPart,
};
use llmsdk_provider::shared::{Headers, Warning};

use super::wire::{MessagesResponse, ResponseContent};
use super::{finish_reason, usage};

/// Parse a successful Messages response.
pub(crate) fn parse_response(
    response: MessagesResponse,
    headers: HashMap<String, String>,
    request_body: Option<serde_json::Value>,
    warnings: Vec<Warning>,
) -> Result<GenerateResult, ProviderError> {
    let mut content = Vec::new();
    for part in response.content {
        match part {
            ResponseContent::Text { text } if !text.is_empty() => {
                content.push(Content::Text(TextPart {
                    text,
                    provider_options: None,
                }));
            }
            ResponseContent::ToolUse { id, name, input } => {
                content.push(Content::ToolCall(ToolCallPart {
                    tool_call_id: id,
                    tool_name: name,
                    input,
                    provider_executed: None,
                    provider_options: None,
                }));
            }
            // Drop: empty text, server-tool variants, anything we don't surface.
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
