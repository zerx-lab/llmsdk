//! Convert a Mistral Chat Completions response to a [`GenerateResult`].
//!
//! Mirrors the `doGenerate` post-processing in
//! `mistral-chat-language-model.ts`. Mistral message content can arrive as a
//! plain string or as an array of typed parts (`text`, `thinking`,
//! `image_url`, `reference`). We surface `text` as [`Content::Text`] and
//! `thinking` as [`Content::Reasoning`]; image / reference parts are dropped.
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;

use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    Content, GenerateResponse, GenerateResult, ReasoningPart, ResponseMetadata, TextPart,
    ToolCallPart,
};
use llmsdk_provider::shared::{Headers, RequestInfo, Warning};

use super::finish_reason;
use super::usage;
use super::wire::{ChatResponse, MistralContent, MistralContentPart, MistralThinkingChunk};

/// Parse a successful (non-streaming) chat response.
///
/// # Errors
///
/// Returns [`ProviderError::no_content_generated`] when the response has no
/// choices.
pub(crate) fn parse_response(
    response: ChatResponse,
    headers: HashMap<String, String>,
    request_body: Option<serde_json::Value>,
    warnings: Vec<Warning>,
) -> Result<GenerateResult, ProviderError> {
    let choice = response
        .choices
        .into_iter()
        .next()
        .ok_or_else(ProviderError::no_content_generated)?;

    let mut content: Vec<Content> = Vec::new();

    // Order-preserving extraction: walk the parts array if present, otherwise
    // fall back to the legacy string form.
    if let Some(c) = choice.message.content {
        match c {
            MistralContent::Parts(parts) => {
                for part in parts {
                    match part {
                        MistralContentPart::Text { text } if !text.is_empty() => {
                            content.push(Content::Text(TextPart {
                                text,
                                provider_options: None,
                            }));
                        }
                        MistralContentPart::Thinking { thinking } => {
                            let reasoning = collect_thinking_text(&thinking);
                            if !reasoning.is_empty() {
                                content.push(Content::Reasoning(ReasoningPart {
                                    text: reasoning,
                                    provider_options: None,
                                }));
                            }
                        }
                        // text/thinking handled; image_url / reference dropped
                        MistralContentPart::Text { .. }
                        | MistralContentPart::ImageUrl { .. }
                        | MistralContentPart::Reference { .. } => {}
                    }
                }
            }
            MistralContent::Text(text) => {
                if !text.is_empty() {
                    content.push(Content::Text(TextPart {
                        text,
                        provider_options: None,
                    }));
                }
            }
        }
    }

    if let Some(tool_calls) = choice.message.tool_calls {
        for tc in tool_calls {
            let input = serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
                .unwrap_or(serde_json::Value::String(tc.function.arguments.clone()));
            content.push(Content::ToolCall(ToolCallPart {
                tool_call_id: tc.id,
                tool_name: tc.function.name,
                input,
                provider_executed: None,
                dynamic: None,
                provider_options: None,
            }));
        }
    }

    let usage_value = response
        .usage
        .as_ref()
        .map_or_else(usage::zero, usage::convert);
    let finish = finish_reason::map(choice.finish_reason.as_deref());

    let response_meta = GenerateResponse {
        metadata: ResponseMetadata {
            id: response.id,
            timestamp: response.created.map(|c| c.to_string()),
            model_id: response.model,
            headers: Some(headers_to_provider(headers)),
        },
        body: None,
    };

    Ok(GenerateResult {
        content,
        finish_reason: finish,
        usage: usage_value,
        provider_metadata: None,
        request: request_body.map(|body| RequestInfo { body: Some(body) }),
        response: Some(response_meta),
        warnings,
    })
}

pub(crate) fn collect_thinking_text(chunks: &[MistralThinkingChunk]) -> String {
    let mut s = String::new();
    for chunk in chunks {
        let MistralThinkingChunk::Text { text } = chunk;
        s.push_str(text);
    }
    s
}

fn headers_to_provider(raw: HashMap<String, String>) -> Headers {
    raw.into_iter().map(|(k, v)| (k, Some(v))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::wire::{
        ChatChoice, ChatChoiceMessage, WireFunctionCall, WireToolCall, WireToolCallKind,
    };
    use llmsdk_provider::language_model::FinishReasonKind;

    fn empty_headers() -> HashMap<String, String> {
        HashMap::new()
    }

    #[test]
    fn parses_string_content() {
        let resp = ChatResponse {
            id: Some("r-1".into()),
            created: Some(1),
            model: Some("mistral-small-latest".into()),
            choices: vec![ChatChoice {
                message: ChatChoiceMessage {
                    content: Some(MistralContent::Text("hello".into())),
                    ..Default::default()
                },
                finish_reason: Some("stop".into()),
                _index: Some(0),
            }],
            ..Default::default()
        };
        let result = parse_response(resp, empty_headers(), None, vec![]).expect("ok");
        assert_eq!(result.content.len(), 1);
        assert_eq!(result.finish_reason.unified, FinishReasonKind::Stop);
    }

    #[test]
    fn parses_thinking_and_text_parts_in_order() {
        let resp = ChatResponse {
            choices: vec![ChatChoice {
                message: ChatChoiceMessage {
                    content: Some(MistralContent::Parts(vec![
                        MistralContentPart::Thinking {
                            thinking: vec![MistralThinkingChunk::Text {
                                text: "thinking...".into(),
                            }],
                        },
                        MistralContentPart::Text {
                            text: "answer".into(),
                        },
                    ])),
                    ..Default::default()
                },
                finish_reason: Some("stop".into()),
                ..Default::default()
            }],
            ..Default::default()
        };
        let result = parse_response(resp, empty_headers(), None, vec![]).unwrap();
        assert_eq!(result.content.len(), 2);
        assert!(matches!(result.content[0], Content::Reasoning(_)));
        assert!(matches!(result.content[1], Content::Text(_)));
    }

    #[test]
    fn parses_tool_calls() {
        let resp = ChatResponse {
            choices: vec![ChatChoice {
                message: ChatChoiceMessage {
                    tool_calls: Some(vec![WireToolCall {
                        id: "call_x".into(),
                        kind: Some(WireToolCallKind::Function),
                        function: WireFunctionCall {
                            name: "get_weather".into(),
                            arguments: r#"{"city":"NYC"}"#.into(),
                        },
                    }]),
                    ..Default::default()
                },
                finish_reason: Some("tool_calls".into()),
                ..Default::default()
            }],
            ..Default::default()
        };
        let result = parse_response(resp, empty_headers(), None, vec![]).unwrap();
        assert_eq!(result.content.len(), 1);
        let Content::ToolCall(tc) = &result.content[0] else {
            panic!("expected ToolCall");
        };
        assert_eq!(tc.tool_call_id, "call_x");
        assert_eq!(tc.tool_name, "get_weather");
        assert_eq!(tc.input["city"], "NYC");
        assert_eq!(result.finish_reason.unified, FinishReasonKind::ToolCalls);
    }

    #[test]
    fn empty_choices_yields_no_content_error() {
        let resp = ChatResponse::default();
        let err = parse_response(resp, empty_headers(), None, vec![]).unwrap_err();
        assert!(format!("{err}").contains("no content"));
    }
}
