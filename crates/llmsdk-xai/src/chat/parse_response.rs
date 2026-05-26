//! Convert an xAI Chat Completions response to a [`GenerateResult`].
//!
//! Mirrors the `doGenerate` post-processing in `xai-chat-language-model.ts`.
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;

use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    Content, GenerateResponse, GenerateResult, ReasoningPart, ResponseMetadata, Source, TextPart,
    ToolCallPart,
};
use llmsdk_provider::shared::{Headers, RequestInfo, Warning};
use llmsdk_provider_utils::time::rfc3339_from_unix_seconds;

use super::finish_reason;
use super::usage;
use super::wire::ChatResponse;

/// Parse a successful (non-streaming) chat response.
///
/// # Errors
///
/// Returns [`ProviderError::api_call`] when xAI returns the 200-error envelope
/// (`{code, error}`), and [`ProviderError::no_content_generated`] when the
/// response has no choices.
pub(crate) fn parse_response(
    response: ChatResponse,
    headers: HashMap<String, String>,
    request_body: Option<serde_json::Value>,
    warnings: Vec<Warning>,
    endpoint_url: &str,
    last_assistant_content: Option<&str>,
    citation_id_seed: &mut u64,
) -> Result<GenerateResult, ProviderError> {
    // xAI sometimes returns `{code, error}` with HTTP 200 instead of a 4xx/5xx.
    if let Some(error_message) = response.error {
        let mut builder =
            ProviderError::api_call_builder(endpoint_url, error_message).status_code(200);
        if response.code.as_deref() == Some("The service is currently unavailable") {
            builder = builder.retryable(true);
        }
        return Err(builder.build());
    }

    let choice = response
        .choices
        .into_iter()
        .next()
        .ok_or_else(ProviderError::no_content_generated)?;

    let mut content = Vec::new();

    // text content (drop the leading-prefix duplicate when prefix continuation is
    // used — matches ai-sdk's `lastMessage.role === 'assistant' && text ===
    // lastMessage.content` shortcut).
    if let Some(text) = choice.message.content
        && !text.is_empty()
        && Some(text.as_str()) != last_assistant_content
    {
        content.push(Content::Text(TextPart {
            text,
            provider_options: None,
        }));
    }

    if let Some(reasoning) = choice.message.reasoning_content
        && !reasoning.is_empty()
    {
        content.push(Content::Reasoning(ReasoningPart {
            text: reasoning,
            provider_options: None,
        }));
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

    // xAI Live Search citations come on the response body, not per-choice.
    if let Some(citations) = response.citations {
        for url in citations {
            content.push(Content::Source(Source::Url {
                id: next_id(citation_id_seed),
                url,
                title: None,
                provider_metadata: None,
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
            timestamp: response.created.map(rfc3339_from_unix_seconds),
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

/// Generate a stable monotonic citation id.
pub(crate) fn next_id(seed: &mut u64) -> String {
    *seed = seed.wrapping_add(1);
    format!("xai-citation-{seed}")
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
    fn parses_plain_text() {
        let resp = ChatResponse {
            id: Some("r-1".into()),
            created: Some(1),
            model: Some("grok-4.3".into()),
            choices: vec![ChatChoice {
                message: ChatChoiceMessage {
                    content: Some("hello".into()),
                    ..Default::default()
                },
                finish_reason: Some("stop".into()),
                _index: Some(0),
            }],
            ..Default::default()
        };
        let mut seed = 0;
        let result = parse_response(
            resp,
            empty_headers(),
            None,
            vec![],
            "https://api.x.ai/v1/chat/completions",
            None,
            &mut seed,
        )
        .unwrap();
        assert_eq!(result.content.len(), 1);
        assert_eq!(result.finish_reason.unified, FinishReasonKind::Stop);
    }

    #[test]
    fn parses_reasoning_content() {
        let resp = ChatResponse {
            choices: vec![ChatChoice {
                message: ChatChoiceMessage {
                    content: Some("answer".into()),
                    reasoning_content: Some("thinking...".into()),
                    ..Default::default()
                },
                finish_reason: Some("stop".into()),
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut seed = 0;
        let result =
            parse_response(resp, empty_headers(), None, vec![], "", None, &mut seed).unwrap();
        assert_eq!(result.content.len(), 2);
        assert!(matches!(result.content[0], Content::Text(_)));
        assert!(matches!(result.content[1], Content::Reasoning(_)));
    }

    #[test]
    fn parses_tool_calls() {
        let resp = ChatResponse {
            choices: vec![ChatChoice {
                message: ChatChoiceMessage {
                    tool_calls: Some(vec![WireToolCall {
                        id: "call_x".into(),
                        kind: WireToolCallKind::Function,
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
        let mut seed = 0;
        let result =
            parse_response(resp, empty_headers(), None, vec![], "", None, &mut seed).unwrap();
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
    fn parses_citations_as_sources() {
        let resp = ChatResponse {
            choices: vec![ChatChoice {
                message: ChatChoiceMessage {
                    content: Some("answer".into()),
                    ..Default::default()
                },
                finish_reason: Some("stop".into()),
                ..Default::default()
            }],
            citations: Some(vec![
                "https://example.com/a".into(),
                "https://example.com/b".into(),
            ]),
            ..Default::default()
        };
        let mut seed = 0;
        let result =
            parse_response(resp, empty_headers(), None, vec![], "", None, &mut seed).unwrap();
        assert_eq!(result.content.len(), 3);
        let Content::Source(Source::Url { url, id, .. }) = &result.content[1] else {
            panic!("expected Source::Url");
        };
        assert_eq!(url, "https://example.com/a");
        assert_eq!(id, "xai-citation-1");
    }

    #[test]
    fn error_envelope_translates_to_api_call_error() {
        let resp = ChatResponse {
            error: Some("rate limited".into()),
            code: Some("rate_limit_exceeded".into()),
            ..Default::default()
        };
        let mut seed = 0;
        let err = parse_response(resp, empty_headers(), None, vec![], "url", None, &mut seed)
            .unwrap_err();
        assert!(format!("{err}").contains("rate limited"));
        assert_eq!(err.status_code(), Some(200));
    }

    #[test]
    fn empty_choices_yields_no_content_error() {
        let resp = ChatResponse::default();
        let mut seed = 0;
        let err =
            parse_response(resp, empty_headers(), None, vec![], "", None, &mut seed).unwrap_err();
        assert!(format!("{err}").contains("no content"));
    }

    #[test]
    fn drops_prefix_duplicate_text() {
        let resp = ChatResponse {
            choices: vec![ChatChoice {
                message: ChatChoiceMessage {
                    content: Some("prefix".into()),
                    ..Default::default()
                },
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut seed = 0;
        let result = parse_response(
            resp,
            empty_headers(),
            None,
            vec![],
            "",
            Some("prefix"),
            &mut seed,
        )
        .unwrap();
        assert_eq!(result.content.len(), 0);
    }
}
