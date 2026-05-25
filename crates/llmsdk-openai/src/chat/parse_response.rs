//! Convert an `OpenAI` Chat Completions response to a [`GenerateResult`].
//!
//! Mirrors the `doGenerate` post-processing in `openai-chat-language-model.ts`.
//! M3 covers text + `tool_calls`; annotations and provider-defined tools are
//! deferred.
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;

use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    Content, GenerateResponse, GenerateResult, ResponseMetadata, Source, ToolCallPart,
};
use llmsdk_provider::shared::{Headers, ProviderMetadata, Warning};
use serde_json::Map;

use super::wire::{Annotation, ChatResponse};
use super::{finish_reason, usage};

/// Parse a successful chat response.
///
/// `headers` is the raw response headers (already lower-cased keys from
/// `provider-utils`). `warnings` carries forward any warnings produced
/// during request building.
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

    let mut content = Vec::new();
    if let Some(text) = choice.message.content
        && !text.is_empty()
    {
        content.push(Content::Text(llmsdk_provider::language_model::TextPart {
            text,
            provider_options: None,
        }));
    }

    if let Some(tool_calls) = choice.message.tool_calls {
        for tc in tool_calls {
            let input = serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
                .unwrap_or(serde_json::Value::String(tc.function.arguments.clone()));
            content.push(Content::ToolCall(ToolCallPart {
                tool_call_id: tc.id.unwrap_or_default(),
                tool_name: tc.function.name,
                input,
                provider_executed: None,
                dynamic: None,
                provider_options: None,
            }));
        }
    }

    // URL citation annotations become `Source` content parts.
    if let Some(annotations) = choice.message.annotations {
        for (idx, ann) in annotations.into_iter().enumerate() {
            if let Some(source) = annotation_to_source(ann, response.id.as_deref(), idx) {
                content.push(Content::Source(source));
            }
        }
    }

    let usage = usage::convert(response.usage.as_ref());
    let finish = finish_reason::map(choice.finish_reason.as_deref());

    // Collect provider metadata: logprobs (when requested) and prediction
    // token counts (`accepted_prediction_tokens` / `rejected_prediction_tokens`,
    // populated for calls that supplied `prediction`).
    let mut openai_meta = Map::new();
    if let Some(logprobs) = choice.logprobs.and_then(|l| l.content) {
        openai_meta.insert("logprobs".to_owned(), logprobs);
    }
    if let Some(details) = response
        .usage
        .as_ref()
        .and_then(|u| u.completion_tokens_details.as_ref())
    {
        let mut prediction = Map::new();
        if let Some(accepted) = details.accepted_prediction_tokens {
            prediction.insert("accepted_tokens".into(), accepted.into());
        }
        if let Some(rejected) = details.rejected_prediction_tokens {
            prediction.insert("rejected_tokens".into(), rejected.into());
        }
        if !prediction.is_empty() {
            openai_meta.insert("prediction".into(), serde_json::Value::Object(prediction));
        }
    }
    let provider_metadata: Option<ProviderMetadata> = if openai_meta.is_empty() {
        None
    } else {
        let mut pm = ProviderMetadata::new();
        pm.insert("openai".to_owned(), openai_meta);
        Some(pm)
    };

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
        usage,
        provider_metadata,
        request: request_body.map(|body| llmsdk_provider::shared::RequestInfo { body: Some(body) }),
        response: Some(response_meta),
        warnings,
    })
}

/// Convert one [`Annotation`] into a [`Source::Url`] citation.
///
/// Generates a stable id by combining the response id (when present) and the
/// annotation's positional index — no extra randomness dependency.
fn annotation_to_source(
    annotation: Annotation,
    response_id: Option<&str>,
    index: usize,
) -> Option<Source> {
    let Annotation::UrlCitation { url_citation } = annotation else {
        return None;
    };
    let id = match response_id {
        Some(r) => format!("{r}:citation:{index}"),
        None => format!("citation:{index}"),
    };
    Some(Source::Url {
        id,
        url: url_citation.url,
        title: url_citation.title,
        provider_metadata: None,
    })
}

fn headers_to_provider(raw: HashMap<String, String>) -> Headers {
    raw.into_iter().map(|(k, v)| (k, Some(v))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::wire::{
        ChatChoice, ChatChoiceMessage, ResponseFunctionCall, ResponseToolCall, WireUsage,
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
            model: Some("gpt-4o-mini".into()),
            choices: vec![ChatChoice {
                message: ChatChoiceMessage {
                    content: Some("hello".into()),
                    ..Default::default()
                },
                finish_reason: Some("stop".into()),
                ..Default::default()
            }],
            usage: Some(WireUsage {
                prompt_tokens: Some(3),
                completion_tokens: Some(2),
                total_tokens: Some(5),
                prompt_tokens_details: None,
                completion_tokens_details: None,
            }),
        };
        let result = parse_response(resp, empty_headers(), None, vec![]).unwrap();
        assert_eq!(result.content.len(), 1);
        assert_eq!(result.finish_reason.unified, FinishReasonKind::Stop);
        assert_eq!(result.usage.input_tokens.total, Some(3));
        assert_eq!(result.usage.output_tokens.total, Some(2));
    }

    #[test]
    fn parses_tool_calls() {
        let resp = ChatResponse {
            id: Some("r-2".into()),
            created: None,
            model: None,
            choices: vec![ChatChoice {
                message: ChatChoiceMessage {
                    tool_calls: Some(vec![ResponseToolCall {
                        id: Some("call_x".into()),
                        _kind: Some("function".into()),
                        function: ResponseFunctionCall {
                            name: "get_weather".into(),
                            arguments: r#"{"city":"NYC"}"#.into(),
                        },
                    }]),
                    ..Default::default()
                },
                finish_reason: Some("tool_calls".into()),
                ..Default::default()
            }],
            usage: None,
        };
        let result = parse_response(resp, empty_headers(), None, vec![]).unwrap();
        assert_eq!(result.content.len(), 1);
        if let Content::ToolCall(tc) = &result.content[0] {
            assert_eq!(tc.tool_call_id, "call_x");
            assert_eq!(tc.tool_name, "get_weather");
            assert_eq!(tc.input["city"], "NYC");
        } else {
            panic!("expected ToolCall");
        }
        assert_eq!(result.finish_reason.unified, FinishReasonKind::ToolCalls);
    }

    #[test]
    fn empty_choices_yields_no_content_error() {
        let resp = ChatResponse {
            id: None,
            created: None,
            model: None,
            choices: vec![],
            usage: None,
        };
        let err = parse_response(resp, empty_headers(), None, vec![]).unwrap_err();
        assert!(format!("{err}").contains("no content"));
    }
}
