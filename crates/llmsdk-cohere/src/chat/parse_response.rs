//! Convert a Cohere chat response into a [`GenerateResult`].
//!
//! Mirrors the `doGenerate` post-processing in `cohere-chat-language-model.ts`.
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;
use std::sync::Arc;

use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    Content, GenerateResponse, GenerateResult, ReasoningPart, ResponseMetadata, Source, TextPart,
    ToolCallPart,
};
use llmsdk_provider::shared::{Headers, ProviderMetadata, RequestInfo, Warning};

use super::finish_reason;
use super::usage;
use super::wire::{ChatResponse, ChatResponseCitation, ChatResponseContent};
use crate::config::GenerateIdFn;

/// Parse a successful (non-streaming) chat response.
///
/// # Errors
///
/// Returns [`ProviderError::no_content_generated`] when the response carries
/// no message content (Cohere does not return an explicit 200-error envelope).
pub(crate) fn parse_response(
    response: ChatResponse,
    headers: HashMap<String, String>,
    request_body: Option<serde_json::Value>,
    warnings: Vec<Warning>,
    citation_id_seed: &mut u64,
    generate_id: Option<&Arc<GenerateIdFn>>,
) -> Result<GenerateResult, ProviderError> {
    let mut content: Vec<Content> = Vec::new();

    for item in response.message.content.unwrap_or_default() {
        match item {
            ChatResponseContent::Text { text } if !text.is_empty() => {
                content.push(Content::Text(TextPart {
                    text,
                    provider_options: None,
                }));
            }
            ChatResponseContent::Thinking { thinking } if !thinking.is_empty() => {
                content.push(Content::Reasoning(ReasoningPart {
                    text: thinking,
                    provider_options: None,
                }));
            }
            ChatResponseContent::Text { .. } | ChatResponseContent::Thinking { .. } => {}
        }
    }

    for citation in response.message.citations.unwrap_or_default() {
        content.push(citation_to_source(citation, citation_id_seed, generate_id));
    }

    if let Some(tool_calls) = response.message.tool_calls {
        for tc in tool_calls {
            // Cohere sometimes returns `"null"` for arguments of zero-arg tools.
            let raw_args = if tc.function.arguments == "null" {
                "{}".to_owned()
            } else {
                tc.function.arguments
            };
            let input = serde_json::from_str::<serde_json::Value>(&raw_args)
                .unwrap_or(serde_json::Value::String(raw_args));
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

    if content.is_empty() && response.finish_reason.is_none() && response.generation_id.is_none() {
        return Err(ProviderError::no_content_generated());
    }

    let usage_value = usage::convert(response.usage.as_ref());
    let finish = finish_reason::map(response.finish_reason.as_deref());

    Ok(GenerateResult {
        content,
        finish_reason: finish,
        usage: usage_value,
        provider_metadata: tool_plan_to_metadata(response.message.tool_plan.as_deref()),
        request: request_body.map(|body| RequestInfo { body: Some(body) }),
        response: Some(GenerateResponse {
            metadata: ResponseMetadata {
                id: response.generation_id,
                timestamp: None,
                model_id: None,
                headers: Some(headers_to_provider(headers)),
            },
            body: None,
        }),
        warnings,
    })
}

/// Build a [`Content::Source`] from a Cohere citation.
pub(crate) fn citation_to_source(
    citation: ChatResponseCitation,
    seed: &mut u64,
    generate_id: Option<&Arc<GenerateIdFn>>,
) -> Content {
    let title = citation
        .sources
        .first()
        .map_or_else(|| "Document".to_owned(), |s| s.document.title.clone());

    let mut cohere = serde_json::Map::new();
    cohere.insert("start".into(), serde_json::Value::from(citation.start));
    cohere.insert("end".into(), serde_json::Value::from(citation.end));
    cohere.insert("text".into(), serde_json::Value::String(citation.text));
    cohere.insert(
        "sources".into(),
        serde_json::to_value(&citation.sources).unwrap_or(serde_json::Value::Null),
    );
    if let Some(kind) = citation.kind {
        cohere.insert("citationType".into(), serde_json::Value::String(kind));
    }

    let mut metadata = ProviderMetadata::new();
    metadata.insert("cohere".into(), cohere);

    Content::Source(Source::Document {
        id: next_id(seed, generate_id),
        media_type: "text/plain".into(),
        title,
        filename: None,
        provider_metadata: Some(metadata),
    })
}

/// Wrap Cohere's `tool_plan` field into the `provider_metadata.cohere.toolPlan` slot.
fn tool_plan_to_metadata(plan: Option<&str>) -> Option<ProviderMetadata> {
    let plan = plan?;
    if plan.is_empty() {
        return None;
    }
    let mut cohere = serde_json::Map::new();
    cohere.insert(
        "toolPlan".into(),
        serde_json::Value::String(plan.to_owned()),
    );
    let mut metadata = ProviderMetadata::new();
    metadata.insert("cohere".into(), cohere);
    Some(metadata)
}

/// Generate a citation id, preferring the caller-supplied generator.
///
/// When `generate_id` is `Some`, the closure is invoked verbatim — this
/// mirrors upstream `this.config.generateId()` in
/// `cohere-chat-language-model.ts:204`. Otherwise the id falls back to the
/// deterministic `cohere-citation-N` counter so offline replay stays stable.
pub(crate) fn next_id(seed: &mut u64, generate_id: Option<&Arc<GenerateIdFn>>) -> String {
    if let Some(f) = generate_id {
        return f();
    }
    *seed = seed.wrapping_add(1);
    format!("cohere-citation-{seed}")
}

fn headers_to_provider(raw: HashMap<String, String>) -> Headers {
    raw.into_iter().map(|(k, v)| (k, Some(v))).collect()
}

// We need to derive Serialize on ChatResponseCitationSource so that
// the citation can be stuffed into provider metadata as-is.
impl serde::Serialize for super::wire::ChatResponseCitationSource {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;
        if let Some(k) = &self.kind {
            map.serialize_entry("type", k)?;
        }
        if let Some(id) = &self.id {
            map.serialize_entry("id", id)?;
        }
        map.serialize_entry("document", &self.document)?;
        map.end()
    }
}

impl serde::Serialize for super::wire::ChatResponseCitationDocument {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;
        if let Some(id) = &self.id {
            map.serialize_entry("id", id)?;
        }
        map.serialize_entry("text", &self.text)?;
        map.serialize_entry("title", &self.title)?;
        map.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::wire::{
        ChatResponseCitationDocument, ChatResponseCitationSource, ChatResponseMessage,
        WireFunctionCall, WireToolCall, WireToolCallKind, WireUsage, WireUsageTokens,
    };
    use llmsdk_provider::language_model::FinishReasonKind;

    fn headers() -> HashMap<String, String> {
        HashMap::new()
    }

    fn base_response() -> ChatResponse {
        ChatResponse {
            generation_id: Some("g1".into()),
            message: ChatResponseMessage {
                role: Some("assistant".into()),
                content: Some(vec![ChatResponseContent::Text {
                    text: "hello".into(),
                }]),
                tool_plan: None,
                tool_calls: None,
                citations: None,
            },
            finish_reason: Some("COMPLETE".into()),
            usage: Some(WireUsage {
                billed_units: Some(WireUsageTokens {
                    input_tokens: Some(5),
                    output_tokens: Some(3),
                }),
                tokens: Some(WireUsageTokens {
                    input_tokens: Some(5),
                    output_tokens: Some(3),
                }),
            }),
        }
    }

    #[test]
    fn parses_text() {
        let mut seed = 0;
        let r = parse_response(base_response(), headers(), None, vec![], &mut seed, None).unwrap();
        assert_eq!(r.content.len(), 1);
        assert!(matches!(r.content[0], Content::Text(_)));
        assert_eq!(r.finish_reason.unified, FinishReasonKind::Stop);
        assert_eq!(r.usage.input_tokens.total, Some(5));
    }

    #[test]
    fn parses_thinking_then_text() {
        let mut resp = base_response();
        resp.message.content = Some(vec![
            ChatResponseContent::Thinking {
                thinking: "let me think".into(),
            },
            ChatResponseContent::Text { text: "42".into() },
        ]);
        let mut seed = 0;
        let r = parse_response(resp, headers(), None, vec![], &mut seed, None).unwrap();
        assert_eq!(r.content.len(), 2);
        assert!(matches!(r.content[0], Content::Reasoning(_)));
        assert!(matches!(r.content[1], Content::Text(_)));
    }

    #[test]
    fn parses_tool_calls() {
        let mut resp = base_response();
        resp.message.content = Some(vec![]);
        resp.message.tool_calls = Some(vec![WireToolCall {
            id: "c1".into(),
            kind: WireToolCallKind::Function,
            function: WireFunctionCall {
                name: "weather".into(),
                arguments: r#"{"city":"NYC"}"#.into(),
            },
        }]);
        resp.finish_reason = Some("TOOL_CALL".into());
        let mut seed = 0;
        let r = parse_response(resp, headers(), None, vec![], &mut seed, None).unwrap();
        assert_eq!(r.content.len(), 1);
        let Content::ToolCall(tc) = &r.content[0] else {
            panic!("expected tool call");
        };
        assert_eq!(tc.tool_call_id, "c1");
        assert_eq!(tc.input["city"], "NYC");
        assert_eq!(r.finish_reason.unified, FinishReasonKind::ToolCalls);
    }

    #[test]
    fn tool_call_with_null_arguments_yields_empty_object() {
        let mut resp = base_response();
        resp.message.content = Some(vec![]);
        resp.message.tool_calls = Some(vec![WireToolCall {
            id: "c2".into(),
            kind: WireToolCallKind::Function,
            function: WireFunctionCall {
                name: "ping".into(),
                arguments: "null".into(),
            },
        }]);
        let mut seed = 0;
        let r = parse_response(resp, headers(), None, vec![], &mut seed, None).unwrap();
        let Content::ToolCall(tc) = &r.content[0] else {
            panic!("expected tool call");
        };
        assert!(tc.input.is_object());
        assert!(tc.input.as_object().unwrap().is_empty());
    }

    #[test]
    fn citations_become_document_sources() {
        let mut resp = base_response();
        resp.message.citations = Some(vec![ChatResponseCitation {
            start: 0,
            end: 5,
            text: "hello".into(),
            sources: vec![ChatResponseCitationSource {
                kind: Some("document".into()),
                id: Some("d0".into()),
                document: ChatResponseCitationDocument {
                    id: Some("d0".into()),
                    text: "hello".into(),
                    title: "Greeting".into(),
                },
            }],
            kind: Some("inline".into()),
        }]);
        let mut seed = 0;
        let r = parse_response(resp, headers(), None, vec![], &mut seed, None).unwrap();
        let Content::Source(Source::Document { title, id, .. }) = &r.content[1] else {
            panic!("expected document source");
        };
        assert_eq!(title, "Greeting");
        assert_eq!(id, "cohere-citation-1");
    }

    #[test]
    fn tool_plan_lands_in_provider_metadata() {
        let mut resp = base_response();
        resp.message.tool_plan = Some("Step 1: call weather".into());
        let mut seed = 0;
        let r = parse_response(resp, headers(), None, vec![], &mut seed, None).unwrap();
        let meta = r.provider_metadata.unwrap();
        let cohere = meta.get("cohere").unwrap();
        assert_eq!(cohere["toolPlan"], "Step 1: call weather");
    }
}
