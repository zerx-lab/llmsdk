//! Synthesize a `StreamPart` sequence from a fully resolved
//! Interactions response.
//!
//! Mirrors `@ai-sdk/google/src/interactions/synthesize-google-interactions-agent-stream.ts`.
//! Used by the `do_stream` background path — agent calls require
//! `background: true`, which is incompatible with `stream: true` on POST, so
//! we POST background, poll until terminal, then deterministically replay
//! the polled outputs as a stream in the same order
//! `build-google-interactions-stream-transform` would produce.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    Content, ReasoningPart, ResponseMetadata, StreamPart, TextPart, ToolCallPart, ToolResult,
};
use llmsdk_provider::shared::{ProviderMetadata, ProviderOptions, Warning};
use serde_json::{Map as JsonMap, Value as JsonValue};

/// Stream the resolved response as a sequence of stream parts. Mirrors the
/// upstream `synthesizeGoogleInteractionsAgentStream` shape — `stream-start`
/// first, then `response-metadata`, then per-content `*-start` / `*-delta`
/// / `*-end` for text + reasoning + a single `tool-call` / `tool-result` for
/// tool blocks + `source` / `file` parts inline, then `finish`.
pub(crate) fn synthesize_response_to_stream(
    response: JsonValue,
    warnings: Vec<Warning>,
    model_id: String,
    header_service_tier: Option<String>,
) -> impl futures::Stream<Item = Result<StreamPart, ProviderError>> + Send {
    async_stream::stream! {
        yield Ok(StreamPart::StreamStart { warnings });

        let interaction_id = response
            .get("id")
            .and_then(JsonValue::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_owned);

        yield Ok(StreamPart::ResponseMetadata(ResponseMetadata {
            id: interaction_id.clone(),
            timestamp: response
                .get("created")
                .and_then(JsonValue::as_str)
                .map(str::to_owned),
            model_id: Some(model_id),
            headers: None,
        }));

        let mut id_n = 0usize;
        let mut next_block_id = || {
            id_n += 1;
            format!(
                "{}:{id_n}",
                interaction_id.as_deref().unwrap_or("agent")
            )
        };

        // Parse the same Content[] we'd produce on `do_generate`, then unfold
        // each into `*-start` / `*-delta` / `*-end`.
        let mut content_buf: Vec<Content> = Vec::new();
        if let Some(steps) = response.get("steps").and_then(JsonValue::as_array) {
            for step in steps {
                super::model::translate_step(step, &mut content_buf);
            }
        }

        let mut has_function_call = false;
        for part in content_buf {
            for emitted in unfold_content_part(&mut next_block_id, part, &mut has_function_call) {
                yield Ok(emitted);
            }
        }

        let service_tier = response
            .get("service_tier")
            .and_then(JsonValue::as_str)
            .map(str::to_owned)
            .or(header_service_tier);

        let usage = super::model::parse_usage(response.get("usage"));
        let status = response
            .get("status")
            .and_then(JsonValue::as_str)
            .map(str::to_owned);
        let mut finish_reason =
            super::model::map_finish_reason_from_status(status.as_deref());
        if has_function_call {
            // Mirrors upstream `mapGoogleInteractionsFinishReason({status:
            // 'completed', hasFunctionCall: true})` → `tool-calls`.
            if status.as_deref() == Some("completed") {
                finish_reason = llmsdk_provider::language_model::FinishReason {
                    unified: llmsdk_provider::language_model::FinishReasonKind::ToolCalls,
                    raw: status,
                };
            }
        }

        let provider_metadata = finish_provider_metadata(
            interaction_id.as_deref(),
            service_tier.as_deref(),
        );

        yield Ok(StreamPart::Finish {
            finish_reason,
            usage,
            provider_metadata,
        });
    }
}

fn unfold_content_part<F: FnMut() -> String>(
    next_id: &mut F,
    part: Content,
    has_function_call: &mut bool,
) -> Vec<StreamPart> {
    let mut out = Vec::new();
    match part {
        Content::Text(TextPart {
            text,
            provider_options,
        }) => {
            let id = next_id();
            out.push(StreamPart::TextStart {
                id: id.clone(),
                provider_metadata: None,
            });
            if !text.is_empty() {
                out.push(StreamPart::TextDelta {
                    id: id.clone(),
                    delta: text,
                    provider_metadata: None,
                });
            }
            out.push(StreamPart::TextEnd {
                id,
                provider_metadata: provider_options.map(provider_metadata_from_options),
            });
        }
        Content::Reasoning(ReasoningPart {
            text,
            provider_options,
        }) => {
            let id = next_id();
            out.push(StreamPart::ReasoningStart {
                id: id.clone(),
                provider_metadata: None,
            });
            if !text.is_empty() {
                out.push(StreamPart::ReasoningDelta {
                    id: id.clone(),
                    delta: text,
                    provider_metadata: None,
                });
            }
            out.push(StreamPart::ReasoningEnd {
                id,
                provider_metadata: provider_options.map(provider_metadata_from_options),
            });
        }
        Content::ToolCall(tc) => {
            *has_function_call = true;
            out.push(StreamPart::ToolInputStart {
                id: tc.tool_call_id.clone(),
                tool_name: tc.tool_name.clone(),
                provider_executed: tc.provider_executed,
                dynamic: tc.dynamic,
                title: None,
                provider_metadata: None,
            });
            let delta = match &tc.input {
                JsonValue::String(s) => s.clone(),
                other => other.to_string(),
            };
            out.push(StreamPart::ToolInputDelta {
                id: tc.tool_call_id.clone(),
                delta,
                provider_metadata: None,
            });
            out.push(StreamPart::ToolInputEnd {
                id: tc.tool_call_id.clone(),
                provider_metadata: None,
            });
            out.push(StreamPart::ToolCall(ToolCallPart {
                tool_call_id: tc.tool_call_id,
                tool_name: tc.tool_name,
                input: tc.input,
                provider_executed: tc.provider_executed,
                dynamic: tc.dynamic,
                provider_options: tc.provider_options,
            }));
        }
        Content::ToolResult(tr) => {
            out.push(StreamPart::ToolResult(ToolResult {
                tool_call_id: tr.tool_call_id,
                tool_name: tr.tool_name,
                output: tr.output,
                preliminary: None,
                provider_metadata: tr.provider_metadata,
            }));
        }
        Content::Source(src) => {
            out.push(StreamPart::Source(src));
        }
        Content::File(f) => {
            out.push(StreamPart::File(f));
        }
        // Other content variants are pass-through-emitted as no-ops in this
        // surface (mirrors upstream's `default: break`).
        _ => {}
    }
    out
}

fn finish_provider_metadata(
    interaction_id: Option<&str>,
    service_tier: Option<&str>,
) -> Option<ProviderMetadata> {
    let mut g = JsonMap::new();
    if let Some(id) = interaction_id {
        g.insert("interactionId".into(), JsonValue::String(id.to_owned()));
    }
    if let Some(t) = service_tier {
        g.insert("serviceTier".into(), JsonValue::String(t.to_owned()));
    }
    if g.is_empty() {
        return None;
    }
    let mut pm = ProviderMetadata::new();
    pm.insert("google".to_owned(), g);
    Some(pm)
}

fn provider_metadata_from_options(po: ProviderOptions) -> ProviderMetadata {
    let mut pm = ProviderMetadata::new();
    for (k, v) in po {
        pm.insert(k, v);
    }
    pm
}
