//! Parse a Bedrock Converse response into llmsdk [`GenerateResult`].
//!
//! Mirrors the post-call content-walking logic inside
//! `amazon-bedrock-chat-language-model.ts`. Reasoning blocks fan into
//! [`Content::Reasoning`] with `provider_metadata.amazonBedrock.signature` /
//! `.redactedData`; tool-use blocks become [`Content::ToolCall`].
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;

use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    Content, GenerateResponse, GenerateResult, ReasoningPart, ResponseMetadata, TextPart,
    ToolCallPart,
};
use llmsdk_provider::shared::{ProviderMetadata, RequestInfo, Warning};
use serde_json::Value;

use super::finish_reason::map_finish_reason;
use super::normalize_tool_call_id::normalize_tool_call_id;
use super::usage::convert_usage;
use super::wire::{
    ConverseOutputContent, ConverseResponse, ResponseReasoningContent, ResponseToolUse,
};
use crate::PROVIDER_ID;

/// Parse the deserialized Converse response.
///
/// # Errors
///
/// Currently infallible — the signature returns `Result` for parity with
/// other provider implementations and to leave room for future validation.
#[allow(
    clippy::unnecessary_wraps,
    reason = "kept for parity with other provider parse-response helpers"
)]
pub(crate) fn parse_response(
    response: ConverseResponse,
    response_headers: HashMap<String, String>,
    request_body: Option<Value>,
    warnings: Vec<Warning>,
    is_mistral: bool,
    uses_json_response_tool: bool,
    generate_id: Option<&std::sync::Arc<crate::config::GenerateIdFn>>,
) -> Result<GenerateResult, ProviderError> {
    let mut content: Vec<Content> = Vec::with_capacity(response.output.message.content.len());
    let mut is_json_response_from_tool = false;

    for block in response.output.message.content {
        let ConverseOutputContent {
            text,
            tool_use,
            reasoning_content,
        } = block;
        if let Some(text_value) = text {
            content.push(Content::Text(TextPart {
                text: text_value,
                provider_options: None,
            }));
        }
        if let Some(reasoning) = reasoning_content {
            push_reasoning(&mut content, reasoning);
        }
        if let Some(tu) = tool_use {
            push_tool_use(
                &mut content,
                tu,
                is_mistral,
                uses_json_response_tool,
                &mut is_json_response_from_tool,
                generate_id,
            );
        }
    }

    let stop_sequence = response
        .additional_model_response_fields
        .as_ref()
        .and_then(|v| v.get("delta"))
        .and_then(|v| v.get("stop_sequence"))
        .and_then(Value::as_str)
        .map(str::to_owned);

    let provider_metadata = build_provider_metadata(
        response.trace.clone(),
        response.performance_config.clone(),
        response.service_tier.clone(),
        response.usage.as_ref().and_then(|u| {
            u.cache_write_input_tokens
                .map(|v| ("cacheWriteInputTokens", v))
        }),
        response
            .usage
            .as_ref()
            .and_then(|u| u.cache_details.clone()),
        stop_sequence,
        is_json_response_from_tool,
    );

    let finish_reason =
        map_finish_reason(response.stop_reason.as_deref(), is_json_response_from_tool);
    let usage = convert_usage(response.usage);

    let response_metadata = ResponseMetadata {
        id: response_headers.get("x-amzn-requestid").cloned(),
        timestamp: response_headers.get("date").cloned(),
        model_id: None,
        headers: Some(
            response_headers
                .into_iter()
                .map(|(k, v)| (k, Some(v)))
                .collect(),
        ),
    };
    let generate_response = GenerateResponse {
        metadata: response_metadata,
        body: None,
    };

    Ok(GenerateResult {
        content,
        finish_reason,
        usage,
        provider_metadata,
        request: Some(RequestInfo { body: request_body }),
        response: Some(generate_response),
        warnings,
    })
}

use std::sync::atomic::{AtomicU64, Ordering};

/// Process-wide counter backing the default tool-use id fallback.
///
/// Bumped only when the caller passes no [`crate::config::GenerateIdFn`],
/// so wiring a custom generator means the counter never advances and stays
/// available for any later code path that opts back to defaults.
static SYNTH_ID_SEQ: AtomicU64 = AtomicU64::new(0);

/// Produce a synthetic tool-use id when Bedrock leaves the wire field blank.
///
/// Prefers the caller-supplied generator (mirroring upstream
/// `this.config.generateId()`); otherwise falls back to a process-wide atomic
/// counter so repeated calls never collide within the same parse.
pub(crate) fn synth_id(
    generate_id: Option<&std::sync::Arc<crate::config::GenerateIdFn>>,
) -> String {
    if let Some(f) = generate_id {
        return f();
    }
    let n = SYNTH_ID_SEQ.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
    format!("bedrock-tooluse-{n}")
}

fn push_reasoning(content: &mut Vec<Content>, reasoning: ResponseReasoningContent) {
    if let Some(text) = reasoning.reasoning_text {
        let provider_options = text.signature.as_ref().map(|sig| {
            let mut po = llmsdk_provider::shared::ProviderOptions::new();
            let payload = serde_json::json!({ "signature": sig });
            let map = payload.as_object().cloned().unwrap_or_default();
            po.insert(PROVIDER_ID.to_owned(), map.clone());
            po.insert("bedrock".to_owned(), map);
            po
        });
        content.push(Content::Reasoning(ReasoningPart {
            text: text.text,
            provider_options,
        }));
    } else if let Some(redacted) = reasoning.redacted_reasoning {
        let data = redacted.data.unwrap_or_default();
        let mut po = llmsdk_provider::shared::ProviderOptions::new();
        let payload = serde_json::json!({ "redactedData": data });
        let map = payload.as_object().cloned().unwrap_or_default();
        po.insert(PROVIDER_ID.to_owned(), map.clone());
        po.insert("bedrock".to_owned(), map);
        content.push(Content::Reasoning(ReasoningPart {
            text: String::new(),
            provider_options: Some(po),
        }));
    }
}

fn push_tool_use(
    content: &mut Vec<Content>,
    tool_use: ResponseToolUse,
    is_mistral: bool,
    uses_json_response_tool: bool,
    is_json_response_from_tool: &mut bool,
    generate_id: Option<&std::sync::Arc<crate::config::GenerateIdFn>>,
) {
    let is_json_response = uses_json_response_tool && tool_use.name.as_deref() == Some("json");
    if is_json_response {
        *is_json_response_from_tool = true;
        let input_json = tool_use.input.unwrap_or(Value::Object(Default::default()));
        content.push(Content::Text(TextPart {
            text: serde_json::to_string(&input_json).unwrap_or_else(|_| "{}".to_owned()),
            provider_options: None,
        }));
        return;
    }
    // Mirrors upstream `amazon-bedrock-chat-language-model.ts:557` / `:561`:
    // when Bedrock omits `toolUseId` (and even `name`) on partial chunks the
    // SDK falls back to `this.config.generateId()` so the surfaced tool call
    // still carries usable identifiers.
    let raw_id = tool_use
        .tool_use_id
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| synth_id(generate_id));
    let normalized = normalize_tool_call_id(&raw_id, is_mistral);
    let name = tool_use
        .name
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| format!("tool-{}", synth_id(generate_id)));
    let input_str = match &tool_use.input {
        Some(Value::String(s)) => s.clone(),
        Some(v) => serde_json::to_string(v).unwrap_or_else(|_| "{}".to_owned()),
        None => "{}".to_owned(),
    };
    let input_value: Value =
        serde_json::from_str(&input_str).unwrap_or(Value::Object(Default::default()));
    content.push(Content::ToolCall(ToolCallPart {
        tool_call_id: normalized,
        tool_name: name,
        input: input_value,
        provider_executed: None,
        dynamic: None,
        provider_options: None,
    }));
}

fn build_provider_metadata(
    trace: Option<Value>,
    performance_config: Option<Value>,
    service_tier: Option<Value>,
    cache_write: Option<(&'static str, u64)>,
    cache_details: Option<Value>,
    stop_sequence: Option<String>,
    is_json_response_from_tool: bool,
) -> Option<ProviderMetadata> {
    let mut payload = serde_json::Map::new();
    if let Some(t) = trace {
        payload.insert("trace".to_owned(), t);
    }
    if let Some(p) = performance_config {
        payload.insert("performanceConfig".to_owned(), p);
    }
    if let Some(s) = service_tier {
        payload.insert("serviceTier".to_owned(), s);
    }
    if cache_write.is_some() || cache_details.is_some() {
        let mut usage = serde_json::Map::new();
        if let Some((name, value)) = cache_write {
            usage.insert(name.to_owned(), Value::from(value));
        }
        if let Some(details) = cache_details {
            usage.insert("cacheDetails".to_owned(), details);
        }
        payload.insert("usage".to_owned(), Value::Object(usage));
    }
    if is_json_response_from_tool {
        payload.insert("isJsonResponseFromTool".to_owned(), Value::Bool(true));
    }
    if let Some(seq) = stop_sequence {
        payload.insert("stopSequence".to_owned(), Value::String(seq));
    } else {
        payload.insert("stopSequence".to_owned(), Value::Null);
    }

    if payload.len() == 1 && payload.get("stopSequence") == Some(&Value::Null) {
        return None;
    }

    let mut meta: ProviderMetadata = HashMap::new();
    meta.insert(PROVIDER_ID.to_owned(), payload.clone());
    meta.insert("bedrock".to_owned(), payload);
    Some(meta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::wire::{BedrockUsage, ConverseOutput, ConverseOutputMessage};

    fn empty_headers() -> HashMap<String, String> {
        HashMap::new()
    }

    #[test]
    fn text_only_response_maps_to_text_content() {
        let response = ConverseResponse {
            output: ConverseOutput {
                message: ConverseOutputMessage {
                    content: vec![ConverseOutputContent {
                        text: Some("hi".into()),
                        ..Default::default()
                    }],
                },
            },
            stop_reason: Some("end_turn".into()),
            additional_model_response_fields: None,
            trace: None,
            performance_config: None,
            service_tier: None,
            usage: Some(BedrockUsage {
                input_tokens: Some(5),
                output_tokens: Some(3),
                ..Default::default()
            }),
        };
        let result =
            parse_response(response, empty_headers(), None, vec![], false, false, None).unwrap();
        assert_eq!(result.content.len(), 1);
        assert!(matches!(result.content[0], Content::Text(_)));
        assert_eq!(
            result.finish_reason.unified,
            llmsdk_provider::language_model::FinishReasonKind::Stop
        );
        assert_eq!(result.usage.input_tokens.no_cache, Some(5));
    }

    #[test]
    fn tool_use_response_maps_to_tool_call() {
        let response = ConverseResponse {
            output: ConverseOutput {
                message: ConverseOutputMessage {
                    content: vec![ConverseOutputContent {
                        tool_use: Some(ResponseToolUse {
                            tool_use_id: Some("tu-1".into()),
                            name: Some("weather".into()),
                            input: Some(serde_json::json!({"city": "NYC"})),
                        }),
                        ..Default::default()
                    }],
                },
            },
            stop_reason: Some("tool_use".into()),
            additional_model_response_fields: None,
            trace: None,
            performance_config: None,
            service_tier: None,
            usage: None,
        };
        let result =
            parse_response(response, empty_headers(), None, vec![], false, false, None).unwrap();
        let Content::ToolCall(tc) = &result.content[0] else {
            panic!("expected tool-call");
        };
        assert_eq!(tc.tool_call_id, "tu-1");
        assert_eq!(tc.tool_name, "weather");
        assert_eq!(tc.input["city"], "NYC");
    }
}
