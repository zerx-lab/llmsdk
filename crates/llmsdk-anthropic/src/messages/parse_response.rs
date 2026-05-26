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
use llmsdk_provider::shared::{Headers, ProviderMetadata, ProviderOptions, Warning};
use serde_json::{Map, Value as JsonValue};

use super::wire::{
    MessagesResponse, ResponseContent, WireAppliedEdit, WireContainerMetadata,
    WireContextManagement,
};
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
    mark_code_execution_dynamic: bool,
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
            ResponseContent::Compaction { content: text } => {
                // Surface compaction as regular text content tagged with
                // `anthropic.type = "compaction"` on `provider_options`.
                // Mirrors upstream anthropic-language-model.ts:959-969.
                let mut anthropic = Map::new();
                anthropic.insert("type".into(), JsonValue::String("compaction".into()));
                let mut po = ProviderOptions::new();
                po.insert("anthropic".into(), anthropic);
                content.push(Content::Text(TextPart {
                    text: text.unwrap_or_default(),
                    provider_options: Some(po),
                }));
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
                // Mark code_execution invocations as dynamic when the request
                // enabled web_*_20260209 without an explicit code_execution
                // tool. Mirrors upstream anthropic-language-model.ts:1028-1031
                // and :1058-1060.
                let dynamic =
                    (mark_code_execution_dynamic && name == "code_execution").then_some(true);
                content.push(Content::ToolCall(ToolCallPart {
                    tool_call_id: id,
                    tool_name: name,
                    input,
                    provider_executed: Some(true),
                    dynamic,
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
                // Detect server-tool error variants. Mirrors upstream
                // anthropic-language-model.ts:1142-1153 / 1185-1196 / 1813-1869:
                // when `content.type` ends in `_tool_result_error`, the tool
                // call failed — surface it via `ErrorJson` so consumers can
                // route on `isError`.
                let is_error = is_tool_result_error(&v);
                let mut anthropic_meta = Map::new();
                if is_error {
                    anthropic_meta.insert("isError".into(), JsonValue::Bool(true));
                }
                let provider_metadata = if anthropic_meta.is_empty() {
                    None
                } else {
                    let mut pm = ProviderMetadata::new();
                    pm.insert("anthropic".into(), anthropic_meta);
                    Some(pm)
                };
                let output = if is_error {
                    ToolResultOutput::ErrorJson {
                        value: v,
                        provider_options: None,
                    }
                } else {
                    ToolResultOutput::Json {
                        value: v,
                        provider_options: None,
                    }
                };
                content.push(Content::ToolResult(ToolResult {
                    tool_call_id,
                    tool_name,
                    output,
                    provider_metadata,
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

    let provider_metadata = build_provider_metadata(
        response.usage.iterations.as_deref(),
        response.container.as_ref(),
        response.context_management.as_ref(),
    );

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

/// Build the `provider_metadata.anthropic.*` block from the response-level
/// metadata fields (`iterations` / `container` / `context_management`).
///
/// Returns `None` when no metadata is present so we keep the on-wire payload
/// minimal for ordinary calls.
fn build_provider_metadata(
    iterations: Option<&[super::wire::WireUsageIteration]>,
    container: Option<&WireContainerMetadata>,
    context_management: Option<&WireContextManagement>,
) -> Option<ProviderMetadata> {
    let mut anthropic = Map::new();

    if let Some(its) = iterations
        && !its.is_empty()
    {
        let value = serde_json::to_value(its).ok()?;
        anthropic.insert("usageIterations".into(), value);
    }

    if let Some(c) = container {
        let mut obj = Map::new();
        obj.insert("expiresAt".into(), JsonValue::String(c.expires_at.clone()));
        obj.insert("id".into(), JsonValue::String(c.id.clone()));
        if let Some(skills) = &c.skills {
            let skill_values: Vec<JsonValue> = skills
                .iter()
                .map(|s| {
                    let mut so = Map::new();
                    so.insert("type".into(), JsonValue::String(s.kind.clone()));
                    so.insert("skillId".into(), JsonValue::String(s.skill_id.clone()));
                    if let Some(v) = &s.version {
                        so.insert("version".into(), JsonValue::String(v.clone()));
                    }
                    JsonValue::Object(so)
                })
                .collect();
            obj.insert("skills".into(), JsonValue::Array(skill_values));
        }
        anthropic.insert("container".into(), JsonValue::Object(obj));
    }

    if let Some(cm) = context_management {
        let edits = cm
            .applied_edits
            .iter()
            .filter_map(applied_edit_to_value)
            .collect::<Vec<_>>();
        let mut obj = Map::new();
        obj.insert("appliedEdits".into(), JsonValue::Array(edits));
        anthropic.insert("contextManagement".into(), JsonValue::Object(obj));
    }

    if anthropic.is_empty() {
        return None;
    }
    let mut pm = ProviderMetadata::new();
    pm.insert("anthropic".into(), anthropic);
    Some(pm)
}

fn applied_edit_to_value(edit: &WireAppliedEdit) -> Option<JsonValue> {
    let mut obj = Map::new();
    match edit {
        WireAppliedEdit::ClearToolUses {
            cleared_tool_uses,
            cleared_input_tokens,
        } => {
            obj.insert(
                "type".into(),
                JsonValue::String("clear_tool_uses_20250919".into()),
            );
            if let Some(n) = cleared_tool_uses {
                obj.insert("clearedToolUses".into(), JsonValue::Number((*n).into()));
            }
            if let Some(n) = cleared_input_tokens {
                obj.insert("clearedInputTokens".into(), JsonValue::Number((*n).into()));
            }
        }
        WireAppliedEdit::ClearThinking {
            cleared_thinking_turns,
            cleared_input_tokens,
        } => {
            obj.insert(
                "type".into(),
                JsonValue::String("clear_thinking_20251015".into()),
            );
            if let Some(n) = cleared_thinking_turns {
                obj.insert(
                    "clearedThinkingTurns".into(),
                    JsonValue::Number((*n).into()),
                );
            }
            if let Some(n) = cleared_input_tokens {
                obj.insert("clearedInputTokens".into(), JsonValue::Number((*n).into()));
            }
        }
        WireAppliedEdit::Compact {
            cleared_input_tokens,
        } => {
            obj.insert("type".into(), JsonValue::String("compact_20260112".into()));
            if let Some(n) = cleared_input_tokens {
                obj.insert("clearedInputTokens".into(), JsonValue::Number((*n).into()));
            }
        }
        WireAppliedEdit::Other => return None,
    }
    Some(JsonValue::Object(obj))
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

/// Detect server-tool error variants by inspecting `content.type` (single
/// object) or scanning `content[].type` (array). Matches the upstream
/// `*_tool_result_error` variants surfaced for `web_search` / `web_fetch` /
/// `code_execution` and friends.
fn is_tool_result_error(v: &JsonValue) -> bool {
    let Some(content) = v.get("content") else {
        return false;
    };
    if let Some(s) = content.get("type").and_then(|x| x.as_str()) {
        return s.ends_with("_tool_result_error");
    }
    if let Some(arr) = content.as_array() {
        return arr.iter().any(|item| {
            item.get("type")
                .and_then(|x| x.as_str())
                .is_some_and(|s| s.ends_with("_tool_result_error"))
        });
    }
    false
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
                iterations: None,
            },
            container: None,
            context_management: None,
        };
        let r = parse_response(resp, empty_headers(), None, vec![], false).unwrap();
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
                iterations: None,
            },
            container: None,
            context_management: None,
        };
        let r = parse_response(resp, empty_headers(), None, vec![], false).unwrap();
        if let Content::ToolCall(tc) = &r.content[0] {
            assert_eq!(tc.tool_call_id, "tu_1");
            assert_eq!(tc.input["city"], "NYC");
        } else {
            panic!("expected tool call");
        }
        assert_eq!(r.finish_reason.unified, FinishReasonKind::ToolCalls);
    }

    #[test]
    fn iterations_surface_under_provider_metadata() {
        use crate::messages::wire::WireUsageIteration;
        let resp = MessagesResponse {
            id: Some("msg_x".into()),
            model: Some("claude-opus-4-7".into()),
            content: vec![ResponseContent::Text {
                text: "ok".into(),
                citations: None,
            }],
            stop_reason: Some("end_turn".into()),
            usage: ResponseUsage {
                input_tokens: 10,
                output_tokens: 5,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
                iterations: Some(vec![
                    WireUsageIteration::Compaction {
                        input_tokens: 200,
                        output_tokens: 80,
                        cache_creation_input_tokens: None,
                        cache_read_input_tokens: None,
                    },
                    WireUsageIteration::AdvisorMessage {
                        model: "claude-opus-4-7".into(),
                        input_tokens: 50,
                        output_tokens: 30,
                        cache_creation_input_tokens: None,
                        cache_read_input_tokens: Some(10),
                    },
                ]),
            },
            container: None,
            context_management: None,
        };
        let r = parse_response(resp, empty_headers(), None, vec![], false).unwrap();
        let pm = r.provider_metadata.unwrap();
        let anthropic = pm.get("anthropic").unwrap();
        let iters = anthropic
            .get("usageIterations")
            .and_then(JsonValue::as_array)
            .unwrap();
        assert_eq!(iters.len(), 2);
        assert_eq!(iters[0]["type"], "compaction");
        assert_eq!(iters[1]["type"], "advisor_message");
        assert_eq!(iters[1]["model"], "claude-opus-4-7");
    }

    #[test]
    fn container_metadata_surfaces() {
        use crate::messages::wire::{WireContainerMetadata, WireContainerSkill};
        let resp = MessagesResponse {
            id: None,
            model: None,
            content: vec![ResponseContent::Text {
                text: "ok".into(),
                citations: None,
            }],
            stop_reason: Some("end_turn".into()),
            usage: ResponseUsage::default(),
            container: Some(WireContainerMetadata {
                expires_at: "2026-05-25T12:00:00Z".into(),
                id: "ctr-xyz".into(),
                skills: Some(vec![WireContainerSkill {
                    kind: "user".into(),
                    skill_id: "skill-1".into(),
                    version: Some("v3".into()),
                }]),
            }),
            context_management: None,
        };
        let r = parse_response(resp, empty_headers(), None, vec![], false).unwrap();
        let pm = r.provider_metadata.unwrap();
        let container = pm.get("anthropic").unwrap().get("container").unwrap();
        assert_eq!(container["id"], "ctr-xyz");
        assert_eq!(container["expiresAt"], "2026-05-25T12:00:00Z");
        assert_eq!(container["skills"][0]["skillId"], "skill-1");
        assert_eq!(container["skills"][0]["version"], "v3");
    }

    #[test]
    fn context_management_three_edit_types_all_surface() {
        let resp = MessagesResponse {
            id: None,
            model: None,
            content: vec![ResponseContent::Text {
                text: "ok".into(),
                citations: None,
            }],
            stop_reason: Some("end_turn".into()),
            usage: ResponseUsage::default(),
            container: None,
            context_management: Some(WireContextManagement {
                applied_edits: vec![
                    WireAppliedEdit::ClearToolUses {
                        cleared_tool_uses: Some(3),
                        cleared_input_tokens: Some(120),
                    },
                    WireAppliedEdit::ClearThinking {
                        cleared_thinking_turns: Some(2),
                        cleared_input_tokens: None,
                    },
                    WireAppliedEdit::Compact {
                        cleared_input_tokens: Some(2000),
                    },
                    WireAppliedEdit::Other,
                ],
            }),
        };
        let r = parse_response(resp, empty_headers(), None, vec![], false).unwrap();
        let pm = r.provider_metadata.unwrap();
        let edits = pm
            .get("anthropic")
            .unwrap()
            .get("contextManagement")
            .unwrap()
            .get("appliedEdits")
            .and_then(JsonValue::as_array)
            .unwrap();
        // Other is filtered out; only 3 known edits.
        assert_eq!(edits.len(), 3);
        assert_eq!(edits[0]["type"], "clear_tool_uses_20250919");
        assert_eq!(edits[0]["clearedToolUses"], 3);
        assert_eq!(edits[0]["clearedInputTokens"], 120);
        assert_eq!(edits[1]["type"], "clear_thinking_20251015");
        assert_eq!(edits[1]["clearedThinkingTurns"], 2);
        assert!(edits[1].get("clearedInputTokens").is_none());
        assert_eq!(edits[2]["type"], "compact_20260112");
        assert_eq!(edits[2]["clearedInputTokens"], 2000);
    }

    #[test]
    fn no_extra_metadata_yields_none() {
        let resp = MessagesResponse {
            id: None,
            model: None,
            content: vec![ResponseContent::Text {
                text: "ok".into(),
                citations: None,
            }],
            stop_reason: Some("end_turn".into()),
            usage: ResponseUsage::default(),
            container: None,
            context_management: None,
        };
        let r = parse_response(resp, empty_headers(), None, vec![], false).unwrap();
        assert!(r.provider_metadata.is_none());
    }

    #[test]
    fn server_tool_use_code_execution_marked_dynamic_when_flag_set() {
        let resp = MessagesResponse {
            id: None,
            model: None,
            content: vec![ResponseContent::ServerToolUse {
                id: "srv_1".into(),
                name: "code_execution".into(),
                input: serde_json::json!({"code": "1+1"}),
            }],
            stop_reason: Some("end_turn".into()),
            usage: ResponseUsage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
                iterations: None,
            },
            container: None,
            context_management: None,
        };
        let r = parse_response(resp, empty_headers(), None, vec![], true).unwrap();
        match &r.content[0] {
            Content::ToolCall(tc) => {
                assert_eq!(tc.dynamic, Some(true));
                assert_eq!(tc.provider_executed, Some(true));
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn server_tool_use_non_code_execution_not_marked_when_flag_set() {
        let resp = MessagesResponse {
            id: None,
            model: None,
            content: vec![ResponseContent::ServerToolUse {
                id: "srv_2".into(),
                name: "web_search".into(),
                input: serde_json::json!({"q": "rust"}),
            }],
            stop_reason: Some("end_turn".into()),
            usage: ResponseUsage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
                iterations: None,
            },
            container: None,
            context_management: None,
        };
        let r = parse_response(resp, empty_headers(), None, vec![], true).unwrap();
        match &r.content[0] {
            Content::ToolCall(tc) => assert_eq!(tc.dynamic, None),
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn web_search_tool_result_error_surfaced_as_error_json() {
        let resp = MessagesResponse {
            id: None,
            model: None,
            content: vec![ResponseContent::WebSearchToolResult(serde_json::json!({
                "tool_use_id": "srv_1",
                "name": "web_search",
                "content": {
                    "type": "web_search_tool_result_error",
                    "error_code": "max_uses_exceeded"
                }
            }))],
            stop_reason: Some("end_turn".into()),
            usage: ResponseUsage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
                iterations: None,
            },
            container: None,
            context_management: None,
        };
        let r = parse_response(resp, empty_headers(), None, vec![], false).unwrap();
        match &r.content[0] {
            Content::ToolResult(tr) => {
                assert!(matches!(tr.output, ToolResultOutput::ErrorJson { .. }));
                let pm = tr.provider_metadata.as_ref().expect("metadata present");
                assert_eq!(
                    pm.get("anthropic").and_then(|b| b.get("isError")),
                    Some(&JsonValue::Bool(true))
                );
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn web_search_tool_result_success_remains_json() {
        let resp = MessagesResponse {
            id: None,
            model: None,
            content: vec![ResponseContent::WebSearchToolResult(serde_json::json!({
                "tool_use_id": "srv_1",
                "name": "web_search",
                "content": [{"type": "web_search_result", "url": "https://x"}]
            }))],
            stop_reason: Some("end_turn".into()),
            usage: ResponseUsage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
                iterations: None,
            },
            container: None,
            context_management: None,
        };
        let r = parse_response(resp, empty_headers(), None, vec![], false).unwrap();
        match &r.content[0] {
            Content::ToolResult(tr) => {
                assert!(matches!(tr.output, ToolResultOutput::Json { .. }));
                assert!(tr.provider_metadata.is_none());
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn compaction_block_surfaced_as_text_with_anthropic_marker() {
        let resp = MessagesResponse {
            id: None,
            model: None,
            content: vec![ResponseContent::Compaction {
                content: Some("[earlier turns compacted]".into()),
            }],
            stop_reason: Some("end_turn".into()),
            usage: ResponseUsage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
                iterations: None,
            },
            container: None,
            context_management: None,
        };
        let r = parse_response(resp, empty_headers(), None, vec![], false).unwrap();
        match &r.content[0] {
            Content::Text(t) => {
                assert_eq!(t.text, "[earlier turns compacted]");
                let po = t.provider_options.as_ref().expect("provider_options set");
                assert_eq!(
                    po.get("anthropic").and_then(|b| b.get("type")),
                    Some(&JsonValue::String("compaction".into()))
                );
            }
            other => panic!("expected Text, got {other:?}"),
        }
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
                iterations: None,
            },
            container: None,
            context_management: None,
        };
        let err = parse_response(resp, empty_headers(), None, vec![], false).unwrap_err();
        assert!(format!("{err}").contains("no content"));
    }
}
