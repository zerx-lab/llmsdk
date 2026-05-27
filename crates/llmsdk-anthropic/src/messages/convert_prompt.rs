//! Convert an [`llmsdk_provider::language_model::Prompt`] into
//! `Anthropic` wire messages + the top-level `system` field.
//!
//! Mirrors `convert-to-anthropic-prompt.ts` (simplified for M6).
//!
//! Two structural shifts from the llmsdk shape:
//!
//! 1. **System collapse** — all [`Message::System`] entries are
//!    concatenated (`"\n\n"`-joined) and pulled into the request's
//!    top-level `system` field.
//! 2. **Tool result fold** — `Anthropic` has no `role:"tool"` message;
//!    each [`Message::Tool`] is folded into a user message whose only
//!    parts are `tool_result` blocks. Consecutive tool messages collapse
//!    into a single user message.
// Rust guideline compliant 2026-02-21

use llmsdk_provider::language_model::{
    AssistantPart, Message, Prompt, ToolCallPart, ToolMessagePart, ToolOutputPart,
    ToolResultOutput, ToolResultPart, UserPart,
};
use llmsdk_provider::shared::{FileBytes, FileData, Warning};

use super::wire::{
    CacheControl, CitationsConfig, WireAssistantPart, WireDocumentSource, WireImageSource,
    WireMessage, WireNestedToolResultContent, WireToolResultContent, WireUserPart,
};

/// Result of [`convert_prompt`].
pub(crate) struct Converted {
    pub system: Option<String>,
    pub messages: Vec<WireMessage>,
    pub warnings: Vec<Warning>,
    /// Beta tokens collected from message conversion (e.g. Files-API
    /// references require `files-api-2025-04-14`). Caller merges these
    /// into the request-level beta header.
    pub betas: std::collections::BTreeSet<String>,
}

/// Convert a prompt; collect warnings about dropped parts.
///
/// `send_reasoning` controls whether assistant `Reasoning` /
/// `ReasoningFile` parts are forwarded to the model. When `false`, both
/// types are silently dropped without warnings (matches ai-sdk semantics
/// for models that don't accept reasoning input).
pub(crate) fn convert_prompt(prompt: &Prompt, send_reasoning: bool) -> Converted {
    let mut systems: Vec<&str> = Vec::new();
    let mut messages: Vec<WireMessage> = Vec::new();
    let mut warnings: Vec<Warning> = Vec::new();
    let mut betas: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut validator = CacheControlValidator::new();

    for message in prompt {
        match message {
            Message::System { content, .. } => systems.push(content.as_str()),
            Message::User { content, .. } => {
                let parts = convert_user(content, &mut warnings, &mut betas, &mut validator);
                push_user(&mut messages, parts);
            }
            Message::Assistant {
                content,
                provider_options,
            } => {
                let parts = convert_assistant(
                    content,
                    provider_options.as_ref(),
                    send_reasoning,
                    &mut warnings,
                    &mut validator,
                );
                messages.push(WireMessage::Assistant { content: parts });
            }
            Message::Tool { content, .. } => {
                let parts = convert_tool(content, &mut warnings, &mut betas, &mut validator);
                push_user(&mut messages, parts);
            }
        }
    }

    let system = if systems.is_empty() {
        None
    } else {
        Some(systems.join("\n\n"))
    };

    warnings.append(&mut validator.warnings);

    Converted {
        system,
        messages,
        warnings,
        betas,
    }
}

/// Push a list of user parts onto `messages`, merging with the trailing
/// user message when present.
fn push_user(messages: &mut Vec<WireMessage>, mut parts: Vec<WireUserPart>) {
    if parts.is_empty() {
        return;
    }
    if let Some(WireMessage::User { content }) = messages.last_mut() {
        content.append(&mut parts);
    } else {
        messages.push(WireMessage::User { content: parts });
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "single dispatcher over all UserPart variants; splitting would scatter the wire-mapping logic"
)]
fn convert_user(
    parts: &[UserPart],
    warnings: &mut Vec<Warning>,
    betas: &mut std::collections::BTreeSet<String>,
    validator: &mut CacheControlValidator,
) -> Vec<WireUserPart> {
    let mut out = Vec::with_capacity(parts.len());
    for part in parts {
        match part {
            UserPart::Text(t) => out.push(WireUserPart::Text {
                text: t.text.clone(),
                cache_control: validator.get(
                    t.provider_options.as_ref(),
                    "user message part",
                    true,
                ),
            }),
            UserPart::File(f) => {
                let cache_control =
                    validator.get(f.provider_options.as_ref(), "user message part", true);
                let citations = read_citations_config(f.provider_options.as_ref());
                let (title, context) = read_document_meta(f.provider_options.as_ref());
                let top = f
                    .media_type
                    .split('/')
                    .next()
                    .unwrap_or(f.media_type.as_str());
                if top == "image" {
                    let source = match &f.data {
                        FileData::Url { url } => WireImageSource::Url { url: url.clone() },
                        FileData::Data { data } => WireImageSource::Base64 {
                            media_type: f.media_type.clone(),
                            data: file_bytes_to_base64(data),
                        },
                        FileData::Reference { reference } => {
                            if let Some(file_id) = resolve_anthropic_file_id(reference) {
                                betas.insert("files-api-2025-04-14".to_owned());
                                WireImageSource::File { file_id }
                            } else {
                                warnings.push(Warning::Unsupported {
                                    feature: "user.file.data".to_owned(),
                                    details: Some(
                                        "image file reference missing `anthropic` provider entry"
                                            .to_owned(),
                                    ),
                                });
                                continue;
                            }
                        }
                        FileData::Text { .. } => {
                            warnings.push(Warning::Unsupported {
                                feature: "user.file.data".to_owned(),
                                details: Some(
                                    "image files only accept Url or inline bytes".to_owned(),
                                ),
                            });
                            continue;
                        }
                    };
                    out.push(WireUserPart::Image {
                        source,
                        cache_control,
                    });
                    continue;
                }
                // Non-image: try document.
                let source = match (f.media_type.as_str(), &f.data) {
                    // Files-API reference works for any non-image document type.
                    (_, FileData::Reference { reference }) => {
                        if let Some(file_id) = resolve_anthropic_file_id(reference) {
                            betas.insert("files-api-2025-04-14".to_owned());
                            WireDocumentSource::File { file_id }
                        } else {
                            warnings.push(Warning::Unsupported {
                                feature: "user.file.data".to_owned(),
                                details: Some(
                                    "document file reference missing `anthropic` provider entry"
                                        .to_owned(),
                                ),
                            });
                            continue;
                        }
                    }
                    ("application/pdf", FileData::Url { url }) => WireDocumentSource::Url {
                        url: url.clone(),
                        media_type: "application/pdf".to_owned(),
                    },
                    ("application/pdf", FileData::Data { data }) => WireDocumentSource::Base64 {
                        media_type: "application/pdf".to_owned(),
                        data: file_bytes_to_base64(data),
                    },
                    ("text/plain", FileData::Url { url }) => WireDocumentSource::Url {
                        url: url.clone(),
                        media_type: "text/plain".to_owned(),
                    },
                    ("text/plain", FileData::Text { text }) => WireDocumentSource::Text {
                        media_type: "text/plain".to_owned(),
                        data: text.clone(),
                    },
                    ("text/plain", FileData::Data { data }) => WireDocumentSource::Base64 {
                        media_type: "text/plain".to_owned(),
                        data: file_bytes_to_base64(data),
                    },
                    (mt, _) if mt.starts_with("audio/") => {
                        warnings.push(Warning::Unsupported {
                            feature: "user.file".to_owned(),
                            details: Some(format!(
                                "Anthropic Messages API does not accept audio files ({mt})"
                            )),
                        });
                        continue;
                    }
                    (mt, _) => {
                        warnings.push(Warning::Unsupported {
                            feature: "user.file".to_owned(),
                            details: Some(format!(
                                "media_type '{mt}' is not supported by llmsdk-anthropic"
                            )),
                        });
                        continue;
                    }
                };
                out.push(WireUserPart::Document {
                    source,
                    title,
                    context,
                    citations,
                    cache_control,
                });
            }
        }
    }
    out
}

/// Pluck the Anthropic file id from a `FileData::Reference` map.
///
/// Mirrors `resolveProviderReference({ reference, provider: 'anthropic' })`
/// in `convert-to-anthropic-prompt.ts`.
fn resolve_anthropic_file_id(
    reference: &serde_json::Map<String, serde_json::Value>,
) -> Option<String> {
    reference
        .get("anthropic")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
}

/// Pluck a `cache_control` block from `provider_options["anthropic"]`.
///
/// Pure read; does not enforce the 4-breakpoint limit. Most call sites
/// should go through [`CacheControlValidator::get`] instead so warnings
/// are surfaced consistently with ai-sdk's `get-cache-control.ts`.
pub(super) fn read_cache_control(
    options: Option<&llmsdk_provider::shared::ProviderOptions>,
) -> Option<CacheControl> {
    let map = options?;
    let bucket = map.get("anthropic")?;
    let cc = bucket
        .get("cacheControl")
        .or_else(|| bucket.get("cache_control"))?;
    let kind = cc
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("ephemeral");
    let ttl = cc.get("ttl").and_then(|v| v.as_str()).map(str::to_owned);
    Some(CacheControl {
        kind: kind.to_owned(),
        ttl,
    })
}

/// Anthropic permits at most 4 cache-control breakpoints per request.
///
/// Mirrors `MAX_CACHE_BREAKPOINTS` from upstream `get-cache-control.ts`.
const MAX_CACHE_BREAKPOINTS: usize = 4;

/// Tracks cache breakpoint usage across one request conversion, emitting
/// warnings for over-the-limit or non-cacheable contexts.
///
/// Mirrors `CacheControlValidator` from upstream `get-cache-control.ts`.
pub(super) struct CacheControlValidator {
    count: usize,
    pub(super) warnings: Vec<Warning>,
}

impl CacheControlValidator {
    pub(super) fn new() -> Self {
        Self {
            count: 0,
            warnings: Vec::new(),
        }
    }

    /// Read `cache_control` from `provider_options`, returning None and pushing
    /// a warning when the breakpoint limit is reached or the context disallows
    /// caching.
    ///
    /// `context_type` is a short human-readable label ("user message part",
    /// "thinking block", …) used in the warning text. `can_cache` mirrors the
    /// upstream `canCache` flag — when false and a `cache_control` is requested,
    /// emit a warning and drop the value without counting toward the limit.
    pub(super) fn get(
        &mut self,
        options: Option<&llmsdk_provider::shared::ProviderOptions>,
        context_type: &str,
        can_cache: bool,
    ) -> Option<CacheControl> {
        let value = read_cache_control(options)?;
        if !can_cache {
            self.warnings.push(Warning::Unsupported {
                feature: "cache_control".to_owned(),
                details: Some(format!(
                    "cache_control cannot be set on {context_type}. It will be ignored."
                )),
            });
            return None;
        }
        self.count += 1;
        if self.count > MAX_CACHE_BREAKPOINTS {
            self.warnings.push(Warning::Unsupported {
                feature: "cache_control".to_owned(),
                details: Some(format!(
                    "Maximum {MAX_CACHE_BREAKPOINTS} cache breakpoints exceeded (found {count}). This breakpoint will be ignored.",
                    count = self.count,
                )),
            });
            return None;
        }
        Some(value)
    }
}

/// Pluck a `citations` config from `provider_options["anthropic"]`.
fn read_citations_config(
    options: Option<&llmsdk_provider::shared::ProviderOptions>,
) -> Option<CitationsConfig> {
    let map = options?;
    let bucket = map.get("anthropic")?;
    let c = bucket.get("citations")?;
    let enabled = c
        .get("enabled")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    Some(CitationsConfig { enabled })
}

/// Pluck `title` / `context` for document blocks.
fn read_document_meta(
    options: Option<&llmsdk_provider::shared::ProviderOptions>,
) -> (Option<String>, Option<String>) {
    let Some(map) = options else {
        return (None, None);
    };
    let Some(bucket) = map.get("anthropic") else {
        return (None, None);
    };
    let title = bucket
        .get("title")
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    let context = bucket
        .get("context")
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    (title, context)
}

fn convert_assistant(
    parts: &[AssistantPart],
    message_provider_options: Option<&llmsdk_provider::shared::ProviderOptions>,
    send_reasoning: bool,
    warnings: &mut Vec<Warning>,
    validator: &mut CacheControlValidator,
) -> Vec<WireAssistantPart> {
    let mut out = Vec::with_capacity(parts.len());
    let parts_count = parts.len();
    for (idx, part) in parts.iter().enumerate() {
        // Cache-control fallback chain mirrors upstream
        // convert-to-anthropic-prompt.ts:548-558:
        // 1. part.providerOptions  → "assistant message part"
        // 2. last part only: message.providerOptions  → "assistant message"
        let is_last = idx + 1 == parts_count;
        let part_provider_options = assistant_part_provider_options(part);
        let part_cc = validator.get(part_provider_options, "assistant message part", true);
        let cache_control = if part_cc.is_some() {
            part_cc
        } else if is_last {
            validator.get(message_provider_options, "assistant message", true)
        } else {
            None
        };

        match part {
            AssistantPart::Text(t) => out.push(WireAssistantPart::Text {
                text: t.text.clone(),
                cache_control,
            }),
            AssistantPart::ToolCall(tc) => {
                if let Some(part) = convert_tool_call(tc, cache_control, warnings) {
                    out.push(part);
                }
            }
            AssistantPart::Reasoning {
                text,
                provider_options,
            } => {
                if !send_reasoning {
                    continue;
                }
                // Thinking blocks cannot carry cache_control. If the user
                // supplied one, validator.get with can_cache=false records
                // a warning and drops it.
                let _ = validator.get(provider_options.as_ref(), "thinking block", false);
                let (signature, redacted_data) = extract_thinking_meta(provider_options.as_ref());
                if let Some(data) = redacted_data {
                    out.push(WireAssistantPart::RedactedThinking { data });
                } else {
                    out.push(WireAssistantPart::Thinking {
                        thinking: text.clone(),
                        signature,
                    });
                }
            }
            AssistantPart::ReasoningFile { .. } => {
                if !send_reasoning {
                    continue;
                }
                warnings.push(Warning::Unsupported {
                    feature: "assistant.reasoning-file".to_owned(),
                    details: Some("Anthropic does not support reasoning files".to_owned()),
                });
            }
            AssistantPart::File(_) => warnings.push(Warning::Unsupported {
                feature: "assistant.file".to_owned(),
                details: None,
            }),
            AssistantPart::Custom { kind, .. } => warnings.push(Warning::Unsupported {
                feature: format!("assistant.custom.{kind}"),
                details: None,
            }),
            AssistantPart::ToolResult(r) => {
                if let Some(part) = convert_assistant_tool_result(r, cache_control, warnings) {
                    out.push(part);
                }
            }
        }
    }
    out
}

/// Pull `signature` / `redactedData` from `provider_options["anthropic"]`.
///
/// Returns `(signature, redacted_data)`; both are `None` when the slot is
/// absent or empty.
fn extract_thinking_meta(
    options: Option<&llmsdk_provider::shared::ProviderOptions>,
) -> (Option<String>, Option<String>) {
    let Some(map) = options else {
        return (None, None);
    };
    let Some(anthropic) = map.get("anthropic") else {
        return (None, None);
    };
    let signature = anthropic
        .get("signature")
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    let redacted_data = anthropic
        .get("redactedData")
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    (signature, redacted_data)
}

fn convert_tool_call(
    tc: &ToolCallPart,
    cache_control: Option<CacheControl>,
    warnings: &mut Vec<llmsdk_provider::shared::Warning>,
) -> Option<WireAssistantPart> {
    // Ensure input is always an object (Anthropic requires JSON-typed input).
    let input = if tc.input.is_null() {
        serde_json::Value::Object(serde_json::Map::new())
    } else {
        tc.input.clone()
    };
    if tc.provider_executed != Some(true) {
        return Some(WireAssistantPart::ToolUse {
            id: tc.tool_call_id.clone(),
            name: tc.tool_name.clone(),
            input,
            cache_control,
        });
    }

    // Mirror upstream `convert-to-anthropic-prompt.ts:647-756` provider-executed
    // tool-call routing. `provider_tool_name` defaults to the tool's name —
    // unlike upstream we don't apply a custom→provider rename map (callers
    // who renamed a typed Anthropic tool will have to set the original name
    // back when echoing the tool-call into a follow-up prompt).
    let provider_tool_name = tc.tool_name.as_str();
    let anthropic_opts = tc
        .provider_options
        .as_ref()
        .and_then(|opts| opts.get("anthropic"));
    let is_mcp = anthropic_opts
        .and_then(|m| m.get("type"))
        .and_then(serde_json::Value::as_str)
        == Some("mcp-tool-use");

    if is_mcp {
        let server_name = anthropic_opts
            .and_then(|m| m.get("serverName"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);
        let Some(server_name) = server_name else {
            warnings.push(llmsdk_provider::shared::Warning::Other {
                message: "mcp tool use server name is required and must be a string".to_owned(),
            });
            return None;
        };
        return Some(WireAssistantPart::McpToolUse {
            id: tc.tool_call_id.clone(),
            name: tc.tool_name.clone(),
            input,
            server_name,
            cache_control,
        });
    }

    // `code_execution_20250825` introduces an envelope with an inner `type`
    // tag that determines the wire `name` (mirrors upstream 677-713).
    if provider_tool_name == "code_execution"
        && let Some(input_obj) = input.as_object()
        && let Some(inner_type) = input_obj.get("type").and_then(serde_json::Value::as_str)
    {
        match inner_type {
            "bash_code_execution" | "text_editor_code_execution" => {
                return Some(WireAssistantPart::ServerToolUse {
                    id: tc.tool_call_id.clone(),
                    name: inner_type.to_owned(),
                    input: input.clone(),
                    cache_control,
                });
            }
            "programmatic-tool-call" => {
                let mut sanitized = input_obj.clone();
                sanitized.remove("type");
                return Some(WireAssistantPart::ServerToolUse {
                    id: tc.tool_call_id.clone(),
                    name: "code_execution".to_owned(),
                    input: serde_json::Value::Object(sanitized),
                    cache_control,
                });
            }
            _ => {}
        }
    }

    match provider_tool_name {
        "code_execution"
        | "web_fetch"
        | "web_search"
        | "tool_search_tool_regex"
        | "tool_search_tool_bm25" => Some(WireAssistantPart::ServerToolUse {
            id: tc.tool_call_id.clone(),
            name: provider_tool_name.to_owned(),
            input,
            cache_control,
        }),
        "advisor" => Some(WireAssistantPart::ServerToolUse {
            id: tc.tool_call_id.clone(),
            name: "advisor".to_owned(),
            input: serde_json::Value::Object(serde_json::Map::new()),
            cache_control,
        }),
        other => {
            warnings.push(llmsdk_provider::shared::Warning::Other {
                message: format!("provider executed tool call for tool {other} is not supported"),
            });
            None
        }
    }
}

/// Convert an inline `AssistantPart::ToolResult` echoed back from an earlier
/// provider-executed tool call.
///
/// Mirrors upstream `convert-to-anthropic-prompt.ts:789-1185` `case
/// 'tool-result'` (assistant scope). Each provider-executed tool gets its own
/// wire `*_tool_result` block; unsupported tools emit a warning and return
/// `None`.
fn convert_assistant_tool_result(
    part: &ToolResultPart,
    cache_control: Option<CacheControl>,
    warnings: &mut Vec<Warning>,
) -> Option<WireAssistantPart> {
    let provider_tool_name = part.tool_name.as_str();
    let tool_use_id = part.tool_call_id.clone();

    // MCP tool result detection: upstream stashes `tool_use_id`s into a
    // `mcpToolUseIds` set while emitting `mcp_tool_use`. Rust echoes the
    // same hint through `provider_options.anthropic.type = "mcp-tool-use"`
    // on the matching ToolCallPart; for `ToolResultPart` we rely on the
    // output shape (only json / error-json are valid for MCP) plus the
    // `provider_options.anthropic.type = "mcp-tool-result"` marker if set.
    let anthropic_opts = part
        .provider_options
        .as_ref()
        .and_then(|opts| opts.get("anthropic"));
    let is_mcp = anthropic_opts
        .and_then(|m| m.get("type"))
        .and_then(serde_json::Value::as_str)
        == Some("mcp-tool-result");

    if is_mcp {
        return convert_mcp_tool_result(part, tool_use_id, cache_control, warnings);
    }

    match provider_tool_name {
        "code_execution" => {
            convert_code_execution_tool_result(part, tool_use_id, cache_control, warnings)
        }
        "web_fetch" => convert_web_fetch_tool_result(part, tool_use_id, cache_control, warnings),
        "web_search" => convert_web_search_tool_result(part, tool_use_id, cache_control, warnings),
        "tool_search_tool_regex" | "tool_search_tool_bm25" => {
            convert_tool_search_tool_result(part, tool_use_id, cache_control, warnings)
        }
        "advisor" => convert_advisor_tool_result(part, tool_use_id, cache_control, warnings),
        other => {
            warnings.push(Warning::Other {
                message: format!("provider executed tool result for tool {other} is not supported"),
            });
            None
        }
    }
}

/// Parse `output.value` for provider-executed error payloads. Accepts both
/// stringified-JSON and plain object forms. Mirrors upstream
/// `extractErrorValue` (`convert-to-anthropic-prompt.ts:46-63`).
fn extract_error_value(value: &serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
    if let Some(s) = value.as_str() {
        if let Ok(serde_json::Value::Object(map)) = serde_json::from_str::<serde_json::Value>(s) {
            return map;
        }
        return serde_json::Map::new();
    }
    if let Some(map) = value.as_object() {
        return map.clone();
    }
    serde_json::Map::new()
}

fn extract_error_code(value: &serde_json::Value) -> String {
    extract_error_value(value)
        .get("errorCode")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unavailable")
        .to_owned()
}

fn convert_mcp_tool_result(
    part: &ToolResultPart,
    tool_use_id: String,
    cache_control: Option<CacheControl>,
    warnings: &mut Vec<Warning>,
) -> Option<WireAssistantPart> {
    let (content, is_error) = match &part.output {
        ToolResultOutput::Json { value, .. } => (value.clone(), false),
        ToolResultOutput::ErrorJson { value, .. } => (value.clone(), true),
        _ => {
            warnings.push(Warning::Other {
                message: format!(
                    "provider executed tool result output type for tool {} is not supported",
                    part.tool_name
                ),
            });
            return None;
        }
    };
    Some(WireAssistantPart::McpToolResult {
        tool_use_id,
        is_error,
        content,
        cache_control,
    })
}

fn convert_code_execution_tool_result(
    part: &ToolResultPart,
    tool_use_id: String,
    cache_control: Option<CacheControl>,
    warnings: &mut Vec<Warning>,
) -> Option<WireAssistantPart> {
    use serde_json::json;

    let output_value = match &part.output {
        ToolResultOutput::ErrorText { .. } | ToolResultOutput::ErrorJson { .. } => {
            // Upstream tries to parse the value for an inner `type` so it can
            // pick between `code_execution_tool_result_error` and
            // `bash_code_execution_tool_result_error`. Mirrors
            // `convert-to-anthropic-prompt.ts:823-855`.
            let parsed = match &part.output {
                ToolResultOutput::ErrorText { value, .. } => {
                    serde_json::from_str::<serde_json::Value>(value.as_str())
                        .unwrap_or(serde_json::Value::Null)
                }
                ToolResultOutput::ErrorJson { value, .. } => value.clone(),
                _ => serde_json::Value::Null,
            };
            let map = extract_error_value(&parsed);
            let inner_type = map
                .get("type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            let error_code = map
                .get("errorCode")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown");
            return if inner_type == "code_execution_tool_result_error" {
                Some(WireAssistantPart::CodeExecutionToolResult {
                    tool_use_id,
                    content: json!({
                        "type": "code_execution_tool_result_error",
                        "error_code": error_code,
                    }),
                    cache_control,
                })
            } else {
                Some(WireAssistantPart::BashCodeExecutionToolResult {
                    tool_use_id,
                    content: json!({
                        "type": "bash_code_execution_tool_result_error",
                        "error_code": error_code,
                    }),
                    cache_control,
                })
            };
        }
        ToolResultOutput::Json { value, .. } => value.clone(),
        _ => {
            warnings.push(Warning::Other {
                message: format!(
                    "provider executed tool result output type for tool {} is not supported",
                    part.tool_name
                ),
            });
            return None;
        }
    };

    // Dispatch on the inner `type` field. Mirrors upstream
    // `convert-to-anthropic-prompt.ts:880-970`. We do not run a full schema
    // validator (upstream uses `validateTypes`) — Anthropic returns
    // well-formed payloads, and partial echoes from clients should also be
    // forwarded verbatim.
    let inner_type = output_value
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    match inner_type {
        // 20250522 envelope + 20260120 encrypted envelope — both ride the
        // `code_execution_tool_result` wire block; the consumer disambiguates
        // via the inner `type` field.
        "code_execution_result" | "encrypted_code_execution_result" => {
            Some(WireAssistantPart::CodeExecutionToolResult {
                tool_use_id,
                content: output_value,
                cache_control,
            })
        }
        // 20250825 bash subtool.
        "bash_code_execution_result" | "bash_code_execution_tool_result_error" => {
            Some(WireAssistantPart::BashCodeExecutionToolResult {
                tool_use_id,
                content: output_value,
                cache_control,
            })
        }
        // 20250825 text-editor subtool (any of view/create/str_replace/error).
        _ => Some(WireAssistantPart::TextEditorCodeExecutionToolResult {
            tool_use_id,
            content: output_value,
            cache_control,
        }),
    }
}

fn convert_web_fetch_tool_result(
    part: &ToolResultPart,
    tool_use_id: String,
    cache_control: Option<CacheControl>,
    warnings: &mut Vec<Warning>,
) -> Option<WireAssistantPart> {
    use serde_json::json;

    match &part.output {
        ToolResultOutput::ErrorJson { value, .. } => Some(WireAssistantPart::WebFetchToolResult {
            tool_use_id,
            content: json!({
                "type": "web_fetch_tool_result_error",
                "error_code": extract_error_code(value),
            }),
            cache_control,
        }),
        ToolResultOutput::Json { value, .. } => {
            // The upstream schema reshapes the camelCase output into wire
            // snake_case. We assume the caller stored the wire-shape (or a
            // best-effort merge) and forward verbatim. This mirrors the
            // permissive read-side semantics already used elsewhere.
            Some(WireAssistantPart::WebFetchToolResult {
                tool_use_id,
                content: value.clone(),
                cache_control,
            })
        }
        _ => {
            warnings.push(Warning::Other {
                message: format!(
                    "provider executed tool result output type for tool {} is not supported",
                    part.tool_name
                ),
            });
            None
        }
    }
}

fn convert_web_search_tool_result(
    part: &ToolResultPart,
    tool_use_id: String,
    cache_control: Option<CacheControl>,
    warnings: &mut Vec<Warning>,
) -> Option<WireAssistantPart> {
    use serde_json::json;

    match &part.output {
        ToolResultOutput::ErrorJson { value, .. } => Some(WireAssistantPart::WebSearchToolResult {
            tool_use_id,
            content: json!({
                "type": "web_search_tool_result_error",
                "error_code": extract_error_code(value),
            }),
            cache_control,
        }),
        ToolResultOutput::Json { value, .. } => Some(WireAssistantPart::WebSearchToolResult {
            tool_use_id,
            content: value.clone(),
            cache_control,
        }),
        _ => {
            warnings.push(Warning::Other {
                message: format!(
                    "provider executed tool result output type for tool {} is not supported",
                    part.tool_name
                ),
            });
            None
        }
    }
}

fn convert_tool_search_tool_result(
    part: &ToolResultPart,
    tool_use_id: String,
    cache_control: Option<CacheControl>,
    warnings: &mut Vec<Warning>,
) -> Option<WireAssistantPart> {
    use serde_json::json;

    let value = if let ToolResultOutput::Json { value, .. } = &part.output {
        value.clone()
    } else {
        warnings.push(Warning::Other {
            message: format!(
                "provider executed tool result output type for tool {} is not supported",
                part.tool_name
            ),
        });
        return None;
    };

    // Tool references — upstream parses `[{toolName: ...}]` into wire
    // `[{type: "tool_reference", tool_name: ...}]`. We accept either shape:
    // if the input is already in wire form, forward verbatim; otherwise
    // rewrite `toolName` → `tool_name`.
    let tool_references: Vec<serde_json::Value> = value
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    let obj = item.as_object()?;
                    let tool_name = obj
                        .get("tool_name")
                        .or_else(|| obj.get("toolName"))
                        .and_then(serde_json::Value::as_str)?
                        .to_owned();
                    Some(json!({
                        "type": "tool_reference",
                        "tool_name": tool_name,
                    }))
                })
                .collect()
        })
        .unwrap_or_default();

    Some(WireAssistantPart::ToolSearchToolResult {
        tool_use_id,
        content: json!({
            "type": "tool_search_tool_search_result",
            "tool_references": tool_references,
        }),
        cache_control,
    })
}

fn convert_advisor_tool_result(
    part: &ToolResultPart,
    tool_use_id: String,
    cache_control: Option<CacheControl>,
    warnings: &mut Vec<Warning>,
) -> Option<WireAssistantPart> {
    use serde_json::json;

    let value = match &part.output {
        ToolResultOutput::Json { value, .. } | ToolResultOutput::ErrorJson { value, .. } => {
            value.clone()
        }
        _ => {
            warnings.push(Warning::Other {
                message: format!(
                    "provider executed tool result output type for tool {} is not supported",
                    part.tool_name
                ),
            });
            return None;
        }
    };
    let inner_type = value
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let content = match inner_type {
        "advisor_result" => {
            let text = value
                .get("text")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_owned();
            json!({ "type": "advisor_result", "text": text })
        }
        "advisor_redacted_result" => {
            let encrypted_content = value
                .get("encryptedContent")
                .or_else(|| value.get("encrypted_content"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_owned();
            json!({
                "type": "advisor_redacted_result",
                "encrypted_content": encrypted_content,
            })
        }
        _ => {
            // Treat as error envelope.
            json!({
                "type": "advisor_tool_result_error",
                "error_code": value
                    .get("errorCode")
                    .or_else(|| value.get("error_code"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown"),
            })
        }
    };
    Some(WireAssistantPart::AdvisorToolResult {
        tool_use_id,
        content,
        cache_control,
    })
}

/// Pluck `provider_options` from an `AssistantPart` variant, regardless of
/// the variant's shape.
fn assistant_part_provider_options(
    part: &AssistantPart,
) -> Option<&llmsdk_provider::shared::ProviderOptions> {
    match part {
        AssistantPart::Text(t) => t.provider_options.as_ref(),
        AssistantPart::ToolCall(tc) => tc.provider_options.as_ref(),
        AssistantPart::Reasoning {
            provider_options, ..
        }
        | AssistantPart::ReasoningFile {
            provider_options, ..
        }
        | AssistantPart::Custom {
            provider_options, ..
        } => provider_options.as_ref(),
        AssistantPart::File(f) => f.provider_options.as_ref(),
        AssistantPart::ToolResult(r) => r.provider_options.as_ref(),
    }
}

fn convert_tool(
    parts: &[ToolMessagePart],
    warnings: &mut Vec<Warning>,
    betas: &mut std::collections::BTreeSet<String>,
    validator: &mut CacheControlValidator,
) -> Vec<WireUserPart> {
    let mut out = Vec::with_capacity(parts.len());
    for part in parts {
        match part {
            ToolMessagePart::ToolResult(r) => {
                let (content, is_error) = tool_result_to_content(r, warnings, betas, validator);
                // Cache control fallback chain matches upstream
                // convert-to-anthropic-prompt.ts:362-376:
                // 1. part.providerOptions (tool result part)
                // 2. output.providerOptions / nested content[].providerOptions
                //    (tool result output) — handled inside
                //    `tool_result_to_content`, but we also need to consider it
                //    here as the message-level cache_control fallback.
                let part_cc = validator.get(r.provider_options.as_ref(), "tool result part", true);
                let output_cc = if part_cc.is_none() {
                    let output_opts = output_provider_options(&r.output);
                    validator.get(output_opts, "tool result output", true)
                } else {
                    None
                };
                out.push(WireUserPart::ToolResult {
                    tool_use_id: r.tool_call_id.clone(),
                    content,
                    is_error,
                    cache_control: part_cc.or(output_cc),
                });
            }
            ToolMessagePart::ToolApprovalResponse(_) => {
                warnings.push(Warning::Unsupported {
                    feature: "feature.approval-response".to_owned(),
                    details: Some("M6 does not relay approval responses".to_owned()),
                });
            }
        }
    }
    out
}

/// Pluck `provider_options` from a `ToolResultOutput` variant.
///
/// Mirrors the `output.providerOptions` / nested `content[].providerOptions`
/// fallback in upstream `convert-to-anthropic-prompt.ts:347-356`.
fn output_provider_options(
    output: &ToolResultOutput,
) -> Option<&llmsdk_provider::shared::ProviderOptions> {
    match output {
        ToolResultOutput::Text {
            provider_options, ..
        }
        | ToolResultOutput::Json {
            provider_options, ..
        }
        | ToolResultOutput::ErrorText {
            provider_options, ..
        }
        | ToolResultOutput::ErrorJson {
            provider_options, ..
        }
        | ToolResultOutput::ExecutionDenied {
            provider_options, ..
        } => provider_options.as_ref(),
        ToolResultOutput::Content { value } => value.iter().find_map(|p| match p {
            ToolOutputPart::Text {
                provider_options, ..
            }
            | ToolOutputPart::File {
                provider_options, ..
            }
            | ToolOutputPart::Custom {
                provider_options, ..
            } => provider_options.as_ref(),
        }),
    }
}

fn tool_result_to_content(
    part: &ToolResultPart,
    warnings: &mut Vec<Warning>,
    betas: &mut std::collections::BTreeSet<String>,
    validator: &mut CacheControlValidator,
) -> (WireToolResultContent, Option<bool>) {
    match &part.output {
        ToolResultOutput::Text { value, .. } => (WireToolResultContent::Text(value.clone()), None),
        ToolResultOutput::Json { value, .. } => (
            WireToolResultContent::Text(
                serde_json::to_string(value).unwrap_or_else(|_| "{}".to_owned()),
            ),
            None,
        ),
        ToolResultOutput::ErrorText { value, .. } => {
            (WireToolResultContent::Text(value.clone()), Some(true))
        }
        ToolResultOutput::ErrorJson { value, .. } => (
            WireToolResultContent::Text(
                serde_json::to_string(value).unwrap_or_else(|_| "{}".to_owned()),
            ),
            Some(true),
        ),
        ToolResultOutput::ExecutionDenied { reason, .. } => (
            WireToolResultContent::Text(
                reason
                    .clone()
                    .unwrap_or_else(|| "Tool call execution denied.".to_owned()),
            ),
            Some(true),
        ),
        ToolResultOutput::Content { value } => {
            let parts: Vec<WireNestedToolResultContent> = value
                .iter()
                .filter_map(|p| convert_nested_tool_output(p, warnings, betas, validator))
                .collect();
            (WireToolResultContent::Parts(parts), None)
        }
    }
}

/// Map one [`ToolOutputPart`] to its nested wire counterpart.
///
/// Returns `None` (with a warning) for unsupported parts, matching ai-sdk's
/// `unsupported tool content part` warning path.
fn convert_nested_tool_output(
    part: &ToolOutputPart,
    warnings: &mut Vec<Warning>,
    betas: &mut std::collections::BTreeSet<String>,
    validator: &mut CacheControlValidator,
) -> Option<WireNestedToolResultContent> {
    match part {
        ToolOutputPart::Text {
            text,
            provider_options,
        } => Some(WireNestedToolResultContent::Text {
            text: text.clone(),
            cache_control: validator.get(provider_options.as_ref(), "tool result output", true),
        }),
        ToolOutputPart::File {
            data,
            media_type,
            provider_options,
            ..
        } => convert_nested_file(
            data,
            media_type,
            provider_options.as_ref(),
            warnings,
            betas,
            validator,
        ),
        ToolOutputPart::Custom { provider_options } => {
            convert_nested_custom(provider_options.as_ref(), warnings)
        }
    }
}

/// `ToolOutputPart::File` → nested wire image / document.
fn convert_nested_file(
    data: &FileData,
    media_type: &str,
    provider_options: Option<&llmsdk_provider::shared::ProviderOptions>,
    warnings: &mut Vec<Warning>,
    betas: &mut std::collections::BTreeSet<String>,
    validator: &mut CacheControlValidator,
) -> Option<WireNestedToolResultContent> {
    let cache_control = validator.get(provider_options, "tool result output", true);
    let top = media_type.split('/').next().unwrap_or(media_type);
    if top == "image" {
        let source = match data {
            FileData::Url { url } => WireImageSource::Url { url: url.clone() },
            FileData::Data { data } => WireImageSource::Base64 {
                media_type: media_type.to_owned(),
                data: file_bytes_to_base64(data),
            },
            FileData::Reference { reference } => {
                if let Some(file_id) = resolve_anthropic_file_id(reference) {
                    betas.insert("files-api-2025-04-14".to_owned());
                    WireImageSource::File { file_id }
                } else {
                    warnings.push(Warning::Unsupported {
                        feature: "feature-result.content.file.data".to_owned(),
                        details: Some(
                            "image reference missing `anthropic` provider entry".to_owned(),
                        ),
                    });
                    return None;
                }
            }
            FileData::Text { .. } => {
                warnings.push(Warning::Unsupported {
                    feature: "feature-result.content.file.data".to_owned(),
                    details: Some(
                        "image files in tool_result accept only Url, inline bytes, or Files-API references".to_owned(),
                    ),
                });
                return None;
            }
        };
        return Some(WireNestedToolResultContent::Image {
            source,
            cache_control,
        });
    }
    let (title, context) = read_document_meta(provider_options);
    let citations = read_citations_config(provider_options);
    let source = match (media_type, data) {
        (_, FileData::Reference { reference }) => {
            if let Some(file_id) = resolve_anthropic_file_id(reference) {
                betas.insert("files-api-2025-04-14".to_owned());
                WireDocumentSource::File { file_id }
            } else {
                warnings.push(Warning::Unsupported {
                    feature: "feature-result.content.file.data".to_owned(),
                    details: Some(
                        "document reference missing `anthropic` provider entry".to_owned(),
                    ),
                });
                return None;
            }
        }
        ("application/pdf", FileData::Url { url }) => WireDocumentSource::Url {
            url: url.clone(),
            media_type: "application/pdf".to_owned(),
        },
        ("application/pdf", FileData::Data { data }) => WireDocumentSource::Base64 {
            media_type: "application/pdf".to_owned(),
            data: file_bytes_to_base64(data),
        },
        ("text/plain", FileData::Url { url }) => WireDocumentSource::Url {
            url: url.clone(),
            media_type: "text/plain".to_owned(),
        },
        ("text/plain", FileData::Text { text }) => WireDocumentSource::Text {
            media_type: "text/plain".to_owned(),
            data: text.clone(),
        },
        ("text/plain", FileData::Data { data }) => WireDocumentSource::Base64 {
            media_type: "text/plain".to_owned(),
            data: file_bytes_to_base64(data),
        },
        (mt, _) => {
            warnings.push(Warning::Unsupported {
                feature: "feature-result.content.file".to_owned(),
                details: Some(format!(
                    "media_type '{mt}' not supported as tool_result nested content"
                )),
            });
            return None;
        }
    };
    Some(WireNestedToolResultContent::Document {
        source,
        title,
        context,
        citations,
    })
}

/// `ToolOutputPart::Custom` → nested wire `tool_reference`, or a warning.
fn convert_nested_custom(
    provider_options: Option<&llmsdk_provider::shared::ProviderOptions>,
    warnings: &mut Vec<Warning>,
) -> Option<WireNestedToolResultContent> {
    // `tool-reference`: emit `{type:"tool_reference", tool_name}` for the
    // tool_search server-tool path. Mirrors upstream
    // convert-to-anthropic-prompt.ts `custom` branch.
    let anthropic = provider_options.and_then(|o| o.get("anthropic"));
    let kind = anthropic
        .and_then(|a| a.get("type"))
        .and_then(|v| v.as_str());
    if kind == Some("tool-reference") {
        if let Some(name) = anthropic
            .and_then(|a| a.get("toolName"))
            .and_then(|v| v.as_str())
        {
            return Some(WireNestedToolResultContent::ToolReference {
                tool_name: name.to_owned(),
            });
        }
        warnings.push(Warning::Unsupported {
            feature: "feature-result.content.custom.tool-reference".to_owned(),
            details: Some("tool-reference requires anthropic.toolName".to_owned()),
        });
        return None;
    }
    warnings.push(Warning::Unsupported {
        feature: "feature-result.content.custom".to_owned(),
        details: None,
    });
    None
}

fn file_bytes_to_base64(bytes: &FileBytes) -> String {
    match bytes {
        FileBytes::Base64(s) => s.clone(),
        FileBytes::Bytes(b) => base64_encode(b),
    }
}

/// Minimal RFC 4648 base64 encoder (same logic as the `OpenAI` provider's
/// copy — kept private to avoid leaking a public re-export).
fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    let mut chunks = bytes.chunks_exact(3);
    for chunk in &mut chunks {
        let n = (u32::from(chunk[0]) << 16) | (u32::from(chunk[1]) << 8) | u32::from(chunk[2]);
        out.push(ALPHABET[((n >> 18) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 6) & 0x3F) as usize] as char);
        out.push(ALPHABET[(n & 0x3F) as usize] as char);
    }
    let rem = chunks.remainder();
    match rem.len() {
        1 => {
            let n = u32::from(rem[0]) << 16;
            out.push(ALPHABET[((n >> 18) & 0x3F) as usize] as char);
            out.push(ALPHABET[((n >> 12) & 0x3F) as usize] as char);
            out.push('=');
            out.push('=');
        }
        2 => {
            let n = (u32::from(rem[0]) << 16) | (u32::from(rem[1]) << 8);
            out.push(ALPHABET[((n >> 18) & 0x3F) as usize] as char);
            out.push(ALPHABET[((n >> 12) & 0x3F) as usize] as char);
            out.push(ALPHABET[((n >> 6) & 0x3F) as usize] as char);
            out.push('=');
        }
        _ => {}
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmsdk_provider::language_model::TextPart;

    #[test]
    fn systems_concatenate_into_top_level() {
        let prompt = vec![
            Message::System {
                content: "First instruction.".into(),
                provider_options: None,
            },
            Message::System {
                content: "Second instruction.".into(),
                provider_options: None,
            },
            Message::User {
                content: vec![UserPart::Text(TextPart {
                    text: "hi".into(),
                    provider_options: None,
                })],
                provider_options: None,
            },
        ];
        let out = convert_prompt(&prompt, true);
        assert_eq!(
            out.system.as_deref(),
            Some("First instruction.\n\nSecond instruction.")
        );
        assert_eq!(out.messages.len(), 1);
    }

    #[test]
    fn tool_message_folds_into_user() {
        let prompt = vec![
            Message::User {
                content: vec![UserPart::Text(TextPart {
                    text: "weather?".into(),
                    provider_options: None,
                })],
                provider_options: None,
            },
            Message::Assistant {
                content: vec![AssistantPart::ToolCall(ToolCallPart {
                    tool_call_id: "tu_1".into(),
                    tool_name: "weather".into(),
                    input: serde_json::json!({"city": "NYC"}),
                    provider_executed: None,
                    dynamic: None,
                    provider_options: None,
                })],
                provider_options: None,
            },
            Message::Tool {
                content: vec![ToolMessagePart::ToolResult(ToolResultPart {
                    tool_call_id: "tu_1".into(),
                    tool_name: "weather".into(),
                    output: ToolResultOutput::Text {
                        value: "Sunny".into(),
                        provider_options: None,
                    },
                    provider_options: None,
                })],
                provider_options: None,
            },
        ];
        let out = convert_prompt(&prompt, true);
        assert_eq!(out.messages.len(), 3);
        // Last message must be a User with a single tool_result part.
        if let WireMessage::User { content } = &out.messages[2]
            && let WireUserPart::ToolResult {
                tool_use_id,
                content: WireToolResultContent::Text(text),
                ..
            } = &content[0]
        {
            assert_eq!(tool_use_id, "tu_1");
            assert_eq!(text, "Sunny");
        } else {
            panic!("expected user/tool_result, got {:?}", out.messages[2]);
        }
    }

    #[test]
    fn consecutive_tool_messages_coalesce() {
        let mk_tool = |id: &str, text: &str| Message::Tool {
            content: vec![ToolMessagePart::ToolResult(ToolResultPart {
                tool_call_id: id.into(),
                tool_name: "x".into(),
                output: ToolResultOutput::Text {
                    value: text.into(),
                    provider_options: None,
                },
                provider_options: None,
            })],
            provider_options: None,
        };
        let prompt = vec![mk_tool("a", "one"), mk_tool("b", "two")];
        let out = convert_prompt(&prompt, true);
        assert_eq!(out.messages.len(), 1);
        if let WireMessage::User { content } = &out.messages[0] {
            assert_eq!(content.len(), 2);
        }
    }

    #[test]
    fn pdf_file_becomes_document_block() {
        let prompt = vec![Message::User {
            content: vec![UserPart::File(llmsdk_provider::language_model::FilePart {
                filename: None,
                data: FileData::Url {
                    url: "https://example.com/a.pdf".into(),
                },
                media_type: "application/pdf".into(),
                provider_options: None,
            })],
            provider_options: None,
        }];
        let out = convert_prompt(&prompt, true);
        assert!(out.warnings.is_empty(), "PDF is supported, no warning");
        if let WireMessage::User { content } = &out.messages[0] {
            assert!(matches!(content[0], WireUserPart::Document { .. }));
        } else {
            panic!("expected user message");
        }
    }

    #[test]
    fn unsupported_audio_file_warns_and_drops() {
        let prompt = vec![Message::User {
            content: vec![UserPart::File(llmsdk_provider::language_model::FilePart {
                filename: None,
                data: FileData::Url {
                    url: "https://example.com/a.mp3".into(),
                },
                media_type: "audio/mpeg".into(),
                provider_options: None,
            })],
            provider_options: None,
        }];
        let out = convert_prompt(&prompt, true);
        assert_eq!(out.warnings.len(), 1);
        assert!(out.messages.is_empty());
    }

    #[test]
    fn tool_result_content_with_tool_reference() {
        use llmsdk_provider::language_model::ToolOutputPart;
        use llmsdk_provider::shared::ProviderOptions;

        let mut po = ProviderOptions::new();
        po.insert(
            "anthropic".into(),
            serde_json::json!({
                "type": "tool-reference",
                "toolName": "get_weather"
            })
            .as_object()
            .unwrap()
            .clone(),
        );

        let prompt = vec![Message::Tool {
            content: vec![ToolMessagePart::ToolResult(ToolResultPart {
                tool_call_id: "srvtoolu_1".into(),
                tool_name: "tool_search_tool_regex".into(),
                output: ToolResultOutput::Content {
                    value: vec![ToolOutputPart::Custom {
                        provider_options: Some(po),
                    }],
                },
                provider_options: None,
            })],
            provider_options: None,
        }];

        let out = convert_prompt(&prompt, true);
        assert!(out.warnings.is_empty(), "tool-reference is supported");
        let WireMessage::User { content } = &out.messages[0] else {
            panic!("expected user message");
        };
        let WireUserPart::ToolResult {
            content: WireToolResultContent::Parts(parts),
            ..
        } = &content[0]
        else {
            panic!("expected ToolResult with Parts");
        };
        assert_eq!(parts.len(), 1);
        let WireNestedToolResultContent::ToolReference { tool_name } = &parts[0] else {
            panic!("expected ToolReference variant");
        };
        assert_eq!(tool_name, "get_weather");
    }

    #[test]
    fn tool_result_content_with_text_and_image() {
        use llmsdk_provider::language_model::ToolOutputPart;

        let prompt = vec![Message::Tool {
            content: vec![ToolMessagePart::ToolResult(ToolResultPart {
                tool_call_id: "tu_x".into(),
                tool_name: "fake".into(),
                output: ToolResultOutput::Content {
                    value: vec![
                        ToolOutputPart::Text {
                            text: "snippet".into(),
                            provider_options: None,
                        },
                        ToolOutputPart::File {
                            data: FileData::Url {
                                url: "https://example.com/x.png".into(),
                            },
                            media_type: "image/png".into(),
                            filename: None,
                            provider_options: None,
                        },
                    ],
                },
                provider_options: None,
            })],
            provider_options: None,
        }];

        let out = convert_prompt(&prompt, true);
        assert!(out.warnings.is_empty());
        let WireMessage::User { content } = &out.messages[0] else {
            panic!("expected user");
        };
        let WireUserPart::ToolResult {
            content: WireToolResultContent::Parts(parts),
            ..
        } = &content[0]
        else {
            panic!("expected Parts");
        };
        assert_eq!(parts.len(), 2);
        assert!(matches!(parts[0], WireNestedToolResultContent::Text { .. }));
        assert!(matches!(
            parts[1],
            WireNestedToolResultContent::Image { .. }
        ));
    }

    #[test]
    fn assistant_text_and_tool_use() {
        let prompt = vec![Message::Assistant {
            content: vec![
                AssistantPart::Text(TextPart {
                    text: "calling".into(),
                    provider_options: None,
                }),
                AssistantPart::ToolCall(ToolCallPart {
                    tool_call_id: "tu_z".into(),
                    tool_name: "calc".into(),
                    input: serde_json::json!({"expr": "1+1"}),
                    provider_executed: None,
                    dynamic: None,
                    provider_options: None,
                }),
            ],
            provider_options: None,
        }];
        let out = convert_prompt(&prompt, true);
        if let WireMessage::Assistant { content } = &out.messages[0] {
            assert_eq!(content.len(), 2);
            assert!(matches!(content[1], WireAssistantPart::ToolUse { .. }));
        }
    }
}
