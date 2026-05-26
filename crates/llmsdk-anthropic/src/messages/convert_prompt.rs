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
                                warnings.push(Warning::UnsupportedSetting {
                                    setting: "user.file.data".to_owned(),
                                    details: Some(
                                        "image file reference missing `anthropic` provider entry"
                                            .to_owned(),
                                    ),
                                });
                                continue;
                            }
                        }
                        FileData::Text { .. } => {
                            warnings.push(Warning::UnsupportedSetting {
                                setting: "user.file.data".to_owned(),
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
                            warnings.push(Warning::UnsupportedSetting {
                                setting: "user.file.data".to_owned(),
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
                        warnings.push(Warning::UnsupportedSetting {
                            setting: "user.file".to_owned(),
                            details: Some(format!(
                                "Anthropic Messages API does not accept audio files ({mt})"
                            )),
                        });
                        continue;
                    }
                    (mt, _) => {
                        warnings.push(Warning::UnsupportedSetting {
                            setting: "user.file".to_owned(),
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
            self.warnings.push(Warning::UnsupportedSetting {
                setting: "cache_control".to_owned(),
                details: Some(format!(
                    "cache_control cannot be set on {context_type}. It will be ignored."
                )),
            });
            return None;
        }
        self.count += 1;
        if self.count > MAX_CACHE_BREAKPOINTS {
            self.warnings.push(Warning::UnsupportedSetting {
                setting: "cache_control".to_owned(),
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
            AssistantPart::ToolCall(tc) => out.push(convert_tool_call(tc, cache_control)),
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
                warnings.push(Warning::UnsupportedSetting {
                    setting: "assistant.reasoning-file".to_owned(),
                    details: Some("Anthropic does not support reasoning files".to_owned()),
                });
            }
            AssistantPart::File(_) => warnings.push(Warning::UnsupportedSetting {
                setting: "assistant.file".to_owned(),
                details: None,
            }),
            AssistantPart::Custom { kind, .. } => warnings.push(Warning::UnsupportedSetting {
                setting: format!("assistant.custom.{kind}"),
                details: None,
            }),
            AssistantPart::ToolResult(_) => warnings.push(Warning::UnsupportedSetting {
                setting: "assistant.tool-result".to_owned(),
                details: Some(
                    "inline tool result on assistant turn not supported; use a Tool message"
                        .to_owned(),
                ),
            }),
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

fn convert_tool_call(tc: &ToolCallPart, cache_control: Option<CacheControl>) -> WireAssistantPart {
    // Ensure input is always an object (Anthropic requires JSON-typed input).
    let input = if tc.input.is_null() {
        serde_json::Value::Object(serde_json::Map::new())
    } else {
        tc.input.clone()
    };
    WireAssistantPart::ToolUse {
        id: tc.tool_call_id.clone(),
        name: tc.tool_name.clone(),
        input,
        cache_control,
    }
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
                warnings.push(Warning::UnsupportedSetting {
                    setting: "tool.approval-response".to_owned(),
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
                    .unwrap_or_else(|| "execution denied".to_owned()),
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
                    warnings.push(Warning::UnsupportedSetting {
                        setting: "tool-result.content.file.data".to_owned(),
                        details: Some(
                            "image reference missing `anthropic` provider entry".to_owned(),
                        ),
                    });
                    return None;
                }
            }
            FileData::Text { .. } => {
                warnings.push(Warning::UnsupportedSetting {
                    setting: "tool-result.content.file.data".to_owned(),
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
                warnings.push(Warning::UnsupportedSetting {
                    setting: "tool-result.content.file.data".to_owned(),
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
            warnings.push(Warning::UnsupportedSetting {
                setting: "tool-result.content.file".to_owned(),
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
        warnings.push(Warning::UnsupportedSetting {
            setting: "tool-result.content.custom.tool-reference".to_owned(),
            details: Some("tool-reference requires anthropic.toolName".to_owned()),
        });
        return None;
    }
    warnings.push(Warning::UnsupportedSetting {
        setting: "tool-result.content.custom".to_owned(),
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
