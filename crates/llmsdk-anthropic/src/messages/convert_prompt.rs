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
    AssistantPart, Message, Prompt, ToolCallPart, ToolMessagePart, ToolResultOutput,
    ToolResultPart, UserPart,
};
use llmsdk_provider::shared::{FileBytes, FileData, Warning};

use super::wire::{WireAssistantPart, WireImageSource, WireMessage, WireUserPart};

/// Result of [`convert_prompt`].
pub(crate) struct Converted {
    pub system: Option<String>,
    pub messages: Vec<WireMessage>,
    pub warnings: Vec<Warning>,
}

/// Convert a prompt; collect warnings about dropped parts.
pub(crate) fn convert_prompt(prompt: &Prompt) -> Converted {
    let mut systems: Vec<&str> = Vec::new();
    let mut messages: Vec<WireMessage> = Vec::new();
    let mut warnings: Vec<Warning> = Vec::new();

    for message in prompt {
        match message {
            Message::System { content, .. } => systems.push(content.as_str()),
            Message::User { content, .. } => {
                let parts = convert_user(content, &mut warnings);
                push_user(&mut messages, parts);
            }
            Message::Assistant { content, .. } => {
                let parts = convert_assistant(content, &mut warnings);
                messages.push(WireMessage::Assistant { content: parts });
            }
            Message::Tool { content, .. } => {
                let parts = convert_tool(content, &mut warnings);
                push_user(&mut messages, parts);
            }
        }
    }

    let system = if systems.is_empty() {
        None
    } else {
        Some(systems.join("\n\n"))
    };

    Converted {
        system,
        messages,
        warnings,
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

fn convert_user(parts: &[UserPart], warnings: &mut Vec<Warning>) -> Vec<WireUserPart> {
    let mut out = Vec::with_capacity(parts.len());
    for part in parts {
        match part {
            UserPart::Text(t) => out.push(WireUserPart::Text {
                text: t.text.clone(),
            }),
            UserPart::File(f) => {
                let top = f
                    .media_type
                    .split('/')
                    .next()
                    .unwrap_or(f.media_type.as_str());
                if top != "image" {
                    warnings.push(Warning::UnsupportedSetting {
                        setting: "user.file".to_owned(),
                        details: Some(format!(
                            "M6 Anthropic provider only supports image/* user files (got {})",
                            f.media_type
                        )),
                    });
                    continue;
                }
                let source = match &f.data {
                    FileData::Url { url } => WireImageSource::Url { url: url.clone() },
                    FileData::Data { data } => WireImageSource::Base64 {
                        media_type: f.media_type.clone(),
                        data: file_bytes_to_base64(data),
                    },
                    FileData::Reference { .. } | FileData::Text { .. } => {
                        warnings.push(Warning::UnsupportedSetting {
                            setting: "user.file.data".to_owned(),
                            details: Some(
                                "M6 does not support provider-reference or inline-text file data"
                                    .to_owned(),
                            ),
                        });
                        continue;
                    }
                };
                out.push(WireUserPart::Image { source });
            }
        }
    }
    out
}

fn convert_assistant(
    parts: &[AssistantPart],
    warnings: &mut Vec<Warning>,
) -> Vec<WireAssistantPart> {
    let mut out = Vec::with_capacity(parts.len());
    for part in parts {
        match part {
            AssistantPart::Text(t) => out.push(WireAssistantPart::Text {
                text: t.text.clone(),
            }),
            AssistantPart::ToolCall(tc) => out.push(convert_tool_call(tc)),
            AssistantPart::Reasoning {
                text,
                provider_options,
            } => {
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

fn convert_tool_call(tc: &ToolCallPart) -> WireAssistantPart {
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
    }
}

fn convert_tool(parts: &[ToolMessagePart], warnings: &mut Vec<Warning>) -> Vec<WireUserPart> {
    let mut out = Vec::with_capacity(parts.len());
    for part in parts {
        match part {
            ToolMessagePart::ToolResult(r) => {
                let (content, is_error) = tool_result_to_string(r, warnings);
                out.push(WireUserPart::ToolResult {
                    tool_use_id: r.tool_call_id.clone(),
                    content,
                    is_error,
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

fn tool_result_to_string(
    part: &ToolResultPart,
    warnings: &mut Vec<Warning>,
) -> (String, Option<bool>) {
    match &part.output {
        ToolResultOutput::Text { value, .. } => (value.clone(), None),
        ToolResultOutput::Json { value, .. } => (
            serde_json::to_string(value).unwrap_or_else(|_| "{}".to_owned()),
            None,
        ),
        ToolResultOutput::ErrorText { value, .. } => (value.clone(), Some(true)),
        ToolResultOutput::ErrorJson { value, .. } => (
            serde_json::to_string(value).unwrap_or_else(|_| "{}".to_owned()),
            Some(true),
        ),
        ToolResultOutput::ExecutionDenied { reason, .. } => (
            reason
                .clone()
                .unwrap_or_else(|| "execution denied".to_owned()),
            Some(true),
        ),
        ToolResultOutput::Content { .. } => {
            warnings.push(Warning::UnsupportedSetting {
                setting: "tool-result.content".to_owned(),
                details: Some("M6 flattens multi-part tool output to empty string".to_owned()),
            });
            (String::new(), None)
        }
    }
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
        let out = convert_prompt(&prompt);
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
        let out = convert_prompt(&prompt);
        assert_eq!(out.messages.len(), 3);
        // Last message must be a User with a single tool_result part.
        if let WireMessage::User { content } = &out.messages[2]
            && let WireUserPart::ToolResult {
                tool_use_id,
                content: text,
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
        let out = convert_prompt(&prompt);
        assert_eq!(out.messages.len(), 1);
        if let WireMessage::User { content } = &out.messages[0] {
            assert_eq!(content.len(), 2);
        }
    }

    #[test]
    fn non_image_file_emits_warning_and_drops_part() {
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
        let out = convert_prompt(&prompt);
        assert_eq!(out.warnings.len(), 1);
        // No usable parts -> push_user drops the message entirely.
        assert!(out.messages.is_empty());
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
                    provider_options: None,
                }),
            ],
            provider_options: None,
        }];
        let out = convert_prompt(&prompt);
        if let WireMessage::Assistant { content } = &out.messages[0] {
            assert_eq!(content.len(), 2);
            assert!(matches!(content[1], WireAssistantPart::ToolUse { .. }));
        }
    }
}
