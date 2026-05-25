//! Convert an [`llmsdk_provider::language_model::Prompt`] into `OpenAI` wire messages.
//!
//! Mirrors `convert-to-openai-chat-messages.ts` (simplified for M3). Anything
//! not yet supported is reported as a [`Warning::UnsupportedSetting`] and
//! dropped — we never silently lose information.
// Rust guideline compliant 2026-02-21

use llmsdk_provider::language_model::{
    AssistantPart, FilePart, Message, Prompt, ToolCallPart, ToolMessagePart, ToolResultOutput,
    ToolResultPart, UserPart,
};
use llmsdk_provider::shared::{FileBytes, FileData, Warning};

use super::model::SystemRole;
use super::wire::{
    WireFunctionCall, WireImageUrl, WireMessage, WireToolCall, WireToolCallKind, WireUserContent,
    WireUserPart,
};

/// Convert a prompt and collect warnings about dropped parts.
///
/// `system_role` selects between the standard `system` role and the
/// reasoning-model `developer` role.
pub(crate) fn convert_prompt(
    prompt: &Prompt,
    system_role: SystemRole,
) -> (Vec<WireMessage>, Vec<Warning>) {
    let mut messages = Vec::with_capacity(prompt.len());
    let mut warnings = Vec::new();

    for message in prompt {
        match message {
            Message::System { content, .. } => match system_role {
                SystemRole::System => messages.push(WireMessage::System {
                    content: content.clone(),
                }),
                SystemRole::Developer => messages.push(WireMessage::Developer {
                    content: content.clone(),
                }),
                SystemRole::Remove => {
                    warnings.push(Warning::Other {
                        message: "system message removed (systemMessageMode = 'remove')".to_owned(),
                    });
                }
            },
            Message::User { content, .. } => {
                messages.push(convert_user(content, &mut warnings));
            }
            Message::Assistant { content, .. } => {
                messages.push(convert_assistant(content, &mut warnings));
            }
            Message::Tool { content, .. } => {
                for part in content {
                    if let Some(msg) = convert_tool_part(part, &mut warnings) {
                        messages.push(msg);
                    }
                }
            }
        }
    }

    (messages, warnings)
}

fn convert_user(parts: &[UserPart], warnings: &mut Vec<Warning>) -> WireMessage {
    // Single text part collapses to plain string (matches ai-sdk).
    if let [UserPart::Text(t)] = parts {
        return WireMessage::User {
            content: WireUserContent::Text(t.text.clone()),
        };
    }

    let mut out = Vec::with_capacity(parts.len());
    for part in parts {
        match part {
            UserPart::Text(t) => out.push(WireUserPart::Text {
                text: t.text.clone(),
            }),
            UserPart::File(f) => {
                if let Some(part) = convert_user_file(f, warnings) {
                    out.push(part);
                }
            }
        }
    }
    WireMessage::User {
        content: WireUserContent::Parts(out),
    }
}

fn convert_user_file(file: &FilePart, warnings: &mut Vec<Warning>) -> Option<WireUserPart> {
    let top_level = file
        .media_type
        .split('/')
        .next()
        .unwrap_or(file.media_type.as_str());
    if top_level != "image" {
        warnings.push(Warning::UnsupportedSetting {
            setting: "user.file".to_owned(),
            details: Some(format!(
                "M3 only supports image/* user files (got {})",
                file.media_type
            )),
        });
        return None;
    }

    let url = match &file.data {
        FileData::Url { url } => url.clone(),
        FileData::Data { data } => data_uri(&file.media_type, data),
        FileData::Reference { .. } | FileData::Text { .. } => {
            warnings.push(Warning::UnsupportedSetting {
                setting: "user.file.data".to_owned(),
                details: Some(
                    "M3 does not support provider-reference or inline-text file data".to_owned(),
                ),
            });
            return None;
        }
    };

    Some(WireUserPart::ImageUrl {
        image_url: WireImageUrl { url },
    })
}

fn data_uri(media_type: &str, bytes: &FileBytes) -> String {
    let payload = match bytes {
        FileBytes::Base64(s) => s.clone(),
        FileBytes::Bytes(b) => base64_encode(b),
    };
    format!("data:{media_type};base64,{payload}")
}

/// Minimal base64 encoder so we don't pull in another dep.
fn base64_encode(bytes: &[u8]) -> String {
    // The OpenAI / Chat-Completions image_url accepts standard base64 with
    // padding. We implement RFC 4648 §4 directly.
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

fn convert_assistant(parts: &[AssistantPart], warnings: &mut Vec<Warning>) -> WireMessage {
    let mut text_buf = String::new();
    let mut tool_calls = Vec::new();

    for part in parts {
        match part {
            AssistantPart::Text(t) => text_buf.push_str(&t.text),
            AssistantPart::ToolCall(tc) => tool_calls.push(convert_tool_call(tc)),
            AssistantPart::Reasoning { .. } => warnings.push(Warning::UnsupportedSetting {
                setting: "assistant.reasoning".to_owned(),
                details: Some("M3 drops reasoning content on outbound messages".to_owned()),
            }),
            AssistantPart::ReasoningFile { .. } => warnings.push(Warning::UnsupportedSetting {
                setting: "assistant.reasoning-file".to_owned(),
                details: None,
            }),
            AssistantPart::File(_) => warnings.push(Warning::UnsupportedSetting {
                setting: "assistant.file".to_owned(),
                details: Some("assistant-side file parts not yet supported".to_owned()),
            }),
            AssistantPart::Custom { kind, .. } => warnings.push(Warning::UnsupportedSetting {
                setting: format!("assistant.custom.{kind}"),
                details: None,
            }),
            AssistantPart::ToolResult(_) => warnings.push(Warning::UnsupportedSetting {
                setting: "assistant.tool-result".to_owned(),
                details: Some(
                    "inline tool result on assistant turn not supported (use role=tool)".to_owned(),
                ),
            }),
        }
    }

    WireMessage::Assistant {
        content: (!text_buf.is_empty()).then_some(text_buf),
        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
    }
}

fn convert_tool_call(tc: &ToolCallPart) -> WireToolCall {
    let arguments = if tc.input.is_null() {
        "{}".to_owned()
    } else if let Some(s) = tc.input.as_str() {
        // Already stringified — pass through.
        s.to_owned()
    } else {
        serde_json::to_string(&tc.input).unwrap_or_else(|_| "{}".to_owned())
    };

    WireToolCall {
        id: tc.tool_call_id.clone(),
        kind: WireToolCallKind::Function,
        function: WireFunctionCall {
            name: tc.tool_name.clone(),
            arguments,
        },
    }
}

fn convert_tool_part(part: &ToolMessagePart, warnings: &mut Vec<Warning>) -> Option<WireMessage> {
    match part {
        ToolMessagePart::ToolResult(r) => Some(WireMessage::Tool {
            tool_call_id: r.tool_call_id.clone(),
            content: tool_result_to_string(r, warnings),
        }),
        ToolMessagePart::ToolApprovalResponse(_) => {
            warnings.push(Warning::UnsupportedSetting {
                setting: "tool.approval-response".to_owned(),
                details: Some("M3 does not relay approval responses".to_owned()),
            });
            None
        }
    }
}

fn tool_result_to_string(part: &ToolResultPart, warnings: &mut Vec<Warning>) -> String {
    match &part.output {
        ToolResultOutput::Text { value, .. } | ToolResultOutput::ErrorText { value, .. } => {
            value.clone()
        }
        ToolResultOutput::Json { value, .. } | ToolResultOutput::ErrorJson { value, .. } => {
            serde_json::to_string(value).unwrap_or_else(|_| "{}".to_owned())
        }
        ToolResultOutput::ExecutionDenied { reason, .. } => reason
            .clone()
            .unwrap_or_else(|| "execution denied".to_owned()),
        ToolResultOutput::Content { .. } => {
            warnings.push(Warning::UnsupportedSetting {
                setting: "tool-result.content".to_owned(),
                details: Some("M3 flattens multi-part tool output to empty string".to_owned()),
            });
            String::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmsdk_provider::language_model::TextPart;

    #[test]
    fn system_message_passthrough() {
        let prompt = vec![Message::System {
            content: "be brief".into(),
            provider_options: None,
        }];
        let (out, warnings) = convert_prompt(&prompt, SystemRole::System);
        assert!(warnings.is_empty());
        assert!(matches!(out[0], WireMessage::System { ref content } if content == "be brief"));
    }

    #[test]
    fn system_message_uses_developer_role_for_reasoning_models() {
        let prompt = vec![Message::System {
            content: "be brief".into(),
            provider_options: None,
        }];
        let (out, _) = convert_prompt(&prompt, SystemRole::Developer);
        assert!(matches!(out[0], WireMessage::Developer { ref content } if content == "be brief"));
    }

    #[test]
    fn single_text_user_collapses_to_string() {
        let prompt = vec![Message::User {
            content: vec![UserPart::Text(TextPart {
                text: "hi".into(),
                provider_options: None,
            })],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(&prompt, SystemRole::System);
        assert!(
            matches!(&out[0], WireMessage::User { content: WireUserContent::Text(s) } if s == "hi")
        );
    }

    #[test]
    fn multi_part_user_uses_parts() {
        let prompt = vec![Message::User {
            content: vec![
                UserPart::Text(TextPart {
                    text: "look".into(),
                    provider_options: None,
                }),
                UserPart::File(FilePart {
                    filename: None,
                    data: FileData::Url {
                        url: "https://example.com/cat.png".into(),
                    },
                    media_type: "image/png".into(),
                    provider_options: None,
                }),
            ],
            provider_options: None,
        }];
        let (out, warnings) = convert_prompt(&prompt, SystemRole::System);
        assert!(warnings.is_empty());
        if let WireMessage::User {
            content: WireUserContent::Parts(parts),
        } = &out[0]
        {
            assert_eq!(parts.len(), 2);
            assert!(matches!(parts[1], WireUserPart::ImageUrl { .. }));
        } else {
            panic!("expected user parts");
        }
    }

    #[test]
    fn non_image_file_produces_warning() {
        let prompt = vec![Message::User {
            content: vec![UserPart::File(FilePart {
                filename: None,
                data: FileData::Url {
                    url: "https://example.com/a.pdf".into(),
                },
                media_type: "application/pdf".into(),
                provider_options: None,
            })],
            provider_options: None,
        }];
        let (out, warnings) = convert_prompt(&prompt, SystemRole::System);
        assert_eq!(warnings.len(), 1);
        if let WireMessage::User {
            content: WireUserContent::Parts(p),
        } = &out[0]
        {
            assert!(p.is_empty());
        }
    }

    #[test]
    fn assistant_text_and_tool_call() {
        let prompt = vec![Message::Assistant {
            content: vec![
                AssistantPart::Text(TextPart {
                    text: "calling now".into(),
                    provider_options: None,
                }),
                AssistantPart::ToolCall(ToolCallPart {
                    tool_call_id: "call_1".into(),
                    tool_name: "weather".into(),
                    input: serde_json::json!({"city": "NYC"}),
                    provider_executed: None,
                    dynamic: None,
                    provider_options: None,
                }),
            ],
            provider_options: None,
        }];
        let (out, warnings) = convert_prompt(&prompt, SystemRole::System);
        assert!(warnings.is_empty());
        if let WireMessage::Assistant {
            content,
            tool_calls,
        } = &out[0]
        {
            assert_eq!(content.as_deref(), Some("calling now"));
            let calls = tool_calls.as_ref().unwrap();
            assert_eq!(calls[0].id, "call_1");
            assert_eq!(calls[0].function.name, "weather");
            assert_eq!(calls[0].function.arguments, r#"{"city":"NYC"}"#);
        } else {
            panic!("expected assistant");
        }
    }

    #[test]
    fn base64_encodes_correctly() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }
}
