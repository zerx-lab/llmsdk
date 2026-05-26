//! Convert an [`llmsdk_provider::language_model::Prompt`] into Mistral wire messages.
//!
//! Mirrors `convert-to-mistral-chat-messages.ts`. Key Mistral-specific behaviour:
//!
//! - User files: `image/*` → `image_url`, `application/pdf` → `document_url`,
//!   anything else is rejected.
//! - When the *last* message is `assistant`, the wire message carries
//!   `prefix: true` so Mistral will continue the assistant message verbatim.
//! - Assistant reasoning parts are flattened into the text content (matches
//!   upstream's `text += part.text` for `reasoning`).
//! - Tool messages: each `ToolResultPart` becomes its own `role=tool` message
//!   (tool message split mirrors upstream behaviour).
// Rust guideline compliant 2026-05-25

use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    AssistantPart, FilePart, Message, Prompt, ToolCallPart, ToolMessagePart, ToolResultOutput,
    ToolResultPart, UserPart,
};
use llmsdk_provider::shared::{FileBytes, FileData, Warning};

use super::wire::{WireFunctionCall, WireMessage, WireToolCall, WireToolCallKind, WireUserPart};

/// Convert a prompt and collect warnings about dropped parts.
///
/// # Errors
///
/// Returns [`ProviderError::unsupported`] for combinations that Mistral rejects
/// hard (matches upstream's `UnsupportedFunctionalityError`).
pub(crate) fn convert_prompt(
    prompt: &Prompt,
) -> Result<(Vec<WireMessage>, Vec<Warning>), ProviderError> {
    let mut messages = Vec::with_capacity(prompt.len());
    let mut warnings = Vec::new();
    let last_index = prompt.len().saturating_sub(1);

    for (i, message) in prompt.iter().enumerate() {
        let is_last = i == last_index;
        match message {
            Message::System { content, .. } => messages.push(WireMessage::System {
                content: content.clone(),
            }),
            Message::User { content, .. } => {
                messages.push(convert_user(content)?);
            }
            Message::Assistant { content, .. } => {
                messages.push(convert_assistant(content, is_last, &mut warnings));
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

    Ok((messages, warnings))
}

fn convert_user(parts: &[UserPart]) -> Result<WireMessage, ProviderError> {
    let mut out = Vec::with_capacity(parts.len());
    for part in parts {
        match part {
            UserPart::Text(t) => out.push(WireUserPart::Text {
                text: t.text.clone(),
            }),
            UserPart::File(f) => out.push(convert_user_file(f)?),
        }
    }
    Ok(WireMessage::User { content: out })
}

fn convert_user_file(file: &FilePart) -> Result<WireUserPart, ProviderError> {
    match &file.data {
        FileData::Reference { .. } => Err(ProviderError::unsupported(
            "file parts with provider references",
        )),
        FileData::Text { .. } => Err(ProviderError::unsupported("text file parts")),
        FileData::Url { url } => {
            if top_level(&file.media_type) == "image" {
                Ok(WireUserPart::ImageUrl {
                    image_url: url.clone(),
                })
            } else if file.media_type == "application/pdf" {
                Ok(WireUserPart::DocumentUrl {
                    document_url: url.clone(),
                })
            } else {
                Err(ProviderError::unsupported(
                    "Only images and PDF file parts are supported",
                ))
            }
        }
        FileData::Data { data } => {
            let payload = match data {
                FileBytes::Base64(s) => s.clone(),
                FileBytes::Bytes(b) => base64_encode(b),
            };
            let data_url = format!("data:{};base64,{}", file.media_type, payload);
            if top_level(&file.media_type) == "image" {
                Ok(WireUserPart::ImageUrl {
                    image_url: data_url,
                })
            } else if file.media_type == "application/pdf" {
                Ok(WireUserPart::DocumentUrl {
                    document_url: data_url,
                })
            } else {
                Err(ProviderError::unsupported(
                    "Only images and PDF file parts are supported",
                ))
            }
        }
    }
}

fn top_level(media_type: &str) -> &str {
    media_type.split('/').next().unwrap_or(media_type)
}

/// Minimal base64 encoder (RFC 4648 §4) — same algorithm as the `OpenAI` port.
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

fn convert_assistant(
    parts: &[AssistantPart],
    is_last_message: bool,
    warnings: &mut Vec<Warning>,
) -> WireMessage {
    let mut text_buf = String::new();
    let mut tool_calls = Vec::new();

    for part in parts {
        match part {
            AssistantPart::Text(t) => text_buf.push_str(&t.text),
            // Mistral flattens reasoning into the text content (upstream parity).
            AssistantPart::Reasoning { text, .. } => text_buf.push_str(text),
            AssistantPart::ToolCall(tc) => tool_calls.push(convert_tool_call(tc)),
            AssistantPart::ReasoningFile { .. } => warnings.push(Warning::Unsupported {
                feature: "assistant.reasoning-file".to_owned(),
                details: None,
            }),
            AssistantPart::File(_) => warnings.push(Warning::Unsupported {
                feature: "assistant.file".to_owned(),
                details: Some("Mistral chat does not support assistant-side files".to_owned()),
            }),
            AssistantPart::Custom { kind, .. } => warnings.push(Warning::Unsupported {
                feature: format!("assistant.custom.{kind}"),
                details: None,
            }),
            AssistantPart::ToolResult(_) => warnings.push(Warning::Unsupported {
                feature: "assistant.feature-result".to_owned(),
                details: Some(
                    "inline tool result on assistant turn not supported (use role=tool)".to_owned(),
                ),
            }),
        }
    }

    WireMessage::Assistant {
        content: text_buf,
        // Only emit `prefix: true` when this is the trailing message — that
        // signals Mistral to continue the assistant message verbatim.
        prefix: if is_last_message { Some(true) } else { None },
        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
    }
}

fn convert_tool_call(tc: &ToolCallPart) -> WireToolCall {
    let arguments = if tc.input.is_null() {
        "{}".to_owned()
    } else if let Some(s) = tc.input.as_str() {
        s.to_owned()
    } else {
        serde_json::to_string(&tc.input).unwrap_or_else(|_| "{}".to_owned())
    };
    WireToolCall {
        id: tc.tool_call_id.clone(),
        kind: Some(WireToolCallKind::Function),
        function: WireFunctionCall {
            name: tc.tool_name.clone(),
            arguments,
        },
    }
}

fn convert_tool_part(part: &ToolMessagePart, warnings: &mut Vec<Warning>) -> Option<WireMessage> {
    match part {
        ToolMessagePart::ToolResult(r) => Some(WireMessage::Tool {
            name: r.tool_name.clone(),
            tool_call_id: r.tool_call_id.clone(),
            content: tool_result_to_string(r, warnings),
        }),
        // Upstream silently skips approval responses (matches the `continue`
        // branch in convert-to-mistral-chat-messages.ts).
        ToolMessagePart::ToolApprovalResponse(_) => None,
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
            .unwrap_or_else(|| "Tool call execution denied.".to_owned()),
        ToolResultOutput::Content { value } => {
            // Upstream stringifies via JSON.stringify; we do the same.
            serde_json::to_string(value).unwrap_or_else(|_| {
                warnings.push(Warning::Unsupported {
                    feature: "feature-result.content".to_owned(),
                    details: Some(
                        "Mistral chat could not serialize multi-part tool output".to_owned(),
                    ),
                });
                String::new()
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmsdk_provider::language_model::TextPart;
    use serde_json::json;

    #[test]
    fn system_message_passthrough() {
        let prompt = vec![Message::System {
            content: "be brief".into(),
            provider_options: None,
        }];
        let (out, warnings) = convert_prompt(&prompt).unwrap();
        assert!(warnings.is_empty());
        assert!(matches!(out[0], WireMessage::System { ref content } if content == "be brief"));
    }

    #[test]
    fn user_text_kept_as_parts_list() {
        let prompt = vec![Message::User {
            content: vec![UserPart::Text(TextPart {
                text: "hi".into(),
                provider_options: None,
            })],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(&prompt).unwrap();
        let WireMessage::User { content } = &out[0] else {
            panic!("expected user");
        };
        assert_eq!(content.len(), 1);
        assert!(matches!(content[0], WireUserPart::Text { .. }));
    }

    #[test]
    fn image_url_pass_through() {
        let prompt = vec![Message::User {
            content: vec![UserPart::File(FilePart {
                filename: None,
                data: FileData::Url {
                    url: "https://example.com/a.png".into(),
                },
                media_type: "image/png".into(),
                provider_options: None,
            })],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(&prompt).unwrap();
        let WireMessage::User { content } = &out[0] else {
            panic!("expected user");
        };
        assert!(matches!(&content[0], WireUserPart::ImageUrl { .. }));
    }

    #[test]
    fn pdf_url_becomes_document_url() {
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
        let (out, _) = convert_prompt(&prompt).unwrap();
        let WireMessage::User { content } = &out[0] else {
            panic!("expected user");
        };
        let WireUserPart::DocumentUrl { document_url } = &content[0] else {
            panic!("expected document_url part");
        };
        assert_eq!(document_url, "https://example.com/a.pdf");
    }

    #[test]
    fn pdf_data_becomes_data_url_document() {
        let prompt = vec![Message::User {
            content: vec![UserPart::File(FilePart {
                filename: None,
                data: FileData::Data {
                    data: FileBytes::Base64("aGVsbG8=".into()),
                },
                media_type: "application/pdf".into(),
                provider_options: None,
            })],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(&prompt).unwrap();
        let WireMessage::User { content } = &out[0] else {
            panic!("expected user");
        };
        let WireUserPart::DocumentUrl { document_url } = &content[0] else {
            panic!("expected document_url part");
        };
        assert!(document_url.starts_with("data:application/pdf;base64,"));
    }

    #[test]
    fn non_image_non_pdf_errors() {
        let prompt = vec![Message::User {
            content: vec![UserPart::File(FilePart {
                filename: None,
                data: FileData::Url {
                    url: "https://example.com/a.txt".into(),
                },
                media_type: "text/plain".into(),
                provider_options: None,
            })],
            provider_options: None,
        }];
        let err = convert_prompt(&prompt).unwrap_err();
        assert!(format!("{err}").contains("Only images and PDF"));
    }

    #[test]
    fn assistant_last_message_has_prefix_true() {
        let prompt = vec![Message::Assistant {
            content: vec![AssistantPart::Text(TextPart {
                text: "partial".into(),
                provider_options: None,
            })],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(&prompt).unwrap();
        let WireMessage::Assistant { prefix, .. } = &out[0] else {
            panic!("expected assistant");
        };
        assert_eq!(*prefix, Some(true));
    }

    #[test]
    fn assistant_non_last_has_no_prefix() {
        let prompt = vec![
            Message::Assistant {
                content: vec![AssistantPart::Text(TextPart {
                    text: "ok".into(),
                    provider_options: None,
                })],
                provider_options: None,
            },
            Message::User {
                content: vec![UserPart::Text(TextPart {
                    text: "more".into(),
                    provider_options: None,
                })],
                provider_options: None,
            },
        ];
        let (out, _) = convert_prompt(&prompt).unwrap();
        let WireMessage::Assistant { prefix, .. } = &out[0] else {
            panic!("expected assistant");
        };
        assert_eq!(*prefix, None);
    }

    #[test]
    fn assistant_reasoning_flattens_into_text() {
        let prompt = vec![Message::Assistant {
            content: vec![
                AssistantPart::Reasoning {
                    text: "let me think. ".into(),
                    provider_options: None,
                },
                AssistantPart::Text(TextPart {
                    text: "answer".into(),
                    provider_options: None,
                }),
            ],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(&prompt).unwrap();
        let WireMessage::Assistant { content, .. } = &out[0] else {
            panic!("expected assistant");
        };
        assert_eq!(content, "let me think. answer");
    }

    #[test]
    fn assistant_tool_calls_emitted() {
        let prompt = vec![Message::Assistant {
            content: vec![AssistantPart::ToolCall(ToolCallPart {
                tool_call_id: "call_1".into(),
                tool_name: "weather".into(),
                input: json!({"city": "NYC"}),
                provider_executed: None,
                dynamic: None,
                provider_options: None,
            })],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(&prompt).unwrap();
        let WireMessage::Assistant { tool_calls, .. } = &out[0] else {
            panic!("expected assistant");
        };
        let calls = tool_calls.as_ref().unwrap();
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].function.arguments, r#"{"city":"NYC"}"#);
    }

    #[test]
    fn tool_result_text_passthrough() {
        let prompt = vec![Message::Tool {
            content: vec![ToolMessagePart::ToolResult(ToolResultPart {
                tool_call_id: "call_1".into(),
                tool_name: "weather".into(),
                output: ToolResultOutput::Text {
                    value: "sunny".into(),
                    provider_options: None,
                },
                provider_options: None,
            })],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(&prompt).unwrap();
        let WireMessage::Tool {
            name,
            tool_call_id,
            content,
        } = &out[0]
        else {
            panic!("expected tool");
        };
        assert_eq!(name, "weather");
        assert_eq!(tool_call_id, "call_1");
        assert_eq!(content, "sunny");
    }
}
