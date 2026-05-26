//! Convert an [`llmsdk_provider::language_model::Prompt`] into xAI wire messages.
//!
//! Mirrors `convert-to-xai-chat-messages.ts`. xAI is OpenAI-compatible with
//! one extension: when a user file part is a [`FileData::Reference`] keyed by
//! `xai`, it serializes to `{type:"file", file:{file_id:"..."}}`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    AssistantPart, FilePart, Message, Prompt, ToolCallPart, ToolMessagePart, ToolResultOutput,
    ToolResultPart, UserPart,
};
use llmsdk_provider::shared::{FileBytes, FileData, Warning};

use super::wire::{
    WireFileRef, WireFunctionCall, WireImageUrl, WireMessage, WireToolCall, WireToolCallKind,
    WireUserContent, WireUserPart,
};

/// Convert a prompt and collect warnings about dropped parts.
///
/// # Errors
///
/// Returns [`ProviderError::unsupported`] for combinations that xAI rejects
/// hard (matches upstream's `UnsupportedFunctionalityError`).
pub(crate) fn convert_prompt(
    prompt: &Prompt,
) -> Result<(Vec<WireMessage>, Vec<Warning>), ProviderError> {
    let mut messages = Vec::with_capacity(prompt.len());
    let mut warnings = Vec::new();

    for message in prompt {
        match message {
            Message::System { content, .. } => messages.push(WireMessage::System {
                content: content.clone(),
            }),
            Message::User { content, .. } => {
                messages.push(convert_user(content)?);
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

    Ok((messages, warnings))
}

fn convert_user(parts: &[UserPart]) -> Result<WireMessage, ProviderError> {
    if let [UserPart::Text(t)] = parts {
        return Ok(WireMessage::User {
            content: WireUserContent::Text(t.text.clone()),
        });
    }

    let mut out = Vec::with_capacity(parts.len());
    for part in parts {
        match part {
            UserPart::Text(t) => out.push(WireUserPart::Text {
                text: t.text.clone(),
            }),
            UserPart::File(f) => out.push(convert_user_file(f)?),
        }
    }
    Ok(WireMessage::User {
        content: WireUserContent::Parts(out),
    })
}

fn convert_user_file(file: &FilePart) -> Result<WireUserPart, ProviderError> {
    match &file.data {
        FileData::Reference { reference } => {
            let file_id = reference
                .get("xai")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    ProviderError::invalid_argument(
                        "file.data.reference",
                        "xAI file reference must contain a string `xai` entry",
                    )
                })?
                .to_owned();
            Ok(WireUserPart::File {
                file: WireFileRef { file_id },
            })
        }
        FileData::Text { .. } => Err(ProviderError::unsupported("text file parts")),
        FileData::Url { url } => {
            if top_level(&file.media_type) == "image" {
                Ok(WireUserPart::ImageUrl {
                    image_url: WireImageUrl { url: url.clone() },
                })
            } else {
                Err(ProviderError::unsupported(format!(
                    "file part media type {}",
                    file.media_type
                )))
            }
        }
        FileData::Data { data } => {
            if top_level(&file.media_type) == "image" {
                let payload = match data {
                    FileBytes::Base64(s) => s.clone(),
                    FileBytes::Bytes(b) => base64_encode(b),
                };
                Ok(WireUserPart::ImageUrl {
                    image_url: WireImageUrl {
                        url: format!("data:{};base64,{}", file.media_type, payload),
                    },
                })
            } else {
                Err(ProviderError::unsupported(format!(
                    "file part media type {}",
                    file.media_type
                )))
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

fn convert_assistant(parts: &[AssistantPart], warnings: &mut Vec<Warning>) -> WireMessage {
    let mut text_buf = String::new();
    let mut tool_calls = Vec::new();

    for part in parts {
        match part {
            AssistantPart::Text(t) => text_buf.push_str(&t.text),
            AssistantPart::ToolCall(tc) => tool_calls.push(convert_tool_call(tc)),
            AssistantPart::Reasoning { .. } => {
                // xAI's chat API does not echo reasoning back into the prompt.
                warnings.push(Warning::Unsupported {
                    feature: "assistant.reasoning".to_owned(),
                    details: Some(
                        "xAI chat drops reasoning blocks on outbound messages".to_owned(),
                    ),
                });
            }
            AssistantPart::ReasoningFile { .. } => warnings.push(Warning::Unsupported {
                feature: "assistant.reasoning-file".to_owned(),
                details: None,
            }),
            AssistantPart::File(_) => warnings.push(Warning::Unsupported {
                feature: "assistant.file".to_owned(),
                details: Some("xAI chat does not support assistant-side files".to_owned()),
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
            warnings.push(Warning::Unsupported {
                feature: "feature.approval-response".to_owned(),
                details: Some("xAI chat does not relay approval responses".to_owned()),
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
            .unwrap_or_else(|| "Tool call execution denied.".to_owned()),
        ToolResultOutput::Content { .. } => {
            warnings.push(Warning::Unsupported {
                feature: "feature-result.content".to_owned(),
                details: Some(
                    "xAI chat flattens multi-part tool output to empty string".to_owned(),
                ),
            });
            String::new()
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
    fn single_text_user_collapses_to_string() {
        let prompt = vec![Message::User {
            content: vec![UserPart::Text(TextPart {
                text: "hi".into(),
                provider_options: None,
            })],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(&prompt).unwrap();
        assert!(matches!(
            &out[0],
            WireMessage::User { content: WireUserContent::Text(s) } if s == "hi"
        ));
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
        let WireMessage::User {
            content: WireUserContent::Parts(p),
        } = &out[0]
        else {
            panic!("expected parts");
        };
        assert!(matches!(&p[0], WireUserPart::ImageUrl { .. }));
    }

    #[test]
    fn file_reference_emits_file_id_part() {
        let mut reference = serde_json::Map::new();
        reference.insert("xai".into(), json!("file_abc123"));
        let prompt = vec![Message::User {
            content: vec![UserPart::File(FilePart {
                filename: None,
                data: FileData::Reference { reference },
                media_type: "application/pdf".into(),
                provider_options: None,
            })],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(&prompt).unwrap();
        let WireMessage::User {
            content: WireUserContent::Parts(p),
        } = &out[0]
        else {
            panic!("expected parts");
        };
        let WireUserPart::File { file } = &p[0] else {
            panic!("expected file part");
        };
        assert_eq!(file.file_id, "file_abc123");
    }

    #[test]
    fn non_image_url_errors() {
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
        let err = convert_prompt(&prompt).unwrap_err();
        assert!(format!("{err}").contains("application/pdf"));
    }

    #[test]
    fn assistant_text_and_tool_calls() {
        let prompt = vec![Message::Assistant {
            content: vec![
                AssistantPart::Text(TextPart {
                    text: "ok".into(),
                    provider_options: None,
                }),
                AssistantPart::ToolCall(ToolCallPart {
                    tool_call_id: "call_1".into(),
                    tool_name: "weather".into(),
                    input: json!({"city": "NYC"}),
                    provider_executed: None,
                    dynamic: None,
                    provider_options: None,
                }),
            ],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(&prompt).unwrap();
        let WireMessage::Assistant {
            content,
            tool_calls,
        } = &out[0]
        else {
            panic!("expected assistant");
        };
        assert_eq!(content, "ok");
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
            tool_call_id,
            content,
        } = &out[0]
        else {
            panic!("expected tool");
        };
        assert_eq!(tool_call_id, "call_1");
        assert_eq!(content, "sunny");
    }
}
