//! Convert a [`Prompt`] into Cohere v2 wire messages + RAG documents.
//!
//! Mirrors `convert-to-cohere-chat-prompt.ts`. Key differences from
//! OpenAI-compatible providers:
//!
//! - Non-image file parts are routed into a separate `documents[]` array
//!   (`{data: {text, title?}}`) and do NOT go into `messages[]`.
//! - When a user message has no image parts, parts collapse to a single
//!   string in `content`; if any image is present, content becomes a list of
//!   parts.
//! - The assistant turn drops `content` whenever `tool_calls` is non-empty.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    AssistantPart, FilePart, Message, Prompt, ToolCallPart, ToolMessagePart, ToolResultOutput,
    ToolResultPart, UserPart,
};
use llmsdk_provider::shared::{FileBytes, FileData, Warning};

use super::options::parse_image_part;
use super::wire::{
    WireDocument, WireDocumentData, WireFunctionCall, WireImageUrl, WireMessage, WireToolCall,
    WireToolCallKind, WireUserContent, WireUserPart,
};

/// Output of [`convert_prompt`].
pub(crate) struct ConvertedPrompt {
    pub messages: Vec<WireMessage>,
    pub documents: Vec<WireDocument>,
    pub warnings: Vec<Warning>,
}

/// Convert a prompt; returns wire messages, `documents[]`, and warnings.
///
/// # Errors
///
/// Returns [`ProviderError::unsupported`] for combinations Cohere rejects.
pub(crate) fn convert_prompt(prompt: &Prompt) -> Result<ConvertedPrompt, ProviderError> {
    let mut messages = Vec::with_capacity(prompt.len());
    let mut documents = Vec::new();
    let mut warnings = Vec::new();

    for message in prompt {
        match message {
            Message::System { content, .. } => messages.push(WireMessage::System {
                content: content.clone(),
            }),
            Message::User { content, .. } => {
                messages.push(convert_user(content, &mut documents)?);
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

    Ok(ConvertedPrompt {
        messages,
        documents,
        warnings,
    })
}

fn convert_user(
    parts: &[UserPart],
    documents: &mut Vec<WireDocument>,
) -> Result<WireMessage, ProviderError> {
    let mut out_parts: Vec<WireUserPart> = Vec::new();
    let mut has_image = false;

    for part in parts {
        match part {
            UserPart::Text(t) => {
                if !t.text.is_empty() {
                    out_parts.push(WireUserPart::Text {
                        text: t.text.clone(),
                    });
                }
            }
            UserPart::File(f) => {
                if top_level(&f.media_type) == "image" {
                    has_image = true;
                    out_parts.push(convert_image(f)?);
                } else {
                    documents.push(file_to_document(f)?);
                }
            }
        }
    }

    if has_image {
        Ok(WireMessage::User {
            content: WireUserContent::Parts(out_parts),
        })
    } else {
        // Concatenate text parts (file documents already split into `documents[]`).
        let joined: String = out_parts
            .into_iter()
            .filter_map(|p| match p {
                WireUserPart::Text { text } => Some(text),
                WireUserPart::ImageUrl { .. } => None,
            })
            .collect();
        Ok(WireMessage::User {
            content: WireUserContent::Text(joined),
        })
    }
}

fn convert_image(file: &FilePart) -> Result<WireUserPart, ProviderError> {
    let url = build_image_url(file)?;
    let opts = parse_image_part(file.provider_options.as_ref());
    Ok(WireUserPart::ImageUrl {
        image_url: WireImageUrl {
            url,
            detail: opts.detail,
        },
    })
}

fn build_image_url(file: &FilePart) -> Result<String, ProviderError> {
    match &file.data {
        FileData::Url { url } => Ok(url.clone()),
        FileData::Data { data } => {
            let payload = match data {
                FileBytes::Base64(s) => s.clone(),
                FileBytes::Bytes(b) => base64_encode(b),
            };
            Ok(format!("data:{};base64,{}", file.media_type, payload))
        }
        FileData::Reference { .. } => Err(ProviderError::unsupported(
            "image file parts with provider references",
        )),
        FileData::Text { .. } => Err(ProviderError::unsupported(
            "image file parts with text data",
        )),
    }
}

fn file_to_document(file: &FilePart) -> Result<WireDocument, ProviderError> {
    let text = match &file.data {
        FileData::Text { text } => text.clone(),
        FileData::Data { data } => match data {
            FileBytes::Base64(s) => s.clone(),
            FileBytes::Bytes(b) => match std::str::from_utf8(b) {
                Ok(s) => s.to_owned(),
                Err(_) => base64_encode(b),
            },
        },
        FileData::Url { .. } => {
            return Err(ProviderError::unsupported(
                "File URL data (URLs should be downloaded before reaching the provider)",
            ));
        }
        FileData::Reference { .. } => {
            return Err(ProviderError::unsupported(
                "file parts with provider references",
            ));
        }
    };
    Ok(WireDocument {
        data: WireDocumentData {
            text,
            title: file.filename.clone(),
        },
    })
}

fn top_level(media_type: &str) -> &str {
    media_type.split('/').next().unwrap_or(media_type)
}

/// Minimal base64 encoder (RFC 4648 §4). Mirrors the helper used by the xAI /
/// `OpenAI` ports — no new dependency.
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
                warnings.push(Warning::UnsupportedSetting {
                    setting: "assistant.reasoning".to_owned(),
                    details: Some(
                        "Cohere chat does not echo reasoning back into the prompt".to_owned(),
                    ),
                });
            }
            AssistantPart::ReasoningFile { .. } => warnings.push(Warning::UnsupportedSetting {
                setting: "assistant.reasoning-file".to_owned(),
                details: None,
            }),
            AssistantPart::File(_) => warnings.push(Warning::UnsupportedSetting {
                setting: "assistant.file".to_owned(),
                details: Some("Cohere chat does not accept assistant-side files".to_owned()),
            }),
            AssistantPart::Custom { kind, .. } => warnings.push(Warning::UnsupportedSetting {
                setting: format!("assistant.custom.{kind}"),
                details: None,
            }),
            AssistantPart::ToolResult(_) => warnings.push(Warning::UnsupportedSetting {
                setting: "assistant.tool-result".to_owned(),
                details: Some("use role=tool for tool results in Cohere chat".to_owned()),
            }),
        }
    }

    let has_calls = !tool_calls.is_empty();
    WireMessage::Assistant {
        content: if has_calls { None } else { Some(text_buf) },
        tool_plan: None,
        tool_calls: has_calls.then_some(tool_calls),
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
            warnings.push(Warning::UnsupportedSetting {
                setting: "tool.approval-response".to_owned(),
                details: Some("Cohere chat does not relay approval responses".to_owned()),
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
            warnings.push(Warning::UnsupportedSetting {
                setting: "tool-result.content".to_owned(),
                details: Some(
                    "Cohere chat flattens multi-part tool output to empty string".to_owned(),
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
        let out = convert_prompt(&prompt).unwrap();
        assert!(out.warnings.is_empty());
        assert!(out.documents.is_empty());
        assert!(
            matches!(&out.messages[0], WireMessage::System { content } if content == "be brief")
        );
    }

    #[test]
    fn user_text_collapses_to_string() {
        let prompt = vec![Message::User {
            content: vec![UserPart::Text(TextPart {
                text: "hi".into(),
                provider_options: None,
            })],
            provider_options: None,
        }];
        let out = convert_prompt(&prompt).unwrap();
        let WireMessage::User {
            content: WireUserContent::Text(s),
        } = &out.messages[0]
        else {
            panic!("expected text user content");
        };
        assert_eq!(s, "hi");
    }

    #[test]
    fn user_with_image_keeps_parts_array() {
        let prompt = vec![Message::User {
            content: vec![
                UserPart::Text(TextPart {
                    text: "look".into(),
                    provider_options: None,
                }),
                UserPart::File(FilePart {
                    filename: None,
                    data: FileData::Url {
                        url: "https://example.com/a.png".into(),
                    },
                    media_type: "image/png".into(),
                    provider_options: None,
                }),
            ],
            provider_options: None,
        }];
        let out = convert_prompt(&prompt).unwrap();
        let WireMessage::User {
            content: WireUserContent::Parts(p),
        } = &out.messages[0]
        else {
            panic!("expected parts");
        };
        assert!(matches!(p[0], WireUserPart::Text { .. }));
        assert!(matches!(p[1], WireUserPart::ImageUrl { .. }));
    }

    #[test]
    fn text_file_routed_to_documents() {
        let prompt = vec![Message::User {
            content: vec![UserPart::File(FilePart {
                filename: Some("notes.txt".into()),
                data: FileData::Text {
                    text: "hello world".into(),
                },
                media_type: "text/plain".into(),
                provider_options: None,
            })],
            provider_options: None,
        }];
        let out = convert_prompt(&prompt).unwrap();
        assert_eq!(out.documents.len(), 1);
        assert_eq!(out.documents[0].data.text, "hello world");
        assert_eq!(out.documents[0].data.title.as_deref(), Some("notes.txt"));
    }

    #[test]
    fn assistant_with_tool_calls_drops_content() {
        let prompt = vec![Message::Assistant {
            content: vec![
                AssistantPart::Text(TextPart {
                    text: "ignored".into(),
                    provider_options: None,
                }),
                AssistantPart::ToolCall(ToolCallPart {
                    tool_call_id: "c1".into(),
                    tool_name: "weather".into(),
                    input: json!({"city": "NYC"}),
                    provider_executed: None,
                    dynamic: None,
                    provider_options: None,
                }),
            ],
            provider_options: None,
        }];
        let out = convert_prompt(&prompt).unwrap();
        let WireMessage::Assistant {
            content,
            tool_calls,
            ..
        } = &out.messages[0]
        else {
            panic!("expected assistant");
        };
        assert!(content.is_none());
        let calls = tool_calls.as_ref().unwrap();
        assert_eq!(calls[0].id, "c1");
        assert_eq!(calls[0].function.arguments, r#"{"city":"NYC"}"#);
    }

    #[test]
    fn tool_role_passthrough() {
        let prompt = vec![Message::Tool {
            content: vec![ToolMessagePart::ToolResult(ToolResultPart {
                tool_call_id: "c1".into(),
                tool_name: "weather".into(),
                output: ToolResultOutput::Text {
                    value: "sunny".into(),
                    provider_options: None,
                },
                provider_options: None,
            })],
            provider_options: None,
        }];
        let out = convert_prompt(&prompt).unwrap();
        let WireMessage::Tool {
            tool_call_id,
            content,
        } = &out.messages[0]
        else {
            panic!("expected tool");
        };
        assert_eq!(tool_call_id, "c1");
        assert_eq!(content, "sunny");
    }
}
