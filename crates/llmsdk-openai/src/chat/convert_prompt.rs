//! Convert an [`llmsdk_provider::language_model::Prompt`] into `OpenAI` wire messages.
//!
//! Mirrors `convert-to-openai-chat-messages.ts` (simplified for M3). Anything
//! not yet supported is reported as a [`Warning::Unsupported`] and
//! dropped — we never silently lose information.
// Rust guideline compliant 2026-02-21

use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    AssistantPart, FilePart, Message, Prompt, ToolCallPart, ToolMessagePart, ToolResultOutput,
    ToolResultPart, UserPart,
};
use llmsdk_provider::shared::{FileBytes, FileData, Warning};

use super::model::SystemRole;
use super::wire::{
    WireFunctionCall, WireImageUrl, WireInputAudio, WireMessage, WireToolCall, WireToolCallKind,
    WireUserContent, WireUserFile, WireUserPart,
};

/// Convert a prompt and collect warnings about dropped parts.
///
/// `system_role` selects between the standard `system` role and the
/// reasoning-model `developer` role.
///
/// # Errors
///
/// Returns [`ProviderError::unsupported`] / [`ProviderError::invalid_argument`]
/// for user file parts that `OpenAI` rejects hard (unsupported media types,
/// URL-sourced audio/PDF, inline-text file data, or provider references that
/// lack an `openai` entry). Mirrors upstream's `UnsupportedFunctionalityError`
/// / `NoSuchProviderReferenceError` in `convert-to-openai-chat-messages.ts`.
pub(crate) fn convert_prompt(
    prompt: &Prompt,
    system_role: SystemRole,
) -> Result<(Vec<WireMessage>, Vec<Warning>), ProviderError> {
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
    // Single text part collapses to plain string (matches ai-sdk).
    if let [UserPart::Text(t)] = parts {
        return Ok(WireMessage::User {
            content: WireUserContent::Text(t.text.clone()),
        });
    }

    let mut out = Vec::with_capacity(parts.len());
    for (idx, part) in parts.iter().enumerate() {
        match part {
            UserPart::Text(t) => out.push(WireUserPart::Text {
                text: t.text.clone(),
            }),
            UserPart::File(f) => out.push(convert_user_file(f, idx)?),
        }
    }
    Ok(WireMessage::User {
        content: WireUserContent::Parts(out),
    })
}

#[allow(
    clippy::too_many_lines,
    reason = "linear dispatch over media-type families mirroring ai-sdk convert-to-openai-chat-messages"
)]
fn convert_user_file(file: &FilePart, index: usize) -> Result<WireUserPart, ProviderError> {
    // Provider-reference data resolves to a previously-uploaded file id.
    if let FileData::Reference { reference, .. } = &file.data {
        // Mirror ai-sdk `resolveProviderReference({ reference, provider: 'openai' })`
        // — the reference must address the `openai` provider, else upstream throws
        // `NoSuchProviderReferenceError`.
        let Some(file_id) = reference.get("openai").and_then(|v| v.as_str()) else {
            return Err(ProviderError::invalid_argument(
                "file.data.reference",
                "OpenAI file reference must contain a string `openai` entry",
            ));
        };
        return Ok(WireUserPart::File {
            file: WireUserFile::Reference {
                file_id: file_id.to_owned(),
            },
        });
    }

    let top_level = file
        .media_type
        .split('/')
        .next()
        .unwrap_or(file.media_type.as_str());

    match top_level {
        "image" => {
            let url = match &file.data {
                FileData::Url { url } => url.clone(),
                FileData::Data { data } => data_uri(&file.media_type, data),
                FileData::Text { .. } => {
                    return Err(ProviderError::unsupported("text file parts"));
                }
                FileData::Reference { .. } => unreachable!("handled above"),
            };
            let detail = file
                .provider_options
                .as_ref()
                .and_then(|po| po.get("openai"))
                .and_then(|openai| openai.get("imageDetail"))
                .and_then(|v| v.as_str())
                .map(str::to_owned);
            Ok(WireUserPart::ImageUrl {
                image_url: WireImageUrl { url, detail },
            })
        }
        "audio" => {
            // OpenAI rejects URL audio; only inline base64 is accepted.
            let data = match &file.data {
                FileData::Data { data } => base64_of(data),
                FileData::Url { .. } => {
                    return Err(ProviderError::unsupported("audio file parts with URLs"));
                }
                FileData::Text { .. } => {
                    return Err(ProviderError::unsupported("text file parts"));
                }
                FileData::Reference { .. } => unreachable!("handled above"),
            };
            let format = match file.media_type.as_str() {
                "audio/wav" => "wav",
                "audio/mp3" | "audio/mpeg" => "mp3",
                other => {
                    return Err(ProviderError::unsupported(format!(
                        "audio content parts with media type {other}"
                    )));
                }
            };
            Ok(WireUserPart::InputAudio {
                input_audio: WireInputAudio {
                    data,
                    format: format.to_owned(),
                },
            })
        }
        _ => {
            // OpenAI only accepts application/pdf for the `file` content part.
            if file.media_type != "application/pdf" {
                return Err(ProviderError::unsupported(format!(
                    "file part media type {}",
                    file.media_type
                )));
            }
            let data = match &file.data {
                FileData::Data { data } => base64_of(data),
                FileData::Url { .. } => {
                    return Err(ProviderError::unsupported("PDF file parts with URLs"));
                }
                FileData::Text { .. } => {
                    return Err(ProviderError::unsupported("text file parts"));
                }
                FileData::Reference { .. } => unreachable!("handled above"),
            };
            let filename = file
                .filename
                .clone()
                .unwrap_or_else(|| format!("part-{index}.pdf"));
            Ok(WireUserPart::File {
                file: WireUserFile::Inline {
                    filename,
                    file_data: format!("data:application/pdf;base64,{data}"),
                },
            })
        }
    }
}

/// Return the base64 payload of inline file bytes, encoding raw bytes if needed.
fn base64_of(bytes: &FileBytes) -> String {
    match bytes {
        FileBytes::Base64(s) => s.clone(),
        FileBytes::Bytes(b) => base64_encode(b),
    }
}

fn data_uri(media_type: &str, bytes: &FileBytes) -> String {
    let payload = base64_of(bytes);
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
            AssistantPart::Reasoning { .. } => warnings.push(Warning::Unsupported {
                feature: "assistant.reasoning".to_owned(),
                details: Some("M3 drops reasoning content on outbound messages".to_owned()),
            }),
            AssistantPart::ReasoningFile { .. } => warnings.push(Warning::Unsupported {
                feature: "assistant.reasoning-file".to_owned(),
                details: None,
            }),
            AssistantPart::File(_) => warnings.push(Warning::Unsupported {
                feature: "assistant.file".to_owned(),
                details: Some("assistant-side file parts not yet supported".to_owned()),
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

    // Mirror upstream `convert-to-openai-chat-messages.ts:204` after #14950 +
    // #13744:
    //
    //   content: toolCalls.length > 0 ? text || null : text
    //
    // - With tool_calls: empty text becomes JSON `null` (OpenAI requires
    //   `content: null` when `tool_calls` is set, otherwise the API throws
    //   "Missing required parameter: tool_calls[].id" — see #13744).
    // - Without tool_calls: empty text becomes `""` (OpenAI rejects
    //   `content: null` with "Invalid value for 'content': expected a
    //   string, got null" — see #14950).
    let content = if tool_calls.is_empty() {
        Some(text_buf)
    } else if text_buf.is_empty() {
        None
    } else {
        Some(text_buf)
    };
    WireMessage::Assistant {
        content,
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
            warnings.push(Warning::Unsupported {
                feature: "feature.approval-response".to_owned(),
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
            .unwrap_or_else(|| "Tool call execution denied.".to_owned()),
        ToolResultOutput::Content { .. } => {
            warnings.push(Warning::Unsupported {
                feature: "feature-result.content".to_owned(),
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
        let (out, warnings) = convert_prompt(&prompt, SystemRole::System).unwrap();
        assert!(warnings.is_empty());
        assert!(matches!(out[0], WireMessage::System { ref content } if content == "be brief"));
    }

    #[test]
    fn system_message_uses_developer_role_for_reasoning_models() {
        let prompt = vec![Message::System {
            content: "be brief".into(),
            provider_options: None,
        }];
        let (out, _) = convert_prompt(&prompt, SystemRole::Developer).unwrap();
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
        let (out, _) = convert_prompt(&prompt, SystemRole::System).unwrap();
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
        let (out, warnings) = convert_prompt(&prompt, SystemRole::System).unwrap();
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
    fn pdf_url_is_rejected() {
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
        let err = convert_prompt(&prompt, SystemRole::System).unwrap_err();
        assert!(format!("{err}").contains("PDF file parts with URLs"));
    }

    #[test]
    fn assistant_tool_only_sends_null_content() {
        // Mirrors upstream
        // `convert-to-openai-chat-messages.test.ts` "should send null content
        // for assistant messages with tool calls" + ai-sdk #14950: when an
        // assistant message has tool_calls and an empty text body, the wire
        // `content` field must be JSON `null`, not omitted, not `""`.
        let prompt = vec![Message::Assistant {
            content: vec![AssistantPart::ToolCall(ToolCallPart {
                tool_call_id: "call_1".into(),
                tool_name: "weather".into(),
                input: serde_json::json!({}),
                provider_executed: None,
                dynamic: None,
                provider_options: None,
            })],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(&prompt, SystemRole::System).unwrap();
        let wire = serde_json::to_value(&out[0]).expect("serializes");
        assert_eq!(
            wire.get("content"),
            Some(&serde_json::Value::Null),
            "content must be explicit null when tool_calls is present and text is empty"
        );
    }

    #[test]
    fn assistant_empty_text_no_tools_sends_empty_string_content() {
        // Mirrors upstream
        // `convert-to-openai-chat-messages.test.ts` "should send empty
        // string content for assistant messages with no tool calls" +
        // ai-sdk #14950: without tool_calls, empty text becomes `""`, not
        // `null` (OpenAI rejects null content when tool_calls is absent).
        let prompt = vec![Message::Assistant {
            content: vec![AssistantPart::Text(TextPart {
                text: String::new(),
                provider_options: None,
            })],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(&prompt, SystemRole::System).unwrap();
        let wire = serde_json::to_value(&out[0]).expect("serializes");
        assert_eq!(
            wire.get("content"),
            Some(&serde_json::Value::String(String::new())),
            "content must be empty string when no tool_calls"
        );
        assert!(
            wire.get("tool_calls").is_none(),
            "tool_calls must be omitted when empty"
        );
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
        let (out, warnings) = convert_prompt(&prompt, SystemRole::System).unwrap();
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
    fn audio_wav_becomes_input_audio() {
        let prompt = vec![Message::User {
            content: vec![UserPart::File(FilePart {
                filename: None,
                data: FileData::Data {
                    data: FileBytes::Bytes(vec![1, 2, 3]),
                },
                media_type: "audio/wav".into(),
                provider_options: None,
            })],
            provider_options: None,
        }];
        let (out, warnings) = convert_prompt(&prompt, SystemRole::System).unwrap();
        assert!(warnings.is_empty());
        if let WireMessage::User {
            content: WireUserContent::Parts(parts),
        } = &out[0]
        {
            assert!(
                matches!(&parts[0], WireUserPart::InputAudio { input_audio } if input_audio.format == "wav" && !input_audio.data.is_empty())
            );
        } else {
            panic!("expected user parts");
        }
    }

    #[test]
    fn audio_mp3_alias_normalizes_to_mp3() {
        let prompt = vec![Message::User {
            content: vec![UserPart::File(FilePart {
                filename: None,
                data: FileData::Data {
                    data: FileBytes::Base64("AAAA".into()),
                },
                media_type: "audio/mpeg".into(),
                provider_options: None,
            })],
            provider_options: None,
        }];
        let (out, warnings) = convert_prompt(&prompt, SystemRole::System).unwrap();
        assert!(warnings.is_empty());
        if let WireMessage::User {
            content: WireUserContent::Parts(parts),
        } = &out[0]
        {
            assert!(
                matches!(&parts[0], WireUserPart::InputAudio { input_audio } if input_audio.format == "mp3")
            );
        }
    }

    #[test]
    fn audio_url_is_rejected() {
        let prompt = vec![Message::User {
            content: vec![UserPart::File(FilePart {
                filename: None,
                data: FileData::Url {
                    url: "https://example.com/a.wav".into(),
                },
                media_type: "audio/wav".into(),
                provider_options: None,
            })],
            provider_options: None,
        }];
        let err = convert_prompt(&prompt, SystemRole::System).unwrap_err();
        assert!(format!("{err}").contains("audio file parts with URLs"));
    }

    #[test]
    fn pdf_becomes_file_part() {
        let prompt = vec![Message::User {
            content: vec![UserPart::File(FilePart {
                filename: Some("report.pdf".into()),
                data: FileData::Data {
                    data: FileBytes::Bytes(b"%PDF-1.4".to_vec()),
                },
                media_type: "application/pdf".into(),
                provider_options: None,
            })],
            provider_options: None,
        }];
        let (out, warnings) = convert_prompt(&prompt, SystemRole::System).unwrap();
        assert!(warnings.is_empty());
        if let WireMessage::User {
            content: WireUserContent::Parts(parts),
        } = &out[0]
        {
            match &parts[0] {
                WireUserPart::File {
                    file:
                        WireUserFile::Inline {
                            filename,
                            file_data,
                        },
                } => {
                    assert_eq!(filename, "report.pdf");
                    assert!(file_data.starts_with("data:application/pdf;base64,"));
                }
                _ => panic!("expected inline pdf file part"),
            }
        }
    }

    #[test]
    fn reference_resolves_to_file_id() {
        let mut reference = serde_json::Map::new();
        reference.insert(
            "openai".into(),
            serde_json::Value::String("file_abc".into()),
        );
        let prompt = vec![Message::User {
            content: vec![UserPart::File(FilePart {
                filename: None,
                data: FileData::Reference { reference },
                media_type: "application/pdf".into(),
                provider_options: None,
            })],
            provider_options: None,
        }];
        let (out, warnings) = convert_prompt(&prompt, SystemRole::System).unwrap();
        assert!(warnings.is_empty());
        if let WireMessage::User {
            content: WireUserContent::Parts(parts),
        } = &out[0]
        {
            assert!(matches!(
                &parts[0],
                WireUserPart::File { file: WireUserFile::Reference { file_id } } if file_id == "file_abc"
            ));
        }
    }

    #[test]
    fn image_detail_provider_option_passes_through() {
        let mut po = llmsdk_provider::shared::ProviderOptions::new();
        po.insert(
            "openai".into(),
            serde_json::json!({"imageDetail": "high"})
                .as_object()
                .cloned()
                .unwrap(),
        );
        let prompt = vec![Message::User {
            content: vec![UserPart::File(FilePart {
                filename: None,
                data: FileData::Url {
                    url: "https://example.com/cat.png".into(),
                },
                media_type: "image/png".into(),
                provider_options: Some(po),
            })],
            provider_options: None,
        }];
        let (out, warnings) = convert_prompt(&prompt, SystemRole::System).unwrap();
        assert!(warnings.is_empty());
        if let WireMessage::User {
            content: WireUserContent::Parts(parts),
        } = &out[0]
        {
            assert!(matches!(
                &parts[0],
                WireUserPart::ImageUrl { image_url } if image_url.detail.as_deref() == Some("high")
            ));
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
