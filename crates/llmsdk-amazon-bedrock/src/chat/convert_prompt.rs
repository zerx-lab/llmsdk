//! Convert an llmsdk [`Prompt`] into Bedrock Converse `system` + `messages`.
//!
//! Mirrors `convert-to-amazon-bedrock-chat-messages.ts`. Bedrock requires
//! alternating user / assistant turns, so adjacent user + tool messages are
//! coalesced into one user turn whose `content[]` carries both raw text and
//! `toolResult` blocks.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    AssistantPart, Message, Prompt, ToolMessagePart, ToolResultOutput, UserPart,
};
use llmsdk_provider::shared::{FileBytes, FileData, Warning};
use serde_json::Value;

use super::normalize_tool_call_id::normalize_tool_call_id;
use super::options::{parse_cache_point, parse_file_part};
use super::reasoning_metadata::parse as parse_reasoning_metadata;
use super::wire::{
    BytesSource, CachePointValue, CitationsConfig, ContentBlock, DocumentBlock, ImageBlock,
    ReasoningContentBlock, ReasoningText, RedactedReasoning, SystemBlock, ToolResultBlock,
    ToolResultPart, ToolUseBlock, WireMessage,
};

/// Result of the converter.
#[derive(Debug)]
pub(crate) struct Converted {
    /// System blocks (text + optional cache markers).
    pub system: Vec<SystemBlock>,
    /// User / assistant message turns.
    pub messages: Vec<WireMessage>,
    /// Warnings accumulated during conversion.
    pub warnings: Vec<Warning>,
}

/// Convert an llmsdk prompt into the Converse wire shape.
///
/// `is_mistral` controls tool-call id normalization (Mistral models require a
/// strict 9-char alphanumeric format).
///
/// # Errors
///
/// Returns [`ProviderError::invalid_argument`] when an input message is
/// structurally invalid (e.g. a system message after the first turn — Bedrock
/// disallows that).
pub(crate) fn convert_prompt(
    prompt: &Prompt,
    is_mistral: bool,
) -> Result<Converted, ProviderError> {
    let mut system: Vec<SystemBlock> = Vec::new();
    let mut messages: Vec<WireMessage> = Vec::new();
    let mut warnings: Vec<Warning> = Vec::new();
    let mut document_counter: u32 = 0;

    let blocks = group_into_blocks(prompt);
    let total_blocks = blocks.len();

    for (block_index, block) in blocks.into_iter().enumerate() {
        let is_last_block = block_index + 1 == total_blocks;
        match block {
            Block::System(messages_in_block) => {
                if !messages.is_empty() {
                    return Err(ProviderError::invalid_argument(
                        "prompt",
                        "Multiple system messages separated by user/assistant turns are not supported by Bedrock.",
                    ));
                }
                for msg in messages_in_block {
                    if let Message::System {
                        content,
                        provider_options,
                    } = msg
                    {
                        system.push(SystemBlock::Text {
                            text: content.clone(),
                        });
                        if let Some((kind, ttl)) = parse_cache_point(provider_options.as_ref()) {
                            system.push(SystemBlock::CachePoint {
                                cache_point: CachePointValue { kind, ttl },
                            });
                        }
                    }
                }
            }
            Block::User(messages_in_block) => {
                let mut content: Vec<ContentBlock> = Vec::new();
                for msg in messages_in_block {
                    match msg {
                        Message::User {
                            content: parts,
                            provider_options,
                        } => {
                            for part in &parts {
                                append_user_part(
                                    part,
                                    is_mistral,
                                    &mut document_counter,
                                    &mut content,
                                    &mut warnings,
                                );
                                if let Some((kind, ttl)) =
                                    parse_cache_point(provider_user_part_options(part))
                                {
                                    content.push(ContentBlock::CachePoint {
                                        cache_point: CachePointValue { kind, ttl },
                                    });
                                }
                            }
                            if let Some((kind, ttl)) = parse_cache_point(provider_options.as_ref())
                            {
                                content.push(ContentBlock::CachePoint {
                                    cache_point: CachePointValue { kind, ttl },
                                });
                            }
                        }
                        Message::Tool {
                            content: parts,
                            provider_options,
                        } => {
                            for part in parts {
                                let ToolMessagePart::ToolResult(tr) = part else {
                                    continue; // tool-approval responses ignored
                                };
                                let payload = encode_tool_result_output(&tr.output);
                                content.push(ContentBlock::ToolResult {
                                    tool_result: ToolResultBlock {
                                        tool_use_id: normalize_tool_call_id(
                                            &tr.tool_call_id,
                                            is_mistral,
                                        ),
                                        content: payload,
                                    },
                                });
                                if let Some((kind, ttl)) =
                                    parse_cache_point(tr.provider_options.as_ref())
                                {
                                    content.push(ContentBlock::CachePoint {
                                        cache_point: CachePointValue { kind, ttl },
                                    });
                                }
                            }
                            if let Some((kind, ttl)) = parse_cache_point(provider_options.as_ref())
                            {
                                content.push(ContentBlock::CachePoint {
                                    cache_point: CachePointValue { kind, ttl },
                                });
                            }
                        }
                        _ => {}
                    }
                }
                messages.push(WireMessage {
                    role: "user".to_owned(),
                    content,
                });
            }
            Block::Assistant(messages_in_block) => {
                let mut content: Vec<ContentBlock> = Vec::new();
                let message_count = messages_in_block.len();
                for (message_index, msg) in messages_in_block.into_iter().enumerate() {
                    let is_last_message = message_index + 1 == message_count;
                    let Message::Assistant { content: parts, .. } = msg else {
                        continue;
                    };
                    let has_reasoning = parts
                        .iter()
                        .any(|p| matches!(p, AssistantPart::Reasoning { .. }));
                    let parts_len = parts.len();
                    for (part_index, part) in parts.into_iter().enumerate() {
                        let is_last_part = part_index + 1 == parts_len;
                        let trim_text = is_last_block && is_last_message && is_last_part;
                        match part {
                            AssistantPart::Text(text_part) => {
                                let trimmed = if trim_text {
                                    text_part.text.trim().to_owned()
                                } else {
                                    text_part.text
                                };
                                if !has_reasoning && trimmed.trim().is_empty() {
                                    continue;
                                }
                                content.push(ContentBlock::Text { text: trimmed });
                            }
                            AssistantPart::Reasoning {
                                text,
                                provider_options,
                            } => {
                                if let Some(meta) =
                                    parse_reasoning_metadata(provider_options.as_ref())
                                {
                                    if let Some(sig) = meta.signature {
                                        content.push(ContentBlock::ReasoningContent {
                                            reasoning_content: ReasoningContentBlock::Text {
                                                reasoning_text: ReasoningText {
                                                    text,
                                                    signature: Some(sig),
                                                },
                                            },
                                        });
                                    } else if let Some(data) = meta.redacted_data {
                                        content.push(ContentBlock::ReasoningContent {
                                            reasoning_content: ReasoningContentBlock::Redacted {
                                                redacted_reasoning: RedactedReasoning { data },
                                            },
                                        });
                                    }
                                }
                                // unsigned reasoning is intentionally dropped — Bedrock will reject it.
                            }
                            AssistantPart::ToolCall(tc) => {
                                let input = if tc.input.is_object() {
                                    tc.input
                                } else {
                                    Value::Object(serde_json::Map::new())
                                };
                                content.push(ContentBlock::ToolUse {
                                    tool_use: ToolUseBlock {
                                        tool_use_id: normalize_tool_call_id(
                                            &tc.tool_call_id,
                                            is_mistral,
                                        ),
                                        name: tc.tool_name,
                                        input,
                                    },
                                });
                            }
                            AssistantPart::File(_)
                            | AssistantPart::ReasoningFile { .. }
                            | AssistantPart::Custom { .. }
                            | AssistantPart::ToolResult(_) => {
                                warnings.push(Warning::Other {
                                    message:
                                        "assistant part dropped — unsupported by Bedrock Converse"
                                            .to_owned(),
                                });
                            }
                        }
                    }
                }
                if !content.is_empty() {
                    messages.push(WireMessage {
                        role: "assistant".to_owned(),
                        content,
                    });
                }
            }
        }
    }

    Ok(Converted {
        system,
        messages,
        warnings,
    })
}

/// Borrow the `provider_options` on a user part.
fn provider_user_part_options(
    part: &UserPart,
) -> Option<&llmsdk_provider::shared::ProviderOptions> {
    match part {
        UserPart::Text(t) => t.provider_options.as_ref(),
        UserPart::File(f) => f.provider_options.as_ref(),
    }
}

/// Encode a tool result output payload into Bedrock `toolResult.content[]`.
fn encode_tool_result_output(output: &ToolResultOutput) -> Vec<ToolResultPart> {
    match output {
        ToolResultOutput::Text { value, .. } | ToolResultOutput::ErrorText { value, .. } => {
            vec![ToolResultPart::Text {
                text: value.clone(),
            }]
        }
        ToolResultOutput::Json { value, .. } | ToolResultOutput::ErrorJson { value, .. } => {
            vec![ToolResultPart::Text {
                text: serde_json::to_string(value).unwrap_or_else(|_| "{}".to_owned()),
            }]
        }
        ToolResultOutput::ExecutionDenied { reason, .. } => vec![ToolResultPart::Text {
            text: reason
                .clone()
                .unwrap_or_else(|| "Tool call execution denied.".to_owned()),
        }],
        ToolResultOutput::Content { value } => {
            let mut parts: Vec<ToolResultPart> = Vec::new();
            for item in value {
                if let llmsdk_provider::language_model::ToolOutputPart::Text { text, .. } = item {
                    parts.push(ToolResultPart::Text { text: text.clone() });
                }
            }
            if parts.is_empty() {
                parts.push(ToolResultPart::Text {
                    text: String::new(),
                });
            }
            parts
        }
    }
}

/// Append a single user-side part to the cumulative content list.
fn append_user_part(
    part: &UserPart,
    _is_mistral: bool,
    document_counter: &mut u32,
    content: &mut Vec<ContentBlock>,
    warnings: &mut Vec<Warning>,
) {
    match part {
        UserPart::Text(t) => {
            content.push(ContentBlock::Text {
                text: t.text.clone(),
            });
        }
        UserPart::File(file) => {
            let media = file.media_type.as_str();
            match &file.data {
                FileData::Url { .. } => {
                    warnings.push(Warning::UnsupportedSetting {
                        setting: "file.url".to_owned(),
                        details: Some(
                            "Bedrock does not accept URL-sourced files; provide bytes inline."
                                .to_owned(),
                        ),
                    });
                }
                FileData::Reference { .. } => {
                    warnings.push(Warning::UnsupportedSetting {
                        setting: "file.reference".to_owned(),
                        details: Some(
                            "Bedrock does not accept provider-reference files in chat content."
                                .to_owned(),
                        ),
                    });
                }
                FileData::Text { text } => {
                    let format = document_format_for_media_type(media).unwrap_or("txt");
                    let name = derive_document_name(file.filename.as_deref(), document_counter);
                    let enable_citations = parse_file_part(file.provider_options.as_ref())
                        .citations
                        .is_some_and(|c| c.enabled);
                    let bytes = base64_encode(text.as_bytes());
                    content.push(ContentBlock::Document {
                        document: DocumentBlock {
                            format: format.to_owned(),
                            name,
                            source: BytesSource { bytes },
                            citations: enable_citations
                                .then_some(CitationsConfig { enabled: true }),
                        },
                    });
                }
                FileData::Data { data } => {
                    let bytes_b64 = match data {
                        FileBytes::Base64(s) => s.clone(),
                        FileBytes::Bytes(b) => base64_encode(b),
                    };
                    if media.starts_with("image/") {
                        let Some(format) = image_format_for_media_type(media) else {
                            warnings.push(Warning::UnsupportedSetting {
                                setting: "file.image".to_owned(),
                                details: Some(format!("unsupported image media type {media}")),
                            });
                            return;
                        };
                        content.push(ContentBlock::Image {
                            image: ImageBlock {
                                format: format.to_owned(),
                                source: BytesSource { bytes: bytes_b64 },
                            },
                        });
                    } else {
                        let format = document_format_for_media_type(media).unwrap_or("txt");
                        let name = derive_document_name(file.filename.as_deref(), document_counter);
                        let enable_citations = parse_file_part(file.provider_options.as_ref())
                            .citations
                            .is_some_and(|c| c.enabled);
                        content.push(ContentBlock::Document {
                            document: DocumentBlock {
                                format: format.to_owned(),
                                name,
                                source: BytesSource { bytes: bytes_b64 },
                                citations: enable_citations
                                    .then_some(CitationsConfig { enabled: true }),
                            },
                        });
                    }
                }
            }
        }
    }
}

/// Map an IANA image media type to Bedrock's `format` string.
fn image_format_for_media_type(media: &str) -> Option<&'static str> {
    match media {
        "image/jpeg" | "image/jpg" => Some("jpeg"),
        "image/png" => Some("png"),
        "image/gif" => Some("gif"),
        "image/webp" => Some("webp"),
        _ => None,
    }
}

/// Map an IANA document media type to Bedrock's `format` string.
fn document_format_for_media_type(media: &str) -> Option<&'static str> {
    match media {
        "application/pdf" => Some("pdf"),
        "text/csv" => Some("csv"),
        "application/msword" => Some("doc"),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => Some("docx"),
        "application/vnd.ms-excel" => Some("xls"),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" => Some("xlsx"),
        "text/html" => Some("html"),
        "text/plain" => Some("txt"),
        "text/markdown" => Some("md"),
        _ => None,
    }
}

fn derive_document_name(filename: Option<&str>, counter: &mut u32) -> String {
    if let Some(name) = filename {
        let trimmed = name.rsplit_once('.').map_or(name, |(stem, _)| stem);
        if !trimmed.is_empty() {
            return trimmed.to_owned();
        }
    }
    *counter = counter.saturating_add(1);
    format!("document-{counter}")
}

/// Minimal base64 encoder (RFC 4648 §4) — used to avoid pulling a new
/// dependency. Mirrors the implementation in `llmsdk-openai` for image bytes.
pub(crate) fn base64_encode(input: &[u8]) -> String {
    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    let mut chunks = input.chunks_exact(3);
    for chunk in chunks.by_ref() {
        let b0 = chunk[0] as usize;
        let b1 = chunk[1] as usize;
        let b2 = chunk[2] as usize;
        out.push(TABLE[(b0 >> 2) & 0x3f] as char);
        out.push(TABLE[((b0 << 4) | (b1 >> 4)) & 0x3f] as char);
        out.push(TABLE[((b1 << 2) | (b2 >> 6)) & 0x3f] as char);
        out.push(TABLE[b2 & 0x3f] as char);
    }
    let remainder = chunks.remainder();
    match remainder.len() {
        1 => {
            let b0 = remainder[0] as usize;
            out.push(TABLE[(b0 >> 2) & 0x3f] as char);
            out.push(TABLE[(b0 << 4) & 0x3f] as char);
            out.push('=');
            out.push('=');
        }
        2 => {
            let b0 = remainder[0] as usize;
            let b1 = remainder[1] as usize;
            out.push(TABLE[(b0 >> 2) & 0x3f] as char);
            out.push(TABLE[((b0 << 4) | (b1 >> 4)) & 0x3f] as char);
            out.push(TABLE[(b1 << 2) & 0x3f] as char);
            out.push('=');
        }
        _ => {}
    }
    out
}

#[derive(Debug)]
enum Block {
    System(Vec<Message>),
    User(Vec<Message>),
    Assistant(Vec<Message>),
}

fn group_into_blocks(prompt: &Prompt) -> Vec<Block> {
    let mut out: Vec<Block> = Vec::new();
    for message in prompt {
        #[allow(
            clippy::match_same_arms,
            reason = "arms differ only by inner block variant; merging breaks symmetry with `User`/`Tool` case"
        )]
        match (out.last_mut(), message) {
            (Some(Block::System(list)), m @ Message::System { .. }) => list.push(m.clone()),
            (Some(Block::Assistant(list)), m @ Message::Assistant { .. }) => list.push(m.clone()),
            (Some(Block::User(list)), m @ (Message::User { .. } | Message::Tool { .. })) => {
                list.push(m.clone());
            }
            (_, m @ Message::System { .. }) => out.push(Block::System(vec![m.clone()])),
            (_, m @ Message::Assistant { .. }) => out.push(Block::Assistant(vec![m.clone()])),
            (_, m @ (Message::User { .. } | Message::Tool { .. })) => {
                out.push(Block::User(vec![m.clone()]));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmsdk_provider::language_model::TextPart;

    fn user_text(text: &str) -> Message {
        Message::User {
            content: vec![UserPart::Text(TextPart {
                text: text.into(),
                provider_options: None,
            })],
            provider_options: None,
        }
    }

    #[test]
    fn simple_user_text_round_trips() {
        let converted = convert_prompt(&vec![user_text("hi")], false).unwrap();
        assert!(converted.system.is_empty());
        assert_eq!(converted.messages.len(), 1);
        assert_eq!(converted.messages[0].role, "user");
        assert!(matches!(
            converted.messages[0].content[0],
            ContentBlock::Text { .. }
        ));
    }

    #[test]
    fn system_message_lands_in_system_block() {
        let prompt = vec![
            Message::System {
                content: "be helpful".into(),
                provider_options: None,
            },
            user_text("hi"),
        ];
        let converted = convert_prompt(&prompt, false).unwrap();
        assert_eq!(converted.system.len(), 1);
        assert!(matches!(converted.system[0], SystemBlock::Text { .. }));
        assert_eq!(converted.messages.len(), 1);
    }

    #[test]
    fn assistant_text_trimmed_when_last() {
        let prompt = vec![
            user_text("hi"),
            Message::Assistant {
                content: vec![AssistantPart::Text(TextPart {
                    text: "hello   ".into(),
                    provider_options: None,
                })],
                provider_options: None,
            },
        ];
        let converted = convert_prompt(&prompt, false).unwrap();
        let last = converted.messages.last().unwrap();
        match &last.content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "hello"),
            _ => panic!("expected text"),
        }
    }

    #[test]
    fn second_system_message_is_rejected() {
        let prompt = vec![
            Message::System {
                content: "a".into(),
                provider_options: None,
            },
            user_text("hi"),
            Message::System {
                content: "b".into(),
                provider_options: None,
            },
            user_text("again"),
        ];
        let err = convert_prompt(&prompt, false).unwrap_err();
        assert!(format!("{err}").contains("system"));
    }

    #[test]
    fn base64_encode_matches_rfc_examples() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }
}
