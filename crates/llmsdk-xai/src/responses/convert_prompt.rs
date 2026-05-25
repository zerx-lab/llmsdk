//! Convert an [`llmsdk_provider::language_model::Prompt`] into xAI Responses
//! input items.
//!
//! Mirrors `convert-to-xai-responses-input.ts`. xAI Responses accepts a flat
//! list of input items (system/developer/user/assistant/function_call/
//! function_call_output/reasoning), not OpenAI-style chat messages.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    AssistantPart, FilePart, Message, Prompt, ToolMessagePart, ToolResultOutput, UserPart,
};
use llmsdk_provider::shared::{FileBytes, FileData, Warning};
use serde_json::{Map, Value, json};

/// Convert a [`Prompt`] into xAI Responses input items.
///
/// # Errors
///
/// Returns [`ProviderError::unsupported`] for combinations the upstream
/// implementation rejects hard (matches `UnsupportedFunctionalityError`).
pub(crate) fn convert_prompt(prompt: &Prompt) -> Result<(Vec<Value>, Vec<Warning>), ProviderError> {
    let mut input: Vec<Value> = Vec::with_capacity(prompt.len());
    let mut warnings: Vec<Warning> = Vec::new();

    for message in prompt {
        match message {
            Message::System { content, .. } => {
                input.push(json!({ "role": "system", "content": content }));
            }
            Message::User { content, .. } => {
                input.push(convert_user(content, &mut warnings)?);
            }
            Message::Assistant { content, .. } => {
                convert_assistant(content, &mut input, &mut warnings);
            }
            Message::Tool { content, .. } => {
                convert_tool_message(content, &mut input);
            }
        }
    }

    Ok((input, warnings))
}

fn convert_user(parts: &[UserPart], warnings: &mut [Warning]) -> Result<Value, ProviderError> {
    let mut content_parts: Vec<Value> = Vec::with_capacity(parts.len());
    for part in parts {
        match part {
            UserPart::Text(t) => {
                content_parts.push(json!({ "type": "input_text", "text": t.text }));
            }
            UserPart::File(f) => {
                if let Some(v) = convert_user_file(f, warnings)? {
                    content_parts.push(v);
                }
            }
        }
    }
    Ok(json!({ "role": "user", "content": content_parts }))
}

fn convert_user_file(
    file: &FilePart,
    _warnings: &mut [Warning],
) -> Result<Option<Value>, ProviderError> {
    match &file.data {
        FileData::Reference { reference } => {
            let file_id = reference
                .get("xai")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ProviderError::invalid_argument(
                        "file.data.reference",
                        "xAI file reference must contain a string `xai` entry",
                    )
                })?
                .to_owned();
            Ok(Some(json!({ "type": "input_file", "file_id": file_id })))
        }
        FileData::Text { .. } => Err(ProviderError::unsupported("text file parts")),
        FileData::Url { url } => {
            if top_level(&file.media_type) == "image" {
                Ok(Some(
                    json!({ "type": "input_image", "image_url": url.clone() }),
                ))
            } else {
                // Non-image documents go through `input_file` with `file_url`.
                // xAI's Responses API supports PDF / text / CSV / ... via URL.
                Ok(Some(
                    json!({ "type": "input_file", "file_url": url.clone() }),
                ))
            }
        }
        FileData::Data { data } => {
            if top_level(&file.media_type) == "image" {
                let payload = match data {
                    FileBytes::Base64(s) => s.clone(),
                    FileBytes::Bytes(b) => base64_encode(b),
                };
                let url = format!("data:{};base64,{}", file.media_type, payload);
                Ok(Some(json!({ "type": "input_image", "image_url": url })))
            } else {
                Err(ProviderError::unsupported(format!(
                    "file part media type {} as inline data (xAI Responses requires a URL or a Files API reference for non-image files)",
                    file.media_type
                )))
            }
        }
    }
}

fn top_level(media_type: &str) -> &str {
    media_type.split('/').next().unwrap_or(media_type)
}

/// Minimal base64 encoder (RFC 4648 §4) — same algorithm as the rest of the
/// crate. Kept in this module to avoid widening visibility from `chat`.
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

fn convert_assistant(parts: &[AssistantPart], input: &mut Vec<Value>, warnings: &mut Vec<Warning>) {
    for part in parts {
        match part {
            AssistantPart::Text(t) => {
                let item_id = xai_item_id(t.provider_options.as_ref());
                let mut msg = Map::new();
                msg.insert("role".into(), json!("assistant"));
                msg.insert("content".into(), json!(t.text));
                if let Some(id) = item_id {
                    msg.insert("id".into(), json!(id));
                }
                input.push(Value::Object(msg));
            }
            AssistantPart::ToolCall(tc) => {
                // Provider-executed tool calls are not echoed back upstream.
                if tc.provider_executed.unwrap_or(false) {
                    continue;
                }
                let item_id = xai_item_id(tc.provider_options.as_ref());
                let arguments =
                    serde_json::to_string(&tc.input).unwrap_or_else(|_| "{}".to_owned());
                let mut call = Map::new();
                call.insert("type".into(), json!("function_call"));
                call.insert(
                    "id".into(),
                    json!(item_id.clone().unwrap_or_else(|| tc.tool_call_id.clone())),
                );
                call.insert("call_id".into(), json!(tc.tool_call_id.clone()));
                call.insert("name".into(), json!(tc.tool_name.clone()));
                call.insert("arguments".into(), json!(arguments));
                call.insert("status".into(), json!("completed"));
                input.push(Value::Object(call));
            }
            AssistantPart::Reasoning {
                text,
                provider_options,
            } => {
                let item_id = xai_item_id(provider_options.as_ref());
                let encrypted = xai_encrypted_content(provider_options.as_ref());
                if item_id.is_none() && encrypted.is_none() {
                    warnings.push(Warning::Other {
                        message: "Reasoning parts without itemId or encrypted content cannot be sent back to xAI. Skipping.".to_owned(),
                    });
                    continue;
                }
                let mut summary: Vec<Value> = Vec::new();
                if !text.is_empty() {
                    summary.push(json!({ "type": "summary_text", "text": text }));
                }
                let mut item = Map::new();
                item.insert("type".into(), json!("reasoning"));
                item.insert("id".into(), json!(item_id.unwrap_or_default()));
                item.insert("summary".into(), json!(summary));
                item.insert("status".into(), json!("completed"));
                if let Some(enc) = encrypted {
                    item.insert("encrypted_content".into(), json!(enc));
                }
                input.push(Value::Object(item));
            }
            AssistantPart::ReasoningFile { .. } => warnings.push(Warning::Other {
                message: "xAI Responses API does not support reasoning-file in assistant messages"
                    .to_owned(),
            }),
            AssistantPart::File(_) => warnings.push(Warning::Other {
                message: "xAI Responses API does not support file in assistant messages".to_owned(),
            }),
            AssistantPart::Custom { kind, .. } => warnings.push(Warning::Other {
                message: format!(
                    "xAI Responses API does not support custom assistant content `{kind}`"
                ),
            }),
            AssistantPart::ToolResult(_) => {
                // ai-sdk drops inline tool-result on assistant role silently.
            }
        }
    }
}

fn convert_tool_message(parts: &[ToolMessagePart], input: &mut Vec<Value>) {
    for part in parts {
        let ToolMessagePart::ToolResult(r) = part else {
            // Approval responses are not relayed by the xAI Responses API.
            continue;
        };
        let output_value = match &r.output {
            ToolResultOutput::Text { value, .. } | ToolResultOutput::ErrorText { value, .. } => {
                value.clone()
            }
            ToolResultOutput::ExecutionDenied { reason, .. } => reason
                .clone()
                .unwrap_or_else(|| "tool execution denied".to_owned()),
            ToolResultOutput::Json { value, .. } | ToolResultOutput::ErrorJson { value, .. } => {
                serde_json::to_string(value).unwrap_or_else(|_| "{}".to_owned())
            }
            ToolResultOutput::Content { value } => {
                // Flatten only the text fragments; non-text parts are dropped.
                // Matches upstream xai-responses convert-to-input behaviour.
                let mut buf = String::new();
                for item in value {
                    let serialized = serde_json::to_value(item).unwrap_or(Value::Null);
                    if serialized.get("type").and_then(Value::as_str) == Some("text")
                        && let Some(text) = serialized.get("text").and_then(Value::as_str)
                    {
                        buf.push_str(text);
                    }
                }
                buf
            }
        };
        input.push(json!({
            "type": "function_call_output",
            "call_id": r.tool_call_id.clone(),
            "output": output_value,
        }));
    }
}

fn xai_item_id(options: Option<&llmsdk_provider::shared::ProviderOptions>) -> Option<String> {
    options?
        .get("xai")?
        .get("itemId")
        .and_then(Value::as_str)
        .map(str::to_owned)
}

fn xai_encrypted_content(
    options: Option<&llmsdk_provider::shared::ProviderOptions>,
) -> Option<String> {
    options?
        .get("xai")?
        .get("reasoningEncryptedContent")
        .and_then(Value::as_str)
        .map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmsdk_provider::language_model::{TextPart, ToolCallPart, ToolResultPart};
    use llmsdk_provider::shared::ProviderOptions;
    use serde_json::json;

    fn make_options(slot: serde_json::Value) -> ProviderOptions {
        let mut po = ProviderOptions::new();
        po.insert("xai".into(), slot.as_object().cloned().unwrap());
        po
    }

    #[test]
    fn system_message_emits_role_system() {
        let prompt = vec![Message::System {
            content: "be brief".into(),
            provider_options: None,
        }];
        let (out, _) = convert_prompt(&prompt).unwrap();
        assert_eq!(out[0]["role"], "system");
        assert_eq!(out[0]["content"], "be brief");
    }

    #[test]
    fn user_text_emits_input_text() {
        let prompt = vec![Message::User {
            content: vec![UserPart::Text(TextPart {
                text: "hi".into(),
                provider_options: None,
            })],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(&prompt).unwrap();
        assert_eq!(out[0]["content"][0]["type"], "input_text");
        assert_eq!(out[0]["content"][0]["text"], "hi");
    }

    #[test]
    fn user_image_url_emits_input_image() {
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
        assert_eq!(out[0]["content"][0]["type"], "input_image");
        assert_eq!(
            out[0]["content"][0]["image_url"],
            "https://example.com/a.png"
        );
    }

    #[test]
    fn user_image_data_inline_base64() {
        let prompt = vec![Message::User {
            content: vec![UserPart::File(FilePart {
                filename: None,
                data: FileData::Data {
                    data: FileBytes::Bytes(vec![1, 2, 3]),
                },
                media_type: "image/png".into(),
                provider_options: None,
            })],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(&prompt).unwrap();
        let url = out[0]["content"][0]["image_url"].as_str().unwrap();
        assert!(url.starts_with("data:image/png;base64,"));
    }

    #[test]
    fn user_pdf_url_emits_input_file_with_file_url() {
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
        assert_eq!(out[0]["content"][0]["type"], "input_file");
        assert_eq!(
            out[0]["content"][0]["file_url"],
            "https://example.com/a.pdf"
        );
    }

    #[test]
    fn user_file_reference_emits_input_file_with_file_id() {
        let mut reference = Map::new();
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
        assert_eq!(out[0]["content"][0]["file_id"], "file_abc123");
    }

    #[test]
    fn assistant_tool_call_emits_function_call_item() {
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
        assert_eq!(out[0]["type"], "function_call");
        assert_eq!(out[0]["call_id"], "call_1");
        assert_eq!(out[0]["arguments"], r#"{"city":"NYC"}"#);
    }

    #[test]
    fn assistant_provider_executed_tool_call_dropped() {
        let prompt = vec![Message::Assistant {
            content: vec![AssistantPart::ToolCall(ToolCallPart {
                tool_call_id: "call_p".into(),
                tool_name: "web_search".into(),
                input: json!({}),
                provider_executed: Some(true),
                dynamic: None,
                provider_options: None,
            })],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(&prompt).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn reasoning_without_metadata_warns_and_drops() {
        let prompt = vec![Message::Assistant {
            content: vec![AssistantPart::Reasoning {
                text: "thinking".into(),
                provider_options: None,
            }],
            provider_options: None,
        }];
        let (out, warnings) = convert_prompt(&prompt).unwrap();
        assert!(out.is_empty());
        assert_eq!(warnings.len(), 1);
    }

    #[test]
    fn reasoning_with_encrypted_content_passes_through() {
        let prompt = vec![Message::Assistant {
            content: vec![AssistantPart::Reasoning {
                text: "thinking".into(),
                provider_options: Some(make_options(json!({
                    "itemId": "rs_1",
                    "reasoningEncryptedContent": "enc_xxx"
                }))),
            }],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(&prompt).unwrap();
        assert_eq!(out[0]["type"], "reasoning");
        assert_eq!(out[0]["id"], "rs_1");
        assert_eq!(out[0]["encrypted_content"], "enc_xxx");
        assert_eq!(out[0]["summary"][0]["text"], "thinking");
    }

    #[test]
    fn tool_result_emits_function_call_output() {
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
        assert_eq!(out[0]["type"], "function_call_output");
        assert_eq!(out[0]["call_id"], "call_1");
        assert_eq!(out[0]["output"], "sunny");
    }
}
