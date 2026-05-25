//! Convert llmsdk [`Prompt`] → Gemini `{ systemInstruction, contents }`.
//!
//! Mirrors `@ai-sdk/google/src/convert-to-google-messages.ts`. Handles:
//!
//! - System messages → `systemInstruction.parts[]` (only at the start;
//!   Gemma models inline them into the first user message).
//! - User parts: text → `text`, file URL → `fileData`, file reference →
//!   `fileData` with resolved URI, inline data → `inlineData`, text data
//!   → base64 `inlineData` of UTF-8 bytes.
//! - Assistant parts: text / reasoning (`thought:true`) / reasoning-file /
//!   file (data / reference) / tool-call (function or server) / tool-result
//!   (server tools only — client results live in the next tool message).
//! - Tool messages: tool-results (multi-part via Gemini 3 `parts[]` array
//!   or legacy single-text fallback) and provider-executed
//!   `toolResponse` (folded into the previous assistant `model` message).
// Rust guideline compliant 2026-05-25

use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    AssistantPart, FilePart, Message, Prompt, ToolMessagePart, ToolOutputPart, ToolResultOutput,
    UserPart,
};
use llmsdk_provider::shared::{FileBytes, FileData, ProviderOptions};
use serde_json::{Map, Value};

/// Result of converting a prompt.
#[derive(Debug, Default)]
pub(crate) struct ConvertedPrompt {
    pub system_instruction: Option<Value>,
    pub contents: Vec<Value>,
}

/// Options modifying the conversion.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ConvertOptions<'a> {
    pub is_gemma_model: bool,
    /// `["google"]` for Gemini API, `["googleVertex","vertex"]` for Vertex.
    pub provider_option_names: &'a [&'a str],
    pub supports_function_response_parts: bool,
}

/// Convert a prompt for the Gemini wire.
///
/// # Errors
///
/// Returns [`ProviderError::unsupported`] / [`ProviderError::invalid_prompt`]
/// for inputs Gemini cannot encode (e.g. file URL in assistant messages,
/// system message in the middle of the conversation).
pub(crate) fn convert_to_google_messages(
    prompt: &Prompt,
    opts: ConvertOptions<'_>,
) -> Result<ConvertedPrompt, ProviderError> {
    let mut system_parts: Vec<Value> = Vec::new();
    let mut contents: Vec<Value> = Vec::new();
    let mut system_allowed = true;
    let is_vertex_like = !opts.provider_option_names.contains(&"google");

    for message in prompt {
        match message {
            Message::System { content, .. } => {
                if !system_allowed {
                    return Err(ProviderError::invalid_prompt(
                        "system messages are only supported at the beginning of the conversation",
                    ));
                }
                let mut m = Map::new();
                m.insert("text".into(), Value::String(content.clone()));
                system_parts.push(Value::Object(m));
            }
            Message::User { content, .. } => {
                system_allowed = false;
                let parts = convert_user_parts(content, is_vertex_like)?;
                contents.push(make_content("user", parts));
            }
            Message::Assistant { content, .. } => {
                system_allowed = false;
                let parts =
                    convert_assistant_parts(content, opts.provider_option_names, is_vertex_like)?;
                contents.push(make_content("model", parts));
            }
            Message::Tool { content, .. } => {
                system_allowed = false;
                let parts = convert_tool_parts(
                    content,
                    opts.provider_option_names,
                    opts.supports_function_response_parts,
                    &mut contents,
                )?;
                contents.push(make_content("user", parts));
            }
        }
    }

    // Gemma quirk: prepend system text into the first user message.
    if opts.is_gemma_model && !system_parts.is_empty() && !contents.is_empty() {
        let first_role = contents[0]
            .as_object()
            .and_then(|o| o.get("role"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if first_role == "user" {
            let merged_text = system_parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n\n")
                + "\n\n";
            if let Some(obj) = contents[0].as_object_mut()
                && let Some(parts) = obj.get_mut("parts").and_then(Value::as_array_mut)
            {
                let mut head = Map::new();
                head.insert("text".into(), Value::String(merged_text));
                parts.insert(0, Value::Object(head));
            }
        }
    }

    let system_instruction = if opts.is_gemma_model || system_parts.is_empty() {
        None
    } else {
        let mut m = Map::new();
        m.insert("parts".into(), Value::Array(system_parts));
        Some(Value::Object(m))
    };

    Ok(ConvertedPrompt {
        system_instruction,
        contents,
    })
}

fn make_content(role: &str, parts: Vec<Value>) -> Value {
    let mut o = Map::new();
    o.insert("role".into(), Value::String(role.into()));
    o.insert("parts".into(), Value::Array(parts));
    Value::Object(o)
}

fn convert_user_parts(parts: &[UserPart], is_vertex: bool) -> Result<Vec<Value>, ProviderError> {
    let mut out = Vec::with_capacity(parts.len());
    for part in parts {
        match part {
            UserPart::Text(t) => {
                let mut o = Map::new();
                o.insert("text".into(), Value::String(t.text.clone()));
                out.push(Value::Object(o));
            }
            UserPart::File(f) => {
                out.push(file_to_part(f, is_vertex, false, None)?);
            }
        }
    }
    Ok(out)
}

fn convert_assistant_parts(
    parts: &[AssistantPart],
    option_names: &[&str],
    is_vertex: bool,
) -> Result<Vec<Value>, ProviderError> {
    let mut out = Vec::with_capacity(parts.len());
    for part in parts {
        let provider_opts = read_provider_options(
            provider_options_of_assistant_part(part),
            option_names,
            is_vertex,
        );
        let thought_signature = provider_opts
            .as_ref()
            .and_then(|o| o.get("thoughtSignature"))
            .and_then(Value::as_str)
            .map(str::to_owned);

        match part {
            AssistantPart::Text(t) => {
                if t.text.is_empty() {
                    continue;
                }
                out.push(text_part(&t.text, false, thought_signature.as_deref()));
            }
            AssistantPart::Reasoning { text, .. } => {
                if text.is_empty() {
                    continue;
                }
                out.push(text_part(text, true, thought_signature.as_deref()));
            }
            AssistantPart::ReasoningFile {
                data, media_type, ..
            } => match data {
                FileData::Url { .. } => {
                    return Err(ProviderError::unsupported(
                        "File data URLs in assistant messages are not supported",
                    ));
                }
                FileData::Data { data } => {
                    out.push(inline_part(
                        media_type,
                        &bytes_to_base64_string(data),
                        true,
                        thought_signature.as_deref(),
                    ));
                }
                _ => {
                    return Err(ProviderError::unsupported(
                        "Only inline data is supported for reasoning files",
                    ));
                }
            },
            AssistantPart::File(f) => {
                let part_value = file_to_part(f, is_vertex, true, thought_signature.as_deref())?;
                out.push(part_value);
            }
            AssistantPart::Custom { .. } => {
                // Custom parts are opaque; skip (matches upstream — they
                // are filtered out by the `undefined` mapper).
            }
            AssistantPart::ToolCall(tc) => {
                let server_tool_call_id = provider_opts
                    .as_ref()
                    .and_then(|o| o.get("serverToolCallId"))
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                let server_tool_type = provider_opts
                    .as_ref()
                    .and_then(|o| o.get("serverToolType"))
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                if let (Some(id), Some(ty)) = (server_tool_call_id, server_tool_type) {
                    let args = if let Value::String(s) = &tc.input {
                        serde_json::from_str::<Value>(s).unwrap_or(Value::Null)
                    } else {
                        tc.input.clone()
                    };
                    let mut o = Map::new();
                    let mut tc_obj = Map::new();
                    tc_obj.insert("toolType".into(), Value::String(ty));
                    if !args.is_null() {
                        tc_obj.insert("args".into(), args);
                    }
                    tc_obj.insert("id".into(), Value::String(id));
                    o.insert("toolCall".into(), Value::Object(tc_obj));
                    if let Some(sig) = thought_signature.as_deref() {
                        o.insert("thoughtSignature".into(), Value::String(sig.to_owned()));
                    }
                    out.push(Value::Object(o));
                } else {
                    let mut o = Map::new();
                    let mut fc = Map::new();
                    if !tc.tool_call_id.is_empty() {
                        fc.insert("id".into(), Value::String(tc.tool_call_id.clone()));
                    }
                    fc.insert("name".into(), Value::String(tc.tool_name.clone()));
                    fc.insert("args".into(), tc.input.clone());
                    o.insert("functionCall".into(), Value::Object(fc));
                    if let Some(sig) = thought_signature.as_deref() {
                        o.insert("thoughtSignature".into(), Value::String(sig.to_owned()));
                    }
                    out.push(Value::Object(o));
                }
            }
            AssistantPart::ToolResult(tr) => {
                // Only emit server-tool results inline; client results go
                // through the next `Message::Tool` block.
                let server_tool_call_id = provider_opts
                    .as_ref()
                    .and_then(|o| o.get("serverToolCallId"))
                    .and_then(Value::as_str);
                let server_tool_type = provider_opts
                    .as_ref()
                    .and_then(|o| o.get("serverToolType"))
                    .and_then(Value::as_str);
                if let (Some(id), Some(ty)) = (server_tool_call_id, server_tool_type) {
                    let resp = match &tr.output {
                        ToolResultOutput::Json { value, .. } => value.clone(),
                        _ => Value::Object(Map::new()),
                    };
                    let mut o = Map::new();
                    let mut wrap = Map::new();
                    wrap.insert("toolType".into(), Value::String(ty.to_owned()));
                    wrap.insert("response".into(), resp);
                    wrap.insert("id".into(), Value::String(id.to_owned()));
                    o.insert("toolResponse".into(), Value::Object(wrap));
                    if let Some(sig) = thought_signature.as_deref() {
                        o.insert("thoughtSignature".into(), Value::String(sig.to_owned()));
                    }
                    out.push(Value::Object(o));
                }
            }
        }
    }
    Ok(out)
}

fn convert_tool_parts(
    parts: &[ToolMessagePart],
    option_names: &[&str],
    supports_multipart: bool,
    contents: &mut [Value],
) -> Result<Vec<Value>, ProviderError> {
    let mut out = Vec::new();
    for part in parts {
        let ToolMessagePart::ToolResult(r) = part else {
            // Approval responses aren't forwarded on the Gemini wire.
            continue;
        };
        let provider_opts = read_provider_options(r.provider_options.as_ref(), option_names, false);
        let server_tool_call_id = provider_opts
            .as_ref()
            .and_then(|o| o.get("serverToolCallId"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        let server_tool_type = provider_opts
            .as_ref()
            .and_then(|o| o.get("serverToolType"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        if let (Some(id), Some(ty)) = (server_tool_call_id, server_tool_type) {
            // Fold into the preceding model message when one is present.
            if let Some(last) = contents.last_mut()
                && last.get("role").and_then(Value::as_str) == Some("model")
                && let Some(arr) = last.get_mut("parts").and_then(Value::as_array_mut)
            {
                let resp = match &r.output {
                    ToolResultOutput::Json { value, .. } => value.clone(),
                    _ => Value::Object(Map::new()),
                };
                let mut wrap = Map::new();
                wrap.insert("toolType".into(), Value::String(ty));
                wrap.insert("response".into(), resp);
                wrap.insert("id".into(), Value::String(id));
                let mut o = Map::new();
                o.insert("toolResponse".into(), Value::Object(wrap));
                arr.push(Value::Object(o));
                continue;
            }
        }

        match &r.output {
            ToolResultOutput::Content { value } => {
                if supports_multipart {
                    append_tool_result_multipart(
                        &mut out,
                        &r.tool_name,
                        value,
                        Some(&r.tool_call_id),
                    );
                } else {
                    append_tool_result_legacy(&mut out, &r.tool_name, value, Some(&r.tool_call_id));
                }
            }
            ToolResultOutput::Text { value, .. } => {
                out.push(function_response(
                    &r.tool_name,
                    Some(&r.tool_call_id),
                    Value::String(value.clone()),
                ));
            }
            ToolResultOutput::Json { value, .. } => {
                out.push(function_response(
                    &r.tool_name,
                    Some(&r.tool_call_id),
                    value.clone(),
                ));
            }
            ToolResultOutput::ExecutionDenied { reason, .. } => {
                let resp = Value::String(
                    reason
                        .clone()
                        .unwrap_or_else(|| "Tool call execution denied.".into()),
                );
                out.push(function_response(&r.tool_name, Some(&r.tool_call_id), resp));
            }
            ToolResultOutput::ErrorText { value, .. } => {
                out.push(function_response(
                    &r.tool_name,
                    Some(&r.tool_call_id),
                    Value::String(value.clone()),
                ));
            }
            ToolResultOutput::ErrorJson { value, .. } => {
                out.push(function_response(
                    &r.tool_name,
                    Some(&r.tool_call_id),
                    value.clone(),
                ));
            }
        }
    }
    Ok(out)
}

fn append_tool_result_multipart(
    out: &mut Vec<Value>,
    tool_name: &str,
    output_value: &[ToolOutputPart],
    tool_call_id: Option<&str>,
) {
    let mut response_text_parts: Vec<String> = Vec::new();
    let mut response_parts: Vec<Value> = Vec::new();

    for part in output_value {
        match part {
            ToolOutputPart::Text { text, .. } => response_text_parts.push(text.clone()),
            ToolOutputPart::File {
                data, media_type, ..
            } => match data {
                FileData::Data { data } => {
                    let mut o = Map::new();
                    let mut inline = Map::new();
                    inline.insert("mimeType".into(), Value::String(media_type.clone()));
                    inline.insert("data".into(), Value::String(bytes_to_base64_string(data)));
                    o.insert("inlineData".into(), Value::Object(inline));
                    response_parts.push(Value::Object(o));
                }
                _ => {
                    response_text_parts.push(serde_json::to_string(part).unwrap_or_default());
                }
            },
            ToolOutputPart::Custom { .. } => {
                response_text_parts.push(serde_json::to_string(part).unwrap_or_default());
            }
        }
    }

    let mut wrap = Map::new();
    if let Some(id) = tool_call_id {
        wrap.insert("id".into(), Value::String(id.to_owned()));
    }
    wrap.insert("name".into(), Value::String(tool_name.to_owned()));
    let mut response = Map::new();
    response.insert("name".into(), Value::String(tool_name.to_owned()));
    response.insert(
        "content".into(),
        Value::String(if response_text_parts.is_empty() {
            "Tool executed successfully.".into()
        } else {
            response_text_parts.join("\n")
        }),
    );
    wrap.insert("response".into(), Value::Object(response));
    if !response_parts.is_empty() {
        wrap.insert("parts".into(), Value::Array(response_parts));
    }
    let mut o = Map::new();
    o.insert("functionResponse".into(), Value::Object(wrap));
    out.push(Value::Object(o));
}

fn append_tool_result_legacy(
    out: &mut Vec<Value>,
    tool_name: &str,
    output_value: &[ToolOutputPart],
    tool_call_id: Option<&str>,
) {
    for part in output_value {
        match part {
            ToolOutputPart::Text { text, .. } => {
                out.push(function_response(
                    tool_name,
                    tool_call_id,
                    Value::String(text.clone()),
                ));
            }
            ToolOutputPart::File {
                data, media_type, ..
            } => {
                if matches!(data, FileData::Data { .. }) && media_type.starts_with("image") {
                    if let FileData::Data { data: bytes } = data {
                        let mut inline = Map::new();
                        inline.insert("mimeType".into(), Value::String(media_type.clone()));
                        inline.insert("data".into(), Value::String(bytes_to_base64_string(bytes)));
                        let mut o1 = Map::new();
                        o1.insert("inlineData".into(), Value::Object(inline));
                        out.push(Value::Object(o1));
                        let mut o2 = Map::new();
                        o2.insert(
                            "text".into(),
                            Value::String(
                                "Tool executed successfully and returned this image as a response"
                                    .into(),
                            ),
                        );
                        out.push(Value::Object(o2));
                    }
                } else {
                    let mut o = Map::new();
                    o.insert(
                        "text".into(),
                        Value::String(serde_json::to_string(part).unwrap_or_default()),
                    );
                    out.push(Value::Object(o));
                }
            }
            ToolOutputPart::Custom { .. } => {
                let mut o = Map::new();
                o.insert(
                    "text".into(),
                    Value::String(serde_json::to_string(part).unwrap_or_default()),
                );
                out.push(Value::Object(o));
            }
        }
    }
}

fn function_response(name: &str, id: Option<&str>, content: Value) -> Value {
    let mut wrap = Map::new();
    if let Some(id) = id {
        wrap.insert("id".into(), Value::String(id.to_owned()));
    }
    wrap.insert("name".into(), Value::String(name.to_owned()));
    let mut response = Map::new();
    response.insert("name".into(), Value::String(name.to_owned()));
    response.insert("content".into(), content);
    wrap.insert("response".into(), Value::Object(response));
    let mut o = Map::new();
    o.insert("functionResponse".into(), Value::Object(wrap));
    Value::Object(o)
}

fn text_part(text: &str, is_thought: bool, sig: Option<&str>) -> Value {
    let mut o = Map::new();
    o.insert("text".into(), Value::String(text.to_owned()));
    if is_thought {
        o.insert("thought".into(), Value::Bool(true));
    }
    if let Some(sig) = sig {
        o.insert("thoughtSignature".into(), Value::String(sig.to_owned()));
    }
    Value::Object(o)
}

fn inline_part(media_type: &str, b64: &str, is_thought: bool, sig: Option<&str>) -> Value {
    let mut o = Map::new();
    let mut inline = Map::new();
    inline.insert("mimeType".into(), Value::String(media_type.to_owned()));
    inline.insert("data".into(), Value::String(b64.to_owned()));
    o.insert("inlineData".into(), Value::Object(inline));
    if is_thought {
        o.insert("thought".into(), Value::Bool(true));
    }
    if let Some(sig) = sig {
        o.insert("thoughtSignature".into(), Value::String(sig.to_owned()));
    }
    Value::Object(o)
}

fn file_to_part(
    f: &FilePart,
    is_vertex: bool,
    in_assistant: bool,
    sig: Option<&str>,
) -> Result<Value, ProviderError> {
    let mut o = Map::new();
    let media = if f.media_type.is_empty() {
        "application/octet-stream".to_owned()
    } else {
        f.media_type.clone()
    };
    match &f.data {
        FileData::Url { url } => {
            if in_assistant {
                return Err(ProviderError::unsupported(
                    "File data URLs in assistant messages are not supported",
                ));
            }
            let mut fd = Map::new();
            fd.insert("mimeType".into(), Value::String(media));
            fd.insert("fileUri".into(), Value::String(url.clone()));
            o.insert("fileData".into(), Value::Object(fd));
        }
        FileData::Reference { reference } => {
            if is_vertex {
                return Err(ProviderError::unsupported(
                    "file parts with provider references",
                ));
            }
            let uri = reference
                .get(crate::PROVIDER_ID)
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ProviderError::invalid_prompt(
                        "missing `google` entry in provider reference for file part",
                    )
                })?
                .to_owned();
            let mut fd = Map::new();
            fd.insert("mimeType".into(), Value::String(media));
            fd.insert("fileUri".into(), Value::String(uri));
            o.insert("fileData".into(), Value::Object(fd));
        }
        FileData::Data { data } => {
            let mut inline = Map::new();
            inline.insert("mimeType".into(), Value::String(media));
            inline.insert("data".into(), Value::String(bytes_to_base64_string(data)));
            o.insert("inlineData".into(), Value::Object(inline));
        }
        FileData::Text { text } => {
            let mut inline = Map::new();
            let mt = if f.media_type.contains('/') {
                f.media_type.clone()
            } else {
                "text/plain".to_owned()
            };
            inline.insert("mimeType".into(), Value::String(mt));
            inline.insert(
                "data".into(),
                Value::String(crate::base64::encode_bytes(text.as_bytes())),
            );
            o.insert("inlineData".into(), Value::Object(inline));
        }
    }
    if let Some(sig) = sig {
        o.insert("thoughtSignature".into(), Value::String(sig.to_owned()));
    }
    Ok(o.into())
}

fn provider_options_of_assistant_part(p: &AssistantPart) -> Option<&ProviderOptions> {
    match p {
        AssistantPart::Text(t) => t.provider_options.as_ref(),
        AssistantPart::Reasoning {
            provider_options, ..
        } => provider_options.as_ref(),
        AssistantPart::ReasoningFile {
            provider_options, ..
        } => provider_options.as_ref(),
        AssistantPart::File(f) => f.provider_options.as_ref(),
        AssistantPart::Custom {
            provider_options, ..
        } => provider_options.as_ref(),
        AssistantPart::ToolCall(t) => t.provider_options.as_ref(),
        AssistantPart::ToolResult(t) => t.provider_options.as_ref(),
    }
}

fn read_provider_options(
    opts: Option<&ProviderOptions>,
    candidates: &[&str],
    is_vertex_like: bool,
) -> Option<Map<String, Value>> {
    let opts = opts?;
    for name in candidates {
        if let Some(v) = opts.get(*name) {
            return Some(v.clone());
        }
    }
    // Cross-namespace fallback.
    if is_vertex_like {
        if let Some(v) = opts.get("google") {
            return Some(v.clone());
        }
    } else {
        if let Some(v) = opts.get("googleVertex").or_else(|| opts.get("vertex")) {
            return Some(v.clone());
        }
    }
    None
}

fn bytes_to_base64_string(b: &FileBytes) -> String {
    match b {
        FileBytes::Base64(s) => s.clone(),
        FileBytes::Bytes(buf) => crate::base64::encode_bytes(buf),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmsdk_provider::language_model::TextPart;

    #[test]
    fn system_to_system_instruction() {
        let prompt = vec![
            Message::System {
                content: "You are helpful".into(),
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
        let r = convert_to_google_messages(
            &prompt,
            ConvertOptions {
                provider_option_names: &["google"],
                supports_function_response_parts: true,
                is_gemma_model: false,
            },
        )
        .unwrap();
        let sys = r.system_instruction.unwrap();
        assert_eq!(sys["parts"][0]["text"], "You are helpful");
        assert_eq!(r.contents.len(), 1);
        assert_eq!(r.contents[0]["role"], "user");
    }

    #[test]
    fn gemma_inlines_system() {
        let prompt = vec![
            Message::System {
                content: "instr".into(),
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
        let r = convert_to_google_messages(
            &prompt,
            ConvertOptions {
                provider_option_names: &["google"],
                supports_function_response_parts: true,
                is_gemma_model: true,
            },
        )
        .unwrap();
        assert!(r.system_instruction.is_none());
        let first_user_first_part = &r.contents[0]["parts"][0]["text"];
        assert!(first_user_first_part.as_str().unwrap().starts_with("instr"));
    }

    #[test]
    fn system_in_middle_errors() {
        let prompt = vec![
            Message::User {
                content: vec![UserPart::Text(TextPart {
                    text: "hi".into(),
                    provider_options: None,
                })],
                provider_options: None,
            },
            Message::System {
                content: "no".into(),
                provider_options: None,
            },
        ];
        let r = convert_to_google_messages(
            &prompt,
            ConvertOptions {
                provider_option_names: &["google"],
                supports_function_response_parts: true,
                is_gemma_model: false,
            },
        );
        assert!(r.is_err());
    }
}
