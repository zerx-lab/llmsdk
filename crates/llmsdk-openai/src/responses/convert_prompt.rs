//! Convert llmsdk `Prompt` → OpenAI Responses `input[]` items.
//!
//! Mirrors `@ai-sdk/openai/src/responses/convert-to-openai-responses-input.ts`.
// Rust guideline compliant 2026-02-21

use llmsdk_provider::language_model::{
    AssistantPart, FilePart, Message, ToolApprovalResponsePart, ToolCallPart, ToolMessagePart,
    ToolResultOutput, ToolResultPart, UserPart,
};
use llmsdk_provider::shared::{FileBytes, FileData, ProviderOptions, Warning};

use super::options::SystemMessageMode;
use super::tools::ids;
use super::wire::request::{
    AssistantContentPart, AssistantRole, FunctionCallOutputBody, InputFile, InputImage, InputItem,
    InputMessage, SystemRole, TypedInputItem, UserContentPart, UserRole,
};
use super::wire::response::{MessagePhase, ReasoningSummary};

/// Per-call settings driving the conversion.
#[derive(Debug, Clone)]
pub struct ConvertCtx<'a> {
    pub system_message_mode: SystemMessageMode,
    /// Provider key (`"openai"` or `"azure"`) used when reading provider-options.
    pub provider_options_name: &'a str,
    pub pass_through_unsupported_files: bool,
    pub store: bool,
    pub has_conversation: bool,
    pub has_previous_response_id: bool,
    pub has_local_shell_tool: bool,
    pub has_shell_tool: bool,
    pub has_apply_patch_tool: bool,
}

impl Default for ConvertCtx<'_> {
    fn default() -> Self {
        Self {
            system_message_mode: SystemMessageMode::System,
            provider_options_name: "openai",
            pass_through_unsupported_files: false,
            store: true,
            has_conversation: false,
            has_previous_response_id: false,
            has_local_shell_tool: false,
            has_shell_tool: false,
            has_apply_patch_tool: false,
        }
    }
}

/// Convert one [`Prompt`] into a flat `input[]` array for the Responses API.
#[must_use]
pub fn convert_prompt(prompt: &[Message], ctx: &ConvertCtx<'_>) -> (Vec<InputItem>, Vec<Warning>) {
    let mut items = Vec::new();
    let mut warnings = Vec::new();

    for message in prompt {
        match message {
            Message::System { content, .. } => match ctx.system_message_mode {
                SystemMessageMode::System => {
                    items.push(InputItem::Message(InputMessage::SystemOrDeveloper {
                        role: SystemRole::System,
                        content: content.clone(),
                    }));
                }
                SystemMessageMode::Developer => {
                    items.push(InputItem::Message(InputMessage::SystemOrDeveloper {
                        role: SystemRole::Developer,
                        content: content.clone(),
                    }));
                }
                SystemMessageMode::Remove => {
                    warnings.push(Warning::Other {
                        message: "system message removed by systemMessageMode='remove'".into(),
                    });
                }
            },
            Message::User { content, .. } => {
                let parts = convert_user_parts(content, ctx, &mut warnings);
                if !parts.is_empty() {
                    items.push(InputItem::Message(InputMessage::User {
                        role: UserRole::User,
                        content: parts,
                    }));
                }
            }
            Message::Assistant {
                content,
                provider_options,
            } => convert_assistant(
                content,
                provider_options.as_ref(),
                ctx,
                &mut items,
                &mut warnings,
            ),
            Message::Tool { content, .. } => {
                convert_tool_message(content, ctx, &mut items, &mut warnings);
            }
        }
    }

    (items, warnings)
}

fn convert_user_parts(
    parts: &[UserPart],
    ctx: &ConvertCtx<'_>,
    warnings: &mut Vec<Warning>,
) -> Vec<UserContentPart> {
    let mut out = Vec::new();
    for part in parts {
        match part {
            UserPart::Text(t) => out.push(UserContentPart::InputText {
                text: t.text.clone(),
            }),
            UserPart::File(f) => {
                if let Some(p) = convert_user_file(f, ctx, warnings) {
                    out.push(p);
                }
            }
        }
    }
    out
}

fn convert_user_file(
    file: &FilePart,
    ctx: &ConvertCtx<'_>,
    warnings: &mut Vec<Warning>,
) -> Option<UserContentPart> {
    let mt = file.media_type.as_str();

    // Image
    if mt.starts_with("image/") {
        let detail = read_image_detail(file.provider_options.as_ref(), ctx);
        let payload = match &file.data {
            FileData::Url { url } => InputImage::Url {
                image_url: url.clone(),
                detail,
            },
            FileData::Data { data } => InputImage::Url {
                image_url: data_uri(mt, data),
                detail,
            },
            FileData::Reference { reference } => {
                let id = reference
                    .get(ctx.provider_options_name)
                    .or_else(|| reference.get("openai"))
                    .and_then(|v| v.as_str());
                match id {
                    Some(id) => InputImage::Reference {
                        file_id: id.to_string(),
                        detail,
                    },
                    None => {
                        warnings.push(Warning::Other {
                            message: format!("user file reference for {mt} missing OpenAI file_id"),
                        });
                        return None;
                    }
                }
            }
            FileData::Text { .. } => {
                warnings.push(Warning::Other {
                    message: "image file with inline text payload is not supported".into(),
                });
                return None;
            }
        };
        return Some(UserContentPart::InputImage(payload));
    }

    // PDF or pass-through file
    let is_pdf = mt == "application/pdf";
    if is_pdf || ctx.pass_through_unsupported_files {
        let payload = match &file.data {
            FileData::Url { url } => InputFile::Url {
                file_url: url.clone(),
            },
            FileData::Data { data } => InputFile::Data {
                filename: file.filename.clone().unwrap_or_else(|| "file".into()),
                file_data: data_uri(mt, data),
            },
            FileData::Reference { reference } => {
                let id = reference
                    .get(ctx.provider_options_name)
                    .or_else(|| reference.get("openai"))
                    .and_then(|v| v.as_str());
                match id {
                    Some(id) => InputFile::Reference {
                        file_id: id.to_string(),
                    },
                    None => {
                        warnings.push(Warning::Other {
                            message: format!("user file reference for {mt} missing OpenAI file_id"),
                        });
                        return None;
                    }
                }
            }
            FileData::Text { .. } => {
                warnings.push(Warning::Other {
                    message: format!("inline text payload for {mt} is not supported"),
                });
                return None;
            }
        };
        return Some(UserContentPart::InputFile(payload));
    }

    warnings.push(Warning::Other {
        message: format!(
            "user file with media type {mt} dropped (set passThroughUnsupportedFiles to keep it)"
        ),
    });
    None
}

fn data_uri(media_type: &str, bytes: &FileBytes) -> String {
    let payload = match bytes {
        FileBytes::Base64(s) => s.clone(),
        FileBytes::Bytes(b) => base64_encode(b),
    };
    format!("data:{media_type};base64,{payload}")
}

/// Minimal RFC 4648 §4 base64 encoder (shared with chat::convert_prompt).
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
            out.push_str("==");
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
    _msg_provider_options: Option<&ProviderOptions>,
    ctx: &ConvertCtx<'_>,
    items: &mut Vec<InputItem>,
    warnings: &mut Vec<Warning>,
) {
    let mut text_buf: Vec<AssistantContentPart> = Vec::new();
    let mut assistant_item_id: Option<String> = None;
    let mut assistant_phase: Option<MessagePhase> = None;

    let flush_text = |items: &mut Vec<InputItem>,
                      buf: &mut Vec<AssistantContentPart>,
                      id: &mut Option<String>,
                      phase: &mut Option<MessagePhase>| {
        if !buf.is_empty() {
            items.push(InputItem::Message(InputMessage::Assistant {
                role: AssistantRole::Assistant,
                content: std::mem::take(buf),
                id: id.take(),
                phase: phase.take(),
            }));
        }
    };

    for part in parts {
        match part {
            AssistantPart::Text(t) => {
                // Skip text parts that already exist in the conversation context
                // to avoid "Duplicate item found" errors. Mirrors upstream
                // convert-to-openai-responses-input.ts:234.
                let text_item_id = read_item_id(t.provider_options.as_ref(), ctx);
                if ctx.has_conversation && text_item_id.is_some() {
                    continue;
                }
                pick_text_metadata(
                    t.provider_options.as_ref(),
                    ctx.provider_options_name,
                    &mut assistant_item_id,
                    &mut assistant_phase,
                );
                text_buf.push(AssistantContentPart::OutputText {
                    text: t.text.clone(),
                });
            }
            AssistantPart::Reasoning {
                text,
                provider_options,
            } => {
                let openai = provider_options.as_ref().and_then(|po| {
                    po.get(ctx.provider_options_name)
                        .or_else(|| po.get("openai"))
                });
                let id = openai
                    .and_then(|m| m.get("itemId"))
                    .and_then(|v| v.as_str())
                    .map(str::to_owned);

                // Skip reasoning items with item IDs when using conversation
                // or previousResponseId — they already exist server-side.
                // Mirrors upstream convert-to-openai-responses-input.ts:513-518.
                if (ctx.has_conversation || ctx.has_previous_response_id) && id.is_some() {
                    continue;
                }

                flush_text(
                    items,
                    &mut text_buf,
                    &mut assistant_item_id,
                    &mut assistant_phase,
                );
                let encrypted = openai
                    .and_then(|m| m.get("reasoningEncryptedContent"))
                    .and_then(|v| v.as_str())
                    .map(str::to_owned);
                items.push(InputItem::Typed(TypedInputItem::Reasoning {
                    id,
                    encrypted_content: encrypted,
                    summary: vec![ReasoningSummary::SummaryText { text: text.clone() }],
                }));
            }
            AssistantPart::ToolCall(tc) => {
                // Skip tool-call parts that already exist in the conversation
                // context. Mirrors upstream convert-to-openai-responses-input.ts:265.
                let call_item_id = read_item_id(tc.provider_options.as_ref(), ctx);
                if ctx.has_conversation && call_item_id.is_some() {
                    continue;
                }
                flush_text(
                    items,
                    &mut text_buf,
                    &mut assistant_item_id,
                    &mut assistant_phase,
                );
                push_assistant_tool_call(tc, ctx, items, warnings);
            }
            AssistantPart::ToolResult(_)
            | AssistantPart::File(_)
            | AssistantPart::ReasoningFile { .. }
            | AssistantPart::Custom { .. } => {
                warnings.push(Warning::Other {
                    message: "unsupported assistant part dropped for Responses input".into(),
                });
            }
        }
    }

    flush_text(
        items,
        &mut text_buf,
        &mut assistant_item_id,
        &mut assistant_phase,
    );
}

/// Pluck `provider_options.<openai|provider_name>.itemId` if present.
fn read_item_id(po: Option<&ProviderOptions>, ctx: &ConvertCtx<'_>) -> Option<String> {
    let map = po?;
    let bucket = map
        .get(ctx.provider_options_name)
        .or_else(|| map.get("openai"))?;
    bucket
        .get("itemId")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
}

/// Pluck `provider_options.<openai|provider_name>.imageDetail` if present.
fn read_image_detail(po: Option<&ProviderOptions>, ctx: &ConvertCtx<'_>) -> Option<String> {
    let map = po?;
    let bucket = map
        .get(ctx.provider_options_name)
        .or_else(|| map.get("openai"))?;
    bucket
        .get("imageDetail")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
}

fn pick_text_metadata(
    po: Option<&ProviderOptions>,
    provider_name: &str,
    item_id: &mut Option<String>,
    phase: &mut Option<MessagePhase>,
) {
    let Some(map) = po.and_then(|po| po.get(provider_name).or_else(|| po.get("openai"))) else {
        return;
    };
    if item_id.is_none()
        && let Some(s) = map.get("itemId").and_then(|v| v.as_str())
    {
        *item_id = Some(s.into());
    }
    if phase.is_none()
        && let Some(s) = map.get("phase").and_then(|v| v.as_str())
    {
        *phase = match s {
            "commentary" => Some(MessagePhase::Commentary),
            "final_answer" => Some(MessagePhase::FinalAnswer),
            _ => None,
        };
    }
}

fn push_assistant_tool_call(
    tc: &ToolCallPart,
    ctx: &ConvertCtx<'_>,
    items: &mut Vec<InputItem>,
    warnings: &mut Vec<Warning>,
) {
    // The Responses API routes provider-executed tool calls back via the
    // matching `*_call` items, which we identify by the tool's name / id
    // hint in `provider_options.openai.providerToolId`. The simplest reliable
    // mapping uses the lower-cased `tool_name` against the upstream id set.
    let provider_tool_id = tc
        .provider_options
        .as_ref()
        .and_then(|po| {
            po.get(ctx.provider_options_name)
                .or_else(|| po.get("openai"))
        })
        .and_then(|m| m.get("providerToolId"))
        .and_then(|v| v.as_str());

    let id = tc
        .provider_options
        .as_ref()
        .and_then(|po| {
            po.get(ctx.provider_options_name)
                .or_else(|| po.get("openai"))
        })
        .and_then(|m| m.get("itemId"))
        .and_then(|v| v.as_str())
        .map(str::to_owned);

    // Mirror upstream `convert-to-openai-responses-input.ts:304-317`.
    // In `store` mode the Responses API retains prior tool calls server-side,
    // so re-emitting the full payload duplicates state. When `id` is known we
    // collapse to an `item_reference`; if the conversation also chains via
    // `previous_response_id`, the prior call is already linked and we drop the
    // emission entirely. Provider-executed calls follow the same shape with no
    // `previous_response_id` skip (upstream lines 304-309).
    if tc.provider_executed == Some(true) {
        if ctx.store {
            if let Some(ref_id) = id {
                items.push(InputItem::Typed(TypedInputItem::ItemReference {
                    id: ref_id,
                }));
            }
        }
        return;
    }

    if let (true, Some(ref_id)) = (ctx.store, id.as_deref()) {
        if ctx.has_previous_response_id {
            return;
        }
        items.push(InputItem::Typed(TypedInputItem::ItemReference {
            id: ref_id.to_owned(),
        }));
        return;
    }

    // Plain client-side function call (no store, or store with no itemId).
    if provider_tool_id.is_none() {
        items.push(InputItem::Typed(TypedInputItem::FunctionCall {
            call_id: tc.tool_call_id.clone(),
            name: tc.tool_name.clone(),
            arguments: stringify_args(&tc.input),
            id,
        }));
        return;
    }

    // Provider-executed: route by id where possible.
    match provider_tool_id.unwrap_or("") {
        ids::APPLY_PATCH if ctx.has_apply_patch_tool => {
            let parsed: Option<super::tools::apply_patch::Input> =
                serde_json::from_value(tc.input.clone()).ok();
            if let Some(input) = parsed {
                items.push(InputItem::Typed(TypedInputItem::ApplyPatchCall {
                    call_id: input.call_id,
                    status: super::wire::response::ApplyPatchCallStatus::Completed,
                    operation: input.operation,
                    id,
                }));
            } else {
                warnings.push(Warning::Other {
                    message: format!(
                        "assistant tool-call {} carried unparsable apply_patch input",
                        tc.tool_call_id
                    ),
                });
            }
        }
        ids::LOCAL_SHELL if ctx.has_local_shell_tool => {
            #[derive(serde::Deserialize)]
            struct LocalShellInput {
                action: super::tools::local_shell::Action,
            }
            let parsed: Option<LocalShellInput> = serde_json::from_value(tc.input.clone()).ok();
            if let Some(input) = parsed {
                items.push(InputItem::Typed(TypedInputItem::LocalShellCall {
                    id: id.unwrap_or_else(|| tc.tool_call_id.clone()),
                    call_id: tc.tool_call_id.clone(),
                    action: input.action,
                }));
            } else {
                warnings.push(Warning::Other {
                    message: format!(
                        "assistant tool-call {} carried unparsable local_shell input",
                        tc.tool_call_id
                    ),
                });
            }
        }
        _ => {
            // Fall back to function_call so non-recognized tool calls still
            // round-trip (matches ai-sdk lenient behavior).
            items.push(InputItem::Typed(TypedInputItem::FunctionCall {
                call_id: tc.tool_call_id.clone(),
                name: tc.tool_name.clone(),
                arguments: stringify_args(&tc.input),
                id,
            }));
        }
    }
}

fn stringify_args(input: &serde_json::Value) -> String {
    if input.is_null() {
        "{}".into()
    } else if let Some(s) = input.as_str() {
        s.to_owned()
    } else {
        serde_json::to_string(input).unwrap_or_else(|_| "{}".into())
    }
}

fn convert_tool_message(
    parts: &[ToolMessagePart],
    ctx: &ConvertCtx<'_>,
    items: &mut Vec<InputItem>,
    warnings: &mut Vec<Warning>,
) {
    for part in parts {
        match part {
            ToolMessagePart::ToolResult(r) => push_tool_result(r, ctx, items, warnings),
            ToolMessagePart::ToolApprovalResponse(r) => {
                push_approval_response(r, items);
            }
        }
    }
}

fn push_tool_result(
    result: &ToolResultPart,
    ctx: &ConvertCtx<'_>,
    items: &mut Vec<InputItem>,
    warnings: &mut Vec<Warning>,
) {
    // Skip tool results that already exist in the conversation context.
    // Mirrors upstream convert-to-openai-responses-input.ts:417.
    if ctx.has_conversation {
        return;
    }

    let output = match &result.output {
        // Content with rich parts → input_text / input_image / input_file array.
        // Mirrors upstream convert-to-openai-responses-input.ts:780-833.
        ToolResultOutput::Content { value } => {
            let parts: Vec<UserContentPart> = value
                .iter()
                .filter_map(|part| convert_tool_output_part(part, ctx, warnings))
                .collect();
            if parts.is_empty() {
                FunctionCallOutputBody::Text(String::new())
            } else {
                FunctionCallOutputBody::Parts(parts)
            }
        }
        _ => FunctionCallOutputBody::Text(tool_output_string(&result.output)),
    };

    items.push(InputItem::Typed(TypedInputItem::FunctionCallOutput {
        call_id: result.tool_call_id.clone(),
        output,
    }));
}

fn convert_tool_output_part(
    part: &llmsdk_provider::language_model::ToolOutputPart,
    ctx: &ConvertCtx<'_>,
    warnings: &mut Vec<Warning>,
) -> Option<UserContentPart> {
    use llmsdk_provider::language_model::ToolOutputPart;

    match part {
        ToolOutputPart::Text { text, .. } => {
            Some(UserContentPart::InputText { text: text.clone() })
        }
        ToolOutputPart::File {
            data,
            media_type,
            filename,
            provider_options,
        } => {
            let mt = media_type.as_str();
            let detail = read_image_detail(provider_options.as_ref(), ctx);

            if mt.starts_with("image/") {
                let payload = match data {
                    FileData::Url { url } => InputImage::Url {
                        image_url: url.clone(),
                        detail,
                    },
                    FileData::Data { data } => InputImage::Url {
                        image_url: data_uri(mt, data),
                        detail,
                    },
                    FileData::Reference { reference } => {
                        let id = reference
                            .get(ctx.provider_options_name)
                            .or_else(|| reference.get("openai"))
                            .and_then(|v| v.as_str());
                        match id {
                            Some(id) => InputImage::Reference {
                                file_id: id.to_string(),
                                detail,
                            },
                            None => {
                                warnings.push(Warning::Other {
                                    message: format!(
                                        "tool-result file reference for {mt} missing OpenAI file_id"
                                    ),
                                });
                                return None;
                            }
                        }
                    }
                    FileData::Text { .. } => {
                        warnings.push(Warning::Other {
                            message: "tool-result image with inline text payload is not supported"
                                .into(),
                        });
                        return None;
                    }
                };
                return Some(UserContentPart::InputImage(payload));
            }

            // Non-image: input_file (PDF or pass-through).
            let payload = match data {
                FileData::Url { url } => InputFile::Url {
                    file_url: url.clone(),
                },
                FileData::Data { data } => InputFile::Data {
                    filename: filename.clone().unwrap_or_else(|| "file".into()),
                    file_data: data_uri(mt, data),
                },
                FileData::Reference { reference } => {
                    let id = reference
                        .get(ctx.provider_options_name)
                        .or_else(|| reference.get("openai"))
                        .and_then(|v| v.as_str());
                    match id {
                        Some(id) => InputFile::Reference {
                            file_id: id.to_string(),
                        },
                        None => {
                            warnings.push(Warning::Other {
                                message: format!(
                                    "tool-result file reference for {mt} missing OpenAI file_id"
                                ),
                            });
                            return None;
                        }
                    }
                }
                FileData::Text { .. } => {
                    warnings.push(Warning::Other {
                        message: format!(
                            "tool-result inline text payload for {mt} is not supported"
                        ),
                    });
                    return None;
                }
            };
            Some(UserContentPart::InputFile(payload))
        }
        ToolOutputPart::Custom { .. } => {
            warnings.push(Warning::Other {
                message: "tool-result custom content part dropped (not supported by Responses)"
                    .into(),
            });
            None
        }
    }
}

fn tool_output_string(output: &ToolResultOutput) -> String {
    match output {
        ToolResultOutput::Text { value, .. } | ToolResultOutput::ErrorText { value, .. } => {
            value.clone()
        }
        ToolResultOutput::Json { value, .. } | ToolResultOutput::ErrorJson { value, .. } => {
            serde_json::to_string(value).unwrap_or_else(|_| "{}".into())
        }
        ToolResultOutput::ExecutionDenied { reason, .. } => reason
            .clone()
            .unwrap_or_else(|| "Tool call execution denied.".into()),
        ToolResultOutput::Content { .. } => String::new(),
    }
}

fn push_approval_response(r: &ToolApprovalResponsePart, items: &mut Vec<InputItem>) {
    items.push(InputItem::Typed(TypedInputItem::McpApprovalResponse {
        approval_request_id: r.approval_id.clone(),
        approve: r.approved,
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmsdk_provider::language_model::TextPart;
    use serde_json::json;

    fn user_text(s: &str) -> Message {
        Message::User {
            content: vec![UserPart::Text(TextPart {
                text: s.into(),
                provider_options: None,
            })],
            provider_options: None,
        }
    }

    #[test]
    fn system_mode_system() {
        let p = vec![Message::System {
            content: "be brief".into(),
            provider_options: None,
        }];
        let (out, _) = convert_prompt(
            &p,
            &ConvertCtx {
                system_message_mode: SystemMessageMode::System,
                ..Default::default()
            },
        );
        let s = serde_json::to_value(&out[0]).unwrap();
        assert_eq!(s["role"], "system");
    }

    #[test]
    fn system_mode_developer() {
        let p = vec![Message::System {
            content: "be brief".into(),
            provider_options: None,
        }];
        let (out, _) = convert_prompt(
            &p,
            &ConvertCtx {
                system_message_mode: SystemMessageMode::Developer,
                ..Default::default()
            },
        );
        let s = serde_json::to_value(&out[0]).unwrap();
        assert_eq!(s["role"], "developer");
    }

    #[test]
    fn system_mode_remove_drops_message() {
        let p = vec![Message::System {
            content: "drop me".into(),
            provider_options: None,
        }];
        let (out, warnings) = convert_prompt(
            &p,
            &ConvertCtx {
                system_message_mode: SystemMessageMode::Remove,
                ..Default::default()
            },
        );
        assert!(out.is_empty());
        assert_eq!(warnings.len(), 1);
    }

    #[test]
    fn user_text_and_image_url() {
        let p = vec![Message::User {
            content: vec![
                UserPart::Text(TextPart {
                    text: "ping".into(),
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
        let (out, _) = convert_prompt(&p, &ConvertCtx::default());
        let v = serde_json::to_value(&out[0]).unwrap();
        assert_eq!(v["content"][0]["type"], "input_text");
        assert_eq!(v["content"][1]["type"], "input_image");
        assert_eq!(v["content"][1]["image_url"], "https://example.com/cat.png");
    }

    #[test]
    fn user_pdf_routes_to_input_file() {
        let p = vec![Message::User {
            content: vec![UserPart::File(FilePart {
                filename: Some("doc.pdf".into()),
                data: FileData::Url {
                    url: "https://x/doc.pdf".into(),
                },
                media_type: "application/pdf".into(),
                provider_options: None,
            })],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(&p, &ConvertCtx::default());
        let v = serde_json::to_value(&out[0]).unwrap();
        assert_eq!(v["content"][0]["type"], "input_file");
        assert_eq!(v["content"][0]["file_url"], "https://x/doc.pdf");
    }

    #[test]
    fn user_text_file_dropped_without_passthrough() {
        let p = vec![Message::User {
            content: vec![UserPart::File(FilePart {
                filename: Some("a.csv".into()),
                data: FileData::Url {
                    url: "https://x/a.csv".into(),
                },
                media_type: "text/csv".into(),
                provider_options: None,
            })],
            provider_options: None,
        }];
        let (out, w) = convert_prompt(&p, &ConvertCtx::default());
        assert!(out.is_empty());
        assert_eq!(w.len(), 1);
    }

    #[test]
    fn user_text_file_passes_through_when_enabled() {
        let p = vec![Message::User {
            content: vec![UserPart::File(FilePart {
                filename: Some("a.csv".into()),
                data: FileData::Url {
                    url: "https://x/a.csv".into(),
                },
                media_type: "text/csv".into(),
                provider_options: None,
            })],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(
            &p,
            &ConvertCtx {
                pass_through_unsupported_files: true,
                ..Default::default()
            },
        );
        let v = serde_json::to_value(&out[0]).unwrap();
        assert_eq!(v["content"][0]["type"], "input_file");
    }

    #[test]
    fn assistant_text_and_function_tool_call() {
        let p = vec![
            user_text("hi"),
            Message::Assistant {
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
            },
        ];
        let (out, _) = convert_prompt(&p, &ConvertCtx::default());
        let assistant = serde_json::to_value(&out[1]).unwrap();
        assert_eq!(assistant["role"], "assistant");
        assert_eq!(assistant["content"][0]["type"], "output_text");
        let call = serde_json::to_value(&out[2]).unwrap();
        assert_eq!(call["type"], "function_call");
        assert_eq!(call["call_id"], "call_1");
        assert_eq!(call["name"], "weather");
        assert_eq!(call["arguments"], "{\"city\":\"NYC\"}");
    }

    #[test]
    fn assistant_reasoning_emits_reasoning_item() {
        let mut po = ProviderOptions::new();
        po.insert(
            "openai".into(),
            json!({"itemId": "r1", "reasoningEncryptedContent": "abc"})
                .as_object()
                .unwrap()
                .clone(),
        );
        let p = vec![Message::Assistant {
            content: vec![AssistantPart::Reasoning {
                text: "thought".into(),
                provider_options: Some(po),
            }],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(&p, &ConvertCtx::default());
        let v = serde_json::to_value(&out[0]).unwrap();
        assert_eq!(v["type"], "reasoning");
        assert_eq!(v["id"], "r1");
        assert_eq!(v["encrypted_content"], "abc");
        assert_eq!(v["summary"][0]["text"], "thought");
    }

    #[test]
    fn tool_message_to_function_call_output() {
        let p = vec![Message::Tool {
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
        let (out, _) = convert_prompt(&p, &ConvertCtx::default());
        let v = serde_json::to_value(&out[0]).unwrap();
        assert_eq!(v["type"], "function_call_output");
        assert_eq!(v["output"], "sunny");
    }

    #[test]
    fn mcp_approval_response_emits_input_item() {
        let p = vec![Message::Tool {
            content: vec![ToolMessagePart::ToolApprovalResponse(
                ToolApprovalResponsePart {
                    approval_id: "appr_1".into(),
                    approved: true,
                    reason: None,
                    provider_options: None,
                },
            )],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(&p, &ConvertCtx::default());
        let v = serde_json::to_value(&out[0]).unwrap();
        assert_eq!(v["type"], "mcp_approval_response");
        assert_eq!(v["approval_request_id"], "appr_1");
        assert_eq!(v["approve"], true);
    }

    #[test]
    fn provider_executed_call_with_item_id_becomes_item_reference() {
        let mut po = ProviderOptions::new();
        po.insert(
            "openai".into(),
            json!({"itemId": "ws_1"}).as_object().unwrap().clone(),
        );
        let p = vec![Message::Assistant {
            content: vec![AssistantPart::ToolCall(ToolCallPart {
                tool_call_id: "ws_1".into(),
                tool_name: "web_search".into(),
                input: json!({}),
                provider_executed: Some(true),
                dynamic: None,
                provider_options: Some(po),
            })],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(&p, &ConvertCtx::default());
        let v = serde_json::to_value(&out[0]).unwrap();
        assert_eq!(v["type"], "item_reference");
        assert_eq!(v["id"], "ws_1");
    }

    // --- M14 fix-pack regression tests ---

    fn po_with_item_id(id: &str) -> ProviderOptions {
        let mut po = ProviderOptions::new();
        po.insert(
            "openai".into(),
            json!({"itemId": id}).as_object().unwrap().clone(),
        );
        po
    }

    #[test]
    fn reasoning_skipped_when_previous_response_id_and_item_id_present() {
        let p = vec![Message::Assistant {
            content: vec![AssistantPart::Reasoning {
                text: "thinking".into(),
                provider_options: Some(po_with_item_id("rs_existing")),
            }],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(
            &p,
            &ConvertCtx {
                has_previous_response_id: true,
                ..Default::default()
            },
        );
        assert!(out.is_empty(), "reasoning with itemId must be skipped");
    }

    #[test]
    fn reasoning_skipped_when_conversation_and_item_id_present() {
        let p = vec![Message::Assistant {
            content: vec![AssistantPart::Reasoning {
                text: "thinking".into(),
                provider_options: Some(po_with_item_id("rs_existing")),
            }],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(
            &p,
            &ConvertCtx {
                has_conversation: true,
                ..Default::default()
            },
        );
        assert!(out.is_empty());
    }

    #[test]
    fn text_skipped_when_conversation_and_item_id_present() {
        let p = vec![Message::Assistant {
            content: vec![AssistantPart::Text(TextPart {
                text: "hello".into(),
                provider_options: Some(po_with_item_id("msg_existing")),
            })],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(
            &p,
            &ConvertCtx {
                has_conversation: true,
                ..Default::default()
            },
        );
        assert!(out.is_empty(), "text with itemId must be skipped");
    }

    #[test]
    fn tool_call_skipped_when_conversation_and_item_id_present() {
        let p = vec![Message::Assistant {
            content: vec![AssistantPart::ToolCall(ToolCallPart {
                tool_call_id: "call_1".into(),
                tool_name: "weather".into(),
                input: json!({"city": "NYC"}),
                provider_executed: None,
                dynamic: None,
                provider_options: Some(po_with_item_id("fc_existing")),
            })],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(
            &p,
            &ConvertCtx {
                has_conversation: true,
                ..Default::default()
            },
        );
        assert!(out.is_empty(), "tool-call with itemId must be skipped");
    }

    #[test]
    fn tool_call_with_store_and_item_id_becomes_item_reference() {
        // Mirrors upstream convert-to-openai-responses-input.test.ts:1280-1303
        // ("should convert multiple tool-call parts with store: true to item_reference"):
        // when `store: true` and the tool call carries an itemId, the Responses
        // API already has the call server-side — emit an `item_reference` rather
        // than re-shipping the full function_call envelope.
        let p = vec![Message::Assistant {
            content: vec![AssistantPart::ToolCall(ToolCallPart {
                tool_call_id: "call_1".into(),
                tool_name: "weather".into(),
                input: json!({"city": "NYC"}),
                provider_executed: None,
                dynamic: None,
                provider_options: Some(po_with_item_id("fc_existing")),
            })],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(&p, &ConvertCtx::default());
        assert_eq!(out.len(), 1, "expected exactly one item_reference");
        let v = serde_json::to_value(&out[0]).unwrap();
        assert_eq!(v["type"], "item_reference");
        assert_eq!(v["id"], "fc_existing");
    }

    #[test]
    fn tool_call_with_store_and_previous_response_id_is_skipped() {
        // Mirrors upstream convert-to-openai-responses-input.ts:311-314 — when
        // `previous_response_id` chains the call, the prior tool_call lives in
        // the linked response and must not be re-emitted, not even as an
        // item_reference.
        let p = vec![Message::Assistant {
            content: vec![AssistantPart::ToolCall(ToolCallPart {
                tool_call_id: "call_1".into(),
                tool_name: "weather".into(),
                input: json!({"city": "NYC"}),
                provider_executed: None,
                dynamic: None,
                provider_options: Some(po_with_item_id("fc_existing")),
            })],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(
            &p,
            &ConvertCtx {
                has_previous_response_id: true,
                ..Default::default()
            },
        );
        assert!(
            out.is_empty(),
            "tool-call with store + previous_response_id + itemId must be skipped"
        );
    }

    #[test]
    fn tool_call_without_store_keeps_function_call() {
        // Negative control: when `store: false`, the conversation does not live
        // on the server, so the full function_call envelope must still ship.
        let p = vec![Message::Assistant {
            content: vec![AssistantPart::ToolCall(ToolCallPart {
                tool_call_id: "call_1".into(),
                tool_name: "weather".into(),
                input: json!({"city": "NYC"}),
                provider_executed: None,
                dynamic: None,
                provider_options: Some(po_with_item_id("fc_existing")),
            })],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(
            &p,
            &ConvertCtx {
                store: false,
                ..Default::default()
            },
        );
        assert_eq!(out.len(), 1);
        let v = serde_json::to_value(&out[0]).unwrap();
        assert_eq!(v["type"], "function_call");
        assert_eq!(v["call_id"], "call_1");
        assert_eq!(v["name"], "weather");
    }

    #[test]
    fn tool_result_skipped_when_conversation() {
        let p = vec![Message::Tool {
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
        let (out, _) = convert_prompt(
            &p,
            &ConvertCtx {
                has_conversation: true,
                ..Default::default()
            },
        );
        assert!(
            out.is_empty(),
            "tool-result must be skipped under hasConversation"
        );
    }

    #[test]
    fn tool_result_content_with_text_serializes_as_parts() {
        use llmsdk_provider::language_model::ToolOutputPart;
        let p = vec![Message::Tool {
            content: vec![ToolMessagePart::ToolResult(ToolResultPart {
                tool_call_id: "call_1".into(),
                tool_name: "search".into(),
                output: ToolResultOutput::Content {
                    value: vec![ToolOutputPart::Text {
                        text: "The weather in SF is 72°F".into(),
                        provider_options: None,
                    }],
                },
                provider_options: None,
            })],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(&p, &ConvertCtx::default());
        let v = serde_json::to_value(&out[0]).unwrap();
        assert_eq!(v["type"], "function_call_output");
        assert_eq!(v["output"][0]["type"], "input_text");
        assert_eq!(v["output"][0]["text"], "The weather in SF is 72°F");
    }

    #[test]
    fn tool_result_content_with_image_data_serializes_with_detail() {
        use llmsdk_provider::language_model::ToolOutputPart;
        let mut po = ProviderOptions::new();
        po.insert(
            "openai".into(),
            json!({"imageDetail": "high"}).as_object().unwrap().clone(),
        );
        let p = vec![Message::Tool {
            content: vec![ToolMessagePart::ToolResult(ToolResultPart {
                tool_call_id: "call_1".into(),
                tool_name: "render".into(),
                output: ToolResultOutput::Content {
                    value: vec![ToolOutputPart::File {
                        data: FileData::Data {
                            data: FileBytes::Base64("AAEC".into()),
                        },
                        media_type: "image/png".into(),
                        filename: None,
                        provider_options: Some(po),
                    }],
                },
                provider_options: None,
            })],
            provider_options: None,
        }];
        let (out, _) = convert_prompt(&p, &ConvertCtx::default());
        let v = serde_json::to_value(&out[0]).unwrap();
        assert_eq!(v["output"][0]["type"], "input_image");
        assert_eq!(v["output"][0]["detail"], "high");
        assert!(
            v["output"][0]["image_url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/png;base64,")
        );
    }
}
