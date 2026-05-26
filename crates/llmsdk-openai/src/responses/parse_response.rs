//! Parse `POST /v1/responses` non-streaming JSON body → `GenerateResult`.
//!
//! Mirrors the `doGenerate` half of
//! `@ai-sdk/openai/src/responses/openai-responses-language-model.ts`.
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;

use llmsdk_provider::ProviderError;
use llmsdk_provider::json::{JsonObject, JsonValue};
use llmsdk_provider::language_model::{
    Content, GenerateResponse, GenerateResult, ReasoningPart, ResponseMetadata, Source, TextPart,
    ToolCallPart, ToolResult, ToolResultOutput,
};
use llmsdk_provider::shared::{ProviderMetadata, ProviderOptions, RequestInfo, Warning};
use serde_json::{Map, Value, json};

use super::finish_reason::map_finish_reason;
use super::tools::ids;
use super::usage::convert_usage;
use super::wire::response::{
    Annotation, ApplyPatchCallItem, CodeInterpreterCallItem, CompactionItem, ComputerCallItem,
    CustomToolCallItem, FileSearchCallItem, FunctionCallItem, ImageGenerationCallItem,
    LocalShellCallItem, McpApprovalRequestItem, McpCallItem, McpListToolsItem, MessageContentPart,
    MessageItem, OutputItem, ReasoningItem, ResponsesResponse, ShellCallItem, ShellCallOutputItem,
    ToolSearchCallItem, ToolSearchOutputItem, WebSearchCallItem,
};

/// Convert a successful Responses API JSON body into [`GenerateResult`].
#[allow(clippy::too_many_lines, reason = "18-way output item switch")]
pub fn parse_response(
    response: ResponsesResponse,
    response_headers: HashMap<String, String>,
    request_body: Option<JsonValue>,
    mut warnings: Vec<Warning>,
    provider_options_name: &str,
    web_search_tool_name: Option<&str>,
    is_shell_provider_executed: bool,
) -> Result<GenerateResult, ProviderError> {
    if let Some(err) = response.error {
        return Err(ProviderError::api_call(
            "/v1/responses".to_owned(),
            err.message,
        ));
    }

    let mut content: Vec<Content> = Vec::new();
    let mut has_function_call = false;
    let mut logprobs: Vec<JsonValue> = Vec::new();
    let mut hosted_tool_search_call_ids: Vec<String> = Vec::new();

    if let Some(output) = response.output {
        for item in output {
            match item {
                OutputItem::Reasoning(r) => emit_reasoning(&r, provider_options_name, &mut content),
                OutputItem::Message(m) => {
                    emit_message(&m, provider_options_name, &mut logprobs, &mut content);
                }
                OutputItem::FunctionCall(f) => {
                    has_function_call = true;
                    emit_function_call(&f, provider_options_name, &mut content);
                }
                OutputItem::CustomToolCall(c) => {
                    has_function_call = true;
                    emit_custom_tool_call(&c, provider_options_name, &mut content);
                }
                OutputItem::WebSearchCall(w) => {
                    emit_web_search_call(&w, web_search_tool_name, &mut content);
                }
                OutputItem::FileSearchCall(f) => emit_file_search_call(&f, &mut content),
                OutputItem::CodeInterpreterCall(c) => emit_code_interpreter_call(&c, &mut content),
                OutputItem::ImageGenerationCall(i) => emit_image_generation_call(&i, &mut content),
                OutputItem::LocalShellCall(l) => {
                    emit_local_shell_call(&l, provider_options_name, &mut content);
                }
                OutputItem::ComputerCall(c) => emit_computer_call(&c, &mut content),
                OutputItem::McpCall(m) => emit_mcp_call(&m, provider_options_name, &mut content),
                OutputItem::McpListTools(_) => { /* skipped (matches ai-sdk) */ }
                OutputItem::McpApprovalRequest(_a) => {
                    warnings.push(Warning::Other {
                        message: "mcp_approval_request observed on non-stream response — \
                                  approval flows are typically streamed; surfacing as warning"
                            .into(),
                    });
                }
                OutputItem::ApplyPatchCall(a) => {
                    has_function_call = true;
                    emit_apply_patch_call(&a, provider_options_name, &mut content);
                }
                OutputItem::ShellCall(s) => {
                    emit_shell_call(
                        &s,
                        provider_options_name,
                        is_shell_provider_executed,
                        &mut content,
                    );
                }
                OutputItem::ShellCallOutput(o) => emit_shell_call_output(&o, &mut content),
                OutputItem::Compaction(c) => {
                    emit_compaction(&c, provider_options_name, &mut content)
                }
                OutputItem::ToolSearchCall(t) => emit_tool_search_call(
                    &t,
                    provider_options_name,
                    &mut hosted_tool_search_call_ids,
                    &mut content,
                ),
                OutputItem::ToolSearchOutput(t) => emit_tool_search_output(
                    &t,
                    provider_options_name,
                    &mut hosted_tool_search_call_ids,
                    &mut content,
                ),
                OutputItem::Unknown => warnings.push(Warning::Other {
                    message: "encountered unknown output item type".into(),
                }),
            }
        }
    }

    let finish_reason = map_finish_reason(
        response
            .incomplete_details
            .as_ref()
            .map(|d| d.reason.as_str()),
        has_function_call,
    );
    let usage = convert_usage(response.usage.as_ref());

    let mut openai_meta = Map::new();
    openai_meta.insert("responseId".into(), json!(response.id));
    if !logprobs.is_empty() {
        openai_meta.insert("logprobs".into(), json!(logprobs));
    }
    if let Some(tier) = &response.service_tier {
        openai_meta.insert("serviceTier".into(), json!(tier));
    }
    let mut provider_metadata = ProviderMetadata::new();
    provider_metadata.insert(provider_options_name.to_string(), openai_meta);

    let response_meta = ResponseMetadata {
        id: response.id.clone(),
        timestamp: response
            .created_at
            .map(|t| chrono_seconds_to_iso(t).unwrap_or_default()),
        model_id: response.model.clone(),
        headers: Some(
            response_headers
                .iter()
                .map(|(k, v)| (k.clone(), Some(v.clone())))
                .collect(),
        ),
    };

    Ok(GenerateResult {
        content,
        finish_reason,
        usage,
        provider_metadata: Some(provider_metadata),
        request: request_body.map(|body| RequestInfo { body: Some(body) }),
        response: Some(GenerateResponse {
            metadata: response_meta,
            body: None,
        }),
        warnings,
    })
}

fn chrono_seconds_to_iso(secs: f64) -> Option<String> {
    use std::time::{Duration, UNIX_EPOCH};
    let when = UNIX_EPOCH.checked_add(Duration::from_secs_f64(secs))?;
    let dur = when.duration_since(UNIX_EPOCH).ok()?;
    let s = dur.as_secs();
    // Hand-rolled RFC 3339 (UTC, second resolution) to avoid adding chrono.
    Some(format_rfc3339_utc(s))
}

fn format_rfc3339_utc(epoch_seconds: u64) -> String {
    let secs_in_day: u64 = 86_400;
    let day = epoch_seconds / secs_in_day;
    let rem = epoch_seconds % secs_in_day;
    let hh = rem / 3600;
    let mm = (rem % 3600) / 60;
    let ss = rem % 60;
    let (y, mo, d) = days_from_epoch_to_ymd(day as i64);
    format!("{y:04}-{mo:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

fn days_from_epoch_to_ymd(mut days: i64) -> (i32, u32, u32) {
    // Howard Hinnant's algorithm (proleptic Gregorian).
    days += 719_468;
    let era = days.div_euclid(146_097);
    let doe = days.rem_euclid(146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = (yoe as i64 + era * 400) as i32;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

fn put_openai(po: &mut ProviderOptions, provider_name: &str, body: Map<String, Value>) {
    po.insert(provider_name.to_string(), body);
}

fn make_openai_po(provider_name: &str, body: Map<String, Value>) -> ProviderOptions {
    let mut po = ProviderOptions::new();
    put_openai(&mut po, provider_name, body);
    po
}

fn make_openai_pm(provider_name: &str, body: Map<String, Value>) -> ProviderMetadata {
    let mut pm = ProviderMetadata::new();
    pm.insert(provider_name.to_string(), body);
    pm
}

fn emit_reasoning(item: &ReasoningItem, provider_name: &str, out: &mut Vec<Content>) {
    let summaries = if item.summary.is_empty() {
        vec![String::new()]
    } else {
        item.summary
            .iter()
            .map(|s| match s {
                super::wire::response::ReasoningSummary::SummaryText { text } => text.clone(),
            })
            .collect()
    };
    for text in summaries {
        let mut body = Map::new();
        body.insert("itemId".into(), json!(item.id));
        body.insert(
            "reasoningEncryptedContent".into(),
            item.encrypted_content
                .as_ref()
                .map_or(JsonValue::Null, |s| json!(s)),
        );
        out.push(Content::Reasoning(ReasoningPart {
            text,
            provider_options: Some(make_openai_po(provider_name, body)),
        }));
    }
}

fn emit_message(
    m: &MessageItem,
    provider_name: &str,
    logprobs: &mut Vec<JsonValue>,
    out: &mut Vec<Content>,
) {
    for part in &m.content {
        let MessageContentPart::OutputText {
            text,
            annotations,
            logprobs: lp,
        } = part;
        if let Some(lp) = lp {
            for entry in lp {
                logprobs.push(serde_json::to_value(entry).unwrap_or(JsonValue::Null));
            }
        }
        let mut body = Map::new();
        body.insert("itemId".into(), json!(m.id));
        if let Some(p) = &m.phase {
            body.insert("phase".into(), json!(p));
        }
        if !annotations.is_empty() {
            body.insert(
                "annotations".into(),
                serde_json::to_value(annotations).unwrap_or(JsonValue::Null),
            );
        }
        out.push(Content::Text(TextPart {
            text: text.clone(),
            provider_options: Some(make_openai_po(provider_name, body)),
        }));
        for annotation in annotations {
            push_annotation_source(annotation, provider_name, out);
        }
    }
}

fn push_annotation_source(annotation: &Annotation, provider_name: &str, out: &mut Vec<Content>) {
    match annotation {
        Annotation::UrlCitation { url, title, .. } => {
            out.push(Content::Source(Source::Url {
                id: generated_id("src"),
                url: url.clone(),
                title: Some(title.clone()),
                provider_metadata: None,
            }));
        }
        Annotation::FileCitation {
            file_id,
            filename,
            index,
        } => {
            let mut body = Map::new();
            body.insert("type".into(), json!("file_citation"));
            body.insert("fileId".into(), json!(file_id));
            body.insert("index".into(), json!(index));
            out.push(Content::Source(Source::Document {
                id: generated_id("src"),
                media_type: "text/plain".into(),
                title: filename.clone(),
                filename: Some(filename.clone()),
                provider_metadata: Some(make_openai_pm(provider_name, body)),
            }));
        }
        Annotation::ContainerFileCitation {
            container_id,
            file_id,
            filename,
            ..
        } => {
            let mut body = Map::new();
            body.insert("type".into(), json!("container_file_citation"));
            body.insert("fileId".into(), json!(file_id));
            body.insert("containerId".into(), json!(container_id));
            out.push(Content::Source(Source::Document {
                id: generated_id("src"),
                media_type: "text/plain".into(),
                title: filename.clone(),
                filename: Some(filename.clone()),
                provider_metadata: Some(make_openai_pm(provider_name, body)),
            }));
        }
        Annotation::FilePath { file_id, index } => {
            let mut body = Map::new();
            body.insert("type".into(), json!("file_path"));
            body.insert("fileId".into(), json!(file_id));
            body.insert("index".into(), json!(index));
            out.push(Content::Source(Source::Document {
                id: generated_id("src"),
                media_type: "application/octet-stream".into(),
                title: file_id.clone(),
                filename: Some(file_id.clone()),
                provider_metadata: Some(make_openai_pm(provider_name, body)),
            }));
        }
    }
}

/// Deterministic-ish id generator (`prefix_<nanos>_<counter>`).
///
/// We avoid an `uuid` dependency; uniqueness within a single response is
/// sufficient because the id is only consumed downstream as a content key.
fn generated_id(prefix: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    format!("{prefix}_{nanos}_{n}")
}

fn emit_function_call(item: &FunctionCallItem, provider_name: &str, out: &mut Vec<Content>) {
    let mut body = Map::new();
    body.insert("itemId".into(), json!(item.id));
    if let Some(ns) = &item.namespace {
        body.insert("namespace".into(), json!(ns));
    }
    out.push(Content::ToolCall(ToolCallPart {
        tool_call_id: item.call_id.clone(),
        tool_name: item.name.clone(),
        input: parse_args(&item.arguments),
        provider_executed: None,
        dynamic: None,
        provider_options: Some(make_openai_po(provider_name, body)),
    }));
}

fn emit_custom_tool_call(item: &CustomToolCallItem, provider_name: &str, out: &mut Vec<Content>) {
    let mut body = Map::new();
    body.insert("itemId".into(), json!(item.id));
    out.push(Content::ToolCall(ToolCallPart {
        tool_call_id: item.call_id.clone(),
        tool_name: item.name.clone(),
        input: JsonValue::String(item.input.clone()),
        provider_executed: None,
        dynamic: None,
        provider_options: Some(make_openai_po(provider_name, body)),
    }));
}

fn emit_web_search_call(
    item: &WebSearchCallItem,
    web_search_tool_name: Option<&str>,
    out: &mut Vec<Content>,
) {
    let tool_name = web_search_tool_name.unwrap_or("web_search").to_string();
    out.push(Content::ToolCall(ToolCallPart {
        tool_call_id: item.id.clone(),
        tool_name: tool_name.clone(),
        input: json!({}),
        provider_executed: Some(true),
        dynamic: None,
        provider_options: None,
    }));
    let action_value = item
        .action
        .as_ref()
        .map(|a| {
            serde_json::to_value(a)
                .ok()
                .and_then(|v| v.as_object().cloned())
                .unwrap_or_default()
        })
        .unwrap_or_default();
    out.push(Content::ToolResult(ToolResult {
        tool_call_id: item.id.clone(),
        tool_name,
        output: ToolResultOutput::Json {
            value: JsonValue::Object(action_value),
            provider_options: None,
        },
        preliminary: None,
        provider_metadata: None,
    }));
}

fn emit_file_search_call(item: &FileSearchCallItem, out: &mut Vec<Content>) {
    out.push(Content::ToolCall(ToolCallPart {
        tool_call_id: item.id.clone(),
        tool_name: ids::FILE_SEARCH.into(),
        input: json!({}),
        provider_executed: Some(true),
        dynamic: None,
        provider_options: None,
    }));
    let mut output = JsonObject::new();
    output.insert("queries".into(), json!(item.queries));
    output.insert(
        "results".into(),
        serde_json::to_value(&item.results).unwrap_or(JsonValue::Null),
    );
    out.push(Content::ToolResult(ToolResult {
        tool_call_id: item.id.clone(),
        tool_name: ids::FILE_SEARCH.into(),
        output: ToolResultOutput::Json {
            value: JsonValue::Object(output),
            provider_options: None,
        },
        preliminary: None,
        provider_metadata: None,
    }));
}

fn emit_code_interpreter_call(item: &CodeInterpreterCallItem, out: &mut Vec<Content>) {
    let mut input = JsonObject::new();
    input.insert("code".into(), json!(item.code));
    input.insert("containerId".into(), json!(item.container_id));
    out.push(Content::ToolCall(ToolCallPart {
        tool_call_id: item.id.clone(),
        tool_name: ids::CODE_INTERPRETER.into(),
        input: JsonValue::Object(input),
        provider_executed: Some(true),
        dynamic: None,
        provider_options: None,
    }));
    let mut output = JsonObject::new();
    output.insert(
        "outputs".into(),
        serde_json::to_value(&item.outputs).unwrap_or(JsonValue::Null),
    );
    out.push(Content::ToolResult(ToolResult {
        tool_call_id: item.id.clone(),
        tool_name: ids::CODE_INTERPRETER.into(),
        output: ToolResultOutput::Json {
            value: JsonValue::Object(output),
            provider_options: None,
        },
        preliminary: None,
        provider_metadata: None,
    }));
}

fn emit_image_generation_call(item: &ImageGenerationCallItem, out: &mut Vec<Content>) {
    out.push(Content::ToolCall(ToolCallPart {
        tool_call_id: item.id.clone(),
        tool_name: ids::IMAGE_GENERATION.into(),
        input: json!({}),
        provider_executed: Some(true),
        dynamic: None,
        provider_options: None,
    }));
    out.push(Content::ToolResult(ToolResult {
        tool_call_id: item.id.clone(),
        tool_name: ids::IMAGE_GENERATION.into(),
        output: ToolResultOutput::Json {
            value: json!({"result": item.result}),
            provider_options: None,
        },
        preliminary: None,
        provider_metadata: None,
    }));
}

fn emit_local_shell_call(item: &LocalShellCallItem, provider_name: &str, out: &mut Vec<Content>) {
    let mut body = Map::new();
    body.insert("itemId".into(), json!(item.id));
    out.push(Content::ToolCall(ToolCallPart {
        tool_call_id: item.call_id.clone(),
        tool_name: ids::LOCAL_SHELL.into(),
        input: json!({"action": item.action}),
        provider_executed: None,
        dynamic: None,
        provider_options: Some(make_openai_po(provider_name, body)),
    }));
}

fn emit_computer_call(item: &ComputerCallItem, out: &mut Vec<Content>) {
    out.push(Content::ToolCall(ToolCallPart {
        tool_call_id: item.id.clone(),
        tool_name: "computer_use".into(),
        input: JsonValue::String(String::new()),
        provider_executed: Some(true),
        dynamic: None,
        provider_options: None,
    }));
    out.push(Content::ToolResult(ToolResult {
        tool_call_id: item.id.clone(),
        tool_name: "computer_use".into(),
        output: ToolResultOutput::Json {
            value: json!({
                "type": "computer_use_tool_result",
                "status": item.status.clone().unwrap_or_else(|| "completed".into()),
            }),
            provider_options: None,
        },
        preliminary: None,
        provider_metadata: None,
    }));
}

fn emit_mcp_call(item: &McpCallItem, provider_name: &str, out: &mut Vec<Content>) {
    let tool_name = format!("mcp.{}", item.name);
    let call_id = item
        .approval_request_id
        .clone()
        .unwrap_or_else(|| item.id.clone());
    out.push(Content::ToolCall(ToolCallPart {
        tool_call_id: call_id.clone(),
        tool_name: tool_name.clone(),
        input: JsonValue::String(item.arguments.clone()),
        provider_executed: Some(true),
        dynamic: Some(true),
        provider_options: None,
    }));
    let mut payload = JsonObject::new();
    payload.insert("type".into(), json!("call"));
    payload.insert("serverLabel".into(), json!(item.server_label));
    payload.insert("name".into(), json!(item.name));
    payload.insert("arguments".into(), json!(item.arguments));
    if let Some(o) = &item.output {
        payload.insert("output".into(), json!(o));
    }
    if let Some(e) = &item.error {
        payload.insert("error".into(), e.clone());
    }
    let mut body = Map::new();
    body.insert("itemId".into(), json!(item.id));
    out.push(Content::ToolResult(ToolResult {
        tool_call_id: call_id,
        tool_name,
        output: ToolResultOutput::Json {
            value: JsonValue::Object(payload),
            provider_options: None,
        },
        preliminary: None,
        provider_metadata: Some(make_openai_pm(provider_name, body)),
    }));
}

fn emit_apply_patch_call(item: &ApplyPatchCallItem, provider_name: &str, out: &mut Vec<Content>) {
    let mut body = Map::new();
    body.insert("itemId".into(), json!(item.id));
    out.push(Content::ToolCall(ToolCallPart {
        tool_call_id: item.call_id.clone(),
        tool_name: ids::APPLY_PATCH.into(),
        input: json!({
            "callId": item.call_id,
            "operation": item.operation,
        }),
        provider_executed: None,
        dynamic: None,
        provider_options: Some(make_openai_po(provider_name, body)),
    }));
}

fn emit_shell_call(
    item: &ShellCallItem,
    provider_name: &str,
    is_provider_executed: bool,
    out: &mut Vec<Content>,
) {
    let mut body = Map::new();
    body.insert("itemId".into(), json!(item.id));
    out.push(Content::ToolCall(ToolCallPart {
        tool_call_id: item.call_id.clone(),
        tool_name: ids::SHELL.into(),
        input: json!({"action": {"commands": item.action.commands}}),
        provider_executed: is_provider_executed.then_some(true),
        dynamic: None,
        provider_options: Some(make_openai_po(provider_name, body)),
    }));
}

fn emit_shell_call_output(item: &ShellCallOutputItem, out: &mut Vec<Content>) {
    out.push(Content::ToolResult(ToolResult {
        tool_call_id: item.call_id.clone(),
        tool_name: ids::SHELL.into(),
        output: ToolResultOutput::Json {
            value: serde_json::to_value(&item.output).unwrap_or(JsonValue::Null),
            provider_options: None,
        },
        preliminary: None,
        provider_metadata: None,
    }));
}

fn emit_compaction(item: &CompactionItem, provider_name: &str, out: &mut Vec<Content>) {
    let mut body = Map::new();
    body.insert("type".into(), json!("compaction"));
    body.insert("itemId".into(), json!(item.id));
    body.insert("encryptedContent".into(), json!(item.encrypted_content));
    out.push(Content::Custom {
        kind: "openai.compaction".into(),
        provider_options: Some(make_openai_po(provider_name, body)),
    });
}

fn emit_tool_search_call(
    item: &ToolSearchCallItem,
    provider_name: &str,
    hosted_call_ids: &mut Vec<String>,
    out: &mut Vec<Content>,
) {
    let is_hosted = matches!(item.execution, super::tools::tool_search::Execution::Server);
    let tool_call_id = item.call_id.clone().unwrap_or_else(|| item.id.clone());
    if is_hosted {
        hosted_call_ids.push(tool_call_id.clone());
    }
    let mut body = Map::new();
    body.insert("itemId".into(), json!(item.id));
    out.push(Content::ToolCall(ToolCallPart {
        tool_call_id,
        tool_name: ids::TOOL_SEARCH.into(),
        input: json!({"arguments": item.arguments, "call_id": item.call_id}),
        provider_executed: is_hosted.then_some(true),
        dynamic: None,
        provider_options: Some(make_openai_po(provider_name, body)),
    }));
}

fn emit_tool_search_output(
    item: &ToolSearchOutputItem,
    provider_name: &str,
    hosted_call_ids: &mut Vec<String>,
    out: &mut Vec<Content>,
) {
    let tool_call_id = item
        .call_id
        .clone()
        .or_else(|| (!hosted_call_ids.is_empty()).then(|| hosted_call_ids.remove(0)))
        .unwrap_or_else(|| item.id.clone());
    let mut body = Map::new();
    body.insert("itemId".into(), json!(item.id));
    out.push(Content::ToolResult(ToolResult {
        tool_call_id,
        tool_name: ids::TOOL_SEARCH.into(),
        output: ToolResultOutput::Json {
            value: json!({"tools": item.tools}),
            provider_options: None,
        },
        preliminary: None,
        provider_metadata: Some(make_openai_pm(provider_name, body)),
    }));
}

#[allow(dead_code, reason = "shared with stream module")]
pub(super) fn parse_args(s: &str) -> JsonValue {
    if s.is_empty() {
        return json!({});
    }
    serde_json::from_str(s).unwrap_or_else(|_| JsonValue::String(s.to_string()))
}

#[allow(
    dead_code,
    reason = "unused warning when McpListToolsItem is feature-gated"
)]
fn _ensure_unused_imports_kept_for_future_use(_a: &McpListToolsItem, _b: &McpApprovalRequestItem) {}

#[cfg(test)]
mod tests {
    use super::*;
    use llmsdk_provider::language_model::FinishReasonKind;

    fn fixture(value: serde_json::Value) -> ResponsesResponse {
        serde_json::from_value(value).expect("decode fixture")
    }

    #[test]
    fn empty_body_is_stop_with_no_content() {
        let r = fixture(serde_json::json!({
            "id": "resp_1",
            "model": "gpt-4o-mini",
            "usage": { "input_tokens": 5, "output_tokens": 0 },
            "output": []
        }));
        let g = parse_response(r, HashMap::new(), None, vec![], "openai", None, false).unwrap();
        assert_eq!(g.finish_reason.unified, FinishReasonKind::Stop);
        assert!(g.content.is_empty());
        assert_eq!(g.usage.input_tokens.total, Some(5));
    }

    #[test]
    fn message_with_text_and_url_citation_emits_source() {
        let r = fixture(serde_json::json!({
            "id": "resp_2",
            "output": [{
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "see https://x",
                    "annotations": [{
                        "type": "url_citation",
                        "start_index": 4,
                        "end_index": 14,
                        "url": "https://x",
                        "title": "X"
                    }]
                }]
            }],
            "usage": { "input_tokens": 0, "output_tokens": 0 }
        }));
        let g = parse_response(r, HashMap::new(), None, vec![], "openai", None, false).unwrap();
        assert_eq!(g.content.len(), 2);
        assert!(matches!(g.content[0], Content::Text(_)));
        assert!(matches!(g.content[1], Content::Source(Source::Url { .. })));
    }

    #[test]
    fn function_call_marks_tool_calls() {
        let r = fixture(serde_json::json!({
            "id": "resp_3",
            "output": [{
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_x",
                "name": "weather",
                "arguments": "{\"city\":\"NYC\"}"
            }],
            "usage": { "input_tokens": 1, "output_tokens": 1 }
        }));
        let g = parse_response(r, HashMap::new(), None, vec![], "openai", None, false).unwrap();
        assert_eq!(g.finish_reason.unified, FinishReasonKind::ToolCalls);
        let Content::ToolCall(tc) = &g.content[0] else {
            panic!("expected tool call");
        };
        assert_eq!(tc.tool_name, "weather");
        assert_eq!(tc.input["city"], "NYC");
    }

    #[test]
    fn reasoning_emits_one_per_summary() {
        let r = fixture(serde_json::json!({
            "id": "r",
            "output": [{
                "type": "reasoning",
                "id": "rsn_1",
                "summary": [
                    { "type": "summary_text", "text": "first" },
                    { "type": "summary_text", "text": "second" }
                ]
            }],
            "usage": { "input_tokens": 0, "output_tokens": 0 }
        }));
        let g = parse_response(r, HashMap::new(), None, vec![], "openai", None, false).unwrap();
        let reasoning: Vec<_> = g
            .content
            .iter()
            .filter(|c| matches!(c, Content::Reasoning(_)))
            .collect();
        assert_eq!(reasoning.len(), 2);
    }

    #[test]
    fn reasoning_empty_summary_still_emits_one() {
        let r = fixture(serde_json::json!({
            "id": "r",
            "output": [{
                "type": "reasoning",
                "id": "rsn_2",
                "summary": []
            }],
            "usage": { "input_tokens": 0, "output_tokens": 0 }
        }));
        let g = parse_response(r, HashMap::new(), None, vec![], "openai", None, false).unwrap();
        let Content::Reasoning(part) = &g.content[0] else {
            panic!("expected reasoning");
        };
        assert_eq!(part.text, "");
    }

    #[test]
    fn web_search_emits_tool_call_and_result() {
        let r = fixture(serde_json::json!({
            "id": "r",
            "output": [{
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {
                    "type": "search",
                    "query": "rust async"
                }
            }],
            "usage": { "input_tokens": 0, "output_tokens": 0 }
        }));
        let g = parse_response(
            r,
            HashMap::new(),
            None,
            vec![],
            "openai",
            Some("web_search"),
            false,
        )
        .unwrap();
        assert!(matches!(g.content[0], Content::ToolCall(_)));
        assert!(matches!(g.content[1], Content::ToolResult(_)));
    }

    #[test]
    fn compaction_becomes_custom_content() {
        let r = fixture(serde_json::json!({
            "id": "r",
            "output": [{
                "type": "compaction",
                "id": "cmp_1",
                "encrypted_content": "enc"
            }],
            "usage": { "input_tokens": 0, "output_tokens": 0 }
        }));
        let g = parse_response(r, HashMap::new(), None, vec![], "openai", None, false).unwrap();
        let Content::Custom { kind, .. } = &g.content[0] else {
            panic!("expected custom");
        };
        assert_eq!(kind, "openai.compaction");
    }

    #[test]
    fn error_body_propagates() {
        let r = fixture(serde_json::json!({
            "id": "r",
            "error": {
                "message": "rate limit",
                "type": "rate_limit_error",
                "code": "rate_limit_exceeded"
            }
        }));
        let result = parse_response(r, HashMap::new(), None, vec![], "openai", None, false);
        assert!(result.is_err());
    }
}
