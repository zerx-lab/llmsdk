//! Convert an xAI Responses non-streaming response to a [`GenerateResult`].
//!
//! Mirrors the post-processing block in `xai-responses-language-model.ts`'s
//! `doGenerate`.
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;

use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    Content, GenerateResponse, GenerateResult, ReasoningPart, ResponseMetadata, Source, TextPart,
    ToolCallPart, ToolResult, ToolResultOutput,
};
use llmsdk_provider::shared::{Headers, ProviderMetadata, ProviderOptions, RequestInfo, Warning};
use llmsdk_provider_utils::time::rfc3339_from_unix_seconds;
use serde_json::{Map, Value, json};

use super::finish_reason;
use super::prepare_tools::ResolvedToolNames;
use super::usage;
use super::wire::{
    FileSearchCallItem, McpCallItem, MessageItem, OutputItem, ReasoningItem, ResponsesResponse,
    ToolCallItem,
};

const WEB_SEARCH_SUB_TOOLS: &[&str] = &["web_search", "web_search_with_snippets", "browse_page"];
const X_SEARCH_SUB_TOOLS: &[&str] = &[
    "x_user_search",
    "x_keyword_search",
    "x_semantic_search",
    "x_thread_fetch",
];

/// Parse a successful (non-streaming) responses payload.
///
/// `citation_seed` is mutated to produce stable monotonic citation ids.
///
/// # Errors
///
/// Currently never fails (xAI returns `output: []` for empty completions and
/// status carries the failure reason). Returns `Result` for symmetry with the
/// other providers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn parse_response(
    response: ResponsesResponse,
    headers: HashMap<String, String>,
    request_body: Option<Value>,
    warnings: Vec<Warning>,
    names: &ResolvedToolNames,
    citation_seed: &mut u64,
) -> Result<GenerateResult, ProviderError> {
    let mut content: Vec<Content> = Vec::new();
    let mut has_function_call = false;

    for item in response.output {
        match item {
            OutputItem::Message(m) => collect_message(m, &mut content, citation_seed),
            OutputItem::Reasoning(r) => collect_reasoning(r, &mut content),
            OutputItem::FunctionCall(f) => {
                has_function_call = true;
                let input = serde_json::from_str::<Value>(&f.arguments)
                    .unwrap_or(Value::String(f.arguments.clone()));
                content.push(Content::ToolCall(ToolCallPart {
                    tool_call_id: f.call_id,
                    tool_name: f.name,
                    input,
                    provider_executed: None,
                    dynamic: None,
                    provider_options: None,
                }));
            }
            OutputItem::FileSearchCall(fs) => collect_file_search(fs, names, &mut content),
            OutputItem::WebSearchCall(t) => {
                collect_server_tool(t, names, ServerToolKind::WebSearch, &mut content)
            }
            OutputItem::XSearchCall(t) => {
                collect_server_tool(t, names, ServerToolKind::XSearch, &mut content)
            }
            OutputItem::CodeInterpreterCall(t) | OutputItem::CodeExecutionCall(t) => {
                collect_server_tool(t, names, ServerToolKind::CodeExecution, &mut content)
            }
            OutputItem::ViewImageCall(t) => {
                collect_server_tool(t, names, ServerToolKind::ViewImage, &mut content)
            }
            OutputItem::ViewXVideoCall(t) => {
                collect_server_tool(t, names, ServerToolKind::ViewXVideo, &mut content)
            }
            OutputItem::CustomToolCall(t) => {
                collect_server_tool(t, names, ServerToolKind::Custom, &mut content)
            }
            OutputItem::McpCall(m) => collect_mcp(m, names, &mut content),
            OutputItem::Other => {}
        }
    }

    let unified = if has_function_call {
        llmsdk_provider::language_model::FinishReason::with_raw(
            llmsdk_provider::language_model::FinishReasonKind::ToolCalls,
            response.status.clone().unwrap_or_default(),
        )
    } else {
        finish_reason::map(response.status.as_deref())
    };

    let usage_value = response
        .usage
        .as_ref()
        .map_or_else(usage::zero, usage::convert);

    let provider_metadata = provider_metadata_from_cost(response.usage.as_ref());

    let response_meta = GenerateResponse {
        metadata: ResponseMetadata {
            id: response.id,
            timestamp: response.created_at.map(rfc3339_from_unix_seconds),
            model_id: response.model,
            headers: Some(headers_to_provider(headers)),
        },
        body: None,
    };

    Ok(GenerateResult {
        content,
        finish_reason: unified,
        usage: usage_value,
        provider_metadata,
        request: request_body.map(|body| RequestInfo { body: Some(body) }),
        response: Some(response_meta),
        warnings,
    })
}

#[derive(Debug, Clone, Copy)]
enum ServerToolKind {
    WebSearch,
    XSearch,
    CodeExecution,
    ViewImage,
    ViewXVideo,
    Custom,
}

fn collect_message(message: MessageItem, content: &mut Vec<Content>, citation_seed: &mut u64) {
    for part in message.content {
        if let Some(text) = &part.text
            && !text.is_empty()
        {
            content.push(Content::Text(TextPart {
                text: text.clone(),
                provider_options: None,
            }));
        }
        if let Some(annotations) = part.annotations {
            for ann in annotations {
                if let Some((url, title)) = ann.as_url_citation() {
                    content.push(Content::Source(Source::Url {
                        id: next_id(citation_seed),
                        url: url.to_owned(),
                        title: Some(title.unwrap_or(url).to_owned()),
                        provider_metadata: None,
                    }));
                }
            }
        }
    }
}

fn collect_reasoning(item: ReasoningItem, content: &mut Vec<Content>) {
    let summary_texts: Vec<String> = if !item.summary.is_empty() {
        item.summary.iter().map(|s| s.text.clone()).collect()
    } else {
        item.content
            .as_ref()
            .map(|cs| cs.iter().map(|c| c.text.clone()).collect())
            .unwrap_or_default()
    };
    let reasoning_text: String = summary_texts
        .iter()
        .filter(|t| !t.is_empty())
        .cloned()
        .collect();

    let has_metadata = item.encrypted_content.is_some() || !item.id.is_empty();
    if reasoning_text.is_empty() && item.encrypted_content.is_none() {
        return;
    }

    let mut po: Option<ProviderOptions> = None;
    if has_metadata {
        let mut xai = Map::new();
        if let Some(enc) = &item.encrypted_content {
            xai.insert("reasoningEncryptedContent".into(), json!(enc));
        }
        if !item.id.is_empty() {
            xai.insert("itemId".into(), json!(item.id));
        }
        let mut outer = ProviderOptions::new();
        outer.insert("xai".into(), xai);
        po = Some(outer);
    }

    content.push(Content::Reasoning(ReasoningPart {
        text: reasoning_text,
        provider_options: po,
    }));
}

fn collect_file_search(
    item: FileSearchCallItem,
    names: &ResolvedToolNames,
    content: &mut Vec<Content>,
) {
    let tool_name = names
        .file_search
        .clone()
        .unwrap_or_else(|| "file_search".to_owned());

    content.push(Content::ToolCall(ToolCallPart {
        tool_call_id: item.id.clone(),
        tool_name: tool_name.clone(),
        input: Value::String(String::new()),
        provider_executed: Some(true),
        dynamic: None,
        provider_options: None,
    }));

    let queries = item.queries.unwrap_or_default();
    let results_value = item
        .results
        .map(|rs| {
            Value::Array(
                rs.into_iter()
                    .map(|r| {
                        json!({
                            "fileId": r.file_id,
                            "filename": r.filename,
                            "score": r.score,
                            "text": r.text,
                        })
                    })
                    .collect(),
            )
        })
        .unwrap_or(Value::Null);

    let output_value = json!({
        "queries": queries,
        "results": results_value,
    });

    content.push(Content::ToolResult(ToolResult {
        tool_call_id: item.id,
        tool_name,
        output: ToolResultOutput::Json {
            value: output_value,
            provider_options: None,
        },
        preliminary: None,
        provider_metadata: None,
    }));
}

fn collect_server_tool(
    item: ToolCallItem,
    names: &ResolvedToolNames,
    kind: ServerToolKind,
    content: &mut Vec<Content>,
) {
    let name = item.name.clone().unwrap_or_default();
    let tool_name = match kind {
        ServerToolKind::WebSearch => names
            .web_search
            .clone()
            .unwrap_or_else(|| "web_search".to_owned()),
        ServerToolKind::XSearch => names
            .x_search
            .clone()
            .unwrap_or_else(|| "x_search".to_owned()),
        ServerToolKind::CodeExecution => names
            .code_execution
            .clone()
            .unwrap_or_else(|| "code_execution".to_owned()),
        ServerToolKind::ViewImage => {
            if name.is_empty() {
                "view_image".to_owned()
            } else {
                name.clone()
            }
        }
        ServerToolKind::ViewXVideo => {
            if name.is_empty() {
                "view_x_video".to_owned()
            } else {
                name.clone()
            }
        }
        ServerToolKind::Custom => {
            if WEB_SEARCH_SUB_TOOLS.iter().any(|s| *s == name.as_str()) {
                names
                    .web_search
                    .clone()
                    .unwrap_or_else(|| "web_search".to_owned())
            } else if X_SEARCH_SUB_TOOLS.iter().any(|s| *s == name.as_str()) {
                names
                    .x_search
                    .clone()
                    .unwrap_or_else(|| "x_search".to_owned())
            } else if name == "code_execution" {
                names
                    .code_execution
                    .clone()
                    .unwrap_or_else(|| "code_execution".to_owned())
            } else if name.is_empty() {
                "custom_tool".to_owned()
            } else {
                name.clone()
            }
        }
    };

    let input_str = match kind {
        ServerToolKind::Custom => item.input.unwrap_or_default(),
        _ => item.arguments.unwrap_or_default(),
    };

    content.push(Content::ToolCall(ToolCallPart {
        tool_call_id: item.id,
        tool_name,
        input: Value::String(input_str),
        provider_executed: Some(true),
        dynamic: None,
        provider_options: None,
    }));
}

fn collect_mcp(item: McpCallItem, names: &ResolvedToolNames, content: &mut Vec<Content>) {
    let tool_name = names
        .mcp
        .clone()
        .or(item.name.clone())
        .unwrap_or_else(|| "mcp".to_owned());

    let input_str = item.arguments.unwrap_or_default();

    content.push(Content::ToolCall(ToolCallPart {
        tool_call_id: item.id,
        tool_name,
        input: Value::String(input_str),
        provider_executed: Some(true),
        dynamic: None,
        provider_options: None,
    }));
}

fn provider_metadata_from_cost(usage: Option<&super::wire::WireUsage>) -> Option<ProviderMetadata> {
    let cost = usage?.cost_in_usd_ticks?;
    let mut xai = Map::new();
    xai.insert("costInUsdTicks".into(), json!(cost));
    let mut outer = ProviderMetadata::new();
    outer.insert("xai".into(), xai);
    Some(outer)
}

/// Generate a stable monotonic citation id.
pub(crate) fn next_id(seed: &mut u64) -> String {
    *seed = seed.wrapping_add(1);
    format!("xai-responses-citation-{seed}")
}

fn headers_to_provider(raw: HashMap<String, String>) -> Headers {
    raw.into_iter().map(|(k, v)| (k, Some(v))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::responses::wire::{
        FileSearchResult, FunctionCallItem, MessageContentPart, WireUsage,
    };
    use llmsdk_provider::language_model::FinishReasonKind;

    fn empty_headers() -> HashMap<String, String> {
        HashMap::new()
    }

    fn names() -> ResolvedToolNames {
        ResolvedToolNames::default()
    }

    #[test]
    fn parses_plain_message() {
        let resp = ResponsesResponse {
            id: Some("resp_1".into()),
            output: vec![OutputItem::Message(MessageItem {
                id: "msg_1".into(),
                role: Some("assistant".into()),
                status: Some("completed".into()),
                content: vec![MessageContentPart {
                    kind: Some("output_text".into()),
                    text: Some("hello".into()),
                    annotations: None,
                }],
            })],
            status: Some("completed".into()),
            ..Default::default()
        };
        let mut seed = 0;
        let r = parse_response(resp, empty_headers(), None, vec![], &names(), &mut seed).unwrap();
        assert_eq!(r.content.len(), 1);
        assert!(matches!(r.content[0], Content::Text(_)));
        assert_eq!(r.finish_reason.unified, FinishReasonKind::Stop);
    }

    #[test]
    fn parses_function_call_marks_tool_calls() {
        let resp = ResponsesResponse {
            output: vec![OutputItem::FunctionCall(FunctionCallItem {
                name: "weather".into(),
                arguments: r#"{"city":"NYC"}"#.into(),
                call_id: "call_1".into(),
                id: "fc_1".into(),
            })],
            status: Some("completed".into()),
            ..Default::default()
        };
        let mut seed = 0;
        let r = parse_response(resp, empty_headers(), None, vec![], &names(), &mut seed).unwrap();
        assert_eq!(r.content.len(), 1);
        let Content::ToolCall(tc) = &r.content[0] else {
            panic!("expected ToolCall");
        };
        assert_eq!(tc.tool_call_id, "call_1");
        assert_eq!(tc.input["city"], "NYC");
        assert_eq!(r.finish_reason.unified, FinishReasonKind::ToolCalls);
    }

    #[test]
    fn parses_reasoning_with_metadata() {
        let resp = ResponsesResponse {
            output: vec![OutputItem::Reasoning(ReasoningItem {
                id: "rs_1".into(),
                summary: vec![super::super::wire::ReasoningSummaryPart {
                    kind: Some("summary_text".into()),
                    text: "thinking".into(),
                }],
                content: None,
                status: Some("completed".into()),
                encrypted_content: Some("enc".into()),
            })],
            status: Some("completed".into()),
            ..Default::default()
        };
        let mut seed = 0;
        let r = parse_response(resp, empty_headers(), None, vec![], &names(), &mut seed).unwrap();
        let Content::Reasoning(rp) = &r.content[0] else {
            panic!("expected Reasoning");
        };
        assert_eq!(rp.text, "thinking");
        let po = rp.provider_options.as_ref().unwrap();
        let xai = po.get("xai").unwrap();
        assert_eq!(xai["itemId"], "rs_1");
        assert_eq!(xai["reasoningEncryptedContent"], "enc");
    }

    #[test]
    fn parses_file_search_emits_call_plus_result() {
        let resp = ResponsesResponse {
            output: vec![OutputItem::FileSearchCall(FileSearchCallItem {
                id: "fs_1".into(),
                status: Some("completed".into()),
                queries: Some(vec!["q".into()]),
                results: Some(vec![FileSearchResult {
                    file_id: "file_1".into(),
                    filename: "a.txt".into(),
                    score: 0.5,
                    text: "snippet".into(),
                }]),
            })],
            status: Some("completed".into()),
            ..Default::default()
        };
        let mut seed = 0;
        let mut nm = names();
        nm.file_search = Some("file_search".into());
        let r = parse_response(resp, empty_headers(), None, vec![], &nm, &mut seed).unwrap();
        assert_eq!(r.content.len(), 2);
        assert!(matches!(r.content[0], Content::ToolCall(_)));
        let Content::ToolResult(tr) = &r.content[1] else {
            panic!("expected ToolResult");
        };
        let ToolResultOutput::Json { value, .. } = &tr.output else {
            panic!("expected Json output");
        };
        assert_eq!(value["queries"][0], "q");
        assert_eq!(value["results"][0]["fileId"], "file_1");
    }

    #[test]
    fn cost_in_usd_ticks_passes_to_provider_metadata() {
        let resp = ResponsesResponse {
            output: vec![],
            usage: Some(WireUsage {
                input_tokens: Some(0),
                output_tokens: Some(0),
                cost_in_usd_ticks: Some(42),
                ..Default::default()
            }),
            status: Some("completed".into()),
            ..Default::default()
        };
        let mut seed = 0;
        let r = parse_response(resp, empty_headers(), None, vec![], &names(), &mut seed).unwrap();
        let pm = r.provider_metadata.unwrap();
        assert_eq!(pm["xai"]["costInUsdTicks"], 42);
    }
}
