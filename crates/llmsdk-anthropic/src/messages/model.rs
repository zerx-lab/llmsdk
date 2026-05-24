//! [`LanguageModel`] implementation for the `Anthropic` Messages API.
// Rust guideline compliant 2026-02-21

use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    CallOptions, GenerateResult, LanguageModel, StreamResult, Tool, ToolChoice,
};
use llmsdk_provider::shared::Warning;
use llmsdk_provider_utils::http::{JsonRequest, post_for_stream, post_json, response_byte_stream};
use llmsdk_provider_utils::sse::{SseEvent, sse_json_stream};

use crate::PROVIDER_ID;
use crate::config::Inner;
use crate::error::rewrite_anthropic_error;

use super::convert_prompt::{Converted, convert_prompt};
use super::options::{AnthropicChatOptions, ThinkingConfig, parse as parse_provider_options};
use super::parse_response::parse_response;
use super::sanitize_json_schema::sanitize_json_schema;
use super::stream::StreamState;
use super::stream_event::StreamEvent;
use super::wire::{
    MessagesRequest, MessagesResponse, WireMessage, WireThinking, WireTool, WireToolChoice,
};

/// Fallback `max_tokens` when the caller did not set one.
///
/// `Anthropic` requires `max_tokens` on every Messages request; we choose
/// a conservative cap that matches ai-sdk's documented fallback.
pub(crate) const DEFAULT_MAX_TOKENS: u32 = 4096;

/// `Anthropic` Messages model handle.
#[derive(Debug, Clone)]
pub struct AnthropicMessagesModel {
    inner: Arc<Inner>,
    model_id: String,
}

impl AnthropicMessagesModel {
    pub(crate) fn new(inner: Arc<Inner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    fn endpoint(&self) -> String {
        format!("{}/messages", self.inner.base_url)
    }

    fn merged_headers(
        &self,
        per_call: Option<&llmsdk_provider::shared::Headers>,
    ) -> std::collections::HashMap<String, Option<String>> {
        let mut headers = self.inner.headers.clone();
        if let Some(h) = per_call {
            for (name, value) in h {
                headers.insert(name.clone(), value.clone());
            }
        }
        headers
    }
}

#[async_trait]
impl LanguageModel for AnthropicMessagesModel {
    fn provider(&self) -> &str {
        PROVIDER_ID
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn do_generate(&self, options: CallOptions) -> Result<GenerateResult, ProviderError> {
        let (request, warnings, betas) = build_request(&self.model_id, &options, false);

        let request_body_value = serde_json::to_value(&request).ok();
        let mut http_request = JsonRequest::new(self.endpoint(), request);
        http_request.headers = self.merged_headers(options.headers.as_ref());
        apply_beta_header(&mut http_request.headers, betas);

        let response = match post_json::<_, MessagesResponse>(&self.inner.http, http_request).await
        {
            Ok(r) => r,
            Err(err) => return Err(rewrite_anthropic_error(err)),
        };

        parse_response(
            response.value,
            response.headers,
            request_body_value,
            warnings,
        )
    }

    async fn do_stream(&self, options: CallOptions) -> Result<StreamResult, ProviderError> {
        let (request, warnings, betas) = build_request(&self.model_id, &options, true);

        let request_body_value = serde_json::to_value(&request).ok();
        let mut http_request = JsonRequest::new(self.endpoint(), request);
        http_request.headers = self.merged_headers(options.headers.as_ref());
        apply_beta_header(&mut http_request.headers, betas);

        let stream_response = match post_for_stream(&self.inner.http, http_request).await {
            Ok(r) => r,
            Err(err) => return Err(rewrite_anthropic_error(err)),
        };

        let stream_headers = stream_response.headers.clone();
        let byte_stream = response_byte_stream(stream_response.response);
        let event_stream = sse_json_stream::<StreamEvent>(byte_stream);

        let state = StreamState::new(warnings);
        let include_raw = options.include_raw_chunks.unwrap_or(false);
        let parts = build_part_stream(state, event_stream, include_raw);

        Ok(StreamResult {
            stream: Box::pin(parts),
            request: Some(llmsdk_provider::shared::RequestInfo {
                body: request_body_value,
            }),
            response: Some(llmsdk_provider::language_model::StreamResponse {
                headers: Some(
                    stream_headers
                        .into_iter()
                        .map(|(k, v)| (k, Some(v)))
                        .collect(),
                ),
            }),
        })
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "single dispatcher mirroring ai-sdk's anthropic-language-model.ts; splitting would obscure the parameter flow"
)]
fn build_request(
    model_id: &str,
    options: &CallOptions,
    stream: bool,
) -> (
    MessagesRequest,
    Vec<Warning>,
    std::collections::BTreeSet<String>,
) {
    let provider_opts = parse_provider_options(options.provider_options.as_ref());
    let send_reasoning = provider_opts.send_reasoning.unwrap_or(true);
    let Converted {
        system,
        messages,
        mut warnings,
    } = convert_prompt(&options.prompt, send_reasoning);

    if options.frequency_penalty.is_some() {
        warnings.push(Warning::UnsupportedSetting {
            setting: "frequencyPenalty".to_owned(),
            details: Some("Anthropic does not support frequencyPenalty".to_owned()),
        });
    }
    if options.presence_penalty.is_some() {
        warnings.push(Warning::UnsupportedSetting {
            setting: "presencePenalty".to_owned(),
            details: Some("Anthropic does not support presencePenalty".to_owned()),
        });
    }
    if options.seed.is_some() {
        warnings.push(Warning::UnsupportedSetting {
            setting: "seed".to_owned(),
            details: Some("Anthropic does not support seed".to_owned()),
        });
    }
    // structuredOutputMode controls whether `response_format` flows into
    // `output_config.format` (outputFormat) or is dropped (other modes).
    let structured_output_mode = provider_opts
        .structured_output_mode
        .as_deref()
        .unwrap_or("auto");
    let output_format = if matches!(structured_output_mode, "outputFormat" | "auto") {
        match &options.response_format {
            Some(llmsdk_provider::language_model::ResponseFormat::Json {
                schema: Some(schema),
                ..
            }) => {
                let raw: serde_json::Value = schema.clone().into();
                Some(serde_json::json!({
                    "type": "json_schema",
                    "schema": sanitize_json_schema(&raw),
                }))
            }
            _ => None,
        }
    } else {
        None
    };
    if options.response_format.is_some() && output_format.is_none() {
        warnings.push(Warning::UnsupportedSetting {
            setting: "responseFormat".to_owned(),
            details: Some("responseFormat ignored under current structuredOutputMode".to_owned()),
        });
    }

    let mut betas: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let tool_streaming_default = provider_opts.tool_streaming.unwrap_or(true);
    let (tools, tool_choice) = convert_tools(
        options.tools.as_deref(),
        options.tool_choice.as_ref(),
        &mut warnings,
        &mut betas,
        provider_opts.disable_parallel_tool_use,
        tool_streaming_default,
    );

    // anthropicBeta extra tokens.
    if let Some(extra) = &provider_opts.anthropic_beta {
        for token in extra {
            betas.insert(token.clone());
        }
    }

    let (thinking, thinking_budget, thinking_enabled) =
        resolve_thinking(provider_opts.thinking.as_ref());

    let base_max = options.max_output_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    // Extended thinking: budget tokens are charged on top of `max_tokens`.
    let max_tokens = if thinking_enabled {
        base_max.saturating_add(thinking_budget.unwrap_or(0))
    } else {
        base_max
    };

    let (mut temperature, mut top_p, mut top_k) =
        (options.temperature, options.top_p, options.top_k);
    if thinking_enabled {
        if temperature.is_some() {
            temperature = None;
            warnings.push(Warning::UnsupportedSetting {
                setting: "temperature".to_owned(),
                details: Some("temperature is not supported when thinking is enabled".to_owned()),
            });
        }
        if top_k.is_some() {
            top_k = None;
            warnings.push(Warning::UnsupportedSetting {
                setting: "topK".to_owned(),
                details: Some("topK is not supported when thinking is enabled".to_owned()),
            });
        }
        if top_p.is_some() {
            top_p = None;
            warnings.push(Warning::UnsupportedSetting {
                setting: "topP".to_owned(),
                details: Some("topP is not supported when thinking is enabled".to_owned()),
            });
        }
    }

    let output_config = build_output_config(&provider_opts, output_format);
    let metadata = provider_opts
        .metadata
        .as_ref()
        .and_then(|m| m.user_id.as_deref())
        .map(|user_id| serde_json::json!({ "user_id": user_id }));
    let mcp_servers = provider_opts.mcp_servers.as_ref().map(|servers| {
        let arr: Vec<serde_json::Value> = servers
            .iter()
            .map(|s| {
                let mut obj = serde_json::Map::new();
                obj.insert("type".to_owned(), serde_json::Value::String(s.kind.clone()));
                obj.insert("name".to_owned(), serde_json::Value::String(s.name.clone()));
                obj.insert("url".to_owned(), serde_json::Value::String(s.url.clone()));
                if let Some(token) = &s.authorization_token {
                    obj.insert(
                        "authorization_token".to_owned(),
                        serde_json::Value::String(token.clone()),
                    );
                }
                if let Some(cfg) = &s.tool_configuration {
                    let mut tc = serde_json::Map::new();
                    if let Some(enabled) = cfg.enabled {
                        tc.insert("enabled".to_owned(), serde_json::Value::Bool(enabled));
                    }
                    if let Some(allowed) = &cfg.allowed_tools {
                        tc.insert(
                            "allowed_tools".to_owned(),
                            serde_json::Value::Array(
                                allowed
                                    .iter()
                                    .cloned()
                                    .map(serde_json::Value::String)
                                    .collect(),
                            ),
                        );
                    }
                    obj.insert(
                        "tool_configuration".to_owned(),
                        serde_json::Value::Object(tc),
                    );
                }
                serde_json::Value::Object(obj)
            })
            .collect();
        serde_json::Value::Array(arr)
    });

    let request = MessagesRequest {
        model: model_id.to_owned(),
        max_tokens,
        messages: ensure_user_first(messages, &mut warnings),
        system,
        temperature,
        top_p,
        top_k,
        stop_sequences: options.stop_sequences.clone(),
        stream: stream.then_some(true),
        tools,
        tool_choice,
        thinking,
        context_management: provider_opts.context_management.clone(),
        container: provider_opts.container.clone(),
        output_config,
        speed: provider_opts.speed.clone(),
        inference_geo: provider_opts.inference_geo.clone(),
        cache_control: provider_opts.cache_control.clone(),
        metadata,
        mcp_servers,
    };

    // Add beta for context_management compaction edits (best-effort guess from
    // wire payload — surfaced as `compact_20260112` in the docs).
    if let Some(cm) = &provider_opts.context_management {
        if cm.to_string().contains("compact_20260112") {
            betas.insert("context-management-2026-01-12".to_owned());
        }
        if cm.to_string().contains("clear_thinking_20251015") {
            betas.insert("clear-thinking-2025-10-15".to_owned());
        }
    }

    (request, warnings, betas)
}

/// Merge collected beta tokens into the `anthropic-beta` header.
///
/// Tokens supplied by the caller (per-call or provider-level headers) are
/// preserved; ours are appended (deduplicated) to the comma-separated list.
fn apply_beta_header(
    headers: &mut std::collections::HashMap<String, Option<String>>,
    betas: std::collections::BTreeSet<String>,
) {
    if betas.is_empty() {
        return;
    }
    let key = "anthropic-beta".to_owned();
    let existing = headers.get(&key).cloned().unwrap_or(None);
    let mut tokens: std::collections::BTreeSet<String> = existing
        .as_deref()
        .map(|s| {
            s.split(',')
                .map(|t| t.trim().to_owned())
                .filter(|t| !t.is_empty())
                .collect()
        })
        .unwrap_or_default();
    for b in betas {
        tokens.insert(b);
    }
    headers.insert(key, Some(tokens.into_iter().collect::<Vec<_>>().join(",")));
}

/// Resolve thinking config into the wire payload plus derived flags.
///
/// Returns `(wire, budget, enabled)` where:
/// - `wire` is the on-wire `thinking` field (or `None` when caller did not set it)
/// - `budget` is the requested thinking budget when enabled
/// - `enabled` flags whether the rest of the request must adjust accordingly
fn resolve_thinking(config: Option<&ThinkingConfig>) -> (Option<WireThinking>, Option<u32>, bool) {
    match config {
        None => (None, None, false),
        Some(ThinkingConfig::Disabled) => (Some(WireThinking::Disabled), None, false),
        Some(ThinkingConfig::Adaptive) => (Some(WireThinking::Adaptive), None, true),
        Some(ThinkingConfig::Enabled { budget_tokens }) => {
            // Default budget when caller did not specify (matches ai-sdk).
            let resolved = budget_tokens.or(Some(DEFAULT_THINKING_BUDGET));
            (
                Some(WireThinking::Enabled {
                    budget_tokens: resolved,
                }),
                resolved,
                true,
            )
        }
    }
}

/// Fallback thinking budget when the caller enabled thinking without an
/// explicit `budgetTokens`. Matches ai-sdk's documented default.
pub(crate) const DEFAULT_THINKING_BUDGET: u32 = 1024;

/// Assemble `output_config` from the provider-option triplet
/// `(effort, taskBudget, response_format → output_format)`.
///
/// Returns `None` when none of them is set, matching ai-sdk's behavior.
fn build_output_config(
    provider_opts: &AnthropicChatOptions,
    output_format: Option<serde_json::Value>,
) -> Option<serde_json::Value> {
    if provider_opts.effort.is_none()
        && provider_opts.task_budget.is_none()
        && output_format.is_none()
    {
        return None;
    }
    let mut obj = serde_json::Map::new();
    if let Some(effort) = &provider_opts.effort {
        obj.insert(
            "effort".to_owned(),
            serde_json::Value::String(effort.clone()),
        );
    }
    if let Some(budget) = &provider_opts.task_budget {
        let mut b = serde_json::Map::new();
        b.insert(
            "type".to_owned(),
            serde_json::Value::String(budget.kind.clone()),
        );
        b.insert("total".to_owned(), serde_json::json!(budget.total));
        if let Some(rem) = budget.remaining {
            b.insert("remaining".to_owned(), serde_json::json!(rem));
        }
        obj.insert("task_budget".to_owned(), serde_json::Value::Object(b));
    }
    if let Some(fmt) = output_format {
        obj.insert("format".to_owned(), fmt);
    }
    Some(serde_json::Value::Object(obj))
}

/// Resolved metadata for a versioned Anthropic provider-defined tool.
struct ServerToolRoute {
    /// On-wire `type` value.
    wire_type: &'static str,
    /// Wire `name` value (overrides the caller-supplied tool name when the
    /// provider mandates a fixed name — e.g. `text_editor` → `"str_replace_editor"`).
    /// `None` means "keep caller-supplied name".
    wire_name: Option<&'static str>,
    /// Beta-header tokens required to enable this tool.
    betas: &'static [&'static str],
}

/// Map a versioned `Tool::Provider.id` (e.g. `"anthropic.web_search_20250305"`)
/// to its wire metadata.
///
/// Mirrors `anthropic-prepare-tools.ts`. The 20 server tools below exhaust
/// the upstream switch; unknown ids return `None` and the caller emits an
/// `UnsupportedTool` warning.
///
/// Args are flattened from `Tool::Provider.args` directly into the wire
/// object using **`snake_case`** field names (e.g. `display_width_px`,
/// not `displayWidthPx`). The upstream ai-sdk `camelCase` → `snake_case`
/// mapping is not replicated; callers supply the wire names verbatim.
fn resolve_anthropic_server_tool(id: &str) -> Option<ServerToolRoute> {
    let r = |wire_type, wire_name, betas: &'static [&'static str]| ServerToolRoute {
        wire_type,
        wire_name,
        betas,
    };
    Some(match id {
        // code_execution family
        "anthropic.code_execution_20250522" => r(
            "code_execution_20250522",
            Some("code_execution"),
            &["code-execution-2025-05-22"],
        ),
        "anthropic.code_execution_20250825" => r(
            "code_execution_20250825",
            Some("code_execution"),
            &["code-execution-2025-08-25"],
        ),
        "anthropic.code_execution_20260120" => {
            r("code_execution_20260120", Some("code_execution"), &[])
        }
        // computer family
        "anthropic.computer_20241022" => r(
            "computer_20241022",
            Some("computer"),
            &["computer-use-2024-10-22"],
        ),
        "anthropic.computer_20250124" => r(
            "computer_20250124",
            Some("computer"),
            &["computer-use-2025-01-24"],
        ),
        "anthropic.computer_20251124" => r(
            "computer_20251124",
            Some("computer"),
            &["computer-use-2025-11-24"],
        ),
        // text_editor family
        "anthropic.text_editor_20241022" => r(
            "text_editor_20241022",
            Some("str_replace_editor"),
            &["computer-use-2024-10-22"],
        ),
        "anthropic.text_editor_20250124" => r(
            "text_editor_20250124",
            Some("str_replace_editor"),
            &["computer-use-2025-01-24"],
        ),
        "anthropic.text_editor_20250429" => r(
            "text_editor_20250429",
            Some("str_replace_based_edit_tool"),
            &["computer-use-2025-01-24"],
        ),
        "anthropic.text_editor_20250728" => r(
            "text_editor_20250728",
            Some("str_replace_based_edit_tool"),
            &[],
        ),
        // bash family
        "anthropic.bash_20241022" => r("bash_20241022", Some("bash"), &["computer-use-2024-10-22"]),
        "anthropic.bash_20250124" => r("bash_20250124", Some("bash"), &["computer-use-2025-01-24"]),
        // memory
        "anthropic.memory_20250818" => r(
            "memory_20250818",
            Some("memory"),
            &["context-management-2025-06-27"],
        ),
        // web_fetch family
        "anthropic.web_fetch_20250910" => r(
            "web_fetch_20250910",
            Some("web_fetch"),
            &["web-fetch-2025-09-10"],
        ),
        "anthropic.web_fetch_20260209" => r(
            "web_fetch_20260209",
            Some("web_fetch"),
            &["code-execution-web-tools-2026-02-09"],
        ),
        // web_search family
        "anthropic.web_search_20250305" => r("web_search_20250305", Some("web_search"), &[]),
        "anthropic.web_search_20260209" => r(
            "web_search_20260209",
            Some("web_search"),
            &["code-execution-web-tools-2026-02-09"],
        ),
        // tool_search family
        "anthropic.tool_search_regex_20251119" => r(
            "tool_search_tool_regex_20251119",
            Some("tool_search_tool_regex"),
            &[],
        ),
        "anthropic.tool_search_bm25_20251119" => r(
            "tool_search_tool_bm25_20251119",
            Some("tool_search_tool_bm25"),
            &[],
        ),
        // advisor
        "anthropic.advisor_20260301" => r(
            "advisor_20260301",
            Some("advisor"),
            &["advisor-tool-2026-03-01"],
        ),
        _ => return None,
    })
}

/// `Anthropic` requires the first message to be a user message. If we
/// produced an empty list (caller only sent system), inject a trivial
/// user message rather than reject — easier to debug than a 400.
fn ensure_user_first(messages: Vec<WireMessage>, warnings: &mut Vec<Warning>) -> Vec<WireMessage> {
    if messages.is_empty() {
        warnings.push(Warning::Other {
            message: "prompt had no non-system messages; inserting empty user turn".to_owned(),
        });
        vec![WireMessage::User {
            content: vec![super::wire::WireUserPart::Text {
                text: String::new(),
                cache_control: None,
            }],
        }]
    } else {
        messages
    }
}

fn convert_tools(
    tools: Option<&[Tool]>,
    choice: Option<&ToolChoice>,
    warnings: &mut Vec<Warning>,
    betas: &mut std::collections::BTreeSet<String>,
    disable_parallel_tool_use: Option<bool>,
    tool_streaming_default: bool,
) -> (Option<Vec<WireTool>>, Option<WireToolChoice>) {
    let Some(tools) = tools else {
        // No tools but disable_parallel_tool_use was still requested — ai-sdk
        // emits a tool_choice anyway. We mirror that for parity.
        return (
            None,
            disable_parallel_tool_use.map(|flag| WireToolChoice::Auto {
                disable_parallel_tool_use: Some(flag),
            }),
        );
    };
    let converted: Vec<_> = tools
        .iter()
        .filter_map(|t| match t {
            Tool::Function(f) => Some(WireTool::Function(super::wire::WireFunctionTool {
                name: f.name.clone(),
                description: f.description.clone(),
                input_schema: f.input_schema.clone().into(),
                eager_input_streaming: tool_streaming_default.then_some(true),
            })),
            Tool::Provider(p) => {
                if let Some(route) = resolve_anthropic_server_tool(&p.id) {
                    for b in route.betas {
                        betas.insert((*b).to_owned());
                    }
                    let name = route
                        .wire_name
                        .map(str::to_owned)
                        .or_else(|| Some(p.name.clone()));
                    Some(WireTool::Server(super::wire::WireServerTool {
                        kind: route.wire_type.to_owned(),
                        name,
                        args: p.args.clone().unwrap_or_default(),
                    }))
                } else {
                    warnings.push(Warning::UnsupportedTool {
                        tool: p.name.clone(),
                        details: Some(format!(
                            "provider-defined tool '{}' not recognized by llmsdk-anthropic",
                            p.id
                        )),
                    });
                    None
                }
            }
        })
        .collect();
    if converted.is_empty() {
        return (None, None);
    }
    let dpu = disable_parallel_tool_use;
    let tool_choice = choice.map_or_else(
        // Caller didn't pick a choice — emit only if disable_parallel_tool_use
        // was set, matching ai-sdk's behavior.
        || {
            dpu.map(|flag| WireToolChoice::Auto {
                disable_parallel_tool_use: Some(flag),
            })
        },
        |c| {
            Some(match c {
                ToolChoice::Auto | ToolChoice::None => {
                    if matches!(c, ToolChoice::None) {
                        warnings.push(Warning::UnsupportedSetting {
                            setting: "toolChoice".to_owned(),
                            details: Some(
                                "Anthropic has no `none` tool choice; downgraded to `auto`"
                                    .to_owned(),
                            ),
                        });
                    }
                    WireToolChoice::Auto {
                        disable_parallel_tool_use: dpu,
                    }
                }
                ToolChoice::Required => WireToolChoice::Any {
                    disable_parallel_tool_use: dpu,
                },
                ToolChoice::Tool { tool_name } => WireToolChoice::Tool {
                    name: tool_name.clone(),
                    disable_parallel_tool_use: dpu,
                },
            })
        },
    );
    (Some(converted), tool_choice)
}

/// Drive an SSE event stream through [`StreamState`].
fn build_part_stream<S>(
    mut state: StreamState,
    events: S,
    include_raw: bool,
) -> impl futures::Stream<Item = Result<llmsdk_provider::language_model::StreamPart, ProviderError>> + Send
where
    S: futures::Stream<Item = Result<SseEvent<StreamEvent>, ProviderError>> + Send + 'static,
{
    async_stream::stream! {
        for part in state.start_frames() {
            yield Ok(part);
        }
        let mut events = Box::pin(events);
        while let Some(event) = futures::StreamExt::next(&mut events).await {
            match event {
                Ok(SseEvent::Data(ev)) => {
                    if include_raw
                        && let Ok(raw_value) = serde_json::to_value(&ev) {
                            yield Ok(llmsdk_provider::language_model::StreamPart::Raw {
                                raw_value,
                            });
                        }
                    for part in state.on_event(ev) {
                        yield Ok(part);
                    }
                }
                Ok(SseEvent::ParseError { raw, message }) => {
                    for part in state.on_parse_error(&raw, &message) {
                        yield Ok(part);
                    }
                }
                Err(e) => {
                    yield Err(e);
                    return;
                }
            }
        }
        for part in state.flush() {
            yield Ok(part);
        }
    }
}
