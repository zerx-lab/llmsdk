//! [`LanguageModel`] implementation for the `Anthropic` Messages API.
// Rust guideline compliant 2026-02-21

use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    CallOptions, FunctionTool, GenerateResult, LanguageModel, StreamResult, Tool, ToolChoice,
};
use llmsdk_provider::shared::Warning;
use llmsdk_provider_utils::http::{JsonRequest, post_for_stream, post_json, response_byte_stream};
use llmsdk_provider_utils::sse::{SseEvent, sse_json_stream};

use crate::PROVIDER_ID;
use crate::auth::apply_request_auth;
use crate::config::Inner;
use crate::error::rewrite_anthropic_error;

use super::convert_prompt::{Converted, convert_prompt, read_cache_control};
use super::options::{AnthropicChatOptions, ThinkingConfig, parse as parse_provider_options};
use super::parse_response::parse_response;
use super::sanitize_json_schema::sanitize_json_schema;
use super::stream::StreamState;
use super::stream_event::StreamEvent;
use super::wire::{
    MessagesRequest, MessagesResponse, WireMessage, WireThinking, WireTool, WireToolChoice,
};
use crate::model_capabilities::model_capabilities;

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
    /// Construct from a fully assembled [`Inner`].
    ///
    /// Public for cross-crate composition (Google Vertex Anthropic, Amazon
    /// Bedrock Anthropic). End-users should prefer
    /// [`crate::Anthropic::messages`].
    #[must_use]
    pub fn new(inner: Arc<Inner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    fn endpoint(&self, is_streaming: bool) -> String {
        self.inner.endpoint_url(&self.model_id, is_streaming)
    }

    /// Serialize the typed request to JSON, then apply the optional
    /// body-transformer hook (Vertex / Bedrock strip `model` and inject
    /// `anthropic_version`). `betas` is forwarded so Bedrock can fold the
    /// full beta list into the request body — Bedrock's Anthropic surface
    /// reads it from the body, not headers.
    fn prepare_body(
        &self,
        request: &MessagesRequest,
        betas: &std::collections::BTreeSet<String>,
    ) -> Result<serde_json::Value, ProviderError> {
        let mut value = serde_json::to_value(request)
            .map_err(|e| ProviderError::json_parse("<request body>", e.to_string()))?;
        self.inner.transform_body(&mut value, betas);
        Ok(value)
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
        let (request, warnings, betas, mark_code_execution_dynamic, uses_json_response_tool) =
            build_request(
                &self.model_id,
                &options,
                false,
                self.inner.supports_native_structured_output(),
                self.inner.supports_strict_tools(),
            );

        let body_value = self.prepare_body(&request, &betas)?;
        let request_body_value = Some(body_value.clone());
        let mut http_request = JsonRequest::new(self.endpoint(false), body_value);
        http_request.headers = self.merged_headers(options.headers.as_ref());
        apply_beta_header(&mut http_request.headers, betas);
        // Serialize once so the auth hook can sign exactly what we send.
        let body_bytes = serde_json::to_vec(&http_request.body)
            .map_err(|e| ProviderError::json_parse("<request body>", e.to_string()))?;
        apply_request_auth(
            self.inner.request_auth.as_ref(),
            &mut http_request.headers,
            "POST",
            &http_request.url,
            &body_bytes,
            Some("application/json"),
        )
        .await?;

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
            mark_code_execution_dynamic,
            uses_json_response_tool,
        )
    }

    async fn do_stream(&self, options: CallOptions) -> Result<StreamResult, ProviderError> {
        let (request, warnings, betas, mark_code_execution_dynamic, uses_json_response_tool) =
            build_request(
                &self.model_id,
                &options,
                true,
                self.inner.supports_native_structured_output(),
                self.inner.supports_strict_tools(),
            );

        let body_value = self.prepare_body(&request, &betas)?;
        let request_body_value = Some(body_value.clone());
        let mut http_request = JsonRequest::new(self.endpoint(true), body_value);
        http_request.headers = self.merged_headers(options.headers.as_ref());
        apply_beta_header(&mut http_request.headers, betas);
        let body_bytes = serde_json::to_vec(&http_request.body)
            .map_err(|e| ProviderError::json_parse("<request body>", e.to_string()))?;
        apply_request_auth(
            self.inner.request_auth.as_ref(),
            &mut http_request.headers,
            "POST",
            &http_request.url,
            &body_bytes,
            Some("application/json"),
        )
        .await?;

        let stream_response = match post_for_stream(&self.inner.http, http_request).await {
            Ok(r) => r,
            Err(err) => return Err(rewrite_anthropic_error(err)),
        };

        let stream_headers = stream_response.headers.clone();
        let byte_stream = response_byte_stream(stream_response.response);
        let event_stream = sse_json_stream::<StreamEvent>(byte_stream);

        let state = StreamState::with_generate_id(
            warnings,
            self.inner.generate_id.clone(),
            mark_code_execution_dynamic,
            uses_json_response_tool,
        );
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
    config_supports_native_structured_output: bool,
    config_supports_strict_tools: bool,
) -> (
    MessagesRequest,
    Vec<Warning>,
    std::collections::BTreeSet<String>,
    bool, // mark_code_execution_dynamic
    bool, // uses_json_response_tool
) {
    let mut provider_opts = parse_provider_options(options.provider_options.as_ref());
    let send_reasoning = provider_opts.send_reasoning.unwrap_or(true);
    let Converted {
        system,
        messages,
        mut warnings,
        betas: prompt_betas,
    } = convert_prompt(&options.prompt, send_reasoning);

    if options.frequency_penalty.is_some() {
        warnings.push(Warning::Unsupported {
            feature: "frequencyPenalty".to_owned(),
            details: Some("Anthropic does not support frequencyPenalty".to_owned()),
        });
    }
    if options.presence_penalty.is_some() {
        warnings.push(Warning::Unsupported {
            feature: "presencePenalty".to_owned(),
            details: Some("Anthropic does not support presencePenalty".to_owned()),
        });
    }
    if options.seed.is_some() {
        warnings.push(Warning::Unsupported {
            feature: "seed".to_owned(),
            details: Some("Anthropic does not support seed".to_owned()),
        });
    }
    // Decide between three response-format strategies based on
    // `structuredOutputMode` + model capabilities (`supportsStructuredOutput`):
    //
    // 1. **outputFormat**: send `response_format.schema` as
    //    `output_config.format` (native structured output).
    // 2. **jsonResponseTool fallback**: synthesize a tool named `json` whose
    //    inputSchema is the user schema, force `tool_choice = {type:"required"}`,
    //    and at parse time render the tool call as text. Used when the model
    //    does not support native structured output but the caller still
    //    requested `responseFormat: 'json'`. Mirrors upstream
    //    `anthropic-language-model.ts:331-355` + `:722-747`.
    // 3. **none**: drop `response_format` entirely (no schema present, or
    //    caller explicitly opted out).
    let caps = model_capabilities(model_id);
    let structured_output_mode = provider_opts
        .structured_output_mode
        .as_deref()
        .unwrap_or("auto");
    // Effective capability merges the per-backend override (Bedrock's
    // `supports_native_structured_output = false` for claude-opus-4-7) with
    // the per-model table. Mirrors upstream
    // `anthropic-language-model.ts:331-333`'s
    // `(config.supportsNativeStructuredOutput ?? true) && modelSupportsStructuredOutput`.
    let supports_structured_output =
        caps.supports_structured_output && config_supports_native_structured_output;
    let use_structured_output = matches!(structured_output_mode, "outputFormat")
        || (matches!(structured_output_mode, "auto") && supports_structured_output);
    let response_format_schema = match &options.response_format {
        Some(llmsdk_provider::language_model::ResponseFormat::Json {
            schema: Some(schema),
            ..
        }) => Some(schema.clone()),
        _ => None,
    };
    let output_format = if use_structured_output {
        response_format_schema.as_ref().map(|schema| {
            let raw: serde_json::Value = schema.clone().into();
            serde_json::json!({
                "type": "json_schema",
                "schema": sanitize_json_schema(&raw),
            })
        })
    } else {
        None
    };
    // jsonResponseTool fires only when the schema is present *and* we did
    // not route through native structured output. Without a schema the model
    // has nothing to constrain against, so we silently drop the request like
    // ai-sdk does.
    let uses_json_response_tool = response_format_schema.is_some() && !use_structured_output;
    let json_response_tool = uses_json_response_tool.then(|| {
        let schema = response_format_schema
            .as_ref()
            .expect("schema presence checked above");
        let raw: serde_json::Value = schema.clone().into();
        Tool::Function(FunctionTool {
            name: "json".to_owned(),
            description: Some("Respond with a JSON object.".to_owned()),
            input_schema: serde_json::from_value(raw).unwrap_or_default(),
            input_examples: None,
            strict: None,
            provider_options: None,
        })
    });

    let mut betas: std::collections::BTreeSet<String> = prompt_betas;
    let tool_streaming_default = provider_opts.tool_streaming.unwrap_or(true);

    // When jsonResponseTool is active, ai-sdk overrides three knobs (see
    // `anthropic-language-model.ts:728-737`): the tool is appended last,
    // tool_choice becomes `{type:"required"}`, and parallel tool use is
    // forced off. The user-provided `tool_choice` is intentionally
    // overridden because the model must call the synthesized `json` tool
    // for the request to produce a parseable response.
    let (combined_tools, effective_tool_choice, effective_disable_parallel) =
        if let Some(json_tool) = &json_response_tool {
            let mut tools_vec: Vec<Tool> = options.tools.clone().unwrap_or_default();
            tools_vec.push(json_tool.clone());
            (Some(tools_vec), Some(ToolChoice::Required), Some(true))
        } else {
            (
                options.tools.clone(),
                options.tool_choice.clone(),
                provider_opts.disable_parallel_tool_use,
            )
        };

    // `supportsStrictTools` mirrors upstream's
    // `(config.supportsStrictTools ?? true) && modelSupportsStructuredOutput`
    // (anthropic-language-model.ts:335-337). It is independent of
    // `supportsStructuredOutput` — wrapping providers may flip either flag.
    let supports_strict_tools = caps.supports_structured_output && config_supports_strict_tools;
    let (tools, tool_choice) = convert_tools(
        combined_tools.as_deref(),
        effective_tool_choice.as_ref(),
        &mut warnings,
        &mut betas,
        effective_disable_parallel,
        tool_streaming_default,
        // When jsonResponseTool fires, upstream forces `supportsStructuredOutput: false`
        // (anthropic-language-model.ts:734) so the synthesized tool is emitted
        // without the `structured-outputs-2025-11-13` beta header, while
        // `supportsStrictTools` continues to gate the per-tool `strict` field.
        if uses_json_response_tool {
            false
        } else {
            supports_structured_output
        },
        supports_strict_tools,
    );

    // anthropicBeta extra tokens.
    if let Some(extra) = &provider_opts.anthropic_beta {
        for token in extra {
            betas.insert(token.clone());
        }
    }

    // Map top-level `options.reasoning` to Anthropic thinking/effort when
    // provider options don't already specify them. Provider options always
    // take precedence (mirrors anthropic-language-model.ts:399-418).
    apply_reasoning_to_provider_opts(
        options.reasoning,
        &caps,
        options.max_output_tokens.unwrap_or(caps.max_output_tokens),
        &mut provider_opts,
        &mut warnings,
    );

    // Warn when adaptive thinking is requested on a model that does not
    // support it; ai-sdk silently strips on Vertex/Bedrock, here we surface.
    if matches!(
        provider_opts.thinking,
        Some(ThinkingConfig::Adaptive { .. })
    ) && !caps.supports_adaptive_thinking
    {
        warnings.push(Warning::Unsupported {
            feature: "thinking.adaptive".to_owned(),
            details: Some(format!(
                "Adaptive thinking is not supported by {model_id}; the server may ignore it"
            )),
        });
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
            warnings.push(Warning::Unsupported {
                feature: "temperature".to_owned(),
                details: Some("temperature is not supported when thinking is enabled".to_owned()),
            });
        }
        if top_k.is_some() {
            top_k = None;
            warnings.push(Warning::Unsupported {
                feature: "topK".to_owned(),
                details: Some("topK is not supported when thinking is enabled".to_owned()),
            });
        }
        if top_p.is_some() {
            top_p = None;
            warnings.push(Warning::Unsupported {
                feature: "topP".to_owned(),
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

    let messages = ensure_user_first(messages, &mut warnings);
    let context_management = provider_opts
        .context_management
        .as_ref()
        .map(|v| normalize_context_management(v, &mut warnings));
    let container = provider_opts
        .container
        .as_ref()
        .map(|v| normalize_container(v, &mut warnings));

    let request = MessagesRequest {
        model: model_id.to_owned(),
        max_tokens,
        messages,
        system,
        temperature,
        top_p,
        top_k,
        stop_sequences: options.stop_sequences.clone(),
        stream: stream.then_some(true),
        tools,
        tool_choice,
        thinking,
        context_management,
        container,
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

    let mark_code_execution_dynamic =
        has_web_tool_20260209_without_code_execution(options.tools.as_deref());

    (
        request,
        warnings,
        betas,
        mark_code_execution_dynamic,
        uses_json_response_tool,
    )
}

/// Known `context_management.edits[].type` strategies.
///
/// Mirrors the switch arms in upstream `anthropic-language-model.ts:540-591`.
/// Unknown strategies are dropped at normalize time with a warning so the
/// wire payload does not get rejected wholesale by the Messages API.
const KNOWN_CONTEXT_EDIT_STRATEGIES: &[&str] = &[
    "clear_tool_uses_20250919",
    "clear_thinking_20251015",
    "compact_20260112",
];

/// Normalize the `context_management` provider-option value into the
/// wire shape Anthropic expects (`snake_case` edit fields).
///
/// Users may pass camelCase (matching ai-sdk option keys); this transform
/// renames the known per-edit fields without altering structure for keys
/// that pass-through unchanged. Edits with an unrecognised `type` are
/// filtered out and reported via `warnings` so callers can react without
/// the request being rejected. Mirrors the inline renames + the unknown
/// strategy filter in upstream `anthropic-language-model.ts:540-591`.
fn normalize_context_management(
    value: &serde_json::Value,
    warnings: &mut Vec<Warning>,
) -> serde_json::Value {
    let serde_json::Value::Object(map) = value else {
        return value.clone();
    };
    let mut out = serde_json::Map::with_capacity(map.len());
    for (key, val) in map {
        if key == "edits"
            && let serde_json::Value::Array(items) = val
        {
            let edits: Vec<serde_json::Value> = items
                .iter()
                .filter_map(|edit| normalize_edit(edit, warnings))
                .collect();
            out.insert("edits".into(), serde_json::Value::Array(edits));
            continue;
        }
        out.insert(key.clone(), val.clone());
    }
    serde_json::Value::Object(out)
}

/// Rename known `camelCase` keys inside one edit entry to `snake_case`.
///
/// Returns `None` (with a warning) when the edit's `type` is not one of
/// the strategies in [`KNOWN_CONTEXT_EDIT_STRATEGIES`].
fn normalize_edit(
    edit: &serde_json::Value,
    warnings: &mut Vec<Warning>,
) -> Option<serde_json::Value> {
    let serde_json::Value::Object(map) = edit else {
        return Some(edit.clone());
    };
    if let Some(serde_json::Value::String(strategy)) = map.get("type")
        && !KNOWN_CONTEXT_EDIT_STRATEGIES.contains(&strategy.as_str())
    {
        warnings.push(Warning::Other {
            message: format!("Unknown context management strategy: {strategy}"),
        });
        return None;
    }
    let mut out = serde_json::Map::with_capacity(map.len());
    for (key, val) in map {
        let renamed = match key.as_str() {
            "clearAtLeast" => "clear_at_least",
            "clearToolInputs" => "clear_tool_inputs",
            "excludeTools" => "exclude_tools",
            "pauseAfterCompaction" => "pause_after_compaction",
            other => other,
        };
        out.insert(renamed.to_owned(), val.clone());
    }
    Some(serde_json::Value::Object(out))
}

/// Normalize the `container` provider-option value into the wire shape
/// Anthropic expects.
///
/// String containers (id-only programmatic tool calling) pass through
/// unchanged. Object containers carry an optional `skills` array whose
/// entries are converted to the wire form `{type, skill_id, version?}`:
///
/// - `type = "anthropic"`: `skill_id` taken from `skillId`.
/// - `type = "custom"`:    `skill_id` resolved from
///   `providerReference["anthropic"]` (mirrors `resolveProviderReference`).
///
/// Entries with an unknown skill `type` or a missing identifier are
/// dropped with a warning. Mirrors upstream
/// `anthropic-language-model.ts:511-533`.
fn normalize_container(
    value: &serde_json::Value,
    warnings: &mut Vec<Warning>,
) -> serde_json::Value {
    let serde_json::Value::Object(map) = value else {
        return value.clone();
    };
    let mut out = serde_json::Map::with_capacity(map.len());
    for (key, val) in map {
        if key == "skills"
            && let serde_json::Value::Array(items) = val
        {
            let skills: Vec<serde_json::Value> = items
                .iter()
                .filter_map(|skill| normalize_skill(skill, warnings))
                .collect();
            out.insert("skills".into(), serde_json::Value::Array(skills));
            continue;
        }
        out.insert(key.clone(), val.clone());
    }
    serde_json::Value::Object(out)
}

/// Convert one `skills[]` entry to its wire shape.
///
/// Returns `None` (with a warning) for unknown skill types or missing
/// identifiers so the resulting wire payload contains only valid entries.
fn normalize_skill(
    skill: &serde_json::Value,
    warnings: &mut Vec<Warning>,
) -> Option<serde_json::Value> {
    let serde_json::Value::Object(map) = skill else {
        return Some(skill.clone());
    };
    let kind = map.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let version = map.get("version").cloned();
    let skill_id = match kind {
        "anthropic" => map.get("skillId").cloned(),
        "custom" => map
            .get("providerReference")
            .and_then(serde_json::Value::as_object)
            .and_then(|r| r.get("anthropic"))
            .cloned(),
        other => {
            warnings.push(Warning::Other {
                message: format!("Unknown container skill type: {other}"),
            });
            return None;
        }
    };
    let Some(skill_id) = skill_id else {
        warnings.push(Warning::Other {
            message: format!(
                "container skill of type '{kind}' is missing its identifier (skillId or providerReference['anthropic'])"
            ),
        });
        return None;
    };
    let mut out = serde_json::Map::with_capacity(3);
    out.insert("type".into(), serde_json::Value::String(kind.to_owned()));
    out.insert("skill_id".into(), skill_id);
    if let Some(v) = version {
        out.insert("version".into(), v);
    }
    Some(serde_json::Value::Object(out))
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
        Some(ThinkingConfig::Adaptive { display }) => (
            Some(WireThinking::Adaptive {
                display: display.clone(),
            }),
            None,
            true,
        ),
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

/// Map a top-level [`llmsdk_provider::language_model::ReasoningEffort`] onto
/// Anthropic's [`ThinkingConfig`] + `effort` provider-option pair.
///
/// Mirrors `resolveAnthropicReasoningConfig` (anthropic-language-model.ts:2686-2732).
///
/// Behavior:
/// - `None`/`ProviderDefault` → no-op.
/// - `None` reasoning level (`'none'`) → `thinking: { type: 'disabled' }`,
///   no `effort`.
/// - Models with `supports_adaptive_thinking` → `thinking: { type: 'adaptive' }`
///   and an `effort` mapped via the upstream effort-map; `xhigh` becomes
///   `"max"` on models without `supports_xhigh_effort`.
/// - Otherwise → `thinking: { type: 'enabled', budgetTokens }` with the
///   budget computed from `mapReasoningToProviderBudget` semantics
///   (percentage of `max_output_tokens_for_model`, clamped to
///   `[1024, max_output_tokens_for_model]`).
///
/// Provider options always take precedence: this function only writes a
/// field when the caller did not set it (and only writes `effort` when the
/// resulting thinking is not disabled).
fn apply_reasoning_to_provider_opts(
    reasoning: Option<llmsdk_provider::language_model::ReasoningEffort>,
    caps: &crate::model_capabilities::ModelCapabilities,
    max_output_tokens_for_model: u32,
    provider_opts: &mut AnthropicChatOptions,
    warnings: &mut Vec<Warning>,
) {
    use llmsdk_provider::language_model::ReasoningEffort;

    // Upstream short-circuits when an explicit `anthropicOptions.effort`
    // is already set (line 399).
    if provider_opts.effort.is_some() {
        return;
    }

    let reasoning = match reasoning {
        // `undefined` or `'provider-default'` ↔ isCustomReasoning(reasoning) === false.
        None | Some(ReasoningEffort::ProviderDefault) => return,
        Some(level) => level,
    };

    // `reasoning === 'none'` ⇒ disable thinking, no effort.
    if matches!(reasoning, ReasoningEffort::None) {
        if provider_opts.thinking.is_none() {
            provider_opts.thinking = Some(ThinkingConfig::Disabled);
        }
        return;
    }

    if caps.supports_adaptive_thinking {
        let (mapped, exact) = match reasoning {
            ReasoningEffort::Minimal => ("low", false),
            ReasoningEffort::Low => ("low", true),
            ReasoningEffort::Medium => ("medium", true),
            ReasoningEffort::High => ("high", true),
            ReasoningEffort::Xhigh => {
                if caps.supports_xhigh_effort {
                    ("xhigh", true)
                } else {
                    ("max", false)
                }
            }
            // already handled above
            ReasoningEffort::None | ReasoningEffort::ProviderDefault => return,
        };
        if !exact {
            warnings.push(Warning::Compatibility {
                feature: "reasoning".to_owned(),
                details: Some(format!(
                    "reasoning \"{}\" is not directly supported by this model. mapped to effort \"{mapped}\".",
                    reasoning_label(reasoning)
                )),
            });
        }
        if provider_opts.thinking.is_none() {
            provider_opts.thinking = Some(ThinkingConfig::Adaptive { display: None });
        }
        // Only write effort when thinking is not disabled (upstream lines 411-415).
        if !matches!(provider_opts.thinking, Some(ThinkingConfig::Disabled)) {
            provider_opts.effort = Some(mapped.to_owned());
        }
        return;
    }

    // Non-adaptive: map to a token budget.
    let pct = match reasoning {
        ReasoningEffort::Minimal => 0.02_f64,
        ReasoningEffort::Low => 0.10,
        ReasoningEffort::Medium => 0.30,
        ReasoningEffort::High => 0.60,
        ReasoningEffort::Xhigh => 0.90,
        ReasoningEffort::None | ReasoningEffort::ProviderDefault => return,
    };
    // Compute percent-of-max-tokens then round-half-away-from-zero.
    // `max_output_tokens_for_model` is u32 ≤ ~10^9, pct ≤ 0.9, so the product
    // fits f64 exactly and the rounded result fits i64.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "max_output_tokens_for_model is u32 ≤ ~10^9; product * pct ≤ 10^9 fits i64 exactly after round"
    )]
    let raw = (f64::from(max_output_tokens_for_model) * pct).round() as i64;
    let clamped = raw.clamp(1024, i64::from(max_output_tokens_for_model));
    // clamped is in [1024, max_output_tokens_for_model] which fits u32 by construction.
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "clamped to [1024, u32::from(max_output_tokens_for_model)]; always non-negative and fits u32"
    )]
    let budget = clamped as u32;
    if provider_opts.thinking.is_none() {
        provider_opts.thinking = Some(ThinkingConfig::Enabled {
            budget_tokens: Some(budget),
        });
    }
}

fn reasoning_label(level: llmsdk_provider::language_model::ReasoningEffort) -> &'static str {
    use llmsdk_provider::language_model::ReasoningEffort;
    match level {
        ReasoningEffort::ProviderDefault => "provider-default",
        ReasoningEffort::None => "none",
        ReasoningEffort::Minimal => "minimal",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::Xhigh => "xhigh",
    }
}

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

/// Whether the request enables `web_search_20260209` or `web_fetch_20260209`
/// **without** an explicit `code_execution` tool.
///
/// When true, the model may implicitly invoke `code_execution` to satisfy
/// the web tool. Such calls must be marked `dynamic: true` so the SDK's
/// generic tool-call validation does not reject them.
///
/// Mirrors `hasWebTool20260209WithoutCodeExecution` in upstream
/// `anthropic-language-model.ts:2661-2683`.
fn has_web_tool_20260209_without_code_execution(tools: Option<&[Tool]>) -> bool {
    let Some(tools) = tools else {
        return false;
    };
    let mut has_web_2026 = false;
    let mut has_code_execution = false;
    for t in tools {
        match t {
            Tool::Provider(p) => {
                if p.id == "anthropic.web_fetch_20260209" || p.id == "anthropic.web_search_20260209"
                {
                    has_web_2026 = true;
                } else if p.id == "anthropic.code_execution_20250522"
                    || p.id == "anthropic.code_execution_20250825"
                    || p.id == "anthropic.code_execution_20260120"
                {
                    has_code_execution = true;
                    break;
                }
            }
            Tool::Function(f) => {
                if f.name == "code_execution" {
                    has_code_execution = true;
                    break;
                }
            }
        }
    }
    has_web_2026 && !has_code_execution
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

#[allow(
    clippy::too_many_lines,
    reason = "single match-statement dispatcher over Anthropic's tool wire surface; splitting obscures flow"
)]
#[allow(
    clippy::too_many_arguments,
    reason = "tool-conversion is a single dispatch point; bundling args into a config struct would hide flow"
)]
fn convert_tools(
    tools: Option<&[Tool]>,
    choice: Option<&ToolChoice>,
    warnings: &mut Vec<Warning>,
    betas: &mut std::collections::BTreeSet<String>,
    disable_parallel_tool_use: Option<bool>,
    tool_streaming_default: bool,
    supports_structured_output: bool,
    supports_strict_tools: bool,
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
            Tool::Function(f) => {
                // Per-tool `provider_options.anthropic.{deferLoading,
                // eagerInputStreaming, allowedCallers}` overrides the
                // model-level `toolStreaming` default.
                let anthropic_opts = f
                    .provider_options
                    .as_ref()
                    .and_then(|po| po.get("anthropic"));
                let defer_loading = anthropic_opts
                    .and_then(|o| o.get("deferLoading"))
                    .and_then(serde_json::Value::as_bool);
                let allowed_callers = anthropic_opts
                    .and_then(|o| o.get("allowedCallers"))
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(str::to_owned))
                            .collect::<Vec<_>>()
                    });
                let per_tool_eager = anthropic_opts
                    .and_then(|o| o.get("eagerInputStreaming"))
                    .and_then(serde_json::Value::as_bool);
                // ai-sdk: eagerInputStreaming = per-tool ?? model-level default;
                // emitted on the wire only when truthy.
                let eager_input_streaming = match per_tool_eager {
                    Some(b) => b.then_some(true),
                    None => tool_streaming_default.then_some(true),
                };
                let input_examples = f.input_examples.as_ref().map(|examples| {
                    examples
                        .iter()
                        .map(|ex| ex.input.clone())
                        .collect::<Vec<_>>()
                });
                let cache_control = read_cache_control(f.provider_options.as_ref());
                let strict = if supports_strict_tools {
                    f.strict
                } else {
                    if let Some(s) = f.strict {
                        warnings.push(Warning::Unsupported {
                            feature: "strict".to_owned(),
                            details: Some(format!(
                                "Tool '{}' has strict: {s}, but strict mode is not supported by this model; ignored",
                                f.name
                            )),
                        });
                    }
                    None
                };
                // Auto-enable beta tokens to match ai-sdk's anthropic-prepare-tools.ts.
                if supports_structured_output {
                    betas.insert("structured-outputs-2025-11-13".to_owned());
                }
                if input_examples.is_some() || allowed_callers.is_some() {
                    betas.insert("advanced-tool-use-2025-11-20".to_owned());
                }
                Some(WireTool::Function(super::wire::WireFunctionTool {
                    name: f.name.clone(),
                    description: f.description.clone(),
                    input_schema: f.input_schema.clone().into(),
                    eager_input_streaming,
                    defer_loading,
                    allowed_callers,
                    input_examples,
                    strict,
                    cache_control,
                }))
            }
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
                    warnings.push(Warning::Unsupported {
                        feature: p.name.clone(),
                        details: Some(format!(
                            "provider-defined feature '{}' not recognized by llmsdk-anthropic",
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
                        warnings.push(Warning::Unsupported {
                            feature: "toolChoice".to_owned(),
                            details: Some(
                                "Anthropic has no `none` feature choice; downgraded to `auto`"
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

#[cfg(test)]
mod tests {
    use super::{normalize_container, normalize_context_management};
    use llmsdk_provider::shared::Warning;
    use serde_json::json;

    #[test]
    fn normalize_container_renames_anthropic_skill_to_snake_case() {
        let mut warnings = Vec::new();
        let input = json!({
            "id": "ctr-1",
            "skills": [{"type": "anthropic", "skillId": "doc-processor", "version": "1.0"}],
        });
        let out = normalize_container(&input, &mut warnings);
        assert_eq!(
            out,
            json!({
                "id": "ctr-1",
                "skills": [{"type": "anthropic", "skill_id": "doc-processor", "version": "1.0"}],
            }),
        );
        assert!(warnings.is_empty());
    }

    #[test]
    fn normalize_container_resolves_custom_provider_reference() {
        let mut warnings = Vec::new();
        let input = json!({
            "skills": [{
                "type": "custom",
                "providerReference": {"anthropic": "skill-abc", "openai": "ignored"},
                "version": "2",
            }],
        });
        let out = normalize_container(&input, &mut warnings);
        assert_eq!(
            out,
            json!({
                "skills": [{"type": "custom", "skill_id": "skill-abc", "version": "2"}],
            }),
        );
        assert!(warnings.is_empty());
    }

    #[test]
    fn normalize_container_drops_skill_missing_identifier_with_warning() {
        let mut warnings = Vec::new();
        let input = json!({
            "skills": [
                {"type": "custom", "providerReference": {"openai": "x"}},
                {"type": "anthropic", "skillId": "ok"},
            ],
        });
        let out = normalize_container(&input, &mut warnings);
        assert_eq!(
            out,
            json!({
                "skills": [{"type": "anthropic", "skill_id": "ok"}],
            }),
        );
        assert_eq!(warnings.len(), 1);
        assert!(matches!(&warnings[0], Warning::Other { message } if message.contains("custom")));
    }

    #[test]
    fn normalize_container_string_id_passes_through_unchanged() {
        let mut warnings = Vec::new();
        let input = json!("ctr-id-only");
        let out = normalize_container(&input, &mut warnings);
        assert_eq!(out, json!("ctr-id-only"));
        assert!(warnings.is_empty());
    }

    #[test]
    fn normalize_context_management_filters_unknown_strategy_with_warning() {
        let mut warnings = Vec::new();
        let input = json!({
            "edits": [
                {"type": "clear_tool_uses_20250919", "clearAtLeast": {"type": "input_tokens", "value": 1000}},
                {"type": "unknown_strategy", "foo": "bar"},
                {"type": "compact_20260112", "pauseAfterCompaction": true},
            ],
        });
        let out = normalize_context_management(&input, &mut warnings);
        assert_eq!(
            out,
            json!({
                "edits": [
                    {"type": "clear_tool_uses_20250919", "clear_at_least": {"type": "input_tokens", "value": 1000}},
                    {"type": "compact_20260112", "pause_after_compaction": true},
                ],
            }),
        );
        assert_eq!(warnings.len(), 1);
        assert!(
            matches!(&warnings[0], Warning::Other { message } if message.contains("unknown_strategy"))
        );
    }

    #[test]
    fn normalize_context_management_renames_all_camel_case_fields() {
        let mut warnings = Vec::new();
        let input = json!({
            "edits": [{
                "type": "clear_tool_uses_20250919",
                "clearAtLeast": {"type": "input_tokens", "value": 100},
                "clearToolInputs": true,
                "excludeTools": ["foo", "bar"],
            }],
        });
        let out = normalize_context_management(&input, &mut warnings);
        assert_eq!(
            out,
            json!({
                "edits": [{
                    "type": "clear_tool_uses_20250919",
                    "clear_at_least": {"type": "input_tokens", "value": 100},
                    "clear_tool_inputs": true,
                    "exclude_tools": ["foo", "bar"],
                }],
            }),
        );
        assert!(warnings.is_empty());
    }

    mod reasoning_mapping {
        use super::super::{
            AnthropicChatOptions, ThinkingConfig, apply_reasoning_to_provider_opts,
        };
        use crate::model_capabilities::model_capabilities;
        use llmsdk_provider::language_model::ReasoningEffort;
        use llmsdk_provider::shared::Warning;

        #[test]
        fn provider_default_is_noop() {
            let caps = model_capabilities("claude-opus-4-7-20251015");
            let mut opts = AnthropicChatOptions::default();
            let mut warnings: Vec<Warning> = Vec::new();
            apply_reasoning_to_provider_opts(
                Some(ReasoningEffort::ProviderDefault),
                &caps,
                64_000,
                &mut opts,
                &mut warnings,
            );
            assert!(opts.thinking.is_none() && opts.effort.is_none());
        }

        #[test]
        fn none_disables_thinking_no_effort() {
            let caps = model_capabilities("claude-opus-4-7-20251015");
            let mut opts = AnthropicChatOptions::default();
            let mut warnings: Vec<Warning> = Vec::new();
            apply_reasoning_to_provider_opts(
                Some(ReasoningEffort::None),
                &caps,
                64_000,
                &mut opts,
                &mut warnings,
            );
            assert_eq!(opts.thinking, Some(ThinkingConfig::Disabled));
            assert_eq!(opts.effort, None);
        }

        #[test]
        fn adaptive_model_maps_low_high_directly() {
            let caps = model_capabilities("claude-opus-4-7-20251015");
            for (level, expected) in [
                (ReasoningEffort::Low, "low"),
                (ReasoningEffort::Medium, "medium"),
                (ReasoningEffort::High, "high"),
                (ReasoningEffort::Xhigh, "xhigh"),
            ] {
                let mut opts = AnthropicChatOptions::default();
                let mut warnings: Vec<Warning> = Vec::new();
                apply_reasoning_to_provider_opts(
                    Some(level),
                    &caps,
                    64_000,
                    &mut opts,
                    &mut warnings,
                );
                assert!(
                    matches!(opts.thinking, Some(ThinkingConfig::Adaptive { .. })),
                    "level {level:?} should pick adaptive thinking"
                );
                assert_eq!(opts.effort.as_deref(), Some(expected));
            }
        }

        #[test]
        fn adaptive_minimal_downgrades_to_low_with_compatibility_warning() {
            let caps = model_capabilities("claude-opus-4-7-20251015");
            let mut opts = AnthropicChatOptions::default();
            let mut warnings: Vec<Warning> = Vec::new();
            apply_reasoning_to_provider_opts(
                Some(ReasoningEffort::Minimal),
                &caps,
                64_000,
                &mut opts,
                &mut warnings,
            );
            assert_eq!(opts.effort.as_deref(), Some("low"));
            assert!(warnings.iter().any(
                |w| matches!(w, Warning::Compatibility { feature, .. } if feature == "reasoning")
            ));
        }

        #[test]
        fn adaptive_xhigh_without_support_maps_to_max() {
            let caps = model_capabilities("claude-sonnet-4-6-20251014");
            let mut opts = AnthropicChatOptions::default();
            let mut warnings: Vec<Warning> = Vec::new();
            apply_reasoning_to_provider_opts(
                Some(ReasoningEffort::Xhigh),
                &caps,
                64_000,
                &mut opts,
                &mut warnings,
            );
            assert_eq!(opts.effort.as_deref(), Some("max"));
            assert!(
                warnings
                    .iter()
                    .any(|w| matches!(w, Warning::Compatibility { .. }))
            );
        }

        #[test]
        fn non_adaptive_model_computes_token_budget() {
            let caps = model_capabilities("claude-opus-4-1");
            let mut opts = AnthropicChatOptions::default();
            let mut warnings: Vec<Warning> = Vec::new();
            apply_reasoning_to_provider_opts(
                Some(ReasoningEffort::Medium),
                &caps,
                32_000,
                &mut opts,
                &mut warnings,
            );
            match opts.thinking {
                Some(ThinkingConfig::Enabled { budget_tokens }) => {
                    let budget = budget_tokens.expect("budget should be populated");
                    // medium → 30% of 32000 = 9600, clamped to [1024, 32000].
                    assert_eq!(budget, 9_600);
                }
                _ => panic!("expected enabled thinking with explicit budget"),
            }
            assert!(opts.effort.is_none());
        }

        #[test]
        fn provider_effort_already_set_short_circuits() {
            let caps = model_capabilities("claude-opus-4-7-20251015");
            let mut opts = AnthropicChatOptions {
                effort: Some("medium".to_owned()),
                ..AnthropicChatOptions::default()
            };
            let mut warnings: Vec<Warning> = Vec::new();
            apply_reasoning_to_provider_opts(
                Some(ReasoningEffort::High),
                &caps,
                64_000,
                &mut opts,
                &mut warnings,
            );
            // No mutation when caller already pinned the effort.
            assert!(opts.thinking.is_none());
            assert_eq!(opts.effort.as_deref(), Some("medium"));
        }

        #[test]
        fn provider_thinking_already_set_is_preserved() {
            let caps = model_capabilities("claude-opus-4-7-20251015");
            let mut opts = AnthropicChatOptions {
                thinking: Some(ThinkingConfig::Enabled {
                    budget_tokens: Some(2_048),
                }),
                ..AnthropicChatOptions::default()
            };
            let mut warnings: Vec<Warning> = Vec::new();
            apply_reasoning_to_provider_opts(
                Some(ReasoningEffort::Low),
                &caps,
                64_000,
                &mut opts,
                &mut warnings,
            );
            // Existing thinking config retained; effort still derived because
            // provider_opts.effort was None and thinking != Disabled.
            assert_eq!(
                opts.thinking,
                Some(ThinkingConfig::Enabled {
                    budget_tokens: Some(2_048),
                })
            );
            assert_eq!(opts.effort.as_deref(), Some("low"));
        }
    }
}
