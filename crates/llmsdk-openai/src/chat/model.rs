//! [`LanguageModel`] implementation for `OpenAI` Chat Completions.
//!
//! Top-level entry: [`OpenAiChatModel`]. Construct via [`crate::OpenAi::chat`].
// Rust guideline compliant 2026-02-21

use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    CallOptions, GenerateResult, LanguageModel, ReasoningEffort, ResponseFormat, StreamResult,
    Tool, ToolChoice,
};
use llmsdk_provider::shared::Warning;
use llmsdk_provider_utils::http::{JsonRequest, post_for_stream, post_json, response_byte_stream};
use llmsdk_provider_utils::sse::{SseEvent, sse_json_stream};

use crate::config::Inner;

use super::capabilities::Capabilities;
use super::options::{LogprobsOption, parse as parse_provider_options};
use super::stream::StreamState;
use super::stream_chunk::ChatChunk;
use super::wire::{
    ChatRequest, ChatResponse, ResponseFormat as WireResponseFormat, StreamOptions, TextOptions,
    WireFunctionDef, WireFunctionKind, WireJsonSchema, WireTool, WireToolCallKind, WireToolChoice,
    WireToolChoiceFunction, WireToolChoiceSimple,
};
use super::{convert_prompt, parse_response};
use crate::error::rewrite_openai_error;

/// `OpenAI` Chat Completions model handle.
///
/// Cheap to clone. Multiple clones share the underlying HTTP client and
/// authentication state via [`OpenAi`]'s `Arc`.
#[derive(Debug, Clone)]
pub struct OpenAiChatModel {
    inner: Arc<Inner>,
    model_id: String,
}

impl OpenAiChatModel {
    /// Construct from a fully assembled [`Inner`].
    ///
    /// Public for cross-crate composition (Azure `OpenAI`). End-users should
    /// prefer [`crate::OpenAi::chat`].
    #[must_use]
    pub fn new(inner: Arc<Inner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    fn endpoint(&self) -> String {
        self.inner.endpoint("/chat/completions", &self.model_id)
    }
}

#[async_trait]
impl LanguageModel for OpenAiChatModel {
    fn provider(&self) -> &str {
        self.inner.provider_id()
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn do_generate(&self, options: CallOptions) -> Result<GenerateResult, ProviderError> {
        let (request, warnings) = build_request(&self.model_id, &options);

        // Merge per-provider headers with per-call headers (call-site wins).
        let mut request_headers = self.inner.headers.clone();
        if let Some(headers) = &options.headers {
            for (name, value) in headers {
                request_headers.insert(name.clone(), value.clone());
            }
        }

        let request_body_value = serde_json::to_value(&request).ok();

        let mut http_request = JsonRequest::new(self.endpoint(), request);
        http_request.headers = request_headers;

        let response = match post_json::<_, ChatResponse>(&self.inner.http, http_request).await {
            Ok(r) => r,
            Err(err) => return Err(rewrite_openai_error(err)),
        };

        parse_response(
            response.value,
            response.headers,
            request_body_value,
            warnings,
        )
    }

    async fn do_stream(&self, options: CallOptions) -> Result<StreamResult, ProviderError> {
        let (mut request, warnings) = build_request(&self.model_id, &options);
        request.stream = Some(true);
        request.stream_options = Some(StreamOptions {
            include_usage: Some(true),
        });

        let request_body_value = serde_json::to_value(&request).ok();

        let mut request_headers = self.inner.headers.clone();
        if let Some(headers) = &options.headers {
            for (name, value) in headers {
                request_headers.insert(name.clone(), value.clone());
            }
        }

        let mut http_request = JsonRequest::new(self.endpoint(), request);
        http_request.headers = request_headers;

        let stream_response = match post_for_stream(&self.inner.http, http_request).await {
            Ok(r) => r,
            Err(err) => return Err(rewrite_openai_error(err)),
        };

        let stream_headers = stream_response.headers.clone();
        let byte_stream = response_byte_stream(stream_response.response);
        let event_stream = sse_json_stream::<ChatChunk>(byte_stream);

        let state = StreamState::new(warnings);
        let parts = build_part_stream(state, event_stream);

        Ok(StreamResult {
            stream: Box::pin(parts),
            request: Some(llmsdk_provider::shared::RequestInfo {
                body: request_body_value,
            }),
            response: Some(llmsdk_provider::language_model::StreamResponse {
                headers: Some(headers_to_provider(stream_headers)),
            }),
        })
    }
}

fn headers_to_provider(
    raw: std::collections::HashMap<String, String>,
) -> llmsdk_provider::shared::Headers {
    raw.into_iter().map(|(k, v)| (k, Some(v))).collect()
}

/// Drive the SSE event stream through [`StreamState`], emitting one
/// [`StreamPart`] at a time and flushing the trailing `Finish` frame.
fn build_part_stream<S>(
    mut state: StreamState,
    events: S,
) -> impl futures::Stream<Item = Result<llmsdk_provider::language_model::StreamPart, ProviderError>> + Send
where
    S: futures::Stream<Item = Result<SseEvent<ChatChunk>, ProviderError>> + Send + 'static,
{
    async_stream::stream! {
        // 1. Initial frames (StreamStart).
        for part in state.start_frames() {
            yield Ok(part);
        }

        // 2. Drain SSE events.
        let mut events = Box::pin(events);
        while let Some(event) = futures::StreamExt::next(&mut events).await {
            match event {
                Ok(SseEvent::Data(chunk)) => {
                    for part in state.on_chunk(chunk) {
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

        // 3. Flush trailing frames.
        for part in state.flush() {
            yield Ok(part);
        }
    }
}

/// Build the wire request and collect warnings about dropped settings.
#[allow(
    clippy::too_many_lines,
    reason = "single dispatcher mirroring ai-sdk's openai-chat-language-model.ts; splitting would obscure the parameter flow"
)]
fn build_request(model_id: &str, options: &CallOptions) -> (ChatRequest, Vec<Warning>) {
    let provider_opts = parse_provider_options(options.provider_options.as_ref());
    let caps = Capabilities::detect(model_id);

    // forceReasoning overrides id-based detection.
    let is_reasoning_model = provider_opts
        .force_reasoning
        .unwrap_or(caps.is_reasoning_model);

    // Resolve reasoning effort: provider-option override wins; otherwise
    // fall back to the top-level `reasoning` field.
    let resolved_reasoning_effort = provider_opts
        .reasoning_effort
        .clone()
        .or_else(|| options.reasoning.and_then(reasoning_effort_wire_value));

    // System-message role: provider option `systemMessageMode` wins; otherwise
    // reasoning models default to `developer`, all others to `system`.
    let auto_role = if is_reasoning_model {
        SystemRole::Developer
    } else {
        SystemRole::System
    };
    let system_role = match provider_opts.system_message_mode.as_deref() {
        Some("system") => SystemRole::System,
        Some("developer") => SystemRole::Developer,
        Some("remove") => SystemRole::Remove,
        _ => auto_role,
    };
    let (messages, mut warnings) = convert_prompt(&options.prompt, system_role);

    if let Some(mode) = provider_opts.system_message_mode.as_deref()
        && !matches!(mode, "system" | "developer" | "remove")
    {
        warnings.push(Warning::UnsupportedSetting {
            setting: "openai.systemMessageMode".to_owned(),
            details: Some(format!(
                "unknown systemMessageMode '{mode}' — must be one of system/developer/remove"
            )),
        });
    }

    if options.top_k.is_some() {
        warnings.push(Warning::UnsupportedSetting {
            setting: "topK".to_owned(),
            details: Some("OpenAI Chat Completions does not accept topK".to_owned()),
        });
    }

    let strict_json_schema = provider_opts.strict_json_schema.unwrap_or(true);
    let response_format = options
        .response_format
        .as_ref()
        .map(|fmt| convert_response_format(fmt, strict_json_schema));
    let (tools, tool_choice) = convert_tools(
        options.tools.as_deref(),
        options.tool_choice.as_ref(),
        &mut warnings,
    );

    // Validate service_tier: format check + per-model capability check (flex/priority).
    let service_tier = provider_opts.service_tier.clone().filter(|tier| {
        match tier.as_str() {
            "auto" | "default" => true,
            "flex" => {
                if caps.supports_flex_processing {
                    true
                } else {
                    warnings.push(Warning::UnsupportedSetting {
                        setting: "serviceTier".to_owned(),
                        details: Some(
                            "flex processing is only available for o3, o4-mini, and gpt-5 models"
                                .to_owned(),
                        ),
                    });
                    false
                }
            }
            "priority" => {
                if caps.supports_priority_processing {
                    true
                } else {
                    warnings.push(Warning::UnsupportedSetting {
                        setting: "serviceTier".to_owned(),
                        details: Some(
                            "priority processing is only available for supported models (gpt-4, gpt-5, gpt-5-mini, o3, o4-mini) and requires Enterprise access. gpt-5-nano is not supported".to_owned(),
                        ),
                    });
                    false
                }
            }
            tier => {
                warnings.push(Warning::UnsupportedSetting {
                    setting: "openai.serviceTier".to_owned(),
                    details: Some(format!(
                        "unknown service_tier '{tier}' — must be one of auto/default/flex/priority"
                    )),
                });
                false
            }
        }
    });

    let text = provider_opts
        .text_verbosity
        .clone()
        .map(|verbosity| TextOptions {
            verbosity: Some(verbosity),
        });

    let mut request = ChatRequest {
        model: model_id.to_owned(),
        messages,
        stream: None,
        stream_options: None,
        max_tokens: options.max_output_tokens,
        max_completion_tokens: provider_opts.max_completion_tokens,
        temperature: options.temperature,
        top_p: options.top_p,
        frequency_penalty: options.frequency_penalty,
        presence_penalty: options.presence_penalty,
        seed: options.seed,
        stop: options.stop_sequences.clone(),
        response_format,
        tools,
        tool_choice,
        reasoning_effort: resolved_reasoning_effort.clone(),
        logprobs: provider_opts.logprobs.as_ref().map(LogprobsOption::enabled),
        top_logprobs: provider_opts.top_logprobs.or_else(|| {
            provider_opts
                .logprobs
                .as_ref()
                .and_then(LogprobsOption::top_logprobs)
        }),
        prediction: provider_opts.prediction.clone(),
        store: provider_opts.store,
        metadata: provider_opts.metadata.clone(),
        service_tier,
        safety_identifier: provider_opts.safety_identifier.clone(),
        prompt_cache_key: provider_opts.prompt_cache_key.clone(),
        parallel_tool_calls: provider_opts.parallel_tool_calls,
        logit_bias: provider_opts.logit_bias.clone(),
        user: provider_opts.user.clone(),
        text,
        prompt_cache_retention: provider_opts.prompt_cache_retention.clone().filter(|v| {
            let ok = matches!(v.as_str(), "in_memory" | "24h");
            if !ok {
                warnings.push(Warning::UnsupportedSetting {
                    setting: "openai.promptCacheRetention".to_owned(),
                    details: Some(format!(
                        "unknown promptCacheRetention '{v}' — must be 'in_memory' or '24h'"
                    )),
                });
            }
            ok
        }),
    };

    if is_reasoning_model {
        apply_reasoning_model_strip(
            &mut request,
            &mut warnings,
            resolved_reasoning_effort.as_deref(),
            caps.supports_non_reasoning_parameters,
        );
    } else if caps.is_search_preview_model {
        apply_search_preview_strip(&mut request, &mut warnings);
    }

    (request, warnings)
}

/// Drop unsupported settings for reasoning models, with a warning per drop.
///
/// Mirrors the `if (isReasoningModel)` block in
/// `openai-chat-language-model.ts`. The order of warnings matches ai-sdk for
/// easier diffing of fixtures.
fn apply_reasoning_model_strip(
    request: &mut ChatRequest,
    warnings: &mut Vec<Warning>,
    resolved_reasoning_effort: Option<&str>,
    supports_non_reasoning_parameters: bool,
) {
    // Reasoning effort `none` on gpt-5.1+ keeps temperature / top_p / logprobs.
    let strip_sampling =
        resolved_reasoning_effort != Some("none") || !supports_non_reasoning_parameters;

    if strip_sampling {
        if request.temperature.is_some() {
            request.temperature = None;
            warnings.push(Warning::UnsupportedSetting {
                setting: "temperature".to_owned(),
                details: Some("temperature is not supported for reasoning models".to_owned()),
            });
        }
        if request.top_p.is_some() {
            request.top_p = None;
            warnings.push(Warning::UnsupportedSetting {
                setting: "topP".to_owned(),
                details: Some("topP is not supported for reasoning models".to_owned()),
            });
        }
        if request.logprobs.is_some() {
            request.logprobs = None;
            warnings.push(Warning::Other {
                message: "logprobs is not supported for reasoning models".to_owned(),
            });
        }
    }

    if request.frequency_penalty.is_some() {
        request.frequency_penalty = None;
        warnings.push(Warning::UnsupportedSetting {
            setting: "frequencyPenalty".to_owned(),
            details: Some("frequencyPenalty is not supported for reasoning models".to_owned()),
        });
    }
    if request.presence_penalty.is_some() {
        request.presence_penalty = None;
        warnings.push(Warning::UnsupportedSetting {
            setting: "presencePenalty".to_owned(),
            details: Some("presencePenalty is not supported for reasoning models".to_owned()),
        });
    }
    if request.top_logprobs.is_some() {
        request.top_logprobs = None;
        warnings.push(Warning::Other {
            message: "topLogprobs is not supported for reasoning models".to_owned(),
        });
    }
    if request.logit_bias.is_some() {
        request.logit_bias = None;
        warnings.push(Warning::Other {
            message: "logit_bias is not supported for reasoning models".to_owned(),
        });
    }

    // Reasoning models use `max_completion_tokens` instead of `max_tokens`.
    if let Some(max) = request.max_tokens.take() {
        request.max_completion_tokens.get_or_insert(max);
    }
}

/// Drop `temperature` for search-preview models.
fn apply_search_preview_strip(request: &mut ChatRequest, warnings: &mut Vec<Warning>) {
    if request.temperature.is_some() {
        request.temperature = None;
        warnings.push(Warning::UnsupportedSetting {
            setting: "temperature".to_owned(),
            details: Some(
                "temperature is not supported for the search preview models and has been removed."
                    .to_owned(),
            ),
        });
    }
}

/// Translate [`ReasoningEffort`] to its on-wire string, returning `None`
/// for [`ReasoningEffort::ProviderDefault`] (let `OpenAI` pick).
fn reasoning_effort_wire_value(effort: ReasoningEffort) -> Option<String> {
    match effort {
        ReasoningEffort::ProviderDefault => None,
        ReasoningEffort::None => Some("none".to_owned()),
        ReasoningEffort::Minimal => Some("minimal".to_owned()),
        ReasoningEffort::Low => Some("low".to_owned()),
        ReasoningEffort::Medium => Some("medium".to_owned()),
        ReasoningEffort::High => Some("high".to_owned()),
        ReasoningEffort::Xhigh => Some("xhigh".to_owned()),
    }
}

/// Where to send system messages — controlled by reasoning-model detection
/// or the `systemMessageMode` provider option.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SystemRole {
    System,
    Developer,
    /// Drop system messages entirely (matches ai-sdk's `'remove'` mode).
    Remove,
}

fn convert_response_format(fmt: &ResponseFormat, strict: bool) -> WireResponseFormat {
    match fmt {
        ResponseFormat::Text => WireResponseFormat::JsonObject, // unreachable in practice; only set when caller asks for JSON
        ResponseFormat::Json {
            schema,
            name,
            description,
        } => match schema {
            Some(schema) => WireResponseFormat::JsonSchema {
                json_schema: WireJsonSchema {
                    name: name.clone().unwrap_or_else(|| "response".to_owned()),
                    description: description.clone(),
                    schema: schema.clone().into(),
                    strict,
                },
            },
            None => WireResponseFormat::JsonObject,
        },
    }
}

fn convert_tools(
    tools: Option<&[Tool]>,
    choice: Option<&ToolChoice>,
    warnings: &mut Vec<Warning>,
) -> (Option<Vec<WireTool>>, Option<WireToolChoice>) {
    let Some(tools) = tools else {
        return (None, None);
    };
    let tools = tools.iter().filter_map(|t| match t {
        Tool::Function(f) => Some(WireTool::Function {
            kind: WireFunctionKind::Function,
            function: WireFunctionDef {
                name: f.name.clone(),
                description: f.description.clone(),
                parameters: f.input_schema.clone().into(),
                strict: f.strict,
            },
        }),
        Tool::Provider(p) => match p.id.as_str() {
            "openai.web_search_preview" => Some(WireTool::Provider {
                kind: "web_search_preview".to_owned(),
                args: p.args.clone().unwrap_or_default(),
            }),
            other => {
                warnings.push(Warning::UnsupportedTool {
                    tool: p.name.clone(),
                    details: Some(format!(
                        "provider-defined tool '{other}' requires the OpenAI Responses API endpoint (not yet supported by llmsdk-openai)"
                    )),
                });
                None
            }
        },
    });
    let tools: Vec<_> = tools.collect();
    if tools.is_empty() {
        return (None, None);
    }

    let tool_choice = choice.map(|c| match c {
        ToolChoice::Auto => WireToolChoice::Simple(WireToolChoiceSimple::Auto),
        ToolChoice::None => WireToolChoice::Simple(WireToolChoiceSimple::None),
        ToolChoice::Required => WireToolChoice::Simple(WireToolChoiceSimple::Required),
        ToolChoice::Tool { tool_name } => WireToolChoice::Tool {
            kind: WireToolCallKind::Function,
            function: WireToolChoiceFunction {
                name: tool_name.clone(),
            },
        },
    });

    (Some(tools), tool_choice)
}
