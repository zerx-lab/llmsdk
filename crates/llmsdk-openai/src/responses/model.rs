//! [`LanguageModel`] implementation for the OpenAI Responses API.
//!
//! Mirrors `@ai-sdk/openai/src/responses/openai-responses-language-model.ts`.
// Rust guideline compliant 2026-02-21

use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    CallOptions, GenerateResult, LanguageModel, ResponseFormat, StreamPart, StreamResponse,
    StreamResult, SupportedUrls, UrlPattern,
};
use llmsdk_provider::shared::{RequestInfo, Warning};
use llmsdk_provider_utils::http::{JsonRequest, post_for_stream, post_json, response_byte_stream};
use llmsdk_provider_utils::sse::{SseEvent, sse_json_stream};
use serde_json::{Value as JsonValue, json};

use super::convert_prompt::{ConvertCtx, convert_prompt};
use super::options::{
    LogprobsOption, ResponsesProviderOptions, SystemMessageMode, TOP_LOGPROBS_MAX, parse, validate,
};
use super::parse_response::parse_response as parse_responses_response;
use super::prepare_tools::{PreparedTools, prepare as prepare_tools};
use super::stream::{StreamSetup, StreamState};
use super::wire::chunk::ResponsesChunk;
use super::wire::request::ResponsesRequest;
use super::wire::response::ResponsesResponse;

use crate::chat::capabilities::Capabilities;
use crate::config::Inner;
use crate::error::rewrite_openai_error;

/// OpenAI Responses API model handle.
///
/// Cheap to clone. Multiple clones share the underlying HTTP client and
/// authentication state via [`crate::OpenAi`]'s `Arc`.
#[derive(Debug, Clone)]
pub struct OpenAiResponsesLanguageModel {
    inner: Arc<Inner>,
    model_id: String,
}

impl OpenAiResponsesLanguageModel {
    /// Construct from a fully assembled [`Inner`].
    ///
    /// Public for cross-crate composition (Azure `OpenAI`). End-users should
    /// prefer [`crate::OpenAi::responses`].
    #[must_use]
    pub fn new(inner: Arc<Inner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    fn endpoint(&self) -> String {
        self.inner.endpoint("/responses", &self.model_id)
    }
}

#[async_trait]
impl LanguageModel for OpenAiResponsesLanguageModel {
    fn provider(&self) -> &str {
        self.inner.provider_id()
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn supported_urls(&self) -> SupportedUrls {
        let mut map = SupportedUrls::new();
        map.insert("image/*".into(), vec![UrlPattern::new(r"^https?://.*$")]);
        map.insert(
            "application/pdf".into(),
            vec![UrlPattern::new(r"^https?://.*$")],
        );
        map
    }

    async fn do_generate(&self, options: CallOptions) -> Result<GenerateResult, ProviderError> {
        let provider_options_name = self.inner.provider_options_name();
        let Built {
            body,
            warnings,
            web_search_tool_name,
            is_shell_provider_executed,
            ..
        } = build_request(&self.model_id, &options, false, provider_options_name);
        let request_body_value = serde_json::to_value(&body).ok();
        let body_bytes = serde_json::to_vec(&body).unwrap_or_default();
        let mut headers = self.inner.headers.clone();
        if let Some(h) = &options.headers {
            for (k, v) in h {
                headers.insert(k.clone(), v.clone());
            }
        }
        let url = self.endpoint();
        self.inner
            .sign_if_needed(&mut headers, "POST", &url, &body_bytes)
            .await?;
        let mut http_request = JsonRequest::new(url, body);
        http_request.headers = headers;

        let response = match post_json::<_, ResponsesResponse>(&self.inner.http, http_request).await
        {
            Ok(r) => r,
            Err(e) => return Err(rewrite_openai_error(e)),
        };

        parse_responses_response(
            response.value,
            response.headers,
            request_body_value,
            warnings,
            provider_options_name,
            web_search_tool_name.as_deref(),
            is_shell_provider_executed,
        )
    }

    async fn do_stream(&self, options: CallOptions) -> Result<StreamResult, ProviderError> {
        let provider_options_name = self.inner.provider_options_name();
        let Built {
            mut body,
            warnings,
            web_search_tool_name,
            is_shell_provider_executed,
            store,
            include_raw_chunks,
            ..
        } = build_request(&self.model_id, &options, true, provider_options_name);
        body.stream = Some(true);
        let request_body_value = serde_json::to_value(&body).ok();
        let body_bytes = serde_json::to_vec(&body).unwrap_or_default();
        let mut headers = self.inner.headers.clone();
        if let Some(h) = &options.headers {
            for (k, v) in h {
                headers.insert(k.clone(), v.clone());
            }
        }
        let url = self.endpoint();
        self.inner
            .sign_if_needed(&mut headers, "POST", &url, &body_bytes)
            .await?;
        let mut http_request = JsonRequest::new(url, body);
        http_request.headers = headers;

        let stream_response = match post_for_stream(&self.inner.http, http_request).await {
            Ok(r) => r,
            Err(e) => return Err(rewrite_openai_error(e)),
        };
        let response_headers = stream_response.headers.clone();
        let byte_stream = response_byte_stream(stream_response.response);
        let event_stream = sse_json_stream::<ResponsesChunk>(byte_stream);

        let state = StreamState::new(StreamSetup {
            warnings,
            provider_options_name,
            store,
            include_raw_chunks,
            web_search_tool_name,
            is_shell_provider_executed,
        });
        let parts = build_part_stream(state, event_stream);

        Ok(StreamResult {
            stream: Box::pin(parts),
            request: Some(RequestInfo {
                body: request_body_value,
            }),
            response: Some(StreamResponse {
                headers: Some(
                    response_headers
                        .into_iter()
                        .map(|(k, v)| (k, Some(v)))
                        .collect(),
                ),
            }),
        })
    }
}

fn build_part_stream<S>(
    mut state: StreamState,
    events: S,
) -> impl futures::Stream<Item = Result<StreamPart, ProviderError>> + Send
where
    S: futures::Stream<Item = Result<SseEvent<ResponsesChunk>, ProviderError>> + Send + 'static,
{
    async_stream::stream! {
        for part in state.start_frames() {
            yield Ok(part);
        }
        let mut events = Box::pin(events);
        while let Some(event) = futures::StreamExt::next(&mut events).await {
            match event {
                Ok(SseEvent::Data(chunk)) => {
                    for part in state.on_chunk(chunk, None) {
                        yield Ok(part);
                    }
                }
                Ok(SseEvent::ParseError { raw, message }) => {
                    yield Ok(StreamPart::Error {
                        error: json!({"message": message, "raw": raw}),
                    });
                }
                Err(e) => {
                    yield Err(e);
                    return;
                }
            }
        }
        yield Ok(state.finish_frame());
    }
}

struct Built {
    body: ResponsesRequest,
    warnings: Vec<Warning>,
    web_search_tool_name: Option<String>,
    is_shell_provider_executed: bool,
    store: bool,
    #[allow(dead_code, reason = "used only by stream path")]
    include_raw_chunks: bool,
}

fn build_request(
    model_id: &str,
    options: &CallOptions,
    _streaming: bool,
    provider_options_name: &'static str,
) -> Built {
    let caps = Capabilities::detect(model_id);
    let mut provider_opts = parse(options.provider_options.as_ref(), provider_options_name);
    let mut warnings = validate(&mut provider_opts, &caps);

    let is_reasoning = provider_opts
        .force_reasoning
        .unwrap_or(caps.is_reasoning_model);

    // System message mode resolution.
    let system_message_mode = provider_opts
        .system_message_mode
        .unwrap_or(if is_reasoning {
            SystemMessageMode::Developer
        } else {
            SystemMessageMode::System
        });

    if options.top_k.is_some() {
        warnings.push(Warning::UnsupportedSetting {
            setting: "topK".into(),
            details: Some("Responses API does not accept topK".into()),
        });
    }
    if options.seed.is_some() {
        warnings.push(Warning::UnsupportedSetting {
            setting: "seed".into(),
            details: Some("Responses API does not accept seed".into()),
        });
    }
    if options.presence_penalty.is_some() {
        warnings.push(Warning::UnsupportedSetting {
            setting: "presencePenalty".into(),
            details: Some("Responses API does not accept presencePenalty".into()),
        });
    }
    if options.frequency_penalty.is_some() {
        warnings.push(Warning::UnsupportedSetting {
            setting: "frequencyPenalty".into(),
            details: Some("Responses API does not accept frequencyPenalty".into()),
        });
    }
    if options.stop_sequences.is_some() {
        warnings.push(Warning::UnsupportedSetting {
            setting: "stopSequences".into(),
            details: Some("Responses API does not accept stopSequences".into()),
        });
    }

    let PreparedTools {
        tools,
        tool_choice,
        warnings: tool_warnings,
        web_search_tool_name,
        is_shell_provider_executed,
        ..
    } = prepare_tools(
        options.tools.as_deref(),
        options.tool_choice.as_ref(),
        provider_opts.allowed_tools.as_ref(),
    );
    warnings.extend(tool_warnings);

    let has_local_shell_tool = tools
        .as_ref()
        .map(|ts| {
            ts.iter()
                .any(|t| t.get("type").and_then(|v| v.as_str()) == Some("local_shell"))
        })
        .unwrap_or(false);
    let has_shell_tool = tools
        .as_ref()
        .map(|ts| {
            ts.iter()
                .any(|t| t.get("type").and_then(|v| v.as_str()) == Some("shell"))
        })
        .unwrap_or(false);
    let has_apply_patch_tool = tools
        .as_ref()
        .map(|ts| {
            ts.iter()
                .any(|t| t.get("type").and_then(|v| v.as_str()) == Some("apply_patch"))
        })
        .unwrap_or(false);

    let store = provider_opts.store.unwrap_or(true);

    let ctx = ConvertCtx {
        system_message_mode,
        provider_options_name,
        pass_through_unsupported_files: provider_opts
            .pass_through_unsupported_files
            .unwrap_or(false),
        store,
        has_conversation: provider_opts.conversation.is_some(),
        has_previous_response_id: provider_opts.previous_response_id.is_some(),
        has_local_shell_tool,
        has_shell_tool,
        has_apply_patch_tool,
    };
    let (input, prompt_warnings) = convert_prompt(&options.prompt, &ctx);
    warnings.extend(prompt_warnings);

    let strict_json_schema = provider_opts.strict_json_schema.unwrap_or(true);

    let mut text_block = serde_json::Map::new();
    if let Some(ResponseFormat::Json {
        schema,
        name,
        description,
    }) = &options.response_format
    {
        let mut fmt = serde_json::Map::new();
        if let Some(s) = schema {
            fmt.insert("type".into(), json!("json_schema"));
            fmt.insert("strict".into(), json!(strict_json_schema));
            fmt.insert(
                "name".into(),
                json!(name.clone().unwrap_or_else(|| "response".into())),
            );
            if let Some(d) = description {
                fmt.insert("description".into(), json!(d));
            }
            fmt.insert(
                "schema".into(),
                serde_json::to_value(s).unwrap_or(JsonValue::Null),
            );
        } else {
            fmt.insert("type".into(), json!("json_object"));
        }
        text_block.insert("format".into(), JsonValue::Object(fmt));
    }
    if let Some(v) = provider_opts.text_verbosity {
        text_block.insert(
            "verbosity".into(),
            serde_json::to_value(v).unwrap_or(JsonValue::Null),
        );
    }
    let text = (!text_block.is_empty()).then_some(JsonValue::Object(text_block));

    // logprobs auto-include + top_logprobs derivation
    let mut include = provider_opts.include.clone();
    let top_logprobs = match provider_opts.logprobs {
        Some(LogprobsOption::Count(n)) => Some(n),
        Some(LogprobsOption::Bool(true)) => Some(TOP_LOGPROBS_MAX),
        _ => None,
    };
    if top_logprobs.is_some() {
        add_include(&mut include, "message.output_text.logprobs");
    }
    if web_search_tool_name.is_some() {
        add_include(&mut include, "web_search_call.action.sources");
    }
    if tools
        .as_ref()
        .map(|ts| {
            ts.iter()
                .any(|t| t.get("type").and_then(|v| v.as_str()) == Some("code_interpreter"))
        })
        .unwrap_or(false)
    {
        add_include(&mut include, "code_interpreter_call.outputs");
    }
    if !store && is_reasoning {
        add_include(&mut include, "reasoning.encrypted_content");
    }

    // Reasoning block
    let resolved_reasoning_effort = provider_opts.reasoning_effort.clone();
    let reasoning_value = if is_reasoning
        && (resolved_reasoning_effort.is_some() || provider_opts.reasoning_summary.is_some())
    {
        let mut obj = serde_json::Map::new();
        if let Some(e) = &resolved_reasoning_effort {
            obj.insert("effort".into(), json!(e));
        }
        if let Some(s) = &provider_opts.reasoning_summary {
            obj.insert("summary".into(), json!(s));
        }
        Some(JsonValue::Object(obj))
    } else {
        None
    };

    // Reasoning models strip temperature / top_p unless effort=none on
    // gpt-5.1+ allowed-parameters family.
    let mut temperature = options.temperature.map(f64::from);
    let mut top_p = options.top_p.map(f64::from);
    if is_reasoning {
        let effort_allows_chat_params = resolved_reasoning_effort.as_deref() == Some("none")
            && caps.supports_non_reasoning_parameters;
        if !effort_allows_chat_params {
            if temperature.is_some() {
                temperature = None;
                warnings.push(Warning::UnsupportedSetting {
                    setting: "temperature".into(),
                    details: Some("temperature is not supported for reasoning models".into()),
                });
            }
            if top_p.is_some() {
                top_p = None;
                warnings.push(Warning::UnsupportedSetting {
                    setting: "topP".into(),
                    details: Some("topP is not supported for reasoning models".into()),
                });
            }
        }
    }

    let body = ResponsesRequest {
        model: model_id.to_owned(),
        input,
        temperature,
        top_p,
        max_output_tokens: options.max_output_tokens,
        text,
        tools,
        tool_choice,
        conversation: provider_opts.conversation,
        max_tool_calls: provider_opts.max_tool_calls,
        metadata: provider_opts.metadata,
        parallel_tool_calls: provider_opts.parallel_tool_calls,
        previous_response_id: provider_opts.previous_response_id,
        store: provider_opts.store,
        user: provider_opts.user,
        instructions: provider_opts.instructions,
        service_tier: provider_opts.service_tier.map(|v| {
            serde_json::to_value(v)
                .ok()
                .and_then(|j| j.as_str().map(str::to_owned))
                .unwrap_or_default()
        }),
        include,
        prompt_cache_key: provider_opts.prompt_cache_key,
        prompt_cache_retention: provider_opts.prompt_cache_retention.map(|v| match v {
            super::options::PromptCacheRetention::InMemory => "in_memory".into(),
            super::options::PromptCacheRetention::H24 => "24h".into(),
        }),
        safety_identifier: provider_opts.safety_identifier,
        top_logprobs,
        truncation: provider_opts.truncation.map(|v| match v {
            super::options::Truncation::Auto => "auto".into(),
            super::options::Truncation::Disabled => "disabled".into(),
        }),
        context_management: provider_opts.context_management.map(|v| {
            v.iter()
                .map(|cm| json!({"type": cm.kind, "compact_threshold": cm.compact_threshold}))
                .collect()
        }),
        reasoning: reasoning_value,
        stream: None,
        extra: Default::default(),
    };

    Built {
        body,
        warnings,
        web_search_tool_name,
        is_shell_provider_executed,
        store,
        include_raw_chunks: options.include_raw_chunks.unwrap_or(false),
    }
}

fn add_include(include: &mut Option<Vec<String>>, key: &str) {
    match include {
        Some(v) => {
            if !v.iter().any(|k| k == key) {
                v.push(key.to_owned());
            }
        }
        None => *include = Some(vec![key.to_owned()]),
    }
}

#[allow(
    dead_code,
    reason = "kept for completeness; unused once stream path lands"
)]
fn _ensure_options_type_imported(_: ResponsesProviderOptions) {}
