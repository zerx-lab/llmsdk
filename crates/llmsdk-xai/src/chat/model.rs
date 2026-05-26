//! [`LanguageModel`] implementation for xAI Chat Completions.
//!
//! Mirrors `xai-chat-language-model.ts`. Entry: [`XaiChatModel::new`] via
//! [`crate::Xai::chat`].
// Rust guideline compliant 2026-05-25

use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    CallOptions, GenerateResult, LanguageModel, ReasoningEffort, ResponseFormat, StreamResult,
    SupportedUrls, UrlPattern,
};
use llmsdk_provider::shared::Warning;
use llmsdk_provider_utils::http::{JsonRequest, post_for_stream, post_json, response_byte_stream};
use llmsdk_provider_utils::sse::{SseEvent, sse_json_stream};

use crate::PROVIDER_ID;
use crate::config::Inner;

use super::convert_prompt::convert_prompt;
use super::options::{XaiChatOptions, parse as parse_xai_options};
use super::parse_response::parse_response;
use super::prepare_tools::prepare as prepare_tools;
use super::stream::StreamState;
use super::wire::{
    ChatChunk, ChatRequest, ChatResponse, ResponseFormat as WireResponseFormat, StreamErrorBody,
    StreamOptions, WireJsonSchema, WireMessage,
};

/// xAI Chat Completions model handle.
///
/// Cheap to clone. Multiple clones share the underlying HTTP client and
/// authentication state via [`Xai`](crate::Xai)'s `Arc`.
#[derive(Debug, Clone)]
pub struct XaiChatModel {
    pub(crate) inner: Arc<Inner>,
    pub(crate) model_id: String,
}

impl XaiChatModel {
    /// Construct from shared provider state and a model id.
    pub(crate) fn new(inner: Arc<Inner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    fn endpoint(&self) -> String {
        format!("{}/chat/completions", self.inner.base_url)
    }
}

#[async_trait]
impl LanguageModel for XaiChatModel {
    fn provider(&self) -> &str {
        PROVIDER_ID
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn supported_urls(&self) -> SupportedUrls {
        // Mirrors `xai-chat-language-model.ts:79-81`: only `image/*` URLs are
        // forwarded natively; PDF / text content must be inlined as data parts.
        let mut map = SupportedUrls::new();
        map.insert("image/*".into(), vec![UrlPattern::new(r"^https?://.*$")]);
        map
    }

    async fn do_generate(&self, options: CallOptions) -> Result<GenerateResult, ProviderError> {
        let last_assistant_content = last_assistant_text(&options);
        let (request, warnings) = build_request(&self.model_id, &options)?;
        let request_body_value = serde_json::to_value(&request).ok();
        let endpoint = self.endpoint();

        let mut request_headers = self.inner.headers.clone();
        if let Some(headers) = &options.headers {
            for (name, value) in headers {
                request_headers.insert(name.clone(), value.clone());
            }
        }

        let mut http_request = JsonRequest::new(endpoint.clone(), request);
        http_request.headers = request_headers;

        let response = post_json::<_, ChatResponse>(&self.inner.http, http_request).await?;

        let mut citation_seed: u64 = 0;
        parse_response(
            response.value,
            response.headers,
            request_body_value,
            warnings,
            &endpoint,
            last_assistant_content.as_deref(),
            &mut citation_seed,
        )
    }

    async fn do_stream(&self, options: CallOptions) -> Result<StreamResult, ProviderError> {
        let last_assistant_content = last_assistant_text(&options);
        let (mut request, warnings) = build_request(&self.model_id, &options)?;
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

        let stream_response = post_for_stream(&self.inner.http, http_request).await?;
        let stream_headers = stream_response.headers.clone();
        let endpoint = self.endpoint();

        // If xAI returns `application/json` for an in-band error instead of
        // text/event-stream, surface it as an in-stream error (preserves the
        // outer stream so the caller still sees `StreamPart::Error +
        // StreamPart::Finish`). Match case-insensitively because header names
        // are case-insensitive per RFC 7230.
        let content_type = stream_headers
            .iter()
            .find_map(|(k, v)| {
                if k.eq_ignore_ascii_case("content-type") {
                    Some(v.clone())
                } else {
                    None
                }
            })
            .unwrap_or_default();
        if content_type.contains("application/json") {
            return drain_json_error_as_stream(
                stream_response,
                warnings,
                request_body_value,
                stream_headers,
                last_assistant_content,
                endpoint,
            )
            .await;
        }

        let byte_stream = response_byte_stream(stream_response.response);
        let event_stream = sse_json_stream::<ChatChunk>(byte_stream);
        let state = StreamState::new(warnings, last_assistant_content);
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

async fn drain_json_error_as_stream(
    stream_response: llmsdk_provider_utils::http::StreamResponse,
    warnings: Vec<Warning>,
    request_body_value: Option<serde_json::Value>,
    stream_headers: std::collections::HashMap<String, String>,
    last_assistant_content: Option<String>,
    _endpoint: String,
) -> Result<StreamResult, ProviderError> {
    let body = stream_response.response.text().await.unwrap_or_default();
    let parsed = serde_json::from_str::<StreamErrorBody>(&body).ok();
    let mut state = StreamState::new(warnings, last_assistant_content);
    let mut parts: Vec<llmsdk_provider::language_model::StreamPart> = state.start_frames();
    if let Some(err) = parsed {
        parts.extend(state.on_error(&err.error, Some(&err.code)));
    } else {
        parts.extend(state.on_parse_error(&body, "Invalid JSON response"));
    }
    parts.extend(state.flush());
    let stream = futures::stream::iter(parts.into_iter().map(Ok));
    Ok(StreamResult {
        stream: Box::pin(stream),
        request: Some(llmsdk_provider::shared::RequestInfo {
            body: request_body_value,
        }),
        response: Some(llmsdk_provider::language_model::StreamResponse {
            headers: Some(headers_to_provider(stream_headers)),
        }),
    })
}

fn last_assistant_text(options: &CallOptions) -> Option<String> {
    let last = options.prompt.last()?;
    if let llmsdk_provider::language_model::Message::Assistant { content, .. } = last {
        use llmsdk_provider::language_model::AssistantPart;
        let mut buf = String::new();
        for part in content {
            if let AssistantPart::Text(t) = part {
                buf.push_str(&t.text);
            }
        }
        if buf.is_empty() { None } else { Some(buf) }
    } else {
        None
    }
}

fn headers_to_provider(
    raw: std::collections::HashMap<String, String>,
) -> llmsdk_provider::shared::Headers {
    raw.into_iter().map(|(k, v)| (k, Some(v))).collect()
}

fn build_part_stream<S>(
    mut state: StreamState,
    events: S,
) -> impl futures::Stream<Item = Result<llmsdk_provider::language_model::StreamPart, ProviderError>> + Send
where
    S: futures::Stream<Item = Result<SseEvent<ChatChunk>, ProviderError>> + Send + 'static,
{
    async_stream::stream! {
        for part in state.start_frames() {
            yield Ok(part);
        }

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

        for part in state.flush() {
            yield Ok(part);
        }
    }
}

/// Build the wire request and collect warnings about dropped settings.
fn build_request(
    model_id: &str,
    options: &CallOptions,
) -> Result<(ChatRequest, Vec<Warning>), ProviderError> {
    let xai_opts = parse_xai_options(options.provider_options.as_ref());
    let mut warnings: Vec<Warning> = Vec::new();

    // xAI does not accept these four sampling parameters.
    for (val, name) in [
        (options.top_k.is_some(), "topK"),
        (options.frequency_penalty.is_some(), "frequencyPenalty"),
        (options.presence_penalty.is_some(), "presencePenalty"),
        (options.stop_sequences.is_some(), "stopSequences"),
    ] {
        if val {
            warnings.push(Warning::UnsupportedSetting {
                setting: name.to_owned(),
                details: Some(format!("xAI chat completions does not accept {name}")),
            });
        }
    }

    let (messages, msg_warnings) = convert_prompt(&options.prompt)?;
    warnings.extend(msg_warnings);

    let prepared = prepare_tools(
        options.tools.as_deref().unwrap_or(&[]),
        options.tool_choice.as_ref(),
    );
    warnings.extend(prepared.warnings);

    let response_format = options
        .response_format
        .as_ref()
        .and_then(convert_response_format);

    let reasoning_effort = resolve_reasoning_effort(&xai_opts, options.reasoning, &mut warnings);

    let logprobs_flag = match (xai_opts.logprobs, xai_opts.top_logprobs) {
        (Some(true), _) | (_, Some(_)) => Some(true),
        _ => xai_opts.logprobs,
    };

    let request = ChatRequest {
        model: model_id.to_owned(),
        messages,
        stream: None,
        stream_options: None,
        max_completion_tokens: options.max_output_tokens,
        temperature: options.temperature,
        top_p: options.top_p,
        seed: options.seed,
        reasoning_effort,
        logprobs: logprobs_flag,
        top_logprobs: xai_opts.top_logprobs,
        parallel_function_calling: xai_opts.parallel_function_calling,
        response_format,
        search_parameters: xai_opts
            .search_parameters
            .as_ref()
            .map(super::options::SearchParameters::to_wire),
        tools: prepared.tools,
        tool_choice: prepared.tool_choice,
    };

    // xAI assistant message: ensure trailing empty content does not lose the
    // role context. Already handled by serde — no-op here.
    let _ = WireMessage::Assistant {
        content: String::new(),
        tool_calls: None,
    };

    Ok((request, warnings))
}

fn convert_response_format(fmt: &ResponseFormat) -> Option<WireResponseFormat> {
    match fmt {
        ResponseFormat::Text => None,
        ResponseFormat::Json { schema, name, .. } => Some(match schema {
            Some(schema) => WireResponseFormat::JsonSchema {
                json_schema: WireJsonSchema {
                    name: name.clone().unwrap_or_else(|| "response".to_owned()),
                    schema: serde_json::to_value(schema).unwrap_or(serde_json::Value::Null),
                    strict: true,
                },
            },
            None => WireResponseFormat::JsonObject,
        }),
    }
}

fn resolve_reasoning_effort(
    xai: &XaiChatOptions,
    top_level: Option<ReasoningEffort>,
    warnings: &mut Vec<Warning>,
) -> Option<String> {
    if let Some(effort) = &xai.reasoning_effort {
        return Some(effort.clone());
    }
    match top_level? {
        ReasoningEffort::ProviderDefault | ReasoningEffort::None => None,
        ReasoningEffort::Minimal | ReasoningEffort::Low => Some("low".to_owned()),
        ReasoningEffort::Medium => Some("medium".to_owned()),
        ReasoningEffort::High | ReasoningEffort::Xhigh => {
            if matches!(top_level, Some(ReasoningEffort::Xhigh)) {
                warnings.push(Warning::Other {
                    message: "xAI does not support 'xhigh' reasoning; coerced to 'high'".to_owned(),
                });
            }
            Some("high".to_owned())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmsdk_provider::language_model::TextPart;
    use llmsdk_provider::language_model::{FunctionTool, Message, Tool, ToolChoice, UserPart};
    use serde_json::json;

    fn opts() -> CallOptions {
        CallOptions {
            prompt: vec![Message::User {
                content: vec![UserPart::Text(TextPart {
                    text: "hi".into(),
                    provider_options: None,
                })],
                provider_options: None,
            }],
            ..Default::default()
        }
    }

    #[test]
    fn warns_on_topk_frequency_presence_stop() {
        let mut o = opts();
        o.top_k = Some(5);
        o.frequency_penalty = Some(0.1);
        o.presence_penalty = Some(0.1);
        o.stop_sequences = Some(vec!["END".into()]);
        let (_, warnings) = build_request("grok-4.3", &o).unwrap();
        assert_eq!(warnings.len(), 4);
    }

    #[test]
    fn maps_reasoning_effort_xhigh_to_high_with_warning() {
        let mut o = opts();
        o.reasoning = Some(ReasoningEffort::Xhigh);
        let (req, warnings) = build_request("grok-4.3", &o).unwrap();
        assert_eq!(req.reasoning_effort.as_deref(), Some("high"));
        assert!(warnings.iter().any(|w| matches!(
            w,
            Warning::Other { message } if message.contains("xhigh")
        )));
    }

    #[test]
    fn provider_options_reasoning_effort_wins() {
        let mut o = opts();
        o.reasoning = Some(ReasoningEffort::Low);
        let mut po = llmsdk_provider::shared::ProviderOptions::new();
        po.insert(
            "xai".into(),
            json!({"reasoningEffort": "high"})
                .as_object()
                .cloned()
                .unwrap(),
        );
        o.provider_options = Some(po);
        let (req, _) = build_request("grok-4.3", &o).unwrap();
        assert_eq!(req.reasoning_effort.as_deref(), Some("high"));
    }

    #[test]
    fn top_logprobs_forces_logprobs_true() {
        let mut o = opts();
        let mut po = llmsdk_provider::shared::ProviderOptions::new();
        po.insert(
            "xai".into(),
            json!({"topLogprobs": 5}).as_object().cloned().unwrap(),
        );
        o.provider_options = Some(po);
        let (req, _) = build_request("grok-4.3", &o).unwrap();
        assert_eq!(req.logprobs, Some(true));
        assert_eq!(req.top_logprobs, Some(5));
    }

    #[test]
    fn search_parameters_serialized_with_snake_case() {
        let mut o = opts();
        let mut po = llmsdk_provider::shared::ProviderOptions::new();
        po.insert(
            "xai".into(),
            json!({
                "searchParameters": {
                    "mode": "auto",
                    "maxSearchResults": 5,
                    "sources": [
                        { "type": "web", "country": "US" }
                    ]
                }
            })
            .as_object()
            .cloned()
            .unwrap(),
        );
        o.provider_options = Some(po);
        let (req, _) = build_request("grok-4.3", &o).unwrap();
        let sp = req.search_parameters.unwrap();
        assert_eq!(sp.mode, "auto");
        assert_eq!(sp.max_search_results, Some(5));
        let wire = serde_json::to_value(&sp).unwrap();
        assert_eq!(wire["max_search_results"], 5);
        assert_eq!(wire["sources"][0]["country"], "US");
    }

    #[test]
    fn function_tool_passthrough_with_tool_choice_required() {
        let mut o = opts();
        o.tools = Some(vec![Tool::Function(FunctionTool {
            name: "weather".into(),
            description: Some("get weather".into()),
            input_schema: serde_json::from_value(
                json!({"type":"object","properties":{"c":{"type":"string"}}}),
            )
            .unwrap(),
            input_examples: None,
            strict: None,
            provider_options: None,
        })]);
        o.tool_choice = Some(ToolChoice::Required);
        let (req, _) = build_request("grok-4.3", &o).unwrap();
        assert!(req.tools.is_some());
        let choice = serde_json::to_value(req.tool_choice.unwrap()).unwrap();
        assert_eq!(choice, json!("required"));
    }

    #[tokio::test]
    async fn supported_urls_advertises_https_image_only() {
        let p = crate::Xai::builder().api_key("k").build().expect("ok");
        let m = p.chat("grok-4.3");
        let urls = m.supported_urls().await;
        let patterns = urls.get("image/*").expect("image/* key");
        assert!(patterns.iter().any(|p| p.0.contains("https?")));
        // Mirrors xai-chat-language-model.ts: PDF / text are inlined, not URL-fetched.
        assert!(!urls.contains_key("application/pdf"));
        assert!(!urls.contains_key("text/*"));
    }

    #[test]
    fn last_assistant_text_extracts_concatenated_text() {
        let mut o = opts();
        o.prompt.push(Message::Assistant {
            content: vec![
                llmsdk_provider::language_model::AssistantPart::Text(TextPart {
                    text: "hello ".into(),
                    provider_options: None,
                }),
                llmsdk_provider::language_model::AssistantPart::Text(TextPart {
                    text: "world".into(),
                    provider_options: None,
                }),
            ],
            provider_options: None,
        });
        let text = last_assistant_text(&o);
        assert_eq!(text.as_deref(), Some("hello world"));
    }
}
