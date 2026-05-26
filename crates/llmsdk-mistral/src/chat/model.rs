//! [`LanguageModel`] implementation for Mistral Chat Completions.
//!
//! Mirrors `mistral-chat-language-model.ts`. Entry: [`MistralChatModel::new`]
//! via [`crate::Mistral::chat`].
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
use super::options::{MistralChatOptions, parse as parse_mistral_options};
use super::parse_response::parse_response;
use super::prepare_tools::prepare as prepare_tools;
use super::stream::StreamState;
use super::wire::{
    ChatChunk, ChatRequest, ChatResponse, ResponseFormat as WireResponseFormat, WireJsonSchema,
};

/// Mistral Chat Completions model handle.
///
/// Cheap to clone. Multiple clones share the underlying HTTP client and
/// authentication state via [`Mistral`](crate::Mistral)'s `Arc`.
#[derive(Debug, Clone)]
pub struct MistralChatModel {
    pub(crate) inner: Arc<Inner>,
    pub(crate) model_id: String,
}

impl MistralChatModel {
    /// Construct from shared provider state and a model id.
    pub(crate) fn new(inner: Arc<Inner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    fn endpoint(&self) -> String {
        format!("{}/chat/completions", self.inner.base_url)
    }
}

#[async_trait]
impl LanguageModel for MistralChatModel {
    fn provider(&self) -> &str {
        PROVIDER_ID
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn supported_urls(&self) -> SupportedUrls {
        // Mirrors upstream's `supportedUrls = { 'application/pdf': [/^https:\/\/.*$/] }`.
        let mut map = SupportedUrls::default();
        map.insert(
            "application/pdf".to_owned(),
            vec![UrlPattern::new("^https://.*$")],
        );
        map
    }

    async fn do_generate(&self, options: CallOptions) -> Result<GenerateResult, ProviderError> {
        let (request, warnings) = build_request(&self.model_id, &options)?;
        let request_body_value = serde_json::to_value(&request).ok();
        let endpoint = self.endpoint();

        let mut request_headers = self.inner.headers.clone();
        if let Some(headers) = &options.headers {
            for (name, value) in headers {
                request_headers.insert(name.clone(), value.clone());
            }
        }

        let mut http_request = JsonRequest::new(endpoint, request);
        http_request.headers = request_headers;

        let response = post_json::<_, ChatResponse>(&self.inner.http, http_request).await?;

        parse_response(
            response.value,
            response.headers,
            request_body_value,
            warnings,
        )
    }

    async fn do_stream(&self, options: CallOptions) -> Result<StreamResult, ProviderError> {
        let (mut request, warnings) = build_request(&self.model_id, &options)?;
        request.stream = Some(true);
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

        let byte_stream = response_byte_stream(stream_response.response);
        let event_stream = sse_json_stream::<ChatChunk>(byte_stream);
        let state = StreamState::with_generate_id(warnings, self.inner.generate_id.clone());
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
    let mistral_opts = parse_mistral_options(options.provider_options.as_ref());
    let mut warnings: Vec<Warning> = Vec::new();

    // Mistral does not accept these three sampling parameters.
    for (val, name) in [
        (options.top_k.is_some(), "topK"),
        (options.frequency_penalty.is_some(), "frequencyPenalty"),
        (options.presence_penalty.is_some(), "presencePenalty"),
    ] {
        if val {
            warnings.push(Warning::Unsupported {
                feature: name.to_owned(),
                details: Some(format!("Mistral chat completions does not accept {name}")),
            });
        }
    }

    let reasoning_effort =
        resolve_reasoning_effort(model_id, &mistral_opts, options.reasoning, &mut warnings);

    let (mut messages, msg_warnings) = convert_prompt(&options.prompt)?;
    warnings.extend(msg_warnings);

    // Mistral needs an explicit instruction when caller asks for a generic
    // JSON response (json mode without a schema) — mirrors ai-sdk's
    // `injectJsonInstructionIntoMessages`. See:
    //   https://docs.mistral.ai/capabilities/structured-output/structured_output_overview/
    if matches!(
        options.response_format.as_ref(),
        Some(ResponseFormat::Json { schema: None, .. })
    ) {
        inject_json_instruction(&mut messages);
    }

    let prepared = prepare_tools(
        options.tools.as_deref().unwrap_or(&[]),
        options.tool_choice.as_ref(),
    );
    warnings.extend(prepared.warnings);

    let response_format = options
        .response_format
        .as_ref()
        .and_then(|fmt| convert_response_format(fmt, &mistral_opts));

    let parallel_tool_calls = if prepared.tools.is_some() {
        mistral_opts.parallel_tool_calls
    } else {
        None
    };

    let request = ChatRequest {
        model: model_id.to_owned(),
        messages,
        stream: None,
        safe_prompt: mistral_opts.safe_prompt,
        max_tokens: options.max_output_tokens,
        temperature: options.temperature,
        top_p: options.top_p,
        stop: options.stop_sequences.clone(),
        random_seed: options.seed,
        reasoning_effort,
        response_format,
        document_image_limit: mistral_opts.document_image_limit,
        document_page_limit: mistral_opts.document_page_limit,
        tools: prepared.tools,
        tool_choice: prepared.tool_choice,
        parallel_tool_calls,
    };

    Ok((request, warnings))
}

/// Append a JSON instruction to the leading system message, creating one if
/// absent. Mirrors `injectJsonInstructionIntoMessages` in
/// `@ai-sdk/provider-utils`.
fn inject_json_instruction(messages: &mut Vec<super::wire::WireMessage>) {
    const SUFFIX: &str = "You MUST answer with JSON.";
    match messages.first_mut() {
        Some(super::wire::WireMessage::System { content }) => {
            if content.is_empty() {
                SUFFIX.clone_into(content);
            } else {
                content.push('\n');
                content.push_str(SUFFIX);
            }
        }
        _ => {
            messages.insert(
                0,
                super::wire::WireMessage::System {
                    content: SUFFIX.to_owned(),
                },
            );
        }
    }
}

fn convert_response_format(
    fmt: &ResponseFormat,
    mistral: &MistralChatOptions,
) -> Option<WireResponseFormat> {
    match fmt {
        ResponseFormat::Text => None,
        ResponseFormat::Json {
            schema,
            name,
            description,
        } => {
            let structured_outputs = mistral.structured_outputs.unwrap_or(true);
            let strict_json_schema = mistral.strict_json_schema.unwrap_or(false);
            Some(match schema {
                Some(schema) if structured_outputs => WireResponseFormat::JsonSchema {
                    json_schema: WireJsonSchema {
                        name: name.clone().unwrap_or_else(|| "response".to_owned()),
                        schema: serde_json::to_value(schema).unwrap_or(serde_json::Value::Null),
                        strict: strict_json_schema,
                        description: description.clone(),
                    },
                },
                _ => WireResponseFormat::JsonObject,
            })
        }
    }
}

fn resolve_reasoning_effort(
    model_id: &str,
    mistral: &MistralChatOptions,
    top_level: Option<ReasoningEffort>,
    warnings: &mut Vec<Warning>,
) -> Option<String> {
    let supports = supports_reasoning_effort(model_id);

    if !supports {
        if top_level.is_some() && !matches!(top_level, Some(ReasoningEffort::ProviderDefault)) {
            warnings.push(Warning::Unsupported {
                feature: "reasoning".to_owned(),
                details: Some("This model does not support reasoning configuration.".to_owned()),
            });
        }
        return None;
    }

    if let Some(effort) = &mistral.reasoning_effort {
        return Some(effort.clone());
    }
    match top_level? {
        ReasoningEffort::ProviderDefault => None,
        ReasoningEffort::None => Some("none".to_owned()),
        ReasoningEffort::Minimal
        | ReasoningEffort::Low
        | ReasoningEffort::Medium
        | ReasoningEffort::High
        | ReasoningEffort::Xhigh => Some("high".to_owned()),
    }
}

/// Mirrors the upstream `supportsReasoningEffort` allowlist.
fn supports_reasoning_effort(model_id: &str) -> bool {
    matches!(
        model_id,
        "mistral-small-latest" | "mistral-small-2603" | "mistral-medium-3" | "mistral-medium-3.5"
    )
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
    fn warns_on_topk_frequency_presence() {
        let mut o = opts();
        o.top_k = Some(5);
        o.frequency_penalty = Some(0.1);
        o.presence_penalty = Some(0.1);
        let (_, warnings) = build_request("mistral-small-latest", &o).unwrap();
        assert_eq!(warnings.len(), 3);
    }

    #[test]
    fn stop_sequences_pass_through_without_warning() {
        let mut o = opts();
        o.stop_sequences = Some(vec!["END".into()]);
        let (req, warnings) = build_request("mistral-small-latest", &o).unwrap();
        assert_eq!(req.stop, Some(vec!["END".into()]));
        assert!(warnings.iter().all(
            |w| !matches!(w, Warning::Unsupported { feature, .. } if feature == "stopSequences")
        ));
    }

    #[test]
    fn seed_serializes_as_random_seed() {
        let mut o = opts();
        o.seed = Some(42);
        let (req, _) = build_request("mistral-small-latest", &o).unwrap();
        assert_eq!(req.random_seed, Some(42));
        let body = serde_json::to_value(&req).unwrap();
        assert_eq!(body["random_seed"], 42);
        assert!(body.get("seed").is_none());
    }

    #[test]
    fn max_output_tokens_serializes_as_max_tokens() {
        let mut o = opts();
        o.max_output_tokens = Some(123);
        let (req, _) = build_request("mistral-small-latest", &o).unwrap();
        assert_eq!(req.max_tokens, Some(123));
    }

    #[test]
    fn safe_prompt_provider_option_pass_through() {
        let mut o = opts();
        let mut po = llmsdk_provider::shared::ProviderOptions::new();
        po.insert(
            "mistral".into(),
            json!({"safePrompt": true}).as_object().cloned().unwrap(),
        );
        o.provider_options = Some(po);
        let (req, _) = build_request("mistral-small-latest", &o).unwrap();
        assert_eq!(req.safe_prompt, Some(true));
    }

    #[test]
    fn unsupported_reasoning_warns_for_non_reasoning_model() {
        let mut o = opts();
        o.reasoning = Some(ReasoningEffort::High);
        let (req, warnings) = build_request("mistral-large-latest", &o).unwrap();
        assert!(req.reasoning_effort.is_none());
        assert!(warnings.iter().any(|w| matches!(
            w,
            Warning::Unsupported { feature, .. } if feature == "reasoning"
        )));
    }

    #[test]
    fn reasoning_effort_coerces_to_high_for_supported_model() {
        let mut o = opts();
        o.reasoning = Some(ReasoningEffort::Low);
        let (req, _) = build_request("mistral-small-latest", &o).unwrap();
        assert_eq!(req.reasoning_effort.as_deref(), Some("high"));
    }

    #[test]
    fn reasoning_effort_none_passes_through() {
        let mut o = opts();
        o.reasoning = Some(ReasoningEffort::None);
        let (req, _) = build_request("mistral-medium-3.5", &o).unwrap();
        assert_eq!(req.reasoning_effort.as_deref(), Some("none"));
    }

    #[test]
    fn provider_options_reasoning_effort_wins() {
        let mut o = opts();
        o.reasoning = Some(ReasoningEffort::Low);
        let mut po = llmsdk_provider::shared::ProviderOptions::new();
        po.insert(
            "mistral".into(),
            json!({"reasoningEffort": "none"})
                .as_object()
                .cloned()
                .unwrap(),
        );
        o.provider_options = Some(po);
        let (req, _) = build_request("mistral-small-latest", &o).unwrap();
        assert_eq!(req.reasoning_effort.as_deref(), Some("none"));
    }

    #[test]
    fn parallel_tool_calls_only_when_tools_present() {
        let mut o = opts();
        let mut po = llmsdk_provider::shared::ProviderOptions::new();
        po.insert(
            "mistral".into(),
            json!({"parallelToolCalls": false})
                .as_object()
                .cloned()
                .unwrap(),
        );
        o.provider_options = Some(po.clone());
        let (req, _) = build_request("mistral-small-latest", &o).unwrap();
        assert_eq!(req.parallel_tool_calls, None);

        o.tools = Some(vec![Tool::Function(FunctionTool {
            name: "weather".into(),
            description: None,
            input_schema: serde_json::from_value(json!({"type":"object"})).unwrap(),
            input_examples: None,
            strict: None,
            provider_options: None,
        })]);
        let (req, _) = build_request("mistral-small-latest", &o).unwrap();
        assert_eq!(req.parallel_tool_calls, Some(false));
    }

    #[test]
    fn function_tool_pass_through_with_tool_choice_required() {
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
        let (req, _) = build_request("mistral-small-latest", &o).unwrap();
        assert!(req.tools.is_some());
        let choice = serde_json::to_value(req.tool_choice.unwrap()).unwrap();
        assert_eq!(choice, json!("any"));
    }

    #[test]
    fn json_response_format_object_default() {
        let mut o = opts();
        o.response_format = Some(ResponseFormat::Json {
            schema: None,
            name: None,
            description: None,
        });
        let (req, _) = build_request("mistral-small-latest", &o).unwrap();
        let body = serde_json::to_value(req.response_format).unwrap();
        assert_eq!(body["type"], "json_object");
    }

    #[test]
    fn json_response_format_schema_when_structured_outputs() {
        let mut o = opts();
        o.response_format = Some(ResponseFormat::Json {
            schema: Some(serde_json::from_value(json!({"type":"object"})).unwrap()),
            name: Some("MySchema".into()),
            description: Some("a schema".into()),
        });
        let (req, _) = build_request("mistral-small-latest", &o).unwrap();
        let body = serde_json::to_value(req.response_format).unwrap();
        assert_eq!(body["type"], "json_schema");
        assert_eq!(body["json_schema"]["name"], "MySchema");
        assert_eq!(body["json_schema"]["description"], "a schema");
        assert_eq!(body["json_schema"]["strict"], false);
    }

    #[test]
    fn json_response_format_strict_pass_through() {
        let mut o = opts();
        o.response_format = Some(ResponseFormat::Json {
            schema: Some(serde_json::from_value(json!({"type":"object"})).unwrap()),
            name: None,
            description: None,
        });
        let mut po = llmsdk_provider::shared::ProviderOptions::new();
        po.insert(
            "mistral".into(),
            json!({"strictJsonSchema": true})
                .as_object()
                .cloned()
                .unwrap(),
        );
        o.provider_options = Some(po);
        let (req, _) = build_request("mistral-small-latest", &o).unwrap();
        let body = serde_json::to_value(req.response_format).unwrap();
        assert_eq!(body["json_schema"]["strict"], true);
    }
}
