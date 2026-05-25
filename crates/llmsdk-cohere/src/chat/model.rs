//! [`LanguageModel`] implementation for Cohere v2 chat.
//!
//! Mirrors `cohere-chat-language-model.ts`. Entry: [`CohereChatModel::new`]
//! via [`crate::Cohere::chat`].
// Rust guideline compliant 2026-05-25

use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    CallOptions, GenerateResult, LanguageModel, ResponseFormat, StreamResult,
};
use llmsdk_provider::shared::Warning;
use llmsdk_provider_utils::http::{JsonRequest, post_for_stream, post_json, response_byte_stream};
use llmsdk_provider_utils::sse::{SseEvent, sse_json_stream};

use crate::PROVIDER_ID;
use crate::config::Inner;

use super::convert_prompt::convert_prompt;
use super::options::parse as parse_options;
use super::parse_response::parse_response;
use super::prepare_tools::prepare as prepare_tools;
use super::stream::StreamState;
use super::wire::{ChatChunk, ChatRequest, ChatResponse, WireResponseFormat, WireThinking};

/// Cohere v2 Chat model handle.
///
/// Cheap to clone; multiple clones share the underlying HTTP client and auth.
#[derive(Debug, Clone)]
pub struct CohereChatModel {
    pub(crate) inner: Arc<Inner>,
    pub(crate) model_id: String,
}

impl CohereChatModel {
    /// Construct from shared provider state and a model id.
    pub(crate) fn new(inner: Arc<Inner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    fn endpoint(&self) -> String {
        format!("{}/chat", self.inner.base_url)
    }
}

#[async_trait]
impl LanguageModel for CohereChatModel {
    fn provider(&self) -> &str {
        PROVIDER_ID
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn do_generate(&self, options: CallOptions) -> Result<GenerateResult, ProviderError> {
        let (request, warnings) = build_request(&self.model_id, &options, false)?;
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
            &mut citation_seed,
        )
    }

    async fn do_stream(&self, options: CallOptions) -> Result<StreamResult, ProviderError> {
        let (request, warnings) = build_request(&self.model_id, &options, true)?;
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
    stream: bool,
) -> Result<(ChatRequest, Vec<Warning>), ProviderError> {
    let cohere_opts = parse_options(options.provider_options.as_ref());
    let mut warnings: Vec<Warning> = Vec::new();

    let converted = convert_prompt(&options.prompt)?;
    warnings.extend(converted.warnings);

    let prepared = prepare_tools(
        options.tools.as_deref().unwrap_or(&[]),
        options.tool_choice.as_ref(),
    );
    warnings.extend(prepared.warnings);

    let response_format = options
        .response_format
        .as_ref()
        .and_then(convert_response_format);

    let thinking = cohere_opts.thinking.as_ref().map(|t| WireThinking {
        kind: t.kind.clone().unwrap_or_else(|| "enabled".to_owned()),
        token_budget: t.token_budget,
    });

    let documents = if converted.documents.is_empty() {
        None
    } else {
        Some(converted.documents)
    };

    let request = ChatRequest {
        model: model_id.to_owned(),
        messages: converted.messages,
        stream: stream.then_some(true),
        max_tokens: options.max_output_tokens,
        temperature: options.temperature,
        p: options.top_p,
        k: options.top_k,
        seed: options.seed,
        frequency_penalty: options.frequency_penalty,
        presence_penalty: options.presence_penalty,
        stop_sequences: options.stop_sequences.clone(),
        response_format,
        tools: prepared.tools,
        tool_choice: prepared.tool_choice,
        documents,
        thinking,
    };

    Ok((request, warnings))
}

fn convert_response_format(fmt: &ResponseFormat) -> Option<WireResponseFormat> {
    match fmt {
        ResponseFormat::Text => None,
        ResponseFormat::Json { schema, .. } => Some(WireResponseFormat::JsonObject {
            json_schema: schema
                .as_ref()
                .map(|s| serde_json::to_value(s).unwrap_or(serde_json::Value::Null)),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmsdk_provider::language_model::{
        FunctionTool, Message, TextPart, Tool, ToolChoice, UserPart,
    };
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
    fn maps_standard_settings_to_cohere_field_names() {
        let mut o = opts();
        o.max_output_tokens = Some(100);
        o.temperature = Some(0.5);
        o.top_p = Some(0.9);
        o.top_k = Some(5);
        o.seed = Some(42);
        o.frequency_penalty = Some(0.1);
        o.presence_penalty = Some(0.2);
        o.stop_sequences = Some(vec!["END".into()]);
        let (req, _) = build_request("command-a-03-2025", &o, false).unwrap();
        assert_eq!(req.max_tokens, Some(100));
        assert_eq!(req.temperature, Some(0.5));
        assert_eq!(req.p, Some(0.9));
        assert_eq!(req.k, Some(5));
        assert_eq!(req.seed, Some(42));
        assert_eq!(req.frequency_penalty, Some(0.1));
        assert_eq!(req.presence_penalty, Some(0.2));
        assert_eq!(req.stop_sequences, Some(vec!["END".to_owned()]));
        assert!(req.stream.is_none());
    }

    #[test]
    fn stream_flag_set_in_streaming_path() {
        let (req, _) = build_request("command-a-03-2025", &opts(), true).unwrap();
        assert_eq!(req.stream, Some(true));
    }

    #[test]
    fn thinking_provider_option_forwarded() {
        let mut o = opts();
        let mut po = llmsdk_provider::shared::ProviderOptions::new();
        po.insert(
            "cohere".into(),
            json!({"thinking": {"type": "enabled", "tokenBudget": 2048}})
                .as_object()
                .cloned()
                .unwrap(),
        );
        o.provider_options = Some(po);
        let (req, _) = build_request("command-a-reasoning-08-2025", &o, false).unwrap();
        let t = req.thinking.expect("thinking forwarded");
        assert_eq!(t.kind, "enabled");
        assert_eq!(t.token_budget, Some(2048));
    }

    #[test]
    fn response_format_json_with_schema() {
        let mut o = opts();
        o.response_format = Some(ResponseFormat::Json {
            schema: Some(serde_json::from_value(json!({"type": "object"})).expect("schema")),
            name: None,
            description: None,
        });
        let (req, _) = build_request("command-a-03-2025", &o, false).unwrap();
        let wire = serde_json::to_value(&req.response_format).unwrap();
        assert_eq!(wire["type"], "json_object");
        assert_eq!(wire["json_schema"]["type"], "object");
    }

    #[test]
    fn function_tool_passthrough_with_required_choice() {
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
        let (req, _) = build_request("command-a-03-2025", &o, false).unwrap();
        assert_eq!(req.tools.as_ref().unwrap().len(), 1);
        let choice = serde_json::to_value(req.tool_choice.unwrap()).unwrap();
        assert_eq!(choice, json!("REQUIRED"));
    }
}
