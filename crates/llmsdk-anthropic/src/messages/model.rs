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
use super::parse_response::parse_response;
use super::stream::StreamState;
use super::stream_event::StreamEvent;
use super::wire::{MessagesRequest, MessagesResponse, WireMessage, WireTool, WireToolChoice};

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
        let (request, warnings) = build_request(&self.model_id, &options, false);

        let request_body_value = serde_json::to_value(&request).ok();
        let mut http_request = JsonRequest::new(self.endpoint(), request);
        http_request.headers = self.merged_headers(options.headers.as_ref());

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
        let (request, warnings) = build_request(&self.model_id, &options, true);

        let request_body_value = serde_json::to_value(&request).ok();
        let mut http_request = JsonRequest::new(self.endpoint(), request);
        http_request.headers = self.merged_headers(options.headers.as_ref());

        let stream_response = match post_for_stream(&self.inner.http, http_request).await {
            Ok(r) => r,
            Err(err) => return Err(rewrite_anthropic_error(err)),
        };

        let stream_headers = stream_response.headers.clone();
        let byte_stream = response_byte_stream(stream_response.response);
        let event_stream = sse_json_stream::<StreamEvent>(byte_stream);

        let state = StreamState::new(warnings);
        let parts = build_part_stream(state, event_stream);

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

fn build_request(
    model_id: &str,
    options: &CallOptions,
    stream: bool,
) -> (MessagesRequest, Vec<Warning>) {
    let Converted {
        system,
        messages,
        mut warnings,
    } = convert_prompt(&options.prompt);

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
    if options.response_format.is_some() {
        warnings.push(Warning::UnsupportedSetting {
            setting: "responseFormat".to_owned(),
            details: Some("M6 does not relay responseFormat to Anthropic".to_owned()),
        });
    }

    let (tools, tool_choice) = convert_tools(
        options.tools.as_deref(),
        options.tool_choice.as_ref(),
        &mut warnings,
    );

    let request = MessagesRequest {
        model: model_id.to_owned(),
        max_tokens: options.max_output_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        messages: ensure_user_first(messages, &mut warnings),
        system,
        temperature: options.temperature,
        top_p: options.top_p,
        top_k: options.top_k,
        stop_sequences: options.stop_sequences.clone(),
        stream: stream.then_some(true),
        tools,
        tool_choice,
    };

    (request, warnings)
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
) -> (Option<Vec<WireTool>>, Option<WireToolChoice>) {
    let Some(tools) = tools else {
        return (None, None);
    };
    let converted: Vec<_> = tools
        .iter()
        .filter_map(|t| match t {
            Tool::Function(f) => Some(WireTool {
                name: f.name.clone(),
                description: f.description.clone(),
                input_schema: f.input_schema.clone(),
            }),
            Tool::Provider(p) => {
                warnings.push(Warning::UnsupportedTool {
                    tool: p.name.clone(),
                    details: Some("M6 Anthropic does not relay provider-defined tools".to_owned()),
                });
                None
            }
        })
        .collect();
    if converted.is_empty() {
        return (None, None);
    }
    let tool_choice = choice.map(|c| match c {
        // Anthropic has no explicit "none" — downgrade to "auto" and warn.
        ToolChoice::Auto | ToolChoice::None => {
            if matches!(c, ToolChoice::None) {
                warnings.push(Warning::UnsupportedSetting {
                    setting: "toolChoice".to_owned(),
                    details: Some(
                        "Anthropic has no `none` tool choice; downgraded to `auto`".to_owned(),
                    ),
                });
            }
            WireToolChoice::Auto
        }
        ToolChoice::Required => WireToolChoice::Any,
        ToolChoice::Tool { tool_name } => WireToolChoice::Tool {
            name: tool_name.clone(),
        },
    });
    (Some(converted), tool_choice)
}

/// Drive an SSE event stream through [`StreamState`].
fn build_part_stream<S>(
    mut state: StreamState,
    events: S,
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
