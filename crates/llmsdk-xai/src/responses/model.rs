//! [`LanguageModel`] implementation for the xAI Responses API.
//!
//! Mirrors `@ai-sdk/xai/src/responses/xai-responses-language-model.ts`.
//! Entry: [`XaiResponsesLanguageModel::new`] via [`crate::Xai::responses`].
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
use serde_json::{Map, Value, json};

use crate::PROVIDER_ID;
use crate::config::Inner;

use super::convert_prompt::convert_prompt;
use super::options::{XaiResponsesOptions, parse as parse_xai_options};
use super::parse_response::parse_response;
use super::prepare_tools::{PreparedTools, ResolvedToolNames, prepare as prepare_tools};
use super::stream::StreamState;
use super::wire::{ResponsesChunk, ResponsesRequest, ResponsesResponse};

/// xAI Responses API model handle.
///
/// Cheap to clone. Multiple clones share the underlying HTTP client and
/// authentication state via [`Xai`](crate::Xai)'s `Arc`.
#[derive(Debug, Clone)]
pub struct XaiResponsesLanguageModel {
    pub(crate) inner: Arc<Inner>,
    pub(crate) model_id: String,
}

impl XaiResponsesLanguageModel {
    /// Construct from shared provider state and a model id.
    pub(crate) fn new(inner: Arc<Inner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    fn endpoint(&self) -> String {
        format!("{}/responses", self.inner.base_url)
    }
}

#[async_trait]
impl LanguageModel for XaiResponsesLanguageModel {
    fn provider(&self) -> &str {
        PROVIDER_ID
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn supported_urls(&self) -> SupportedUrls {
        let mut map = SupportedUrls::new();
        let any_https = UrlPattern::new(r"^https?://.*$");
        map.insert("image/*".into(), vec![any_https.clone()]);
        map.insert("application/pdf".into(), vec![any_https.clone()]);
        map.insert("text/*".into(), vec![any_https]);
        map
    }

    async fn do_generate(&self, options: CallOptions) -> Result<GenerateResult, ProviderError> {
        let Built {
            request,
            warnings,
            names,
        } = build_request(&self.model_id, &options, false)?;
        let request_body_value = serde_json::to_value(&request).ok();

        let mut headers = self.inner.headers.clone();
        if let Some(extra) = &options.headers {
            for (k, v) in extra {
                headers.insert(k.clone(), v.clone());
            }
        }

        let mut http_request = JsonRequest::new(self.endpoint(), request);
        http_request.headers = headers;

        let response = post_json::<_, ResponsesResponse>(&self.inner.http, http_request).await?;

        let mut citation_seed: u64 = 0;
        parse_response(
            response.value,
            response.headers,
            request_body_value,
            warnings,
            &names,
            &mut citation_seed,
        )
    }

    async fn do_stream(&self, options: CallOptions) -> Result<StreamResult, ProviderError> {
        let include_raw_chunks = options.include_raw_chunks.unwrap_or(false);
        let Built {
            mut request,
            warnings,
            names,
        } = build_request(&self.model_id, &options, true)?;
        request.stream = Some(true);
        let request_body_value = serde_json::to_value(&request).ok();

        let mut headers = self.inner.headers.clone();
        if let Some(extra) = &options.headers {
            for (k, v) in extra {
                headers.insert(k.clone(), v.clone());
            }
        }

        let mut http_request = JsonRequest::new(self.endpoint(), request);
        http_request.headers = headers;

        let stream_response = post_for_stream(&self.inner.http, http_request).await?;
        let stream_headers = stream_response.headers.clone();
        let byte_stream = response_byte_stream(stream_response.response);
        let event_stream = sse_json_stream::<ResponsesChunk>(byte_stream);

        let state = StreamState::new(warnings, names, include_raw_chunks);
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

fn build_part_stream<S>(
    mut state: StreamState,
    events: S,
) -> impl futures::Stream<Item = Result<llmsdk_provider::language_model::StreamPart, ProviderError>> + Send
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

fn headers_to_provider(
    raw: std::collections::HashMap<String, String>,
) -> llmsdk_provider::shared::Headers {
    raw.into_iter().map(|(k, v)| (k, Some(v))).collect()
}

struct Built {
    request: ResponsesRequest,
    warnings: Vec<Warning>,
    names: ResolvedToolNames,
}

fn build_request(
    model_id: &str,
    options: &CallOptions,
    _streaming: bool,
) -> Result<Built, ProviderError> {
    let xai_opts = parse_xai_options(options.provider_options.as_ref());
    let mut warnings: Vec<Warning> = Vec::new();

    if options.stop_sequences.is_some() {
        warnings.push(Warning::UnsupportedSetting {
            setting: "stopSequences".into(),
            details: Some("xAI Responses API does not accept stopSequences".into()),
        });
    }

    let PreparedTools {
        tools,
        tool_choice,
        warnings: tool_warnings,
        names,
    } = prepare_tools(
        options.tools.as_deref().unwrap_or(&[]),
        options.tool_choice.as_ref(),
    );
    warnings.extend(tool_warnings);

    let (input, prompt_warnings) = convert_prompt(&options.prompt)?;
    warnings.extend(prompt_warnings);

    let mut include = xai_opts.include.clone();
    if xai_opts.store == Some(false) {
        match include.as_mut() {
            Some(v) => {
                if !v.iter().any(|k| k == "reasoning.encrypted_content") {
                    v.push("reasoning.encrypted_content".to_owned());
                }
            }
            None => include = Some(vec!["reasoning.encrypted_content".to_owned()]),
        }
    }

    let resolved_reasoning_effort =
        resolve_reasoning_effort(&xai_opts, options.reasoning, &mut warnings);

    let text = build_text_field(options.response_format.as_ref());
    let reasoning = build_reasoning_field(
        resolved_reasoning_effort.as_deref(),
        xai_opts.reasoning_summary.as_deref(),
    );

    let logprobs_flag = match (xai_opts.logprobs, xai_opts.top_logprobs) {
        (Some(true), _) | (_, Some(_)) => Some(true),
        _ => xai_opts.logprobs,
    };

    let request = ResponsesRequest {
        model: model_id.to_owned(),
        input,
        max_output_tokens: options.max_output_tokens,
        temperature: options.temperature,
        top_p: options.top_p,
        seed: options.seed,
        logprobs: logprobs_flag,
        top_logprobs: xai_opts.top_logprobs,
        text,
        reasoning,
        // upstream only emits `store` when it's explicitly `false`
        store: match xai_opts.store {
            Some(false) => Some(false),
            _ => None,
        },
        include,
        previous_response_id: xai_opts.previous_response_id.clone(),
        tools,
        tool_choice,
        stream: None,
    };

    Ok(Built {
        request,
        warnings,
        names,
    })
}

fn resolve_reasoning_effort(
    xai: &XaiResponsesOptions,
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

fn build_text_field(fmt: Option<&ResponseFormat>) -> Option<Value> {
    let Some(ResponseFormat::Json {
        schema,
        name,
        description,
    }) = fmt
    else {
        return None;
    };
    let mut fmt_obj = Map::new();
    if let Some(s) = schema {
        fmt_obj.insert("type".into(), json!("json_schema"));
        fmt_obj.insert("strict".into(), json!(true));
        fmt_obj.insert(
            "name".into(),
            json!(name.clone().unwrap_or_else(|| "response".into())),
        );
        if let Some(d) = description {
            fmt_obj.insert("description".into(), json!(d));
        }
        fmt_obj.insert(
            "schema".into(),
            serde_json::to_value(s).unwrap_or(Value::Null),
        );
    } else {
        fmt_obj.insert("type".into(), json!("json_object"));
    }
    let mut outer = Map::new();
    outer.insert("format".into(), Value::Object(fmt_obj));
    Some(Value::Object(outer))
}

fn build_reasoning_field(effort: Option<&str>, summary: Option<&str>) -> Option<Value> {
    if effort.is_none() && summary.is_none() {
        return None;
    }
    let mut obj = Map::new();
    if let Some(e) = effort {
        obj.insert("effort".into(), json!(e));
    }
    if let Some(s) = summary {
        obj.insert("summary".into(), json!(s));
    }
    Some(Value::Object(obj))
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmsdk_provider::language_model::{
        FunctionTool, Message, ProviderTool, TextPart, Tool, UserPart,
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
    fn stop_sequences_emits_warning() {
        let mut o = opts();
        o.stop_sequences = Some(vec!["END".into()]);
        let built = build_request("grok-4.3", &o, false).unwrap();
        assert!(built.warnings.iter().any(|w| matches!(
            w,
            Warning::UnsupportedSetting { setting, .. } if setting == "stopSequences"
        )));
    }

    #[test]
    fn reasoning_effort_xhigh_maps_to_high_with_warning() {
        let mut o = opts();
        o.reasoning = Some(ReasoningEffort::Xhigh);
        let built = build_request("grok-4.3", &o, false).unwrap();
        let r = built.request.reasoning.as_ref().unwrap();
        assert_eq!(r["effort"], "high");
        assert!(built.warnings.iter().any(|w| matches!(
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
            json!({"reasoningEffort": "high", "reasoningSummary": "concise"})
                .as_object()
                .cloned()
                .unwrap(),
        );
        o.provider_options = Some(po);
        let built = build_request("grok-4.3", &o, false).unwrap();
        let r = built.request.reasoning.as_ref().unwrap();
        assert_eq!(r["effort"], "high");
        assert_eq!(r["summary"], "concise");
    }

    #[test]
    fn store_false_auto_includes_encrypted_content() {
        let mut o = opts();
        let mut po = llmsdk_provider::shared::ProviderOptions::new();
        po.insert(
            "xai".into(),
            json!({"store": false}).as_object().cloned().unwrap(),
        );
        o.provider_options = Some(po);
        let built = build_request("grok-4.3", &o, false).unwrap();
        assert_eq!(built.request.store, Some(false));
        let include = built.request.include.unwrap();
        assert!(include.contains(&"reasoning.encrypted_content".to_owned()));
    }

    #[test]
    fn previous_response_id_passes_through() {
        let mut o = opts();
        let mut po = llmsdk_provider::shared::ProviderOptions::new();
        po.insert(
            "xai".into(),
            json!({"previousResponseId": "resp_xyz"})
                .as_object()
                .cloned()
                .unwrap(),
        );
        o.provider_options = Some(po);
        let built = build_request("grok-4.3", &o, false).unwrap();
        assert_eq!(
            built.request.previous_response_id.as_deref(),
            Some("resp_xyz")
        );
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
        let built = build_request("grok-4.3", &o, false).unwrap();
        assert_eq!(built.request.logprobs, Some(true));
        assert_eq!(built.request.top_logprobs, Some(5));
    }

    #[test]
    fn function_tool_routes_to_responses_function_shape() {
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
        let built = build_request("grok-4.3", &o, false).unwrap();
        let tools = built.request.tools.unwrap();
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["name"], "weather");
    }

    #[test]
    fn provider_tool_web_search_captures_name() {
        let mut o = opts();
        o.tools = Some(vec![Tool::Provider(ProviderTool {
            id: "xai.web_search".into(),
            name: "web_search".into(),
            args: None,
            provider_options: None,
        })]);
        let built = build_request("grok-4.3", &o, false).unwrap();
        assert_eq!(built.names.web_search.as_deref(), Some("web_search"));
        let tools = built.request.tools.unwrap();
        assert_eq!(tools[0]["type"], "web_search");
    }
}
