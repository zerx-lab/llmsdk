//! `LanguageModel` implementation for the legacy `OpenAI` Completions API
//! (`POST /v1/completions`).
//!
//! Mirrors `@ai-sdk/openai/src/completion/openai-completion-language-model.ts`.
//! Used for instruction-tuned legacy models such as `gpt-3.5-turbo-instruct`.
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    AssistantPart, CallOptions, Content, FinishReason, FinishReasonKind, GenerateResponse,
    GenerateResult, InputTokenUsage, LanguageModel, Message, OutputTokenUsage, ResponseFormat,
    ResponseMetadata, StreamPart, StreamResponse, StreamResult, TextPart, Usage, UserPart,
};
use llmsdk_provider::shared::{ProviderMetadata, RequestInfo, Warning};
use llmsdk_provider_utils::http::{JsonRequest, post_for_stream, post_json, response_byte_stream};
use llmsdk_provider_utils::sse::{SseEvent, sse_json_stream};
use llmsdk_provider_utils::time::rfc3339_from_unix_seconds;
use serde::{Deserialize, Serialize};

use crate::chat::finish_reason::map as map_finish_reason;
use crate::config::Inner;
use crate::error::rewrite_openai_error;

const TEXT_BLOCK_ID: &str = "0";

/// `OpenAI` legacy Completions model handle.
#[derive(Debug, Clone)]
pub struct OpenAiCompletionLanguageModel {
    inner: Arc<Inner>,
    model_id: String,
}

impl OpenAiCompletionLanguageModel {
    /// Construct from a fully assembled [`Inner`]. Public for cross-crate
    /// composition (Azure `OpenAI`). End-users should prefer the provider
    /// builder's `completion(...)` factory.
    #[must_use]
    pub fn new(inner: Arc<Inner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    fn endpoint(&self) -> String {
        self.inner.endpoint("/completions", &self.model_id)
    }
}

#[async_trait]
impl LanguageModel for OpenAiCompletionLanguageModel {
    fn provider(&self) -> &str {
        self.inner.provider_id()
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn do_generate(&self, options: CallOptions) -> Result<GenerateResult, ProviderError> {
        let provider_options_name = self.inner.provider_options_name();
        let (request, warnings) =
            build_request(&self.model_id, &options, false, provider_options_name)?;

        let mut headers = self.inner.headers.clone();
        if let Some(extra) = &options.headers {
            for (n, v) in extra {
                headers.insert(n.clone(), v.clone());
            }
        }

        let request_body = serde_json::to_value(&request).ok();
        let body_bytes = serde_json::to_vec(&request).unwrap_or_default();
        let url = self.endpoint();
        self.inner
            .sign_if_needed(&mut headers, "POST", &url, &body_bytes)
            .await?;
        let mut http = JsonRequest::new(url, request);
        http.headers = headers;

        let response = match post_json::<_, CompletionResponse>(&self.inner.http, http).await {
            Ok(r) => r,
            Err(err) => return Err(rewrite_openai_error(err)),
        };

        parse_completion(
            response.value,
            response.headers,
            request_body,
            warnings,
            provider_options_name,
        )
    }

    async fn do_stream(&self, options: CallOptions) -> Result<StreamResult, ProviderError> {
        let provider_options_name = self.inner.provider_options_name();
        let (request, warnings) =
            build_request(&self.model_id, &options, true, provider_options_name)?;

        let mut headers = self.inner.headers.clone();
        if let Some(extra) = &options.headers {
            for (n, v) in extra {
                headers.insert(n.clone(), v.clone());
            }
        }
        let request_body = serde_json::to_value(&request).ok();
        let body_bytes = serde_json::to_vec(&request).unwrap_or_default();
        let url = self.endpoint();
        self.inner
            .sign_if_needed(&mut headers, "POST", &url, &body_bytes)
            .await?;
        let mut http = JsonRequest::new(url, request);
        http.headers = headers;

        let stream_response = match post_for_stream(&self.inner.http, http).await {
            Ok(r) => r,
            Err(err) => return Err(rewrite_openai_error(err)),
        };
        let stream_headers = stream_response.headers.clone();
        let byte_stream = response_byte_stream(stream_response.response);
        let event_stream = sse_json_stream::<CompletionChunk>(byte_stream);

        let parts = drive_completion_stream(warnings, event_stream, provider_options_name);

        Ok(StreamResult {
            stream: Box::pin(parts),
            request: Some(RequestInfo { body: request_body }),
            response: Some(StreamResponse {
                headers: Some(headers_to_provider(stream_headers)),
            }),
        })
    }
}

fn headers_to_provider(raw: HashMap<String, String>) -> llmsdk_provider::shared::Headers {
    raw.into_iter().map(|(k, v)| (k, Some(v))).collect()
}

// -------- request build --------------------------------------------------

#[derive(Debug, Clone, Serialize)]
struct CompletionRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    echo: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    suffix: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logit_bias: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
}

#[derive(Debug, Clone, Serialize)]
struct StreamOptions {
    include_usage: bool,
}

/// Parsed `provider_options["openai"]` slot for completion calls.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct CompletionProviderOptions {
    echo: Option<bool>,
    logit_bias: Option<serde_json::Map<String, serde_json::Value>>,
    suffix: Option<String>,
    user: Option<String>,
    /// `true` -> 0 alternatives, `false` -> omit, a number -> top-N alternatives.
    logprobs: Option<LogprobsOption>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum LogprobsOption {
    Flag(bool),
    Count(u32),
}

impl LogprobsOption {
    fn to_wire(&self) -> Option<u32> {
        match self {
            Self::Flag(true) => Some(0),
            Self::Flag(false) => None,
            Self::Count(n) => Some(*n),
        }
    }
}

fn parse_provider_options(
    options: &CallOptions,
    provider_options_name: &str,
) -> CompletionProviderOptions {
    let Some(po) = options.provider_options.as_ref() else {
        return CompletionProviderOptions::default();
    };
    // Mirror upstream merge semantics: parse `openai` first as base, then
    // layer `provider_options_name` (e.g. `azure`) on top so Azure-scoped
    // fields override the canonical OpenAI ones. See
    // `openai-completion-language-model.ts:106-117`.
    let mut merged = serde_json::Map::new();
    if let Some(base) = po.get("openai") {
        for (k, v) in base {
            merged.insert(k.clone(), v.clone());
        }
    }
    if provider_options_name != "openai"
        && let Some(slot) = po.get(provider_options_name)
    {
        for (k, v) in slot {
            merged.insert(k.clone(), v.clone());
        }
    }
    if merged.is_empty() {
        return CompletionProviderOptions::default();
    }
    serde_json::from_value::<CompletionProviderOptions>(serde_json::Value::Object(merged))
        .unwrap_or_default()
}

fn build_request(
    model_id: &str,
    options: &CallOptions,
    stream: bool,
    provider_options_name: &str,
) -> Result<(CompletionRequest, Vec<Warning>), ProviderError> {
    let provider_opts = parse_provider_options(options, provider_options_name);
    let mut warnings = Vec::new();

    if options.top_k.is_some() {
        warnings.push(Warning::Unsupported {
            feature: "topK".to_owned(),
            details: None,
        });
    }
    if options.tools.as_ref().is_some_and(|t| !t.is_empty()) {
        warnings.push(Warning::Unsupported {
            feature: "tools".to_owned(),
            details: Some("the legacy Completions endpoint does not support tools".to_owned()),
        });
    }
    if options.tool_choice.is_some() {
        warnings.push(Warning::Unsupported {
            feature: "toolChoice".to_owned(),
            details: None,
        });
    }
    if matches!(options.response_format, Some(ResponseFormat::Json { .. })) {
        warnings.push(Warning::Unsupported {
            feature: "responseFormat".to_owned(),
            details: Some("JSON response format is not supported on /v1/completions".to_owned()),
        });
    }

    let (prompt, derived_stop) = convert_completion_prompt(&options.prompt)?;
    let mut stop_sequences = derived_stop;
    if let Some(extra) = &options.stop_sequences {
        stop_sequences.extend(extra.iter().cloned());
    }
    let stop = (!stop_sequences.is_empty()).then_some(stop_sequences);

    Ok((
        CompletionRequest {
            model: model_id.to_owned(),
            prompt,
            max_tokens: options.max_output_tokens,
            temperature: options.temperature,
            top_p: options.top_p,
            frequency_penalty: options.frequency_penalty,
            presence_penalty: options.presence_penalty,
            seed: options.seed,
            stop,
            echo: provider_opts.echo,
            suffix: provider_opts.suffix,
            user: provider_opts.user,
            logprobs: provider_opts
                .logprobs
                .as_ref()
                .and_then(LogprobsOption::to_wire),
            logit_bias: provider_opts.logit_bias,
            stream: stream.then_some(true),
            stream_options: stream.then_some(StreamOptions {
                include_usage: true,
            }),
        },
        warnings,
    ))
}

/// Flatten a chat-style prompt into a single string, mirroring
/// `convert-to-openai-completion-prompt.ts`.
fn convert_completion_prompt(prompt: &[Message]) -> Result<(String, Vec<String>), ProviderError> {
    let mut text = String::new();
    let mut iter = prompt.iter().peekable();

    // Leading system message becomes the preface.
    if let Some(Message::System { content, .. }) = iter.peek() {
        text.push_str(content);
        text.push_str("\n\n");
        iter.next();
    }

    for msg in iter {
        match msg {
            Message::System { .. } => {
                return Err(ProviderError::invalid_argument(
                    "prompt",
                    "unexpected non-leading system message in completion prompt",
                ));
            }
            Message::User { content, .. } => {
                let mut user_text = String::new();
                for part in content {
                    if let UserPart::Text(TextPart { text, .. }) = part {
                        user_text.push_str(text);
                    }
                }
                text.push_str("user:\n");
                text.push_str(&user_text);
                text.push_str("\n\n");
            }
            Message::Assistant { content, .. } => {
                let mut assistant_text = String::new();
                for part in content {
                    if let AssistantPart::Text(TextPart { text, .. }) = part {
                        assistant_text.push_str(text);
                    } else {
                        return Err(ProviderError::invalid_argument(
                            "prompt",
                            "tool / file / reasoning parts are not supported by /v1/completions",
                        ));
                    }
                }
                text.push_str("assistant:\n");
                text.push_str(&assistant_text);
                text.push_str("\n\n");
            }
            Message::Tool { .. } => {
                return Err(ProviderError::invalid_argument(
                    "prompt",
                    "tool messages are not supported by /v1/completions",
                ));
            }
        }
    }

    // Assistant prefix the model is expected to continue from.
    text.push_str("assistant:\n");

    Ok((text, vec!["\nuser:".to_owned()]))
}

// -------- response parsing -----------------------------------------------

#[derive(Debug, Clone, Deserialize)]
struct CompletionResponse {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    created: Option<u64>,
    #[serde(default)]
    model: Option<String>,
    choices: Vec<CompletionChoice>,
    #[serde(default)]
    usage: Option<CompletionUsage>,
}

#[derive(Debug, Clone, Deserialize)]
struct CompletionChoice {
    text: String,
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    logprobs: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(
    clippy::struct_field_names,
    reason = "field names mirror OpenAI's wire schema verbatim"
)]
struct CompletionUsage {
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens: Option<u64>,
    /// Captured for wire parity; we recompute the unified total ourselves.
    #[serde(default, rename = "total_tokens")]
    _total_tokens: Option<u64>,
}

fn convert_usage(u: Option<&CompletionUsage>) -> Usage {
    let input_total = u.and_then(|x| x.prompt_tokens);
    let output_total = u.and_then(|x| x.completion_tokens);
    Usage {
        input_tokens: InputTokenUsage {
            total: input_total,
            ..InputTokenUsage::default()
        },
        output_tokens: OutputTokenUsage {
            total: output_total,
            text: output_total,
            ..OutputTokenUsage::default()
        },
        raw: None,
    }
}

fn parse_completion(
    response: CompletionResponse,
    headers: HashMap<String, String>,
    request_body: Option<serde_json::Value>,
    warnings: Vec<Warning>,
    provider_options_name: &str,
) -> Result<GenerateResult, ProviderError> {
    let choice = response.choices.first().ok_or_else(|| {
        ProviderError::type_validation(
            "choices",
            serde_json::Value::Null,
            "OpenAI completion response had no choices",
        )
    })?;

    let mut provider_metadata = None;
    if let Some(lp) = &choice.logprobs {
        let mut openai = serde_json::Map::new();
        openai.insert("logprobs".to_owned(), lp.clone());
        let mut pm = ProviderMetadata::new();
        pm.insert(provider_options_name.to_owned(), openai);
        provider_metadata = Some(pm);
    }

    let content = vec![Content::Text(TextPart {
        text: choice.text.clone(),
        provider_options: None,
    })];

    let finish_reason = map_finish_reason(choice.finish_reason.as_deref());

    Ok(GenerateResult {
        content,
        finish_reason,
        usage: convert_usage(response.usage.as_ref()),
        provider_metadata,
        warnings,
        request: Some(RequestInfo { body: request_body }),
        response: Some(GenerateResponse {
            metadata: ResponseMetadata {
                id: response.id,
                timestamp: response.created.map(rfc3339_from_unix_seconds),
                model_id: response.model,
                headers: Some(headers_to_provider(headers)),
            },
            body: None,
        }),
    })
}

// -------- streaming ------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum CompletionChunk {
    Ok(CompletionChunkOk),
    Err(CompletionChunkErr),
}

#[derive(Debug, Clone, Deserialize)]
struct CompletionChunkOk {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    created: Option<u64>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    choices: Vec<CompletionChunkChoice>,
    #[serde(default)]
    usage: Option<CompletionUsage>,
}

#[derive(Debug, Clone, Deserialize)]
struct CompletionChunkErr {
    error: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize)]
struct CompletionChunkChoice {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    logprobs: Option<serde_json::Value>,
}

fn drive_completion_stream<S>(
    warnings: Vec<Warning>,
    events: S,
    provider_options_name: &'static str,
) -> impl futures::Stream<Item = Result<StreamPart, ProviderError>> + Send
where
    S: futures::Stream<Item = Result<SseEvent<CompletionChunk>, ProviderError>> + Send + 'static,
{
    async_stream::stream! {
        yield Ok(StreamPart::StreamStart { warnings });

        let mut events = Box::pin(events);
        let mut text_started = false;
        let mut metadata_emitted = false;
        let mut last_finish: FinishReason = FinishReason::new(FinishReasonKind::Other);
        let mut last_usage: Option<CompletionUsage> = None;
        let mut last_logprobs: Option<serde_json::Value> = None;

        while let Some(event) = events.next().await {
            match event {
                Ok(SseEvent::Data(CompletionChunk::Err(err))) => {
                    last_finish = FinishReason::new(FinishReasonKind::Error);
                    yield Ok(StreamPart::Error { error: err.error });
                }
                Ok(SseEvent::Data(CompletionChunk::Ok(chunk))) => {
                    if !metadata_emitted
                        && (chunk.id.is_some() || chunk.created.is_some() || chunk.model.is_some())
                    {
                        metadata_emitted = true;
                        yield Ok(StreamPart::ResponseMetadata(ResponseMetadata {
                            id: chunk.id.clone(),
                            timestamp: chunk.created.map(rfc3339_from_unix_seconds),
                            model_id: chunk.model.clone(),
                            headers: None,
                        }));
                    }
                    if let Some(u) = chunk.usage {
                        last_usage = Some(u);
                    }
                    if let Some(choice) = chunk.choices.into_iter().next() {
                        if let Some(reason) = choice.finish_reason {
                            last_finish = map_finish_reason(Some(reason.as_str()));
                        }
                        if let Some(lp) = choice.logprobs {
                            last_logprobs = Some(lp);
                        }
                        if let Some(delta) = choice.text.filter(|s| !s.is_empty()) {
                            if !text_started {
                                text_started = true;
                                yield Ok(StreamPart::TextStart {
                                    id: TEXT_BLOCK_ID.to_owned(),
                                    provider_metadata: None,
                                });
                            }
                            yield Ok(StreamPart::TextDelta {
                                id: TEXT_BLOCK_ID.to_owned(),
                                delta,
                                provider_metadata: None,
                            });
                        }
                    }
                }
                Ok(SseEvent::ParseError { raw, message }) => {
                    last_finish = FinishReason::new(FinishReasonKind::Error);
                    yield Ok(StreamPart::Error {
                        error: serde_json::json!({ "message": message, "raw": raw }),
                    });
                }
                Err(e) => {
                    yield Err(e);
                    return;
                }
            }
        }

        if text_started {
            yield Ok(StreamPart::TextEnd {
                id: TEXT_BLOCK_ID.to_owned(),
                provider_metadata: None,
            });
        }

        let provider_metadata = last_logprobs.map(|lp| {
            let mut openai = serde_json::Map::new();
            openai.insert("logprobs".to_owned(), lp);
            let mut pm = ProviderMetadata::new();
            pm.insert(provider_options_name.to_owned(), openai);
            pm
        });

        yield Ok(StreamPart::Finish {
            usage: convert_usage(last_usage.as_ref()),
            finish_reason: last_finish,
            provider_metadata,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flattens_simple_prompt() {
        let prompt = vec![
            Message::System {
                content: "You are helpful.".into(),
                provider_options: None,
            },
            Message::User {
                content: vec![UserPart::Text(TextPart {
                    text: "Hi".into(),
                    provider_options: None,
                })],
                provider_options: None,
            },
        ];
        let (text, stop) = convert_completion_prompt(&prompt).unwrap();
        assert!(text.starts_with("You are helpful.\n\nuser:\nHi\n\nassistant:\n"));
        assert_eq!(stop, vec!["\nuser:".to_owned()]);
    }

    #[test]
    fn rejects_tool_messages() {
        let prompt = vec![Message::Tool {
            content: vec![],
            provider_options: None,
        }];
        assert!(convert_completion_prompt(&prompt).is_err());
    }

    #[test]
    fn provider_options_merge_openai_base_then_namespace_override() {
        // Mirrors upstream merge semantics in
        // `openai-completion-language-model.ts:106-117`: parse "openai" first,
        // then layer the namespace-specific scope (e.g. "azure") on top.
        let mut po = llmsdk_provider::shared::ProviderOptions::new();
        let mut openai = serde_json::Map::new();
        openai.insert("user".into(), serde_json::json!("alice"));
        openai.insert("echo".into(), serde_json::json!(false));
        po.insert("openai".into(), openai);
        let mut azure = serde_json::Map::new();
        azure.insert("echo".into(), serde_json::json!(true));
        po.insert("azure".into(), azure);

        let opts = CallOptions {
            provider_options: Some(po),
            ..Default::default()
        };
        let parsed = parse_provider_options(&opts, "azure");
        assert_eq!(parsed.user.as_deref(), Some("alice"));
        assert_eq!(parsed.echo, Some(true));
    }

    #[test]
    fn provider_options_openai_only_namespace_returns_base() {
        let mut po = llmsdk_provider::shared::ProviderOptions::new();
        let mut openai = serde_json::Map::new();
        openai.insert("user".into(), serde_json::json!("bob"));
        po.insert("openai".into(), openai);
        let opts = CallOptions {
            provider_options: Some(po),
            ..Default::default()
        };
        let parsed = parse_provider_options(&opts, "openai");
        assert_eq!(parsed.user.as_deref(), Some("bob"));
    }
}
