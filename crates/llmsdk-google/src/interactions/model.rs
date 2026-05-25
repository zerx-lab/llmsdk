//! [`LanguageModel`] implementation for Gemini Interactions API.
//!
//! Mirrors `google-interactions-language-model.ts` plus its supporting
//! conversion / parsing / polling modules. The wire surface is large and
//! intentionally lenient (`.loose()` upstream); we type the fields the SDK
//! reads + writes and pass everything else through as `serde_json::Value`.
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    AssistantPart, CallOptions, Content, FilePart, FinishReason, FinishReasonKind,
    GenerateResponse, GenerateResult, InputTokenUsage, LanguageModel, Message, OutputTokenUsage,
    ReasoningPart, ResponseFormat, ResponseMetadata, Source, StreamPart, StreamResponse,
    StreamResult, TextPart, Tool, ToolCallPart, ToolMessagePart, ToolResultOutput, Usage, UserPart,
};
use llmsdk_provider::shared::{
    FileBytes, FileData, ProviderMetadata, ProviderOptions, RequestInfo, Warning,
};
use llmsdk_provider_utils::http::{
    JsonRequest, get_json, post_for_stream, post_json, response_byte_stream,
};
use llmsdk_provider_utils::sse::{SseEvent, sse_json_stream};
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value as JsonValue, json};

use crate::config::Inner;
use crate::error::rewrite_google_error;

const INTERACTIONS_API_REVISION: &str = "v1";
const INTERACTIONS_REVISION_HEADER: &str = "x-goog-api-revision";

const DEFAULT_INITIAL_DELAY_MS: u64 = 1000;
const DEFAULT_MAX_DELAY_MS: u64 = 10_000;
const DEFAULT_TIMEOUT_MS: u64 = 30 * 60 * 1000;

const TEXT_BLOCK_ID: &str = "text-0";

// -------- Public model handle ------------------------------------------

/// Either a Gemini model id (`"gemini-2.5-flash"`) or a managed agent name
/// (`"projects/.../agents/..."`). Mirrors the upstream `model | { agent } |
/// { managedAgent }` discriminated union.
#[derive(Debug, Clone)]
pub enum GoogleInteractionsAgent {
    /// Plain model id.
    Model(String),
    /// `{ agent: "<resource>" }` form.
    Agent(String),
    /// `{ managedAgent: "<resource>" }` form.
    ManagedAgent(String),
}

impl GoogleInteractionsAgent {
    fn id_for_provider(&self) -> &str {
        match self {
            Self::Model(s) | Self::Agent(s) | Self::ManagedAgent(s) => s,
        }
    }

    fn apply_to_request(&self, body: &mut JsonMap<String, JsonValue>) {
        match self {
            Self::Model(id) => {
                body.insert("model".into(), JsonValue::String(id.clone()));
            }
            Self::Agent(id) => {
                body.insert("agent".into(), JsonValue::String(id.clone()));
            }
            Self::ManagedAgent(id) => {
                body.insert("managed_agent".into(), JsonValue::String(id.clone()));
            }
        }
    }
}

/// Interaction lifecycle status returned by the API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoogleInteractionsStatus {
    /// Still running on the server.
    InProgress,
    /// Awaiting a tool / approval response from the client.
    RequiresAction,
    /// Finished successfully.
    Completed,
    /// Server-side failure.
    Failed,
    /// Cancelled by the client.
    Cancelled,
    /// Stopped before completion (token limit / safety).
    Incomplete,
}

impl GoogleInteractionsStatus {
    fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Cancelled | Self::Incomplete
        )
    }
}

/// `LanguageModel` implementation talking to `POST /v1beta/interactions`.
#[derive(Debug, Clone)]
pub struct GoogleInteractionsLanguageModel {
    inner: Arc<Inner>,
    agent: GoogleInteractionsAgent,
}

impl GoogleInteractionsLanguageModel {
    /// Construct from a fully assembled [`Inner`] plus model / agent identifier.
    #[must_use]
    pub fn new(inner: Arc<Inner>, agent: GoogleInteractionsAgent) -> Self {
        Self { inner, agent }
    }

    fn endpoint(&self) -> String {
        format!("{}/interactions", self.inner.base_url)
    }

    fn poll_endpoint(&self, id: &str) -> String {
        format!("{}/interactions/{id}", self.inner.base_url)
    }

    fn cancel_endpoint(&self, id: &str) -> String {
        format!("{}/interactions/{id}:cancel", self.inner.base_url)
    }

    fn add_revision_header(headers: &mut HashMap<String, Option<String>>) {
        headers
            .entry(INTERACTIONS_REVISION_HEADER.into())
            .or_insert_with(|| Some(INTERACTIONS_API_REVISION.to_owned()));
    }
}

#[async_trait]
impl LanguageModel for GoogleInteractionsLanguageModel {
    fn provider(&self) -> &str {
        &self.inner.provider
    }

    fn model_id(&self) -> &str {
        self.agent.id_for_provider()
    }

    async fn do_generate(&self, options: CallOptions) -> Result<GenerateResult, ProviderError> {
        let provider_opts = parse_provider_options(options.provider_options.as_ref());

        let (body, warnings) = build_request_body(&self.agent, &options, &provider_opts, false)?;
        let request_body_value = JsonValue::Object(body.clone());

        let mut headers = self.inner.headers.clone();
        if let Some(extra) = &options.headers {
            for (n, v) in extra {
                headers.insert(n.clone(), v.clone());
            }
        }
        Self::add_revision_header(&mut headers);

        let mut http_request = JsonRequest::new(self.endpoint(), JsonValue::Object(body));
        http_request.headers = headers.clone();

        let envelope = match post_json::<JsonValue, JsonValue>(&self.inner.http, http_request).await
        {
            Ok(r) => r,
            Err(err) => return Err(rewrite_google_error(err)),
        };

        let mut response = envelope.value;
        let mut response_headers = envelope.headers;
        let mut current_id = response
            .get("id")
            .and_then(JsonValue::as_str)
            .map(str::to_owned);

        // Background interactions return non-terminal — poll until terminal
        // (or until the configured timeout elapses).
        if let Some(id) = current_id.clone() {
            let status = response_status(&response);
            if status.is_some_and(|s| !s.is_terminal()) {
                let (polled, polled_headers) = poll_until_terminal(
                    &self.inner,
                    &self.poll_endpoint(&id),
                    &headers,
                    &provider_opts,
                )
                .await?;
                response = polled;
                response_headers = polled_headers;
                current_id = response
                    .get("id")
                    .and_then(JsonValue::as_str)
                    .map(str::to_owned);
            }
        }

        parse_response(
            response,
            response_headers,
            Some(request_body_value),
            current_id,
            warnings,
            self.model_id().to_owned(),
        )
    }

    async fn do_stream(&self, options: CallOptions) -> Result<StreamResult, ProviderError> {
        let provider_opts = parse_provider_options(options.provider_options.as_ref());
        let (body, warnings) = build_request_body(&self.agent, &options, &provider_opts, true)?;
        let request_body_value = JsonValue::Object(body.clone());

        let mut headers = self.inner.headers.clone();
        if let Some(extra) = &options.headers {
            for (n, v) in extra {
                headers.insert(n.clone(), v.clone());
            }
        }
        Self::add_revision_header(&mut headers);

        let mut http_request = JsonRequest::new(self.endpoint(), JsonValue::Object(body));
        http_request.headers = headers;

        let stream_response = match post_for_stream(&self.inner.http, http_request).await {
            Ok(r) => r,
            Err(err) => return Err(rewrite_google_error(err)),
        };
        let stream_headers = stream_response.headers.clone();
        let byte_stream = response_byte_stream(stream_response.response);
        let event_stream = sse_json_stream::<JsonValue>(byte_stream);

        let model_id = self.model_id().to_owned();
        let parts = drive_stream(warnings, model_id, event_stream);

        Ok(StreamResult {
            stream: Box::pin(parts),
            request: Some(RequestInfo {
                body: Some(request_body_value),
            }),
            response: Some(StreamResponse {
                headers: Some(headers_to_provider(stream_headers)),
            }),
        })
    }
}

// -------- Provider options --------------------------------------------

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct InteractionsProviderOptions {
    previous_interaction_id: Option<String>,
    store: Option<bool>,
    agent: Option<String>,
    agent_config: Option<JsonValue>,
    thinking_level: Option<String>,
    thinking_summaries: Option<String>,
    response_format: Option<JsonValue>,
    image_config: Option<JsonValue>,
    media_resolution: Option<String>,
    response_modalities: Option<Vec<String>>,
    service_tier: Option<String>,
    system_instruction: Option<String>,
    polling_timeout_ms: Option<u64>,
    background: Option<bool>,
    environment: Option<JsonValue>,
}

fn parse_provider_options(po: Option<&ProviderOptions>) -> InteractionsProviderOptions {
    let Some(map) = po else {
        return InteractionsProviderOptions::default();
    };
    let Some(slot) = map.get("google") else {
        return InteractionsProviderOptions::default();
    };
    serde_json::from_value::<InteractionsProviderOptions>(JsonValue::Object(slot.clone()))
        .unwrap_or_default()
}

// -------- Request build -----------------------------------------------

fn build_request_body(
    agent: &GoogleInteractionsAgent,
    options: &CallOptions,
    provider_opts: &InteractionsProviderOptions,
    stream: bool,
) -> Result<(JsonMap<String, JsonValue>, Vec<Warning>), ProviderError> {
    let mut body = JsonMap::new();
    let mut warnings = Vec::new();

    agent.apply_to_request(&mut body);
    if stream {
        body.insert("stream".into(), JsonValue::Bool(true));
    }

    // Generation config (max_output_tokens / temperature / top_p / top_k /
    // stop_sequences / seed / frequency_penalty / presence_penalty).
    let mut gen_config = JsonMap::new();
    if let Some(v) = options.max_output_tokens {
        gen_config.insert("max_output_tokens".into(), json!(v));
    }
    if let Some(v) = options.temperature {
        gen_config.insert("temperature".into(), json!(v));
    }
    if let Some(v) = options.top_p {
        gen_config.insert("top_p".into(), json!(v));
    }
    if let Some(v) = options.top_k {
        gen_config.insert("top_k".into(), json!(v));
    }
    if let Some(v) = options.frequency_penalty {
        gen_config.insert("frequency_penalty".into(), json!(v));
    }
    if let Some(v) = options.presence_penalty {
        gen_config.insert("presence_penalty".into(), json!(v));
    }
    if let Some(seq) = &options.stop_sequences {
        if !seq.is_empty() {
            gen_config.insert("stop_sequences".into(), json!(seq));
        }
    }
    if let Some(seed) = options.seed {
        gen_config.insert("seed".into(), json!(seed));
    }
    if let Some(level) = &provider_opts.thinking_level {
        gen_config.insert("thinking_level".into(), json!(level));
    }
    if let Some(summaries) = &provider_opts.thinking_summaries {
        gen_config.insert("thinking_summaries".into(), json!(summaries));
    }
    if let Some(mode) = &provider_opts.media_resolution {
        gen_config.insert("media_resolution".into(), json!(mode));
    }
    if let Some(modalities) = &provider_opts.response_modalities {
        gen_config.insert("response_modalities".into(), json!(modalities));
    }
    if let Some(tier) = &provider_opts.service_tier {
        gen_config.insert("service_tier".into(), json!(tier));
    }
    if !gen_config.is_empty() {
        body.insert("generation_config".into(), JsonValue::Object(gen_config));
    }

    if let Some(v) = &provider_opts.previous_interaction_id {
        body.insert("previous_interaction_id".into(), json!(v));
    }
    if let Some(v) = provider_opts.store {
        body.insert("store".into(), json!(v));
    }
    if let Some(v) = provider_opts.background {
        body.insert("background".into(), json!(v));
    }
    if let Some(v) = &provider_opts.agent_config {
        body.insert("agent_config".into(), v.clone());
    }
    if let Some(v) = &provider_opts.environment {
        body.insert("environment".into(), v.clone());
    }

    // response_format from provider options (multi-entry typed list).
    let mut response_format_entries: Vec<JsonValue> = provider_opts
        .response_format
        .as_ref()
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    // Auto-inject `{type: "text", schema, mime_type: "application/json"}` for
    // call-level JSON response_format mirroring upstream.
    if let Some(ResponseFormat::Json { schema, .. }) = &options.response_format {
        let mut entry = JsonMap::new();
        entry.insert("type".into(), JsonValue::String("text".into()));
        entry.insert(
            "mime_type".into(),
            JsonValue::String("application/json".into()),
        );
        if let Some(s) = schema {
            entry.insert(
                "schema".into(),
                serde_json::to_value(s).unwrap_or(JsonValue::Null),
            );
        }
        response_format_entries.push(JsonValue::Object(entry));
    }

    // Deprecated imageConfig fallback (one image entry, warned).
    if let Some(image_config) = &provider_opts.image_config {
        let has_image = response_format_entries.iter().any(|e| {
            e.get("type")
                .and_then(JsonValue::as_str)
                .map(|t| t == "image")
                .unwrap_or(false)
        });
        warnings.push(Warning::Other {
            message: "provider_options.google.imageConfig is deprecated; use responseFormat with a {type:'image', ...} entry instead".to_owned(),
        });
        if !has_image {
            let mut entry = JsonMap::new();
            entry.insert("type".into(), JsonValue::String("image".into()));
            if let Some(ar) = image_config.get("aspectRatio") {
                entry.insert("aspect_ratio".into(), ar.clone());
            }
            if let Some(sz) = image_config.get("imageSize") {
                entry.insert("image_size".into(), sz.clone());
            }
            response_format_entries.push(JsonValue::Object(entry));
        }
    }

    if !response_format_entries.is_empty() {
        // ai-sdk uses snake_case `aspect_ratio`/`image_size`/`mime_type` —
        // normalize camelCase fields to snake_case in the typed entries we
        // emit. (Loose `.loose()` entries from provider options are kept
        // verbatim, matching the upstream pass-through behavior.)
        for entry in &mut response_format_entries {
            if let Some(obj) = entry.as_object_mut() {
                if let Some(v) = obj.remove("aspectRatio") {
                    obj.insert("aspect_ratio".into(), v);
                }
                if let Some(v) = obj.remove("imageSize") {
                    obj.insert("image_size".into(), v);
                }
                if let Some(v) = obj.remove("mimeType") {
                    obj.insert("mime_type".into(), v);
                }
            }
        }
        body.insert(
            "response_format".into(),
            JsonValue::Array(response_format_entries),
        );
    }

    // System instruction: SDK-level system message wins; provider option is
    // used only if no system message is present.
    let mut sdk_system_msg: Option<String> = None;
    for msg in &options.prompt {
        if let Message::System { content, .. } = msg {
            if sdk_system_msg.is_some() {
                warnings.push(Warning::Other {
                    message: "multiple system messages collapsed into the first".to_owned(),
                });
            }
            sdk_system_msg.get_or_insert(content.clone());
        }
    }
    if sdk_system_msg.is_some() && provider_opts.system_instruction.is_some() {
        warnings.push(Warning::Other {
            message: "provider_options.google.systemInstruction ignored: SDK system message takes precedence".to_owned(),
        });
    }
    if let Some(sys) = sdk_system_msg.or_else(|| provider_opts.system_instruction.clone()) {
        body.insert("system_instruction".into(), JsonValue::String(sys));
    }

    // Tools / tool_choice (function tools only — typed provider-defined tools
    // are not exposed on the Interactions surface in this minimal mapping).
    if let Some(tools) = &options.tools {
        let mut wire_tools = Vec::new();
        for t in tools {
            match t {
                Tool::Function(f) => {
                    let mut entry = JsonMap::new();
                    entry.insert("type".into(), JsonValue::String("function".into()));
                    entry.insert(
                        "function".into(),
                        json!({
                            "name": f.name,
                            "description": f.description,
                            "parameters": serde_json::to_value(&f.input_schema)
                                .unwrap_or(JsonValue::Null),
                        }),
                    );
                    wire_tools.push(JsonValue::Object(entry));
                }
                Tool::Provider(_) => {
                    warnings.push(Warning::Other {
                        message:
                            "provider-defined tools are not yet routed for Google Interactions"
                                .to_owned(),
                    });
                }
            }
        }
        if !wire_tools.is_empty() {
            body.insert("tools".into(), JsonValue::Array(wire_tools));
        }
    }

    // Input messages (everything after the system message).
    let input = convert_prompt_to_input(&options.prompt, &mut warnings);
    body.insert("input".into(), JsonValue::Array(input));

    Ok((body, warnings))
}

/// Convert llmsdk Prompt -> Interactions `input[]`. Each non-system message
/// becomes one entry with `role` + typed `content` array.
fn convert_prompt_to_input(prompt: &[Message], warnings: &mut Vec<Warning>) -> Vec<JsonValue> {
    let mut out = Vec::new();
    for msg in prompt {
        match msg {
            Message::System { .. } => {} // emitted via system_instruction.
            Message::User { content, .. } => {
                let mut blocks = Vec::new();
                for part in content {
                    match part {
                        UserPart::Text(TextPart { text, .. }) => {
                            blocks.push(json!({"type": "text", "text": text}));
                        }
                        UserPart::File(f) => blocks.push(file_to_block(f, warnings)),
                    }
                }
                out.push(json!({ "role": "user", "content": blocks }));
            }
            Message::Assistant { content, .. } => {
                let mut blocks = Vec::new();
                let mut tool_calls: Vec<JsonValue> = Vec::new();
                for part in content {
                    match part {
                        AssistantPart::Text(TextPart { text, .. }) => {
                            blocks.push(json!({"type": "text", "text": text}));
                        }
                        AssistantPart::Reasoning {
                            text,
                            provider_options,
                            ..
                        } => {
                            let signature = provider_options
                                .as_ref()
                                .and_then(|po| po.get("google"))
                                .and_then(|g| g.get("signature"))
                                .and_then(JsonValue::as_str)
                                .map(str::to_owned);
                            let mut t = JsonMap::new();
                            t.insert("type".into(), JsonValue::String("thought".into()));
                            t.insert("summary".into(), json!([{ "type": "text", "text": text }]));
                            if let Some(sig) = signature {
                                t.insert("signature".into(), JsonValue::String(sig));
                            }
                            blocks.push(JsonValue::Object(t));
                        }
                        AssistantPart::ToolCall(tc) => {
                            let mut entry = JsonMap::new();
                            entry.insert("type".into(), JsonValue::String("function_call".into()));
                            entry.insert("id".into(), JsonValue::String(tc.tool_call_id.clone()));
                            entry.insert("name".into(), JsonValue::String(tc.tool_name.clone()));
                            entry.insert("arguments".into(), tc.input.clone());
                            if let Some(sig) = tc
                                .provider_options
                                .as_ref()
                                .and_then(|po| po.get("google"))
                                .and_then(|g| g.get("signature"))
                                .and_then(JsonValue::as_str)
                            {
                                entry.insert("signature".into(), JsonValue::String(sig.to_owned()));
                            }
                            tool_calls.push(JsonValue::Object(entry));
                        }
                        AssistantPart::File(_)
                        | AssistantPart::ReasoningFile { .. }
                        | AssistantPart::ToolResult(_)
                        | AssistantPart::Custom { .. } => {
                            warnings.push(Warning::Other {
                                message:
                                    "unsupported assistant part skipped for Google Interactions"
                                        .to_owned(),
                            });
                        }
                    }
                }
                let mut entry = JsonMap::new();
                entry.insert("role".into(), JsonValue::String("model".into()));
                entry.insert("content".into(), JsonValue::Array(blocks));
                if !tool_calls.is_empty() {
                    entry.insert("tool_calls".into(), JsonValue::Array(tool_calls));
                }
                out.push(JsonValue::Object(entry));
            }
            Message::Tool { content, .. } => {
                for part in content {
                    if let ToolMessagePart::ToolResult(r) = part {
                        let result_value = match &r.output {
                            ToolResultOutput::Text { value, .. }
                            | ToolResultOutput::ErrorText { value, .. } => {
                                JsonValue::String(value.clone())
                            }
                            ToolResultOutput::Json { value, .. }
                            | ToolResultOutput::ErrorJson { value, .. } => value.clone(),
                            ToolResultOutput::ExecutionDenied { reason, .. } => {
                                JsonValue::String(reason.clone().unwrap_or_default())
                            }
                            ToolResultOutput::Content { .. } => JsonValue::Null,
                        };
                        let is_error = matches!(
                            r.output,
                            ToolResultOutput::ErrorText { .. } | ToolResultOutput::ErrorJson { .. }
                        );
                        out.push(json!({
                            "role": "tool",
                            "content": [
                                {
                                    "type": "function_result",
                                    "call_id": r.tool_call_id,
                                    "result": result_value,
                                    "is_error": is_error,
                                }
                            ],
                        }));
                    }
                }
            }
        }
    }
    out
}

fn file_to_block(file: &FilePart, warnings: &mut Vec<Warning>) -> JsonValue {
    let top_level = file
        .media_type
        .split('/')
        .next()
        .unwrap_or(&file.media_type);
    let kind = match top_level {
        "image" => "image",
        "audio" => "audio",
        "video" => "video",
        _ => "document",
    };
    let mut entry = JsonMap::new();
    entry.insert("type".into(), JsonValue::String(kind.into()));
    entry.insert(
        "mime_type".into(),
        JsonValue::String(file.media_type.clone()),
    );
    match &file.data {
        FileData::Url { url } => {
            entry.insert("uri".into(), JsonValue::String(url.clone()));
        }
        FileData::Data { data } => {
            let payload = match data {
                FileBytes::Base64(s) => s.clone(),
                FileBytes::Bytes(_) => {
                    warnings.push(Warning::Other {
                        message: "raw bytes file part requires base64 — encode before passing"
                            .to_owned(),
                    });
                    return JsonValue::Object(entry);
                }
            };
            entry.insert("data".into(), JsonValue::String(payload));
        }
        FileData::Text { text } => {
            entry.insert("data".into(), JsonValue::String(text.clone()));
        }
        FileData::Reference { reference } => {
            if let Some(uri) = reference.get("google").and_then(JsonValue::as_str) {
                entry.insert("uri".into(), JsonValue::String(uri.to_owned()));
            } else {
                warnings.push(Warning::Other {
                    message: "file reference missing `google` resolver — dropped".to_owned(),
                });
            }
        }
    }
    JsonValue::Object(entry)
}

// -------- Response parsing --------------------------------------------

fn response_status(response: &JsonValue) -> Option<GoogleInteractionsStatus> {
    response
        .get("status")
        .and_then(JsonValue::as_str)
        .and_then(|s| match s {
            "in_progress" => Some(GoogleInteractionsStatus::InProgress),
            "requires_action" => Some(GoogleInteractionsStatus::RequiresAction),
            "completed" => Some(GoogleInteractionsStatus::Completed),
            "failed" => Some(GoogleInteractionsStatus::Failed),
            "cancelled" => Some(GoogleInteractionsStatus::Cancelled),
            "incomplete" => Some(GoogleInteractionsStatus::Incomplete),
            _ => None,
        })
}

fn map_finish_reason(status: Option<GoogleInteractionsStatus>) -> FinishReason {
    let (kind, raw) = match status {
        Some(GoogleInteractionsStatus::Completed) => (FinishReasonKind::Stop, "completed"),
        Some(GoogleInteractionsStatus::Failed) => (FinishReasonKind::Error, "failed"),
        Some(GoogleInteractionsStatus::Cancelled) => (FinishReasonKind::Other, "cancelled"),
        Some(GoogleInteractionsStatus::Incomplete) => (FinishReasonKind::Length, "incomplete"),
        Some(GoogleInteractionsStatus::RequiresAction) => {
            (FinishReasonKind::ToolCalls, "requires_action")
        }
        _ => (FinishReasonKind::Other, "in_progress"),
    };
    FinishReason {
        unified: kind,
        raw: Some(raw.to_owned()),
    }
}

fn parse_response(
    response: JsonValue,
    headers: HashMap<String, String>,
    request_body: Option<JsonValue>,
    response_id: Option<String>,
    warnings: Vec<Warning>,
    model_id: String,
) -> Result<GenerateResult, ProviderError> {
    let status = response_status(&response);
    let finish_reason = map_finish_reason(status);
    let usage = parse_usage(response.get("usage"));

    let mut content: Vec<Content> = Vec::new();
    if let Some(steps) = response.get("steps").and_then(JsonValue::as_array).cloned() {
        for step in steps {
            translate_step(&step, &mut content);
        }
    }

    let mut provider_metadata = ProviderMetadata::new();
    let mut google_meta = JsonMap::new();
    if let Some(id) = response_id.clone() {
        google_meta.insert("interactionId".into(), JsonValue::String(id));
    }
    if let Some(status_str) = response.get("status").cloned() {
        google_meta.insert("status".into(), status_str);
    }
    if let Some(modes) = response.get("response_modalities").cloned() {
        google_meta.insert("responseModalities".into(), modes);
    }
    if let Some(tier) = response.get("service_tier").cloned() {
        google_meta.insert("serviceTier".into(), tier);
    }
    if let Some(prev) = response.get("previous_interaction_id").cloned() {
        google_meta.insert("previousInteractionId".into(), prev);
    }
    if !google_meta.is_empty() {
        provider_metadata.insert("google".to_owned(), google_meta);
    }

    let response_meta = ResponseMetadata {
        id: response_id,
        timestamp: response
            .get("created")
            .and_then(JsonValue::as_str)
            .map(str::to_owned),
        model_id: Some(model_id),
        headers: Some(headers_to_provider(headers)),
    };

    Ok(GenerateResult {
        content,
        finish_reason,
        usage,
        provider_metadata: (!provider_metadata.is_empty()).then_some(provider_metadata),
        warnings,
        request: Some(RequestInfo { body: request_body }),
        response: Some(GenerateResponse {
            metadata: response_meta,
            body: None,
        }),
    })
}

fn translate_step(step: &JsonValue, out: &mut Vec<Content>) {
    let Some(step_type) = step.get("type").and_then(JsonValue::as_str) else {
        return;
    };
    match step_type {
        "model_output" => {
            if let Some(blocks) = step.get("content").and_then(JsonValue::as_array) {
                for block in blocks {
                    translate_block(block, out);
                }
            }
        }
        "function_call" => {
            let id = step.get("id").and_then(JsonValue::as_str).unwrap_or("");
            let name = step.get("name").and_then(JsonValue::as_str).unwrap_or("");
            let arguments = step.get("arguments").cloned().unwrap_or(JsonValue::Null);
            let mut po: Option<ProviderOptions> = None;
            if let Some(sig) = step.get("signature").and_then(JsonValue::as_str) {
                let mut google = JsonMap::new();
                google.insert("signature".into(), JsonValue::String(sig.to_owned()));
                let mut map = ProviderOptions::new();
                map.insert("google".to_owned(), google);
                po = Some(map);
            }
            out.push(Content::ToolCall(ToolCallPart {
                tool_call_id: id.to_owned(),
                tool_name: name.to_owned(),
                input: arguments,
                provider_executed: None,
                dynamic: None,
                provider_options: po,
            }));
        }
        "thought" => {
            let mut text = String::new();
            if let Some(items) = step.get("summary").and_then(JsonValue::as_array) {
                for item in items {
                    if let Some(t) = item.get("text").and_then(JsonValue::as_str) {
                        text.push_str(t);
                    }
                }
            }
            let mut po: Option<ProviderOptions> = None;
            if let Some(sig) = step.get("signature").and_then(JsonValue::as_str) {
                let mut google = JsonMap::new();
                google.insert("signature".into(), JsonValue::String(sig.to_owned()));
                let mut map = ProviderOptions::new();
                map.insert("google".to_owned(), google);
                po = Some(map);
            }
            out.push(Content::Reasoning(ReasoningPart {
                text,
                provider_options: po,
            }));
        }
        // user_input echo on GET is ignored.
        "user_input" => {}
        // Built-in tool steps surfaced via Content::Source for citations
        // where possible; otherwise dropped silently (mirrors upstream's
        // pass-through behavior).
        _ => {}
    }
}

fn translate_block(block: &JsonValue, out: &mut Vec<Content>) {
    let Some(kind) = block.get("type").and_then(JsonValue::as_str) else {
        return;
    };
    if kind == "text" {
        let Some(text) = block.get("text").and_then(JsonValue::as_str) else {
            return;
        };
        out.push(Content::Text(TextPart {
            text: text.to_owned(),
            provider_options: None,
        }));
        if let Some(annotations) = block.get("annotations").and_then(JsonValue::as_array) {
            for ann in annotations {
                if let Some(source) = annotation_to_source(ann) {
                    out.push(Content::Source(source));
                }
            }
        }
    }
    // Image / audio / video blocks could be mapped to Content::File when
    // the SDK exposes them on outputs; left as a TODO since llmsdk's
    // assistant-side File variant is not yet wired across providers.
}

fn annotation_to_source(ann: &JsonValue) -> Option<Source> {
    let kind = ann.get("type").and_then(JsonValue::as_str)?;
    match kind {
        "url_citation" => {
            let url = ann.get("url").and_then(JsonValue::as_str)?.to_owned();
            let title = ann
                .get("title")
                .and_then(JsonValue::as_str)
                .map(str::to_owned);
            Some(Source::Url {
                id: url.clone(),
                url,
                title,
                provider_metadata: None,
            })
        }
        "file_citation" => {
            let id = ann
                .get("media_id")
                .or_else(|| ann.get("document_uri"))
                .and_then(JsonValue::as_str)?
                .to_owned();
            let media_type = ann
                .get("file_name")
                .and_then(JsonValue::as_str)
                .map(|_| "application/octet-stream".to_owned())
                .unwrap_or_else(|| "application/octet-stream".to_owned());
            let filename = ann
                .get("file_name")
                .and_then(JsonValue::as_str)
                .map(str::to_owned);
            Some(Source::Document {
                id,
                media_type,
                title: filename.unwrap_or_else(|| "document".to_owned()),
                filename: None,
                provider_metadata: None,
            })
        }
        _ => None,
    }
}

fn parse_usage(value: Option<&JsonValue>) -> Usage {
    let Some(u) = value else {
        return Usage::default();
    };
    let input = u.get("total_input_tokens").and_then(JsonValue::as_u64);
    let output = u.get("total_output_tokens").and_then(JsonValue::as_u64);
    let cached = u.get("total_cached_tokens").and_then(JsonValue::as_u64);
    let reasoning = u.get("total_thought_tokens").and_then(JsonValue::as_u64);
    Usage {
        input_tokens: InputTokenUsage {
            total: input,
            cache_read: cached,
            ..InputTokenUsage::default()
        },
        output_tokens: OutputTokenUsage {
            total: output,
            text: output,
            reasoning,
        },
        raw: None,
    }
}

// -------- Polling -----------------------------------------------------

async fn poll_until_terminal(
    inner: &Inner,
    poll_url: &str,
    headers: &HashMap<String, Option<String>>,
    provider_opts: &InteractionsProviderOptions,
) -> Result<(JsonValue, HashMap<String, String>), ProviderError> {
    let timeout_ms = provider_opts
        .polling_timeout_ms
        .unwrap_or(DEFAULT_TIMEOUT_MS);
    let started = std::time::Instant::now();
    let mut delay = DEFAULT_INITIAL_DELAY_MS;
    let max_delay = DEFAULT_MAX_DELAY_MS;

    loop {
        if started.elapsed().as_millis() > u128::from(timeout_ms) {
            return Err(ProviderError::api_call_builder(
                poll_url,
                format!("Google Interactions poll timed out after {timeout_ms} ms"),
            )
            .build());
        }
        tokio::time::sleep(Duration::from_millis(delay)).await;

        let response = match get_json::<JsonValue, _>(&inner.http, poll_url, headers).await {
            Ok(r) => r,
            Err(err) => return Err(rewrite_google_error(err)),
        };
        let status = response_status(&response.value);
        if status.is_some_and(GoogleInteractionsStatus::is_terminal) {
            return Ok((response.value, response.headers));
        }
        delay = (delay * 2).min(max_delay);
    }
}

// -------- Cancel ------------------------------------------------------

impl GoogleInteractionsLanguageModel {
    /// Cancel an in-flight (background) interaction by id. Returns the final
    /// server-side state (which may still be non-terminal if cancellation
    /// is observed asynchronously).
    ///
    /// # Errors
    ///
    /// Propagates [`ProviderError`] on transport / HTTP failure.
    pub async fn cancel(&self, interaction_id: &str) -> Result<JsonValue, ProviderError> {
        let url = self.cancel_endpoint(interaction_id);
        let mut headers = self.inner.headers.clone();
        Self::add_revision_header(&mut headers);
        let mut http_request = JsonRequest::new(url, JsonValue::Object(JsonMap::new()));
        http_request.headers = headers;
        match post_json::<JsonValue, JsonValue>(&self.inner.http, http_request).await {
            Ok(r) => Ok(r.value),
            Err(err) => Err(rewrite_google_error(err)),
        }
    }
}

// -------- Stream forwarding ------------------------------------------

fn drive_stream<S>(
    warnings: Vec<Warning>,
    model_id: String,
    events: S,
) -> impl futures::Stream<Item = Result<StreamPart, ProviderError>> + Send
where
    S: futures::Stream<Item = Result<SseEvent<JsonValue>, ProviderError>> + Send + 'static,
{
    async_stream::stream! {
        yield Ok(StreamPart::StreamStart { warnings });

        let mut text_open = false;
        let mut metadata_emitted = false;
        let mut last_status: Option<GoogleInteractionsStatus> = None;
        let mut last_usage: Option<JsonValue> = None;
        let mut events = Box::pin(events);

        while let Some(event) = events.next().await {
            match event {
                Ok(SseEvent::Data(value)) => {
                    let event_type = value
                        .get("event_type")
                        .and_then(JsonValue::as_str)
                        .unwrap_or_default()
                        .to_owned();

                    if !metadata_emitted {
                        let id = value
                            .pointer("/interaction/id")
                            .and_then(JsonValue::as_str)
                            .map(str::to_owned);
                        if id.is_some() {
                            metadata_emitted = true;
                            yield Ok(StreamPart::ResponseMetadata(ResponseMetadata {
                                id,
                                timestamp: None,
                                model_id: Some(model_id.clone()),
                                headers: None,
                            }));
                        }
                    }

                    if let Some(s) = value
                        .pointer("/interaction/status")
                        .and_then(JsonValue::as_str)
                        .map(str::to_owned)
                        .and_then(|s| match s.as_str() {
                            "completed" => Some(GoogleInteractionsStatus::Completed),
                            "failed" => Some(GoogleInteractionsStatus::Failed),
                            "cancelled" => Some(GoogleInteractionsStatus::Cancelled),
                            "incomplete" => Some(GoogleInteractionsStatus::Incomplete),
                            "in_progress" => Some(GoogleInteractionsStatus::InProgress),
                            "requires_action" => Some(GoogleInteractionsStatus::RequiresAction),
                            _ => None,
                        })
                    {
                        last_status = Some(s);
                    }
                    if let Some(u) = value.pointer("/interaction/usage").cloned() {
                        last_usage = Some(u);
                    }

                    match event_type.as_str() {
                        "content.delta" => {
                            if let Some(delta) = value
                                .pointer("/delta/text")
                                .and_then(JsonValue::as_str)
                                .filter(|s| !s.is_empty())
                            {
                                if !text_open {
                                    text_open = true;
                                    yield Ok(StreamPart::TextStart {
                                        id: TEXT_BLOCK_ID.to_owned(),
                                        provider_metadata: None,
                                    });
                                }
                                yield Ok(StreamPart::TextDelta {
                                    id: TEXT_BLOCK_ID.to_owned(),
                                    delta: delta.to_owned(),
                                    provider_metadata: None,
                                });
                            }
                        }
                        "step.done" => {
                            if text_open {
                                text_open = false;
                                yield Ok(StreamPart::TextEnd {
                                    id: TEXT_BLOCK_ID.to_owned(),
                                    provider_metadata: None,
                                });
                            }
                            if let Some(step) = value.get("step") {
                                let mut content_buf = Vec::new();
                                translate_step(step, &mut content_buf);
                                for item in content_buf {
                                    if let Content::ToolCall(tc) = item {
                                        yield Ok(StreamPart::ToolCall(tc));
                                    }
                                }
                            }
                        }
                        "interaction.completed"
                        | "interaction.failed"
                        | "interaction.cancelled"
                        | "interaction.incomplete"
                            if text_open =>
                        {
                            text_open = false;
                            yield Ok(StreamPart::TextEnd {
                                id: TEXT_BLOCK_ID.to_owned(),
                                provider_metadata: None,
                            });
                        }
                        _ => {}
                    }
                }
                Ok(SseEvent::ParseError { raw, message }) => {
                    yield Ok(StreamPart::Error {
                        error: json!({ "message": message, "raw": raw }),
                    });
                }
                Err(e) => {
                    yield Err(e);
                    return;
                }
            }
        }

        if text_open {
            yield Ok(StreamPart::TextEnd {
                id: TEXT_BLOCK_ID.to_owned(),
                provider_metadata: None,
            });
        }

        yield Ok(StreamPart::Finish {
            finish_reason: map_finish_reason(last_status),
            usage: parse_usage(last_usage.as_ref()),
            provider_metadata: None,
        });
    }
}

// -------- Helpers ----------------------------------------------------

fn headers_to_provider(raw: HashMap<String, String>) -> llmsdk_provider::shared::Headers {
    raw.into_iter().map(|(k, v)| (k, Some(v))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_routes_id_into_request_body() {
        let mut body = JsonMap::new();
        GoogleInteractionsAgent::Model("gemini-2.5-flash".into()).apply_to_request(&mut body);
        assert_eq!(
            body.get("model").and_then(JsonValue::as_str),
            Some("gemini-2.5-flash")
        );

        let mut body = JsonMap::new();
        GoogleInteractionsAgent::Agent("agents/foo".into()).apply_to_request(&mut body);
        assert_eq!(
            body.get("agent").and_then(JsonValue::as_str),
            Some("agents/foo")
        );

        let mut body = JsonMap::new();
        GoogleInteractionsAgent::ManagedAgent("managed/bar".into()).apply_to_request(&mut body);
        assert_eq!(
            body.get("managed_agent").and_then(JsonValue::as_str),
            Some("managed/bar")
        );
    }

    #[test]
    fn status_terminal_check() {
        assert!(!GoogleInteractionsStatus::InProgress.is_terminal());
        assert!(GoogleInteractionsStatus::Completed.is_terminal());
        assert!(GoogleInteractionsStatus::Failed.is_terminal());
        assert!(GoogleInteractionsStatus::Incomplete.is_terminal());
    }

    #[test]
    fn finish_reason_mapping() {
        let fr = map_finish_reason(Some(GoogleInteractionsStatus::Completed));
        assert_eq!(fr.unified, FinishReasonKind::Stop);
        let fr = map_finish_reason(Some(GoogleInteractionsStatus::Incomplete));
        assert_eq!(fr.unified, FinishReasonKind::Length);
        let fr = map_finish_reason(Some(GoogleInteractionsStatus::RequiresAction));
        assert_eq!(fr.unified, FinishReasonKind::ToolCalls);
    }

    #[test]
    fn parse_usage_handles_missing_fields() {
        let u = parse_usage(Some(
            &json!({"total_input_tokens": 10, "total_output_tokens": 5}),
        ));
        assert_eq!(u.input_tokens.total, Some(10));
        assert_eq!(u.output_tokens.total, Some(5));
    }

    #[test]
    fn url_citation_becomes_source() {
        let ann = json!({"type": "url_citation", "url": "https://x", "title": "X"});
        let src = annotation_to_source(&ann).expect("source");
        if let Source::Url { url, title, .. } = src {
            assert_eq!(url, "https://x");
            assert_eq!(title.as_deref(), Some("X"));
        } else {
            panic!("expected url source");
        }
    }
}
