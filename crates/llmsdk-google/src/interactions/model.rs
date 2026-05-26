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
use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    AssistantPart, CallOptions, Content, FilePart, FinishReason, FinishReasonKind,
    GenerateResponse, GenerateResult, InputTokenUsage, LanguageModel, Message, OutputTokenUsage,
    ReasoningPart, ResponseFormat, ResponseMetadata, StreamResponse, StreamResult, TextPart,
    ToolCallPart, ToolMessagePart, ToolResultOutput, Usage, UserPart,
};
use llmsdk_provider::shared::{
    FileBytes, FileData, ProviderMetadata, ProviderOptions, RequestInfo, Warning,
};
use llmsdk_provider_utils::http::{
    JsonRequest, get_json, post_for_stream, post_json, response_byte_stream,
};
use llmsdk_provider_utils::sse::sse_json_stream;
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value as JsonValue, json};

use crate::config::Inner;
use crate::error::rewrite_google_error;

// Mirrors upstream `google-interactions-language-model.ts:761-763` —
// `'Api-Revision': '2026-05-20'`. Pins the wire schema the SDK targets.
const INTERACTIONS_API_REVISION: &str = "2026-05-20";
const INTERACTIONS_REVISION_HEADER: &str = "Api-Revision";

const DEFAULT_INITIAL_DELAY_MS: u64 = 1000;
const DEFAULT_MAX_DELAY_MS: u64 = 10_000;
const DEFAULT_TIMEOUT_MS: u64 = 30 * 60 * 1000;

/// Percent-encode a string for use in a URL path segment. Encodes anything
/// outside the RFC 3986 `unreserved` set (`ALPHA / DIGIT / - . _ ~`). Mirrors
/// upstream's `encodeURIComponent(interactionId)`.
fn percent_encode_path_segment(input: &str) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(input.len());
    for b in input.as_bytes() {
        let ch = *b;
        if ch.is_ascii_alphanumeric() || ch == b'-' || ch == b'_' || ch == b'.' || ch == b'~' {
            out.push(ch as char);
        } else {
            // Writing into an existing `String` via `write!` avoids the
            // allocation `format!(..)` does for the percent encoding.
            let _ = write!(out, "%{ch:02X}");
        }
    }
    out
}

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

    /// True when this handle targets an agent (builtin or managed), false
    /// when it targets a plain model id. Drives the same request-body branch
    /// as upstream's `isAgent` flag — agent calls must use `agent_config`
    /// instead of `generation_config`, cannot send `tools` or `responseFormat`,
    /// and require `background: true`.
    #[must_use]
    pub fn is_agent(&self) -> bool {
        matches!(self, Self::Agent(_) | Self::ManagedAgent(_))
    }

    fn apply_to_request(&self, body: &mut JsonMap<String, JsonValue>) {
        match self {
            Self::Model(id) => {
                body.insert("model".into(), JsonValue::String(id.clone()));
            }
            // Mirrors upstream `google-interactions-language-model.ts:112-117`:
            // both `{agent}` and `{managedAgent}` forms route to the wire
            // `agent` field; the API does not have a `managed_agent` field.
            Self::Agent(id) | Self::ManagedAgent(id) => {
                body.insert("agent".into(), JsonValue::String(id.clone()));
            }
        }
    }
}

/// Built-in agent names declared by the Interactions API. Mirrors upstream
/// `google-interactions-agent.ts`'s string-literal union — passing one of
/// these into [`GoogleInteractionsAgent::Agent`] targets a hosted agent
/// without needing to type the literal yourself.
pub mod builtin_agent {
    /// Deep Research (December 2025 preview, "pro" tier).
    pub const DEEP_RESEARCH_PRO_PREVIEW_12_2025: &str = "deep-research-pro-preview-12-2025";
    /// Deep Research (April 2026 preview).
    pub const DEEP_RESEARCH_PREVIEW_04_2026: &str = "deep-research-preview-04-2026";
    /// Deep Research (April 2026 preview, "max" tier).
    pub const DEEP_RESEARCH_MAX_PREVIEW_04_2026: &str = "deep-research-max-preview-04-2026";
    /// Antigravity (May 2026 preview).
    pub const ANTIGRAVITY_PREVIEW_05_2026: &str = "antigravity-preview-05-2026";
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
        format!(
            "{}/interactions/{}",
            self.inner.base_url,
            percent_encode_path_segment(id)
        )
    }

    // Mirrors upstream `cancel-google-interaction.ts:34` — path segment is
    // `/cancel`, not the GCP-style `:cancel` verb.
    fn cancel_endpoint(&self, id: &str) -> String {
        format!(
            "{}/interactions/{}/cancel",
            self.inner.base_url,
            percent_encode_path_segment(id)
        )
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
        let is_background = provider_opts.background == Some(true);

        // `background: true` is incompatible with `stream: true` on POST
        // (mirrors upstream `doStreamBackground`). Drive agent calls via
        // POST background → poll until terminal → synthesize the polled
        // response as a deterministic stream sequence.
        if is_background {
            return self.do_stream_background(options, provider_opts).await;
        }

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
        // Mirrors upstream `doStream` reading `x-gemini-service-tier` as a
        // defensive fallback while the body event remains primary
        // (`interaction.completed.service_tier`).
        let header_service_tier = stream_headers.get("x-gemini-service-tier").cloned();
        let byte_stream = response_byte_stream(stream_response.response);
        let event_stream = sse_json_stream::<JsonValue>(byte_stream);

        let model_id = self.model_id().to_owned();
        let parts =
            super::stream::drive_stream(warnings, model_id, header_service_tier, event_stream);

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

impl GoogleInteractionsLanguageModel {
    /// Drive `do_stream` for background-mode interactions (required by agent
    /// calls). Mirrors upstream `doStreamBackground` minus the live GET-SSE
    /// reconnect loop: we POST background, poll until terminal, then
    /// synthesize the polled response as a deterministic stream.
    async fn do_stream_background(
        &self,
        options: CallOptions,
        provider_opts: InteractionsProviderOptions,
    ) -> Result<StreamResult, ProviderError> {
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

        let id = response
            .get("id")
            .and_then(JsonValue::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_owned);
        if id.is_none() {
            return Err(ProviderError::api_call_builder(
                self.endpoint(),
                "background POST response did not include an interaction id; cannot stream",
            )
            .build());
        }
        let id = id.expect("checked above");

        // Poll the run until terminal so we have a complete payload to
        // synthesize from.
        let initial_status = response_status(&response);
        if initial_status.is_some_and(|s| !s.is_terminal()) {
            let (polled, polled_headers) = poll_until_terminal(
                &self.inner,
                &self.poll_endpoint(&id),
                &headers,
                &provider_opts,
            )
            .await?;
            response = polled;
            response_headers = polled_headers;
        }

        let header_service_tier = response_headers.get("x-gemini-service-tier").cloned();
        let model_id = self.model_id().to_owned();
        let parts = super::synthesize_stream::synthesize_response_to_stream(
            response,
            warnings,
            model_id,
            header_service_tier,
        );

        Ok(StreamResult {
            stream: Box::pin(parts),
            request: Some(RequestInfo {
                body: Some(request_body_value),
            }),
            response: Some(StreamResponse {
                headers: Some(headers_to_provider(response_headers)),
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

/// Rewrite agent_config field names from `camelCase` (ai-sdk wire-format
/// from JS callers) to `snake_case` (Interactions API on-wire). Mirrors
/// the explicit mapping in
/// `google-interactions-language-model.ts:345-358`.
fn normalize_agent_config(value: JsonValue) -> JsonValue {
    let JsonValue::Object(map) = value else {
        return value;
    };
    let mut out = JsonMap::new();
    for (key, value) in map {
        let renamed = match key.as_str() {
            "thinkingSummaries" => "thinking_summaries".to_owned(),
            "collaborativePlanning" => "collaborative_planning".to_owned(),
            // `type`, `visualization`, and any other snake_case key pass through.
            _ => key,
        };
        out.insert(renamed, value);
    }
    JsonValue::Object(out)
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

    let is_agent = agent.is_agent();

    // Generation config (max_output_tokens / temperature / top_p / top_k /
    // stop_sequences / seed / frequency_penalty / presence_penalty).
    // When an agent is set, the API rejects these fields — drop them and
    // emit a single warning naming each dropped field, matching ai-sdk's
    // `google-interactions-language-model.ts:277-298`.
    if is_agent {
        let mut dropped: Vec<&str> = Vec::new();
        if options.temperature.is_some() {
            dropped.push("temperature");
        }
        if options.top_p.is_some() {
            dropped.push("topP");
        }
        if options.seed.is_some() {
            dropped.push("seed");
        }
        if options
            .stop_sequences
            .as_ref()
            .is_some_and(|s| !s.is_empty())
        {
            dropped.push("stopSequences");
        }
        if options.max_output_tokens.is_some() {
            dropped.push("maxOutputTokens");
        }
        if provider_opts.thinking_level.is_some() {
            dropped.push("thinkingLevel");
        }
        if provider_opts.thinking_summaries.is_some() {
            dropped.push("thinkingSummaries");
        }
        if provider_opts.image_config.is_some() {
            dropped.push("imageConfig");
        }
        if !dropped.is_empty() {
            let verb = if dropped.len() == 1 { "is" } else { "are" };
            warnings.push(Warning::Other {
                message: format!(
                    "google.interactions: {} {verb} not supported when an agent is set; use providerOptions.google.agentConfig instead. Dropped from the request body.",
                    dropped.join(", ")
                ),
            });
        }
    } else {
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
    if is_agent {
        if let Some(v) = &provider_opts.agent_config {
            body.insert("agent_config".into(), normalize_agent_config(v.clone()));
        }
    } else if provider_opts.agent_config.is_some() {
        warnings.push(Warning::Other {
            message:
                "google.interactions: agentConfig is only supported when an agent is set; ignored."
                    .to_owned(),
        });
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

    // Tools / tool_choice routing — covers function tools + all 8 typed
    // Google provider-defined tools (`google.google_search`, `code_execution`,
    // `url_context`, `file_search`, `google_maps`, `computer_use`,
    // `mcp_server`, `retrieval`). See `prepare_tools.rs`.
    {
        let prepared = super::prepare_tools::prepare_tools(
            options.tools.as_deref(),
            options.tool_choice.as_ref(),
        );
        warnings.extend(prepared.warnings);
        if let Some(t) = prepared.tools {
            body.insert("tools".into(), JsonValue::Array(t));
        }
        if let Some(tc) = prepared.tool_choice {
            // `tool_choice` is sent at the generation_config layer (mirrors
            // upstream google-interactions-language-model.ts:311).
            if let Some(JsonValue::Object(gc)) = body.get_mut("generation_config") {
                gc.insert("tool_choice".into(), tc);
            } else {
                let mut gc = JsonMap::new();
                gc.insert("tool_choice".into(), tc);
                body.insert("generation_config".into(), JsonValue::Object(gc));
            }
        }
    }

    // Compaction (mirrors upstream `compactPromptForPreviousInteraction`):
    // when `previousInteractionId` is set and `store !== false`, drop
    // assistant turns whose parts carry a matching
    // `providerOptions.google.interactionId`, plus the tool-result parts
    // whose `toolCallId` came from a dropped assistant turn. The combo
    // `previousInteractionId + store:false` is incoherent (the server has
    // no record to reference); we warn but still send the full history,
    // matching upstream.
    let compact_buffer;
    let prompt_slice: &[Message] = match (
        provider_opts.previous_interaction_id.as_deref(),
        provider_opts.store,
    ) {
        (Some(prev), store) if store != Some(false) => {
            compact_buffer = compact_prompt_for_previous_interaction(&options.prompt, prev);
            &compact_buffer
        }
        (Some(_), Some(false)) => {
            warnings.push(Warning::Other {
                message: "provider_options.google.previousInteractionId was set together with store: false; the full history will be sent and previous_interaction_id will still be emitted".to_owned(),
            });
            &options.prompt
        }
        _ => &options.prompt,
    };

    // Input messages (everything after the system message).
    let input = convert_prompt_to_input(prompt_slice, &mut warnings);
    body.insert("input".into(), JsonValue::Array(input));

    Ok((body, warnings))
}

/// Drop assistant turns whose parts carry
/// `providerOptions.google.interactionId == previousInteractionId`. Also
/// prunes the tool-result parts whose `toolCallId` matches a dropped
/// assistant tool-call. Mirrors upstream
/// `convert-to-google-interactions-input.ts:326-375`.
fn compact_prompt_for_previous_interaction(
    prompt: &[Message],
    previous_interaction_id: &str,
) -> Vec<Message> {
    let mut out = Vec::with_capacity(prompt.len());
    let mut dropped_tool_call_ids: std::collections::HashSet<String> =
        std::collections::HashSet::new();

    for msg in prompt {
        match msg {
            Message::Assistant { content, .. } => {
                let matches_linked = content.iter().any(|part| match part {
                    AssistantPart::Text(TextPart {
                        provider_options, ..
                    })
                    | AssistantPart::Reasoning {
                        provider_options, ..
                    } => {
                        part_has_interaction_id(provider_options.as_ref(), previous_interaction_id)
                    }
                    AssistantPart::ToolCall(tc) => part_has_interaction_id(
                        tc.provider_options.as_ref(),
                        previous_interaction_id,
                    ),
                    AssistantPart::File(f) => part_has_interaction_id(
                        f.provider_options.as_ref(),
                        previous_interaction_id,
                    ),
                    AssistantPart::ReasoningFile {
                        provider_options, ..
                    } => {
                        part_has_interaction_id(provider_options.as_ref(), previous_interaction_id)
                    }
                    AssistantPart::ToolResult(tr) => part_has_interaction_id(
                        tr.provider_options.as_ref(),
                        previous_interaction_id,
                    ),
                    AssistantPart::Custom { .. } => false,
                });
                if matches_linked {
                    for part in content {
                        if let AssistantPart::ToolCall(tc) = part {
                            dropped_tool_call_ids.insert(tc.tool_call_id.clone());
                        }
                    }
                    continue;
                }
                out.push(msg.clone());
            }
            Message::Tool { content, .. } => {
                let remaining: Vec<ToolMessagePart> = content
                    .iter()
                    .filter(|part| {
                        if let ToolMessagePart::ToolResult(r) = part {
                            !dropped_tool_call_ids.contains(&r.tool_call_id)
                        } else {
                            true
                        }
                    })
                    .cloned()
                    .collect();
                if remaining.is_empty() {
                    continue;
                }
                if let Message::Tool {
                    provider_options, ..
                } = msg
                {
                    out.push(Message::Tool {
                        content: remaining,
                        provider_options: provider_options.clone(),
                    });
                }
            }
            other => out.push(other.clone()),
        }
    }

    out
}

fn part_has_interaction_id(provider_options: Option<&ProviderOptions>, expected: &str) -> bool {
    provider_options
        .and_then(|po| po.get("google"))
        .and_then(|g| g.get("interactionId"))
        .and_then(JsonValue::as_str)
        == Some(expected)
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

/// Wire-string variant of [`map_finish_reason`]: takes the raw `status`
/// string off a response/event and maps it through the same table. Mirrors
/// upstream `mapGoogleInteractionsFinishReason({status, hasFunctionCall})`
/// minus the `hasFunctionCall` branch — `completed` always maps to `Stop`
/// in the non-tool case; the stream layer overrides to `ToolCalls` itself
/// when it has seen a function call (kept inside stream state).
pub(super) fn map_finish_reason_from_status(status: Option<&str>) -> FinishReason {
    let parsed = status.and_then(|s| match s {
        "in_progress" => Some(GoogleInteractionsStatus::InProgress),
        "requires_action" => Some(GoogleInteractionsStatus::RequiresAction),
        "completed" => Some(GoogleInteractionsStatus::Completed),
        "failed" => Some(GoogleInteractionsStatus::Failed),
        "cancelled" => Some(GoogleInteractionsStatus::Cancelled),
        "incomplete" => Some(GoogleInteractionsStatus::Incomplete),
        _ => None,
    });
    map_finish_reason(parsed)
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

pub(super) fn translate_step(step: &JsonValue, out: &mut Vec<Content>) {
    translate_step_with(
        step,
        out,
        &mut default_id_gen(),
        &mut std::collections::HashSet::new(),
    )
}

fn default_id_gen() -> impl FnMut() -> String {
    let mut n = 0usize;
    move || {
        n += 1;
        format!("gi-{n}")
    }
}

fn translate_step_with<F: FnMut() -> String>(
    step: &JsonValue,
    out: &mut Vec<Content>,
    id_gen: &mut F,
    seen_sources: &mut std::collections::HashSet<String>,
) {
    let Some(step_type) = step.get("type").and_then(JsonValue::as_str) else {
        return;
    };
    // Built-in tool steps surface a Source list (mirroring upstream's
    // `builtinToolResultToSources`); they don't translate to a content
    // block on their own. Handled before the user-facing branches below.
    if matches!(
        step_type,
        "url_context_result" | "google_search_result" | "google_maps_result" | "file_search_result"
    ) {
        let sources = super::extract_sources::builtin_tool_result_to_sources(
            step_type,
            step.get("result"),
            id_gen,
        );
        for src in sources {
            let key = super::extract_sources::source_key(&src);
            if seen_sources.insert(key) {
                out.push(Content::Source(src));
            }
        }
        return;
    }
    match step_type {
        "model_output" => {
            if let Some(blocks) = step.get("content").and_then(JsonValue::as_array) {
                for block in blocks {
                    translate_block_with(block, out, id_gen, seen_sources);
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

fn translate_block_with<F: FnMut() -> String>(
    block: &JsonValue,
    out: &mut Vec<Content>,
    id_gen: &mut F,
    seen_sources: &mut std::collections::HashSet<String>,
) {
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
        let sources = super::extract_sources::annotations_to_sources(
            block.get("annotations"),
            id_gen,
            seen_sources,
        );
        for src in sources {
            out.push(Content::Source(src));
        }
    } else if kind == "image" {
        let media_type = block
            .get("mime_type")
            .and_then(JsonValue::as_str)
            .unwrap_or("image/png")
            .to_owned();
        if let Some(data) = block.get("data").and_then(JsonValue::as_str) {
            if !data.is_empty() {
                out.push(Content::File(FilePart {
                    media_type,
                    data: FileData::Data {
                        data: FileBytes::Base64(data.to_owned()),
                    },
                    filename: None,
                    provider_options: None,
                }));
                return;
            }
        }
        if let Some(uri) = block.get("uri").and_then(JsonValue::as_str) {
            if !uri.is_empty() {
                out.push(Content::File(FilePart {
                    media_type,
                    data: FileData::Url {
                        url: uri.to_owned(),
                    },
                    filename: None,
                    provider_options: None,
                }));
            }
        }
    }
    // Audio / video / document blocks remain pass-through until the SDK
    // surfaces a typed `AssistantPart` variant for them.
}

pub(super) fn parse_usage(value: Option<&JsonValue>) -> Usage {
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

// -------- Helpers ----------------------------------------------------

fn headers_to_provider(raw: HashMap<String, String>) -> llmsdk_provider::shared::Headers {
    raw.into_iter().map(|(k, v)| (k, Some(v))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmsdk_provider::language_model::ToolResultPart;

    fn assistant_text(text: &str, interaction_id: Option<&str>) -> Message {
        let provider_options = interaction_id.map(|id| {
            let mut po = ProviderOptions::new();
            let mut google = JsonMap::new();
            google.insert("interactionId".into(), JsonValue::String(id.into()));
            po.insert("google".to_owned(), google);
            po
        });
        Message::Assistant {
            content: vec![AssistantPart::Text(TextPart {
                text: text.into(),
                provider_options,
            })],
            provider_options: None,
        }
    }

    fn user_text(text: &str) -> Message {
        Message::User {
            content: vec![UserPart::Text(TextPart {
                text: text.into(),
                provider_options: None,
            })],
            provider_options: None,
        }
    }

    #[test]
    fn compaction_drops_assistant_turns_tagged_with_previous_interaction() {
        let prompt = vec![
            user_text("hi"),
            assistant_text("prior reply", Some("int-1")),
            user_text("follow up"),
        ];
        let compacted = compact_prompt_for_previous_interaction(&prompt, "int-1");
        assert_eq!(compacted.len(), 2);
        assert!(matches!(compacted[0], Message::User { .. }));
        assert!(matches!(compacted[1], Message::User { .. }));
    }

    #[test]
    fn compaction_drops_orphaned_tool_results() {
        let assistant_with_tool_call = Message::Assistant {
            content: vec![AssistantPart::ToolCall(ToolCallPart {
                tool_call_id: "tc-1".into(),
                tool_name: "search".into(),
                input: json!({}),
                provider_executed: None,
                dynamic: None,
                provider_options: Some({
                    let mut po = ProviderOptions::new();
                    let mut g = JsonMap::new();
                    g.insert("interactionId".into(), JsonValue::String("int-1".into()));
                    po.insert("google".to_owned(), g);
                    po
                }),
            })],
            provider_options: None,
        };
        let tool_msg = Message::Tool {
            content: vec![ToolMessagePart::ToolResult(ToolResultPart {
                tool_call_id: "tc-1".into(),
                tool_name: "search".into(),
                output: ToolResultOutput::Text {
                    value: "result".into(),
                    provider_options: None,
                },
                provider_options: None,
            })],
            provider_options: None,
        };
        let prompt = vec![user_text("ask"), assistant_with_tool_call, tool_msg];
        let compacted = compact_prompt_for_previous_interaction(&prompt, "int-1");
        // Assistant with matching id dropped → its tool-result orphan also pruned.
        assert_eq!(compacted.len(), 1);
        assert!(matches!(compacted[0], Message::User { .. }));
    }

    #[test]
    fn compaction_keeps_unrelated_assistant_turns() {
        let prompt = vec![
            assistant_text("kept", Some("int-2")),
            assistant_text("dropped", Some("int-1")),
        ];
        let compacted = compact_prompt_for_previous_interaction(&prompt, "int-1");
        assert_eq!(compacted.len(), 1);
        if let Message::Assistant { content, .. } = &compacted[0] {
            if let AssistantPart::Text(TextPart { text, .. }) = &content[0] {
                assert_eq!(text, "kept");
            }
        }
    }

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

        // Both `Agent` and `ManagedAgent` route to the wire `agent` field,
        // matching upstream `google-interactions-language-model.ts:112-117`.
        let mut body = JsonMap::new();
        GoogleInteractionsAgent::ManagedAgent("managed/bar".into()).apply_to_request(&mut body);
        assert_eq!(
            body.get("agent").and_then(JsonValue::as_str),
            Some("managed/bar")
        );
        assert!(body.get("managed_agent").is_none());
    }

    #[test]
    fn percent_encode_path_segment_handles_special_chars() {
        assert_eq!(percent_encode_path_segment("abc"), "abc");
        assert_eq!(
            percent_encode_path_segment("agents/foo bar"),
            "agents%2Ffoo%20bar"
        );
        assert_eq!(
            percent_encode_path_segment("int-2025_05.06~v1"),
            "int-2025_05.06~v1"
        );
    }

    #[test]
    fn cancel_and_poll_endpoints_are_url_encoded() {
        let inner = Arc::new(
            Inner::builder()
                .base_url("https://api.example")
                .build()
                .expect("inner build"),
        );
        let model = GoogleInteractionsLanguageModel::new(
            inner,
            GoogleInteractionsAgent::Model("ignored".into()),
        );
        // Mirrors upstream `encodeURIComponent` use in cancel-google-interaction.ts:34
        // and poll-google-interactions.ts:76. Special characters in the
        // interaction id must be percent-encoded so the URL stays valid.
        assert_eq!(
            model.cancel_endpoint("int/with slash"),
            "https://api.example/interactions/int%2Fwith%20slash/cancel"
        );
        assert_eq!(
            model.poll_endpoint("int/with slash"),
            "https://api.example/interactions/int%2Fwith%20slash"
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
}
