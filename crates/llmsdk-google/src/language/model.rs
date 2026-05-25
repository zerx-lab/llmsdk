//! [`LanguageModel`] implementation for Gemini.
//!
//! Mirrors `@ai-sdk/google/src/google-language-model.ts`. Owns the
//! provider-shared [`crate::config::Inner`] state, builds the wire body
//! via [`super::convert_prompt`] / [`super::prepare_tools`], and parses
//! the response via [`super::parse_response`] / [`super::stream`].
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    CallOptions, GenerateResponse, GenerateResult, LanguageModel, ResponseFormat, StreamResponse,
    StreamResult, SupportedUrls, UrlPattern,
};
use llmsdk_provider::shared::{Headers, RequestInfo, Warning};
use llmsdk_provider_utils::http::{
    JsonRequest, JsonResponse, post_for_stream, post_json, response_byte_stream,
};
use serde_json::{Map, Value};

use crate::PROVIDER_ID;
use crate::config::Inner;
use crate::error::rewrite_google_error;
use crate::schema::convert_json_schema_to_openapi_nested;

use super::convert_prompt::{ConvertOptions, convert_to_google_messages};
use super::finish_reason::map_finish_reason;
use super::parse_response::{build_content, build_provider_metadata};
use super::prepare_tools::prepare_tools;
use super::stream::make_stream;
use super::usage::convert_usage;
use super::wire::WireResponse;

/// Gemini language-model handle.
///
/// Returned by [`crate::Google::language_model`]. Cheap to clone.
#[derive(Debug, Clone)]
pub struct GoogleLanguageModel {
    inner: Arc<Inner>,
    model_id: String,
    /// Counter for unique synthetic ids when the upstream does not return
    /// one (tool calls without an `id`, source ids, ...).
    counter: Arc<AtomicU64>,
}

impl GoogleLanguageModel {
    /// Construct from a fully assembled [`Inner`].
    ///
    /// Public for cross-crate composition (Google Vertex). End-users should
    /// prefer [`crate::Google::language_model`].
    #[must_use]
    pub fn new(inner: Arc<Inner>, model_id: String) -> Self {
        Self {
            inner,
            model_id,
            counter: Arc::new(AtomicU64::new(0)),
        }
    }

    #[allow(
        dead_code,
        reason = "kept for future reuse alongside the inline next_id closures"
    )]
    fn next_id(&self) -> String {
        format!("g-{}", self.counter.fetch_add(1, Ordering::Relaxed))
    }

    fn model_path(&self) -> String {
        if self.model_id.contains('/') {
            self.model_id.clone()
        } else {
            format!("models/{}", self.model_id)
        }
    }

    fn provider_option_keys(&self) -> Vec<&'static str> {
        if self.inner.provider.contains("vertex") {
            vec!["googleVertex", "vertex"]
        } else {
            vec!["google"]
        }
    }

    fn is_vertex(&self) -> bool {
        self.inner.provider.starts_with("google.vertex.")
    }

    fn build_url(&self, method: &str) -> String {
        format!("{}/{}:{}", self.inner.base_url, self.model_path(), method)
    }

    fn merged_headers(&self, extra: Option<&Headers>) -> HashMap<String, Option<String>> {
        let mut h = self.inner.headers.clone();
        if let Some(extra) = extra {
            for (k, v) in extra {
                h.insert(k.clone(), v.clone());
            }
        }
        h
    }
}

#[async_trait]
impl LanguageModel for GoogleLanguageModel {
    fn provider(&self) -> &str {
        &self.inner.provider
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn supported_urls(&self) -> SupportedUrls {
        let files_pattern = format!("^{}/files/.*$", regex_escape(&self.inner.base_url));
        [(
            "*".to_owned(),
            vec![
                UrlPattern(files_pattern),
                UrlPattern(
                    r"^https://(?:www\.)?youtube\.com/watch\?v=[\w-]+(?:&[\w=&.-]*)?$".into(),
                ),
                UrlPattern(r"^https://youtu\.be/[\w-]+(?:\?[\w=&.-]*)?$".into()),
            ],
        )]
        .into_iter()
        .collect()
    }

    async fn do_generate(&self, options: CallOptions) -> Result<GenerateResult, ProviderError> {
        let prepared = self.prepare_request(&options, false)?;
        let url = self.build_url("generateContent");
        let mut headers = self.merged_headers(options.headers.as_ref());
        for (k, v) in &prepared.extra_headers {
            headers.insert(k.clone(), v.clone());
        }

        let mut req = JsonRequest::new(url, prepared.body.clone());
        req.headers = headers;

        let envelope: JsonResponse<WireResponse> = match post_json(&self.inner.http, req).await {
            Ok(r) => r,
            Err(e) => return Err(rewrite_google_error(e)),
        };

        let response = envelope.value;
        let response_headers = envelope.headers;

        let candidate = response.candidates.first().cloned().unwrap_or_default();
        let parts = candidate
            .content
            .as_ref()
            .and_then(|c| c.parts.as_ref())
            .cloned()
            .unwrap_or_default();

        let provider_keys = prepared.provider_keys.clone();
        let provider_keys_ref: Vec<&str> = provider_keys.iter().map(String::as_str).collect();
        let counter = Arc::clone(&self.counter);
        let next_id = move || format!("g-{}", counter.fetch_add(1, Ordering::Relaxed));

        let (content, has_client_tool) = build_content(
            &parts,
            candidate.grounding_metadata.as_ref(),
            &provider_keys_ref,
            next_id,
        );

        let unified = map_finish_reason(candidate.finish_reason.as_deref(), has_client_tool);
        let finish_reason = match candidate.finish_reason.clone() {
            Some(raw) => llmsdk_provider::language_model::FinishReason::with_raw(unified, raw),
            None => llmsdk_provider::language_model::FinishReason::new(unified),
        };

        Ok(GenerateResult {
            content,
            finish_reason,
            usage: convert_usage(response.usage_metadata.as_ref()),
            provider_metadata: Some(build_provider_metadata(&response, &provider_keys_ref)),
            request: Some(RequestInfo {
                body: Some(prepared.body),
            }),
            response: Some(GenerateResponse {
                metadata: Default::default(),
                body: Some(serde_json::to_value(&response).unwrap_or(Value::Null)),
            })
            .map(|mut gr| {
                gr.metadata.headers = Some(
                    response_headers
                        .into_iter()
                        .map(|(k, v)| (k, Some(v)))
                        .collect(),
                );
                gr
            }),
            warnings: prepared.warnings,
        })
    }

    async fn do_stream(&self, options: CallOptions) -> Result<StreamResult, ProviderError> {
        let prepared = self.prepare_request(&options, true)?;
        let url = self.build_url("streamGenerateContent?alt=sse");
        let mut headers = self.merged_headers(options.headers.as_ref());
        for (k, v) in &prepared.extra_headers {
            headers.insert(k.clone(), v.clone());
        }

        let mut req = JsonRequest::new(url, prepared.body.clone());
        req.headers = headers;

        let stream_resp = match post_for_stream(&self.inner.http, req).await {
            Ok(r) => r,
            Err(e) => return Err(rewrite_google_error(e)),
        };
        let response_headers = stream_resp.headers.clone();
        let bytes = response_byte_stream(stream_resp.response);

        let counter = Arc::clone(&self.counter);
        let next_id = move || format!("g-{}", counter.fetch_add(1, Ordering::Relaxed));

        let parts_stream = make_stream(
            bytes,
            prepared.warnings,
            options.include_raw_chunks.unwrap_or(false),
            prepared.provider_keys.clone(),
            next_id,
        );

        Ok(StreamResult {
            stream: parts_stream,
            request: Some(RequestInfo {
                body: Some(prepared.body),
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

struct PreparedRequest {
    body: Value,
    warnings: Vec<Warning>,
    provider_keys: Vec<String>,
    extra_headers: Headers,
}

impl GoogleLanguageModel {
    fn prepare_request(
        &self,
        options: &CallOptions,
        is_streaming: bool,
    ) -> Result<PreparedRequest, ProviderError> {
        let mut warnings = Vec::new();

        // Provider option keys: Vertex uses both `googleVertex` and `vertex`
        // (new key first); Google uses `google` only.
        let option_keys: Vec<&'static str> = self.provider_option_keys();
        let is_vertex = self.is_vertex();

        let mut google_options =
            super::options::parse(options.provider_options.as_ref(), &option_keys)?;
        if google_options.is_none() && !option_keys.contains(&"google") {
            google_options = super::options::parse(options.provider_options.as_ref(), &["google"])?;
        }

        let google_options = google_options.unwrap_or_default();

        // Capability mismatch warnings.
        if let Some(tools) = options.tools.as_deref() {
            for t in tools {
                if let llmsdk_provider::language_model::Tool::Provider(p) = t {
                    if p.id == "google.vertex_rag_store" && !is_vertex {
                        warnings.push(Warning::Other {
                            message: format!(
                                "The 'vertex_rag_store' tool is only supported with the Google Vertex provider \
                                 and might not be supported or could behave unexpectedly with the current Google provider ({})",
                                self.inner.provider
                            ),
                        });
                    }
                }
            }
        }
        if google_options.stream_function_call_arguments == Some(true) && !is_vertex {
            warnings.push(Warning::Other {
                message: format!(
                    "'streamFunctionCallArguments' is only supported on the Vertex AI API and will be ignored with the current Google provider ({})",
                    self.inner.provider
                ),
            });
        }
        if google_options.service_tier.is_some() && is_vertex {
            warnings.push(Warning::Other {
                message: "'serviceTier' is a Gemini API option and is not supported on Vertex AI."
                    .into(),
            });
        }
        if (google_options.shared_request_type.is_some() || google_options.request_type.is_some())
            && !is_vertex
        {
            warnings.push(Warning::Other {
                message: format!(
                    "'sharedRequestType' and 'requestType' are Vertex AI options and are ignored with the current Google provider ({})",
                    self.inner.provider
                ),
            });
        }

        let body_service_tier = if is_vertex {
            None
        } else {
            google_options.service_tier.clone()
        };

        let is_gemma = self.model_id.to_lowercase().starts_with("gemma-");
        let supports_function_response_parts = self.model_id.starts_with("gemini-3");

        let converted = convert_to_google_messages(
            &options.prompt,
            ConvertOptions {
                is_gemma_model: is_gemma,
                provider_option_names: &option_keys,
                supports_function_response_parts,
            },
        )?;

        let prepared_tools = prepare_tools(
            options.tools.as_deref(),
            options.tool_choice.as_ref(),
            &self.model_id,
            is_vertex,
        );
        warnings.extend(prepared_tools.warnings);

        // generationConfig
        let mut gen_config = Map::new();
        if let Some(v) = options.max_output_tokens {
            gen_config.insert("maxOutputTokens".into(), Value::from(v));
        }
        if let Some(v) = options.temperature {
            gen_config.insert("temperature".into(), Value::from(v));
        }
        if let Some(v) = options.top_k {
            gen_config.insert("topK".into(), Value::from(v));
        }
        if let Some(v) = options.top_p {
            gen_config.insert("topP".into(), Value::from(v));
        }
        if let Some(v) = options.frequency_penalty {
            gen_config.insert("frequencyPenalty".into(), Value::from(v));
        }
        if let Some(v) = options.presence_penalty {
            gen_config.insert("presencePenalty".into(), Value::from(v));
        }
        if let Some(ref v) = options.stop_sequences {
            gen_config.insert("stopSequences".into(), Value::from(v.clone()));
        }
        if let Some(v) = options.seed {
            gen_config.insert("seed".into(), Value::from(v));
        }
        // response format
        if let Some(ResponseFormat::Json { schema, .. }) = &options.response_format {
            gen_config.insert(
                "responseMimeType".into(),
                Value::String("application/json".into()),
            );
            let allow_schema = google_options.structured_outputs.unwrap_or(true);
            if allow_schema {
                if let Some(s) = schema {
                    let s_value = serde_json::to_value(s).unwrap_or(Value::Null);
                    gen_config.insert(
                        "responseSchema".into(),
                        convert_json_schema_to_openapi_nested(&s_value),
                    );
                }
            }
        }
        if let Some(v) = google_options.audio_timestamp {
            gen_config.insert("audioTimestamp".into(), Value::Bool(v));
        }
        if let Some(modalities) = &google_options.response_modalities {
            gen_config.insert(
                "responseModalities".into(),
                Value::Array(
                    modalities
                        .iter()
                        .map(|s| Value::String(s.clone()))
                        .collect(),
                ),
            );
        }
        let thinking_config = build_thinking_config(
            &options.reasoning,
            &self.model_id,
            &google_options,
            &mut warnings,
        );
        if let Some(tc) = thinking_config {
            gen_config.insert("thinkingConfig".into(), tc);
        }
        if let Some(mr) = &google_options.media_resolution {
            gen_config.insert("mediaResolution".into(), Value::String(mr.clone()));
        }
        if let Some(ic) = &google_options.image_config {
            gen_config.insert("imageConfig".into(), ic.clone());
        }

        // toolConfig merge
        let mut tool_config_out: Option<Value> = prepared_tools.tool_config.clone();
        let stream_fn_args = if is_streaming && is_vertex {
            google_options
                .stream_function_call_arguments
                .unwrap_or(false)
        } else {
            false
        };
        if stream_fn_args || google_options.retrieval_config.is_some() {
            let mut tc_map: Map<String, Value> = tool_config_out
                .clone()
                .and_then(|v| v.as_object().cloned())
                .unwrap_or_default();
            if stream_fn_args {
                let mut fcc: Map<String, Value> = tc_map
                    .get("functionCallingConfig")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                fcc.insert("streamFunctionCallArguments".into(), Value::Bool(true));
                tc_map.insert("functionCallingConfig".into(), Value::Object(fcc));
            }
            if let Some(rc) = &google_options.retrieval_config {
                tc_map.insert("retrievalConfig".into(), rc.clone());
            }
            tool_config_out = Some(Value::Object(tc_map));
        }

        // Assemble body
        let mut body = Map::new();
        body.insert("generationConfig".into(), Value::Object(gen_config));
        body.insert("contents".into(), Value::Array(converted.contents));
        if !is_gemma {
            if let Some(si) = converted.system_instruction {
                body.insert("systemInstruction".into(), si);
            }
        }
        if let Some(safety) = &google_options.safety_settings {
            body.insert("safetySettings".into(), Value::Array(safety.clone()));
        }
        if let Some(tools_val) = prepared_tools.tools {
            body.insert("tools".into(), tools_val);
        }
        if let Some(tc_val) = tool_config_out {
            body.insert("toolConfig".into(), tc_val);
        }
        if let Some(cached) = &google_options.cached_content {
            body.insert("cachedContent".into(), Value::String(cached.clone()));
        }
        if let Some(labels) = &google_options.labels {
            body.insert("labels".into(), labels.clone());
        }
        if let Some(st) = body_service_tier {
            body.insert("serviceTier".into(), Value::String(st));
        }

        let mut extra_headers = Headers::new();
        if is_vertex {
            if let Some(srt) = google_options.shared_request_type.as_deref() {
                extra_headers.insert(
                    "X-Vertex-AI-LLM-Shared-Request-Type".into(),
                    Some(srt.to_owned()),
                );
            }
            if let Some(rt) = google_options.request_type.as_deref() {
                extra_headers.insert("X-Vertex-AI-LLM-Request-Type".into(), Some(rt.to_owned()));
            }
        }

        Ok(PreparedRequest {
            body: Value::Object(body),
            warnings,
            provider_keys: option_keys.iter().map(|s| (*s).to_owned()).collect(),
            extra_headers,
        })
    }
}

fn build_thinking_config(
    reasoning: &Option<llmsdk_provider::language_model::ReasoningEffort>,
    model_id: &str,
    google_options: &super::options::GoogleOptions,
    warnings: &mut Vec<Warning>,
) -> Option<Value> {
    use llmsdk_provider::language_model::ReasoningEffort;
    let is_gemini3 = model_id.contains("gemini-3") && !model_id.contains("gemini-3-pro-image");

    let resolved = match reasoning {
        None | Some(ReasoningEffort::ProviderDefault) => None,
        Some(eff) => {
            if is_gemini3 {
                let lvl = match eff {
                    ReasoningEffort::None => Some("minimal"),
                    ReasoningEffort::Minimal => Some("minimal"),
                    ReasoningEffort::Low => Some("low"),
                    ReasoningEffort::Medium => Some("medium"),
                    ReasoningEffort::High | ReasoningEffort::Xhigh => Some("high"),
                    ReasoningEffort::ProviderDefault => None,
                };
                lvl.map(|v| {
                    let mut m = Map::new();
                    m.insert("thinkingLevel".into(), Value::String(v.into()));
                    m
                })
            } else {
                // Gemini 2.5 budget-based
                let budget = match eff {
                    ReasoningEffort::None => Some(0i64),
                    ReasoningEffort::Minimal => Some(512),
                    ReasoningEffort::Low => Some(2048),
                    ReasoningEffort::Medium => Some(8192),
                    ReasoningEffort::High => Some(16384),
                    ReasoningEffort::Xhigh => Some(24576),
                    ReasoningEffort::ProviderDefault => None,
                };
                budget.map(|v| {
                    let mut m = Map::new();
                    m.insert("thinkingBudget".into(), Value::from(v));
                    m
                })
            }
        }
    };

    if let Some(tc) = &google_options.thinking_config {
        if resolved.is_some() && tc.thinking_budget.is_some() && !is_gemini3 {
            warnings.push(Warning::Other {
                message:
                    "Both `reasoning` effort and explicit `thinkingBudget` are set; explicit wins."
                        .into(),
            });
        }
        let mut m = resolved.unwrap_or_default();
        if let Some(b) = tc.thinking_budget {
            m.insert("thinkingBudget".into(), Value::from(b));
        }
        if let Some(it) = tc.include_thoughts {
            m.insert("includeThoughts".into(), Value::Bool(it));
        }
        if let Some(level) = &tc.thinking_level {
            m.insert("thinkingLevel".into(), Value::String(level.clone()));
        }
        return Some(Value::Object(m));
    }
    resolved.map(Value::Object)
}

fn regex_escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for c in input.chars() {
        match c {
            '.' | '+' | '*' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '^' | '$' | '\\'
            | '/' => {
                out.push('\\');
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out
}

// PROVIDER_ID is reachable from sibling modules via `crate::PROVIDER_ID`.
#[allow(dead_code, reason = "kept for future routing hooks")]
fn _provider_id_alive() -> &'static str {
    PROVIDER_ID
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regex_escape_basic() {
        assert_eq!(
            regex_escape("https://example.com/v1beta"),
            r"https:\/\/example\.com\/v1beta"
        );
    }

    #[test]
    fn provider_keys_google_default() {
        let cfg = Inner {
            provider: "google".into(),
            base_url: "https://x".into(),
            headers: HashMap::new(),
            http: llmsdk_provider_utils::http::HttpClient::default(),
        };
        let m = GoogleLanguageModel::new(Arc::new(cfg), "gemini-2.5-flash".into());
        assert_eq!(m.provider_option_keys(), vec!["google"]);
    }
}
