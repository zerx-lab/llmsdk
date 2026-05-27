//! [`LanguageModel`] implementation for the Bedrock Converse API.
// Rust guideline compliant 2026-05-25

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::TryStreamExt;
use llmsdk_provider::ProviderError;
use llmsdk_provider::language_model::{
    CallOptions, GenerateResult, LanguageModel, ResponseFormat, StreamResponse, StreamResult,
};
use llmsdk_provider::shared::{Headers, RequestInfo, Warning};
use llmsdk_provider_utils::http::RawRequest;
use llmsdk_provider_utils::http::{HttpClient, JsonResponse, post_raw, response_byte_stream};
use reqwest::Method;
use serde_json::Value;

use crate::PROVIDER_ID;
use crate::config::Inner;

use super::convert_prompt::{Converted, convert_prompt};
use super::normalize_tool_call_id::is_mistral_model;
use super::options::parse as parse_options;
use super::parse_response::parse_response;
use super::prepare_tools::{PreparedTools, prepare_tools};
use super::stream::build_stream;
use super::wire::{ConverseRequest, ConverseResponse, ServiceTier};

/// Bedrock chat (Converse API) model handle.
///
/// Cheap to clone; the underlying HTTP client and auth are shared with the
/// parent [`crate::AmazonBedrock`] provider.
#[derive(Debug, Clone)]
pub struct AmazonBedrockChatModel {
    pub(crate) inner: Arc<Inner>,
    pub(crate) model_id: String,
}

impl AmazonBedrockChatModel {
    pub(crate) fn new(inner: Arc<Inner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    fn url(&self, suffix: &str) -> String {
        let encoded = encode_path_segment(&self.model_id);
        format!(
            "{}/model/{}/{}",
            self.inner.runtime_base_url, encoded, suffix
        )
    }

    #[allow(dead_code, reason = "exposed for downstream wrapping providers")]
    pub(crate) fn http(&self) -> &HttpClient {
        &self.inner.http
    }
}

#[async_trait]
impl LanguageModel for AmazonBedrockChatModel {
    fn provider(&self) -> &str {
        PROVIDER_ID
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn do_generate(&self, options: CallOptions) -> Result<GenerateResult, ProviderError> {
        let prepared = build_request(&self.model_id, &options, false)?;
        let url = self.url("converse");
        let body_bytes = serde_json::to_vec(&prepared.request)
            .map_err(|e| ProviderError::json_parse("<bedrock-request>", e.to_string()))?;
        let request_body_value = serde_json::from_slice::<Value>(&body_bytes).ok();

        let mut headers = self.inner.extra_headers.clone();
        if let Some(per_call) = options.headers.as_ref() {
            for (k, v) in per_call {
                headers.insert(k.clone(), v.clone());
            }
        }
        self.inner
            .auth
            .apply(&mut headers, &Method::POST, &url, &body_bytes)
            .await?;

        let mut raw = RawRequest::new(url.clone(), body_bytes, "application/json");
        raw.headers = headers;
        let response: JsonResponse<ConverseResponse> =
            post_raw::<ConverseResponse>(&self.inner.http, raw).await?;

        parse_response(
            response.value,
            response.headers,
            request_body_value,
            prepared.warnings,
            prepared.is_mistral,
            prepared.uses_json_response_tool,
            self.inner.generate_id.as_ref(),
        )
    }

    async fn do_stream(&self, options: CallOptions) -> Result<StreamResult, ProviderError> {
        let prepared = build_request(&self.model_id, &options, true)?;
        let url = self.url("converse-stream");
        let body_bytes = serde_json::to_vec(&prepared.request)
            .map_err(|e| ProviderError::json_parse("<bedrock-request>", e.to_string()))?;
        let request_body_value = serde_json::from_slice::<Value>(&body_bytes).ok();

        let mut headers = self.inner.extra_headers.clone();
        if let Some(per_call) = options.headers.as_ref() {
            for (k, v) in per_call {
                headers.insert(k.clone(), v.clone());
            }
        }
        self.inner
            .auth
            .apply(&mut headers, &Method::POST, &url, &body_bytes)
            .await?;

        // Dispatch the streaming request via reqwest directly so we can keep
        // the binary EventStream body intact (provider-utils only knows about
        // SSE today).
        let mut builder = self
            .inner
            .http
            .reqwest()
            .request(Method::POST, &url)
            .header("content-type", "application/json")
            .body(Bytes::from(body_bytes.clone()));
        for (name, value) in &headers {
            if let Some(v) = value {
                builder = builder.header(name, v);
            }
        }
        let http_response = builder.send().await.map_err(|e| {
            ProviderError::api_call_builder(&url, format!("transport error: {e}"))
                .retryable(true)
                .build()
        })?;
        let status = http_response.status();
        let mut response_headers =
            std::collections::HashMap::with_capacity(http_response.headers().len());
        for (name, value) in http_response.headers() {
            if let Ok(v) = value.to_str() {
                response_headers.insert(name.as_str().to_owned(), v.to_owned());
            }
        }
        if !status.is_success() {
            let body_text = http_response.text().await.unwrap_or_default();
            return Err(ProviderError::api_call_builder(
                &url,
                format!(
                    "HTTP {} {}",
                    status.as_u16(),
                    status.canonical_reason().unwrap_or("")
                ),
            )
            .status_code(status.as_u16())
            .response_body(body_text)
            .response_headers(response_headers.clone())
            .request_body(request_body_value.clone().unwrap_or(Value::Null))
            .build());
        }
        let model_id = self.model_id.clone();
        let bytes_stream = response_byte_stream(http_response).map_err(|e| e);
        let stream_parts = build_stream(
            bytes_stream,
            prepared.warnings,
            prepared.is_mistral,
            prepared.uses_json_response_tool,
            response_headers.clone(),
            model_id,
            options.include_raw_chunks.unwrap_or(false),
            self.inner.generate_id.clone(),
        );

        let stream_headers: Headers = response_headers
            .into_iter()
            .map(|(k, v)| (k, Some(v)))
            .collect();

        Ok(StreamResult {
            stream: Box::pin(stream_parts),
            request: Some(RequestInfo {
                body: request_body_value,
            }),
            response: Some(StreamResponse {
                headers: Some(stream_headers),
            }),
        })
    }
}

/// Output of [`build_request`].
pub(crate) struct PreparedRequest {
    pub request: ConverseRequest,
    pub warnings: Vec<Warning>,
    pub uses_json_response_tool: bool,
    pub is_mistral: bool,
}

/// Build the on-wire Converse request from llmsdk [`CallOptions`].
///
/// Handles every dropped / coerced standard parameter (frequency_penalty,
/// presence_penalty, seed, out-of-range temperature, response_format
/// other than text / json, ...) and surfaces a [`Warning`] for each.
pub(crate) fn build_request(
    model_id: &str,
    options: &CallOptions,
    _stream: bool,
) -> Result<PreparedRequest, ProviderError> {
    let mut bedrock_opts = parse_options(options.provider_options.as_ref());
    let is_mistral = is_mistral_model(model_id);

    let Converted {
        system,
        messages,
        mut warnings,
    } = convert_prompt(&options.prompt, is_mistral)?;

    if options.frequency_penalty.is_some() {
        warnings.push(Warning::Unsupported {
            feature: "frequencyPenalty".into(),
            details: None,
        });
    }
    if options.presence_penalty.is_some() {
        warnings.push(Warning::Unsupported {
            feature: "presencePenalty".into(),
            details: None,
        });
    }
    if options.seed.is_some() {
        warnings.push(Warning::Unsupported {
            feature: "seed".into(),
            details: None,
        });
    }
    let mut temperature = options.temperature;
    if let Some(t) = temperature {
        if t > 1.0 {
            warnings.push(Warning::Unsupported {
                feature: "temperature".into(),
                details: Some(format!(
                    "{t} exceeds bedrock maximum of 1.0; clamped to 1.0"
                )),
            });
            temperature = Some(1.0);
        } else if t < 0.0 {
            warnings.push(Warning::Unsupported {
                feature: "temperature".into(),
                details: Some(format!("{t} below bedrock minimum of 0; clamped to 0")),
            });
            temperature = Some(0.0);
        }
    }
    if let Some(format) = options.response_format.as_ref() {
        let supported = matches!(format, ResponseFormat::Text | ResponseFormat::Json { .. });
        if !supported {
            warnings.push(Warning::Unsupported {
                feature: "responseFormat".into(),
                details: Some("Only text and json response formats are supported.".into()),
            });
        }
    }

    let uses_json_response_tool = matches!(
        options.response_format.as_ref(),
        Some(ResponseFormat::Json {
            schema: Some(_),
            ..
        })
    ) && !model_id.contains("anthropic.");
    // Inject the synthetic `json` function tool for non-Anthropic models that
    // ask for structured JSON output.
    let mut effective_tools = options.tools.clone();
    let mut effective_choice = options.tool_choice.clone();
    if uses_json_response_tool
        && let Some(ResponseFormat::Json {
            schema: Some(schema),
            ..
        }) = options.response_format.as_ref()
    {
        let mut list = effective_tools.unwrap_or_default();
        list.push(llmsdk_provider::language_model::Tool::Function(
            llmsdk_provider::language_model::FunctionTool {
                name: "json".into(),
                description: Some("Respond with a JSON object.".into()),
                input_schema: schema.clone(),
                input_examples: None,
                strict: None,
                provider_options: None,
            },
        ));
        effective_tools = Some(list);
        effective_choice = Some(llmsdk_provider::language_model::ToolChoice::Required);
    }

    let PreparedTools {
        tool_config,
        warnings: tool_warnings,
        additional_tools,
        betas,
    } = prepare_tools(
        effective_tools.as_deref(),
        effective_choice.as_ref(),
        model_id,
    );
    warnings.extend(tool_warnings);

    // Resolve top-level `CallOptions.reasoning` (and any user-supplied
    // `provider_options.amazonBedrock.reasoningConfig` overrides) into the
    // final `reasoning_config` block. Mirrors upstream
    // `resolveAmazonBedrockReasoningConfig`
    // (amazon-bedrock-chat-language-model.ts:1220-1294).
    bedrock_opts.reasoning_config = super::reasoning_mapper::resolve_reasoning_config(
        options.reasoning,
        bedrock_opts.reasoning_config.take(),
        model_id,
        &mut warnings,
    );

    // Mirror upstream amazon-bedrock-chat-language-model.ts:239-298:
    // capture the per-axis thinking knobs from the resolved reasoning
    // config and route them into either `additionalModelRequestFields`
    // (Anthropic + thinking) or surface warnings.
    let is_anthropic_model = model_id.contains("anthropic.") || model_id.contains("claude");
    let is_openai_model = model_id.contains("openai.");
    let thinking_type = bedrock_opts
        .reasoning_config
        .as_ref()
        .and_then(|rc| rc.kind.clone());
    let thinking_budget = match thinking_type.as_deref() {
        Some("enabled") => bedrock_opts
            .reasoning_config
            .as_ref()
            .and_then(|rc| rc.budget_tokens),
        _ => None,
    };
    let thinking_display = match thinking_type.as_deref() {
        Some("adaptive") => bedrock_opts
            .reasoning_config
            .as_ref()
            .and_then(|rc| rc.display.clone()),
        _ => None,
    };
    let is_thinking_enabled = matches!(thinking_type.as_deref(), Some("enabled" | "adaptive"));
    let is_anthropic_thinking_enabled = is_anthropic_model && is_thinking_enabled;
    let max_reasoning_effort = bedrock_opts
        .reasoning_config
        .as_ref()
        .and_then(|rc| rc.max_reasoning_effort.clone());

    // Anthropic thinking ('enabled' | 'adaptive') is incompatible with
    // temperature / topP / topK on Anthropic-on-Bedrock — strip all three
    // with a warning. Mirrors amazon-bedrock-chat-language-model.ts:345-372.
    let mut top_k = options.top_k;
    let mut top_p = options.top_p;
    if is_anthropic_thinking_enabled {
        if temperature.is_some() {
            warnings.push(Warning::Unsupported {
                feature: "temperature".into(),
                details: Some(
                    "temperature is not supported when Anthropic thinking is enabled; dropped"
                        .into(),
                ),
            });
            temperature = None;
        }
        if top_k.is_some() {
            warnings.push(Warning::Unsupported {
                feature: "topK".into(),
                details: Some(
                    "topK is not supported when Anthropic thinking is enabled; dropped".into(),
                ),
            });
            top_k = None;
        }
        if top_p.is_some() {
            warnings.push(Warning::Unsupported {
                feature: "topP".into(),
                details: Some(
                    "topP is not supported when Anthropic thinking is enabled; dropped".into(),
                ),
            });
            top_p = None;
        }
    } else if !is_anthropic_model {
        if bedrock_opts
            .reasoning_config
            .as_ref()
            .and_then(|rc| rc.budget_tokens)
            .is_some()
        {
            warnings.push(Warning::Unsupported {
                feature: "budgetTokens".into(),
                details: Some(
                    "budgetTokens applies only to Anthropic models on Bedrock and will be ignored \
                     for this model."
                        .into(),
                ),
            });
        }
        if matches!(thinking_type.as_deref(), Some("adaptive")) {
            warnings.push(Warning::Unsupported {
                feature: "adaptive thinking".into(),
                details: Some(
                    "adaptive thinking type applies only to Anthropic models on Bedrock.".into(),
                ),
            });
        }
    }

    // Compute the final `max_tokens`. When Anthropic thinking is enabled
    // with an explicit budget, the budget must be *added* to the user's
    // `max_output_tokens` (or default 4096 when absent). Mirrors upstream
    // amazon-bedrock-chat-language-model.ts:260-263.
    let max_tokens_resolved = match (options.max_output_tokens, thinking_budget) {
        (Some(mx), Some(b)) if is_anthropic_thinking_enabled => Some(mx + b),
        (None, Some(b)) if is_anthropic_thinking_enabled => Some(b + 4096),
        (mx, _) => mx,
    };

    let mut inference_config = super::wire::InferenceConfig {
        max_tokens: max_tokens_resolved,
        temperature,
        top_p,
        top_k,
        stop_sequences: options.stop_sequences.clone(),
    };
    let inference_emit = if inference_config.is_empty() {
        None
    } else {
        Some(std::mem::take(&mut inference_config))
    };

    let service_tier = bedrock_opts
        .service_tier
        .take()
        .map(|kind| ServiceTier { kind });

    // Merge collected extras into additionalModelRequestFields:
    //   - prepare_tools `additional_tools` (anthropic tool_choice)
    //   - prepare_tools `betas` + provider option `anthropic_beta`
    //     → `anthropic_beta` array
    let user_extra = bedrock_opts.additional_model_request_fields.take();
    let mut merged: Option<serde_json::Map<String, serde_json::Value>> = match user_extra {
        Some(serde_json::Value::Object(m)) => Some(m),
        Some(other) => {
            warnings.push(Warning::Other {
                message: format!(
                    "additionalModelRequestFields must be an object; got {other}. Dropped."
                ),
            });
            None
        }
        None => None,
    };
    let mut all_betas = betas;
    if let Some(extra) = bedrock_opts.anthropic_beta.take() {
        for t in extra {
            all_betas.insert(t);
        }
    }
    if let Some(extras) = additional_tools {
        let m = merged.get_or_insert_with(serde_json::Map::new);
        for (k, v) in extras {
            m.insert(k, v);
        }
    }
    if !all_betas.is_empty() {
        let m = merged.get_or_insert_with(serde_json::Map::new);
        let list = serde_json::Value::Array(
            all_betas
                .into_iter()
                .map(serde_json::Value::String)
                .collect(),
        );
        m.insert("anthropic_beta".to_owned(), list);
    }

    // Inject `thinking` into `additionalModelRequestFields` for
    // Anthropic + thinking-enabled requests. Mirrors upstream
    // amazon-bedrock-chat-language-model.ts:258-298.
    if is_anthropic_thinking_enabled {
        let m = merged.get_or_insert_with(serde_json::Map::new);
        if let Some(b) = thinking_budget {
            // `thinking: { type: 'enabled', budget_tokens: <budget> }`
            let mut block = serde_json::Map::new();
            block.insert(
                "type".to_owned(),
                serde_json::Value::String("enabled".to_owned()),
            );
            block.insert("budget_tokens".to_owned(), serde_json::Value::from(b));
            m.insert("thinking".to_owned(), serde_json::Value::Object(block));
        } else if matches!(thinking_type.as_deref(), Some("adaptive")) {
            // `thinking: { type: 'adaptive', display?: <display> }`
            let mut block = serde_json::Map::new();
            block.insert(
                "type".to_owned(),
                serde_json::Value::String("adaptive".to_owned()),
            );
            if let Some(d) = thinking_display {
                block.insert("display".to_owned(), serde_json::Value::String(d));
            }
            m.insert("thinking".to_owned(), serde_json::Value::Object(block));
        }
    }

    // Route `max_reasoning_effort` based on model family. Mirrors upstream
    // amazon-bedrock-chat-language-model.ts:300-330.
    if let Some(eff) = max_reasoning_effort {
        let m = merged.get_or_insert_with(serde_json::Map::new);
        if is_anthropic_model {
            // Anthropic on Bedrock: nest under `output_config.effort`.
            let mut oc = match m.get("output_config") {
                Some(serde_json::Value::Object(existing)) => existing.clone(),
                _ => serde_json::Map::new(),
            };
            oc.insert("effort".to_owned(), serde_json::Value::String(eff));
            m.insert("output_config".to_owned(), serde_json::Value::Object(oc));
        } else if is_openai_model {
            // OpenAI on Bedrock: flat `reasoning_effort`.
            m.insert(
                "reasoning_effort".to_owned(),
                serde_json::Value::String(eff),
            );
        } else {
            // Other models (e.g. Nova 2): use `reasoningConfig` envelope.
            let mut rc = serde_json::Map::new();
            if let Some(t) = thinking_type.as_deref().filter(|t| *t != "adaptive") {
                rc.insert("type".to_owned(), serde_json::Value::String(t.to_owned()));
            }
            rc.insert(
                "maxReasoningEffort".to_owned(),
                serde_json::Value::String(eff),
            );
            m.insert("reasoningConfig".to_owned(), serde_json::Value::Object(rc));
        }
    }

    let merged_extras = merged.map(serde_json::Value::Object);

    let request = ConverseRequest {
        system,
        messages,
        tool_config,
        inference_config: inference_emit,
        additional_model_request_fields: merged_extras,
        additional_model_response_field_paths: model_id
            .contains("anthropic.")
            .then(|| vec!["/delta/stop_sequence".to_owned()]),
        service_tier,
        guardrail_config: bedrock_opts.guardrail_config.take(),
        performance_config: bedrock_opts.performance_config.take(),
        request_metadata: bedrock_opts.request_metadata.take(),
        prompt_variables: bedrock_opts.prompt_variables.take(),
    };

    Ok(PreparedRequest {
        request,
        warnings,
        uses_json_response_tool,
        is_mistral,
    })
}

/// Percent-encode the model id when it contains characters that conflict
/// with URL path segments (`:` is common on Bedrock model ids).
pub(crate) fn encode_path_segment(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(byte as char);
            }
            _ => {
                use std::fmt::Write as _;
                let _ = write!(out, "%{byte:02X}");
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmsdk_provider::language_model::{Message, TextPart, UserPart};

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
    fn temperature_above_one_is_clamped_with_warning() {
        let mut o = opts();
        o.temperature = Some(2.5);
        let prepared = build_request("amazon.nova-lite-v1:0", &o, false).unwrap();
        let warn = prepared
            .warnings
            .iter()
            .find(|w| matches!(w, Warning::Unsupported { feature, .. } if feature == "temperature"))
            .expect("expected temperature warning");
        let Warning::Unsupported { details, .. } = warn else {
            unreachable!()
        };
        assert!(details.as_deref().unwrap().contains("1.0"));
        let inference = prepared.request.inference_config.unwrap();
        assert!((inference.temperature.unwrap() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn percent_encodes_colon_in_model_id() {
        let encoded = encode_path_segment("amazon.titan-embed-text-v2:0");
        assert!(encoded.contains("%3A"));
    }

    #[test]
    fn anthropic_model_emits_additional_response_field_paths() {
        let prepared =
            build_request("anthropic.claude-3-haiku-20240307-v1:0", &opts(), false).unwrap();
        let paths = prepared
            .request
            .additional_model_response_field_paths
            .unwrap();
        assert!(paths.iter().any(|p| p == "/delta/stop_sequence"));
    }

    fn opts_with_provider(map: serde_json::Map<String, serde_json::Value>) -> CallOptions {
        let mut po = llmsdk_provider::shared::ProviderOptions::new();
        po.insert("amazonBedrock".into(), map);
        let mut o = opts();
        o.provider_options = Some(po);
        o.max_output_tokens = Some(1024);
        o
    }

    #[test]
    fn anthropic_thinking_enabled_injects_thinking_and_inflates_max_tokens() {
        // Mirrors upstream amazon-bedrock-chat-language-model.ts:258-271 +
        // ai-sdk #14582: `reasoningConfig: { type: 'enabled', budgetTokens }`
        // must surface as additionalModelRequestFields.thinking and shift
        // max_tokens by budget.
        let mut rc = serde_json::Map::new();
        rc.insert("type".into(), serde_json::json!("enabled"));
        rc.insert("budgetTokens".into(), serde_json::json!(2048));
        let mut p = serde_json::Map::new();
        p.insert("reasoningConfig".into(), serde_json::Value::Object(rc));
        let prepared = build_request(
            "anthropic.claude-3-7-sonnet-20250219-v1:0",
            &opts_with_provider(p),
            false,
        )
        .unwrap();
        let extras = prepared
            .request
            .additional_model_request_fields
            .as_ref()
            .and_then(|v| v.as_object())
            .expect("extras object present");
        let thinking = extras.get("thinking").expect("thinking block injected");
        assert_eq!(thinking["type"], serde_json::json!("enabled"));
        assert_eq!(thinking["budget_tokens"], serde_json::json!(2048));
        // 1024 user max + 2048 budget = 3072.
        let inference = prepared.request.inference_config.unwrap();
        assert_eq!(inference.max_tokens, Some(3072));
    }

    #[test]
    fn anthropic_adaptive_thinking_with_display_routes_to_thinking_adaptive() {
        // Mirrors upstream ai-sdk #14582: `reasoningConfig: { type:
        // 'adaptive', display }` → `thinking: { type: 'adaptive', display }`.
        let mut rc = serde_json::Map::new();
        rc.insert("type".into(), serde_json::json!("adaptive"));
        rc.insert("display".into(), serde_json::json!("summarized"));
        let mut p = serde_json::Map::new();
        p.insert("reasoningConfig".into(), serde_json::Value::Object(rc));
        let prepared = build_request(
            "anthropic.claude-opus-4-7-20251001-v1:0",
            &opts_with_provider(p),
            false,
        )
        .unwrap();
        let extras = prepared
            .request
            .additional_model_request_fields
            .as_ref()
            .and_then(|v| v.as_object())
            .expect("extras object present");
        let thinking = extras.get("thinking").expect("thinking block injected");
        assert_eq!(thinking["type"], serde_json::json!("adaptive"));
        assert_eq!(thinking["display"], serde_json::json!("summarized"));
    }
}
