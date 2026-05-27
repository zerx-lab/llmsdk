//! `OpenAI` Speech model.
//!
//! Mirrors `@ai-sdk/openai/src/speech/openai-speech-model.ts`. Posts a
//! JSON body to `/v1/audio/speech` and returns the raw audio bytes.
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_provider::error::Result;
use llmsdk_provider::shared::{ProviderOptions, RequestInfo, Warning};
use llmsdk_provider::{SpeechModel, SpeechOptions, SpeechResponseInfo, SpeechResult};
use llmsdk_provider_utils::http::{JsonRequest, post_json_for_bytes};
use llmsdk_provider_utils::time::rfc3339_now;
use serde::{Deserialize, Serialize};

use crate::config::Inner;
use crate::error::rewrite_openai_error;

const SUPPORTED_FORMATS: &[&str] = &["mp3", "opus", "aac", "flac", "wav", "pcm"];

/// `OpenAI` text-to-speech model handle.
#[derive(Debug, Clone)]
pub struct OpenAiSpeechModel {
    inner: Arc<Inner>,
    model_id: String,
}

impl OpenAiSpeechModel {
    /// Construct from a fully assembled [`Inner`]. Public for cross-crate
    /// composition. End-users should prefer the provider builder's
    /// `speech(...)` factory.
    #[must_use]
    pub fn new(inner: Arc<Inner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    fn endpoint(&self) -> String {
        self.inner.endpoint("/audio/speech", &self.model_id)
    }
}

#[derive(Debug, Clone, Serialize)]
struct SpeechRequest<'a> {
    model: &'a str,
    input: &'a str,
    voice: &'a str,
    response_format: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<&'a str>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct OpenAiSpeechProviderOptions {
    instructions: Option<String>,
    speed: Option<f32>,
}

fn parse_options(opts: Option<&ProviderOptions>) -> OpenAiSpeechProviderOptions {
    let Some(map) = opts else {
        return OpenAiSpeechProviderOptions::default();
    };
    let Some(slot) = map.get("openai") else {
        return OpenAiSpeechProviderOptions::default();
    };
    serde_json::from_value::<OpenAiSpeechProviderOptions>(serde_json::Value::Object(slot.clone()))
        .unwrap_or_default()
}

#[async_trait]
impl SpeechModel for OpenAiSpeechModel {
    fn provider(&self) -> &str {
        self.inner.provider_id()
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn do_generate(&self, options: SpeechOptions) -> Result<SpeechResult> {
        let mut warnings = Vec::new();
        let provider_opts = parse_options(options.provider_options.as_ref());

        if options.language.is_some() {
            warnings.push(Warning::Unsupported {
                feature: "language".to_owned(),
                details: Some("OpenAI speech models do not support language selection".to_owned()),
            });
        }

        let voice = options.voice.as_deref().unwrap_or("alloy");
        let mut response_format = "mp3";
        if let Some(fmt) = options.output_format.as_deref() {
            if SUPPORTED_FORMATS.contains(&fmt) {
                response_format = fmt;
            } else {
                warnings.push(Warning::Unsupported {
                    feature: "outputFormat".to_owned(),
                    details: Some(format!(
                        "Unsupported output format: {fmt}. Using mp3 instead."
                    )),
                });
            }
        }

        let speed = provider_opts.speed.or(options.speed);
        let instructions = provider_opts
            .instructions
            .clone()
            .or_else(|| options.instructions.clone());

        let request = SpeechRequest {
            model: &self.model_id,
            input: &options.text,
            voice,
            response_format,
            speed,
            instructions: instructions.as_deref(),
        };
        let request_body = serde_json::to_value(&request).ok();

        let mut headers = self.inner.headers.clone();
        if let Some(extra) = &options.headers {
            for (n, v) in extra {
                headers.insert(n.clone(), v.clone());
            }
        }

        let body_bytes = serde_json::to_vec(&request).unwrap_or_default();
        let url = self.endpoint();
        self.inner
            .sign_if_needed(&mut headers, "POST", &url, &body_bytes)
            .await?;
        let mut http = JsonRequest::new(url, &request);
        http.headers = headers;

        let response = match post_json_for_bytes(&self.inner.http, http).await {
            Ok(r) => r,
            Err(err) => return Err(rewrite_openai_error(err)),
        };

        // Mirror upstream `openai-speech-model.ts:122-150`: the raw audio
        // body is exposed under `response.body` for debugging / observability,
        // separate from the parsed `audio` payload.
        let audio_bytes = response.bytes.to_vec();
        Ok(SpeechResult {
            audio: audio_bytes.clone(),
            warnings,
            request: Some(RequestInfo { body: request_body }),
            response: SpeechResponseInfo {
                timestamp: rfc3339_now(),
                model_id: self.model_id.clone(),
                headers: Some(headers_to_provider(response.headers)),
                body: Some(llmsdk_provider::shared::FileBytes::Bytes(audio_bytes)),
            },
            provider_metadata: None,
        })
    }
}

fn headers_to_provider(raw: HashMap<String, String>) -> llmsdk_provider::shared::Headers {
    raw.into_iter().map(|(k, v)| (k, Some(v))).collect()
}
