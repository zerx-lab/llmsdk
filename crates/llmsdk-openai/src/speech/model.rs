//! `OpenAI` Speech model.
//!
//! Mirrors `@ai-sdk/openai/src/speech/openai-speech-model.ts`. Posts a
//! JSON body to `/v1/audio/speech` and returns the raw audio bytes.
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use llmsdk_provider::error::Result;
use llmsdk_provider::shared::{ProviderOptions, RequestInfo, Warning};
use llmsdk_provider::{SpeechModel, SpeechOptions, SpeechResponseInfo, SpeechResult};
use llmsdk_provider_utils::http::{JsonRequest, post_json_for_bytes};
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

        Ok(SpeechResult {
            audio: response.bytes.to_vec(),
            warnings,
            request: Some(RequestInfo { body: request_body }),
            response: SpeechResponseInfo {
                timestamp: rfc3339_now(),
                model_id: self.model_id.clone(),
                headers: Some(headers_to_provider(response.headers)),
                body: None,
            },
            provider_metadata: None,
        })
    }
}

fn headers_to_provider(raw: HashMap<String, String>) -> llmsdk_provider::shared::Headers {
    raw.into_iter().map(|(k, v)| (k, Some(v))).collect()
}

/// Render the current wall-clock time as an RFC 3339 string.
fn rfc3339_now() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let nsecs = now.subsec_nanos();
    rfc3339_from_unix(secs, nsecs)
}

/// Minimal Unix epoch -> RFC 3339 converter (UTC). Avoids pulling in `chrono`.
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    reason = "civil-from-days algorithm (Howard Hinnant) trades safety lints for arithmetic clarity; values stay in u32 / i64 ranges for any plausible epoch"
)]
fn rfc3339_from_unix(secs: u64, nsecs: u32) -> String {
    // Algorithm: civil_from_days, per Howard Hinnant.
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let h = (rem / 3600) as u32;
    let m = ((rem % 3600) / 60) as u32;
    let s = (rem % 60) as u32;

    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = y + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{d:02}T{h:02}:{m:02}:{s:02}.{ms:03}Z",
        ms = nsecs / 1_000_000
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc3339_zero_epoch() {
        assert_eq!(rfc3339_from_unix(0, 0), "1970-01-01T00:00:00.000Z");
    }

    #[test]
    fn rfc3339_known_value() {
        // 1_700_000_000s = 2023-11-14T22:13:20Z
        assert_eq!(
            rfc3339_from_unix(1_700_000_000, 0),
            "2023-11-14T22:13:20.000Z"
        );
    }
}
