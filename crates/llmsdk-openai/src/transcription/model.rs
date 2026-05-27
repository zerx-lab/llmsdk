//! `OpenAI` Transcription model.
//!
//! Mirrors `@ai-sdk/openai/src/transcription/openai-transcription-model.ts`.
//! Posts multipart audio to `/v1/audio/transcriptions` and returns text plus
//! per-segment timing.
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_provider::error::Result;
use llmsdk_provider::shared::{FileBytes, ProviderOptions};
use llmsdk_provider::{
    TranscriptionModel, TranscriptionOptions, TranscriptionResponseInfo, TranscriptionResult,
    TranscriptionSegment,
};
use llmsdk_provider_utils::http::{RawRequest, post_raw};
use llmsdk_provider_utils::multipart::Multipart;
use llmsdk_provider_utils::time::rfc3339_now;
use serde::Deserialize;

use crate::config::Inner;
use crate::error::rewrite_openai_error;

/// Models that prefer the lighter `json` response format over `verbose_json`.
///
/// Mirrors upstream `openai-transcription-model.ts:162-165`. Note that
/// `gpt-4o-transcribe-diarize` is intentionally **not** listed — upstream
/// treats it like every other non-whisper model and routes it through
/// `verbose_json` so segment timings survive.
const GPT_4O_TRANSCRIBE_MODELS: &[&str] = &["gpt-4o-transcribe", "gpt-4o-mini-transcribe"];

/// `OpenAI` speech-to-text model handle.
#[derive(Debug, Clone)]
pub struct OpenAiTranscriptionModel {
    inner: Arc<Inner>,
    model_id: String,
}

impl OpenAiTranscriptionModel {
    /// Construct from a fully assembled [`Inner`]. Public for cross-crate
    /// composition. End-users should prefer the provider builder's
    /// `transcription(...)` factory.
    #[must_use]
    pub fn new(inner: Arc<Inner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    fn endpoint(&self) -> String {
        self.inner.endpoint("/audio/transcriptions", &self.model_id)
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct OpenAiTranscriptionProviderOptions {
    include: Option<Vec<String>>,
    language: Option<String>,
    prompt: Option<String>,
    temperature: Option<f32>,
    /// `"word"` / `"segment"`.
    timestamp_granularities: Option<Vec<String>>,
}

/// Returns `Some(_)` only when the caller actually supplied
/// `provider_options.openai.*`. Mirrors upstream `if (openAIOptions)` gate
/// in `openai-transcription-model.ts:161` — without that gate the zod-style
/// defaults (`temperature: 0`, `timestamp_granularities: ['segment']`) would
/// leak into the wire even for callers that omitted the namespace entirely.
fn parse_options(opts: Option<&ProviderOptions>) -> Option<OpenAiTranscriptionProviderOptions> {
    let map = opts?;
    let slot = map.get("openai")?;
    let mut parsed = serde_json::from_value::<OpenAiTranscriptionProviderOptions>(
        serde_json::Value::Object(slot.clone()),
    )
    .unwrap_or_default();
    // Apply the same defaults the upstream zod schema applies on parse —
    // see `openai-transcription-model-options.ts:41` (`.default(0)`) and
    // `:47-49` (`.default(['segment'])`). These only fire once we know the
    // caller opted into the namespace, matching upstream's `if (openAIOptions)`
    // gate.
    if parsed.temperature.is_none() {
        parsed.temperature = Some(0.0);
    }
    if parsed.timestamp_granularities.is_none() {
        parsed.timestamp_granularities = Some(vec!["segment".to_owned()]);
    }
    Some(parsed)
}

#[async_trait]
impl TranscriptionModel for OpenAiTranscriptionModel {
    fn provider(&self) -> &str {
        self.inner.provider_id()
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn do_generate(&self, options: TranscriptionOptions) -> Result<TranscriptionResult> {
        let provider_opts = parse_options(options.provider_options.as_ref());

        let bytes = audio_to_bytes(&options.audio);
        let extension = media_type_to_extension(&options.media_type);

        let mut mp = Multipart::new();
        mp.file(
            "file",
            &format!("audio.{extension}"),
            Some(&options.media_type),
            &bytes,
        );
        mp.text("model", &self.model_id);

        // Mirror upstream `openai-transcription-model.ts:156-178`: whisper-1
        // always pins `verbose_json`, but every other model only emits
        // `response_format` when the caller actually supplied
        // `provider_options.openai.*` — without that gate the server-side
        // default would never be reachable.
        let is_gpt_4o = GPT_4O_TRANSCRIBE_MODELS.contains(&self.model_id.as_str());
        if self.model_id == "whisper-1" {
            mp.text("response_format", "verbose_json");
        } else if provider_opts.is_some() {
            mp.text(
                "response_format",
                if is_gpt_4o { "json" } else { "verbose_json" },
            );
        }

        if let Some(opts) = &provider_opts {
            if let Some(includes) = &opts.include {
                for v in includes {
                    mp.text("include[]", v);
                }
            }
            if let Some(lang) = &opts.language {
                mp.text("language", lang);
            }
            if let Some(prompt) = &opts.prompt {
                mp.text("prompt", prompt);
            }
            if let Some(temp) = opts.temperature {
                mp.text("temperature", &temp.to_string());
            }
            if let Some(granularities) = &opts.timestamp_granularities {
                for v in granularities {
                    mp.text("timestamp_granularities[]", v);
                }
            }
        }

        let (boundary, body) = mp.finish();
        let content_type = format!("multipart/form-data; boundary={boundary}");

        let mut headers = self.inner.headers.clone();
        if let Some(extra) = &options.headers {
            for (n, v) in extra {
                headers.insert(n.clone(), v.clone());
            }
        }

        let url = self.endpoint();
        self.inner
            .sign_if_needed(&mut headers, "POST", &url, &body)
            .await?;
        let mut req = RawRequest::new(url, body, content_type);
        req.headers = headers;

        let envelope = match post_raw::<WireTranscriptionResponse>(&self.inner.http, req).await {
            Ok(r) => r,
            Err(err) => return Err(rewrite_openai_error(err)),
        };
        let response = envelope.value;

        // Prefer segments; fall back to per-word timings; else empty.
        let segments = if let Some(segs) = &response.segments {
            segs.iter()
                .map(|s| TranscriptionSegment {
                    text: s.text.clone(),
                    start_second: s.start,
                    end_second: s.end,
                })
                .collect()
        } else if let Some(words) = &response.words {
            words
                .iter()
                .map(|w| TranscriptionSegment {
                    text: w.word.clone(),
                    start_second: w.start,
                    end_second: w.end,
                })
                .collect()
        } else {
            Vec::new()
        };

        let language = response
            .language
            .as_deref()
            .and_then(language_name_to_iso639_1)
            .or(response.language);

        Ok(TranscriptionResult {
            text: response.text,
            segments,
            language,
            duration_in_seconds: response.duration,
            warnings: Vec::new(),
            // OpenAI transcription uses multipart/form-data; the raw body is
            // binary and cannot be losslessly stringified, so we leave the
            // upstream-aligned telemetry slot empty here (matches upstream
            // `openai-transcription-model.ts` not setting `request.body`).
            request: None,
            response: TranscriptionResponseInfo {
                timestamp: rfc3339_now(),
                model_id: self.model_id.clone(),
                headers: Some(headers_to_provider(envelope.headers)),
                body: None,
            },
            provider_metadata: None,
        })
    }
}

fn audio_to_bytes(audio: &FileBytes) -> Vec<u8> {
    match audio {
        FileBytes::Bytes(b) => b.clone(),
        FileBytes::Base64(s) => decode_base64(s).unwrap_or_default(),
    }
}

fn decode_base64(input: &str) -> std::result::Result<Vec<u8>, &'static str> {
    let bytes = input.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return Err("length not a multiple of 4");
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks_exact(4) {
        let (b0, _) = decode_byte(chunk[0])?;
        let (b1, _) = decode_byte(chunk[1])?;
        let (b2, p2) = decode_byte(chunk[2])?;
        let (b3, p3) = decode_byte(chunk[3])?;
        out.push((b0 << 2) | (b1 >> 4));
        if !p2 {
            out.push(((b1 & 0x0F) << 4) | (b2 >> 2));
        }
        if !p3 {
            out.push(((b2 & 0x03) << 6) | b3);
        }
    }
    Ok(out)
}

fn decode_byte(c: u8) -> std::result::Result<(u8, bool), &'static str> {
    match c {
        b'A'..=b'Z' => Ok((c - b'A', false)),
        b'a'..=b'z' => Ok((c - b'a' + 26, false)),
        b'0'..=b'9' => Ok((c - b'0' + 52, false)),
        b'+' => Ok((62, false)),
        b'/' => Ok((63, false)),
        b'=' => Ok((0, true)),
        _ => Err("invalid base64 byte"),
    }
}

fn media_type_to_extension(media_type: &str) -> &'static str {
    match media_type {
        "audio/wav" | "audio/x-wav" | "audio/wave" => "wav",
        "audio/mp3" | "audio/mpeg" => "mp3",
        "audio/m4a" | "audio/mp4" | "audio/x-m4a" => "m4a",
        "audio/webm" => "webm",
        "audio/ogg" | "audio/opus" => "ogg",
        "audio/flac" => "flac",
        _ => "bin",
    }
}

fn language_name_to_iso639_1(name: &str) -> Option<String> {
    // Mirror ai-sdk's `languageMap` verbatim
    // (openai-transcription-model.ts:41-99). The Whisper response surfaces
    // language as the lowercase English name; this table maps it back to the
    // ISO 639-1 code that the API also accepts on input.
    Some(
        match name.to_ascii_lowercase().as_str() {
            "afrikaans" => "af",
            "arabic" => "ar",
            "armenian" => "hy",
            "azerbaijani" => "az",
            "belarusian" => "be",
            "bosnian" => "bs",
            "bulgarian" => "bg",
            "catalan" => "ca",
            "chinese" => "zh",
            "croatian" => "hr",
            "czech" => "cs",
            "danish" => "da",
            "dutch" => "nl",
            "english" => "en",
            "estonian" => "et",
            "finnish" => "fi",
            "french" => "fr",
            "galician" => "gl",
            "german" => "de",
            "greek" => "el",
            "hebrew" => "he",
            "hindi" => "hi",
            "hungarian" => "hu",
            "icelandic" => "is",
            "indonesian" => "id",
            "italian" => "it",
            "japanese" => "ja",
            "kannada" => "kn",
            "kazakh" => "kk",
            "korean" => "ko",
            "latvian" => "lv",
            "lithuanian" => "lt",
            "macedonian" => "mk",
            "malay" => "ms",
            "marathi" => "mr",
            "maori" => "mi",
            "nepali" => "ne",
            "norwegian" => "no",
            "persian" => "fa",
            "polish" => "pl",
            "portuguese" => "pt",
            "romanian" => "ro",
            "russian" => "ru",
            "serbian" => "sr",
            "slovak" => "sk",
            "slovenian" => "sl",
            "spanish" => "es",
            "swahili" => "sw",
            "swedish" => "sv",
            "tagalog" => "tl",
            "tamil" => "ta",
            "thai" => "th",
            "turkish" => "tr",
            "ukrainian" => "uk",
            "urdu" => "ur",
            "vietnamese" => "vi",
            "welsh" => "cy",
            _ => return None,
        }
        .to_owned(),
    )
}

fn headers_to_provider(raw: HashMap<String, String>) -> llmsdk_provider::shared::Headers {
    raw.into_iter().map(|(k, v)| (k, Some(v))).collect()
}

// -------- wire types ----------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
struct WireTranscriptionResponse {
    text: String,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    duration: Option<f64>,
    #[serde(default)]
    words: Option<Vec<WireWord>>,
    #[serde(default)]
    segments: Option<Vec<WireSegment>>,
}

#[derive(Debug, Clone, Deserialize)]
struct WireWord {
    word: String,
    start: f64,
    end: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct WireSegment {
    text: String,
    start: f64,
    end: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_known_language_names() {
        assert_eq!(language_name_to_iso639_1("english"), Some("en".into()));
        assert_eq!(language_name_to_iso639_1("Chinese"), Some("zh".into()));
        assert_eq!(language_name_to_iso639_1("unknown"), None);
    }

    #[test]
    fn extension_for_known_audio_types() {
        assert_eq!(media_type_to_extension("audio/wav"), "wav");
        assert_eq!(media_type_to_extension("audio/mpeg"), "mp3");
        assert_eq!(media_type_to_extension("audio/m4a"), "m4a");
    }

    #[test]
    fn decode_base64_passes_audio_through() {
        // Base64 "AAEC" = [0,1,2]
        assert_eq!(decode_base64("AAEC").unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn parse_options_applies_zod_defaults_when_namespace_present() {
        // Mirrors upstream openai-transcription-model-options.ts:41 + :47-49:
        // when the caller opts into provider_options.openai (even an empty
        // map), zod fills `temperature: 0` and `timestampGranularities:
        // ['segment']`. Without the second default the multipart form omits
        // `timestamp_granularities[]` and verbose_json responses come back
        // without segment timestamps.
        let mut po = ProviderOptions::new();
        po.insert("openai".into(), serde_json::Map::new());
        let parsed = parse_options(Some(&po)).expect("namespace present must yield Some");
        assert_eq!(parsed.temperature, Some(0.0));
        assert_eq!(
            parsed.timestamp_granularities,
            Some(vec!["segment".to_owned()])
        );
    }

    #[test]
    fn parse_options_skips_when_namespace_absent() {
        // Mirrors upstream `if (openAIOptions)` gate — no namespace, no
        // defaults applied (so callers who never opt in stay out of the
        // wire entirely).
        let po = ProviderOptions::new();
        assert!(parse_options(Some(&po)).is_none());
        assert!(parse_options(None).is_none());
    }

    #[test]
    fn gpt4o_transcribe_list_excludes_diarize() {
        // `gpt-4o-transcribe-diarize` is NOT a member of the upstream
        // `isGpt4oTranscribeModel` array (upstream lines 162-165). It must
        // therefore route through `verbose_json` like every other non-whisper
        // model, otherwise segment timings are lost.
        assert!(GPT_4O_TRANSCRIBE_MODELS.contains(&"gpt-4o-transcribe"));
        assert!(GPT_4O_TRANSCRIBE_MODELS.contains(&"gpt-4o-mini-transcribe"));
        assert!(!GPT_4O_TRANSCRIBE_MODELS.contains(&"gpt-4o-transcribe-diarize"));
    }
}
