//! xAI image-generation wire types + inline base64 decoder.
//!
//! Mirrors the request/response shapes embedded inside
//! `xai-image-model.ts`. xAI does **not** speak the `OpenAI` image wire shape
//! — every field is provider-specific.
// Rust guideline compliant 2026-05-25

use serde::{Deserialize, Serialize};

/// Request body for `POST /v1/images/generations` and `POST /v1/images/edits`.
///
/// Fields with `skip_serializing_if = "Option::is_none"` are dropped from
/// the wire when not set — xAI rejects `null` for several of them.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ImageRequest {
    pub model: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    pub response_format: ResponseFormat,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sync_mode: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Single source file (`/images/edits` only — set when exactly one
    /// `ImageOptions::files` entry is present).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<ImageReference>,
    /// Multiple source files (`/images/edits` only — set when two or more
    /// `ImageOptions::files` entries are present).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<ImageReference>>,
}

/// Wire-form image reference. Matches the upstream
/// `{ url, type: 'image_url' }` shape — `url` may be either a plain URL or a
/// `data:<media>;base64,<payload>` URI.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ImageReference {
    pub url: String,
    #[serde(rename = "type")]
    pub ref_type: &'static str,
}

impl ImageReference {
    /// Build a `{ url, type: "image_url" }` reference.
    pub(crate) fn image_url(url: String) -> Self {
        Self {
            url,
            ref_type: "image_url",
        }
    }
}

/// `response_format` field — xAI accepts `b64_json` only. We always send it
/// to keep the response shape deterministic.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ResponseFormat {
    B64Json,
}

/// Response body for both image endpoints (subset of upstream).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct ImageResponse {
    pub data: Vec<ImageData>,
    #[serde(default)]
    pub usage: Option<ImageUsage>,
}

/// One image entry inside [`ImageResponse::data`].
///
/// xAI returns either `b64_json` (preferred) or `url` per entry. The model
/// downloads the URL when no inline payload is present.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct ImageData {
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub b64_json: Option<String>,
    #[serde(default)]
    pub revised_prompt: Option<String>,
}

/// `usage` block returned by xAI image endpoints.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct ImageUsage {
    /// xAI-specific cost-in-USD ticks counter.
    #[serde(default)]
    pub cost_in_usd_ticks: Option<u64>,
}

// ---- base64 decoding --------------------------------------------------

/// Minimal RFC 4648 base64 decoder.
///
/// We intentionally avoid a third-party `base64` dependency to honor the
/// project's no-new-deps rule. Accepts standard alphabet (`+/`) with
/// optional `=` padding; rejects whitespace, urlsafe alphabet, and any
/// non-alphabet byte with [`Base64Error`].
pub(crate) fn base64_decode(input: &str) -> Result<Vec<u8>, Base64Error> {
    let bytes = input.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return Err(Base64Error::Length);
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks_exact(4) {
        let (b0, p0) = decode_byte(chunk[0])?;
        let (b1, p1) = decode_byte(chunk[1])?;
        let (b2, p2) = decode_byte(chunk[2])?;
        let (b3, p3) = decode_byte(chunk[3])?;
        // Padding may only appear at positions 2 and/or 3, never earlier.
        if p0 || p1 {
            return Err(Base64Error::Padding);
        }
        let n =
            (u32::from(b0) << 18) | (u32::from(b1) << 12) | (u32::from(b2) << 6) | u32::from(b3);
        // Mask before casting so clippy's truncation lint is satisfied;
        // the masks are no-ops at runtime (each byte fits in 8 bits).
        out.push(((n >> 16) & 0xFF) as u8);
        if !p2 {
            out.push(((n >> 8) & 0xFF) as u8);
        }
        if !p3 {
            // p2 without p3 is illegal: data after padding.
            if p2 {
                return Err(Base64Error::Padding);
            }
            out.push((n & 0xFF) as u8);
        }
    }
    Ok(out)
}

/// Encode a byte slice as standard RFC 4648 base64 (`+/`, with `=` padding).
///
/// Used to build `data:<media>;base64,...` URIs for `/images/edits`
/// requests, mirroring upstream's [`convertImageModelFileToDataUri`].
pub(crate) fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    let mut chunks = input.chunks_exact(3);
    for chunk in chunks.by_ref() {
        let n = (u32::from(chunk[0]) << 16) | (u32::from(chunk[1]) << 8) | u32::from(chunk[2]);
        out.push(ALPHABET[((n >> 18) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 6) & 0x3F) as usize] as char);
        out.push(ALPHABET[(n & 0x3F) as usize] as char);
    }
    let rem = chunks.remainder();
    match rem.len() {
        0 => {}
        1 => {
            let n = u32::from(rem[0]) << 16;
            out.push(ALPHABET[((n >> 18) & 0x3F) as usize] as char);
            out.push(ALPHABET[((n >> 12) & 0x3F) as usize] as char);
            out.push('=');
            out.push('=');
        }
        2 => {
            let n = (u32::from(rem[0]) << 16) | (u32::from(rem[1]) << 8);
            out.push(ALPHABET[((n >> 18) & 0x3F) as usize] as char);
            out.push(ALPHABET[((n >> 12) & 0x3F) as usize] as char);
            out.push(ALPHABET[((n >> 6) & 0x3F) as usize] as char);
            out.push('=');
        }
        _ => unreachable!("chunks_exact remainder is always < 3"),
    }
    out
}

/// Map one base64 byte to its 6-bit value; the `bool` flags `=` (padding).
fn decode_byte(b: u8) -> Result<(u8, bool), Base64Error> {
    Ok(match b {
        b'A'..=b'Z' => (b - b'A', false),
        b'a'..=b'z' => (b - b'a' + 26, false),
        b'0'..=b'9' => (b - b'0' + 52, false),
        b'+' => (62, false),
        b'/' => (63, false),
        b'=' => (0, true),
        _ => return Err(Base64Error::Byte(b)),
    })
}

/// Reasons [`base64_decode`] can fail.
#[derive(Debug)]
pub(crate) enum Base64Error {
    Length,
    Padding,
    Byte(u8),
}

impl std::fmt::Display for Base64Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Length => f.write_str("input length is not a multiple of 4"),
            Self::Padding => f.write_str("misplaced padding"),
            Self::Byte(b) => write!(f, "non-alphabet byte 0x{b:02x}"),
        }
    }
}

impl std::error::Error for Base64Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_round_trips_known_vectors() {
        // RFC 4648 §10 test vectors.
        let cases: &[(&str, &[u8])] = &[
            ("", b""),
            ("Zg==", b"f"),
            ("Zm8=", b"fo"),
            ("Zm9v", b"foo"),
            ("Zm9vYg==", b"foob"),
            ("Zm9vYmE=", b"fooba"),
            ("Zm9vYmFy", b"foobar"),
        ];
        for (encoded, raw) in cases {
            let decoded = base64_decode(encoded).expect("valid base64");
            assert_eq!(&decoded, raw, "decode vector {encoded}");
            let re_encoded = base64_encode(raw);
            assert_eq!(re_encoded.as_str(), *encoded, "encode vector {encoded}");
        }
    }

    #[test]
    fn base64_rejects_invalid_input() {
        assert!(base64_decode("abc").is_err()); // wrong length
        assert!(base64_decode("ab=c").is_err()); // misplaced padding
        assert!(base64_decode("ab!d").is_err()); // bad byte
    }

    #[test]
    fn image_request_skips_optional_none_fields() {
        let req = ImageRequest {
            model: "grok-imagine-image".into(),
            prompt: "a hat".into(),
            n: None,
            response_format: ResponseFormat::B64Json,
            aspect_ratio: None,
            output_format: None,
            sync_mode: None,
            resolution: None,
            quality: None,
            user: None,
            image: None,
            images: None,
        };
        let value = serde_json::to_value(&req).unwrap();
        let obj = value.as_object().unwrap();
        assert!(obj.contains_key("model"));
        assert!(obj.contains_key("prompt"));
        assert!(obj.contains_key("response_format"));
        assert!(!obj.contains_key("n"));
        assert!(!obj.contains_key("aspect_ratio"));
        assert!(!obj.contains_key("image"));
        assert!(!obj.contains_key("images"));
    }

    #[test]
    fn image_reference_serializes_with_type_tag() {
        let r = ImageReference::image_url("data:image/png;base64,iVBOR".into());
        let value = serde_json::to_value(&r).unwrap();
        assert_eq!(value["type"], "image_url");
        assert_eq!(value["url"], "data:image/png;base64,iVBOR");
    }

    #[test]
    fn image_response_decodes_url_or_b64_or_revised_prompt() {
        let body = serde_json::json!({
            "data": [
                { "url": "https://x.ai/img/1.png" },
                { "b64_json": "Zg==", "revised_prompt": "a fancy fox" }
            ],
            "usage": { "cost_in_usd_ticks": 42 }
        });
        let parsed: ImageResponse = serde_json::from_value(body).unwrap();
        assert_eq!(parsed.data.len(), 2);
        assert_eq!(
            parsed.data[0].url.as_deref(),
            Some("https://x.ai/img/1.png")
        );
        assert!(parsed.data[0].b64_json.is_none());
        assert_eq!(parsed.data[1].b64_json.as_deref(), Some("Zg=="));
        assert_eq!(
            parsed.data[1].revised_prompt.as_deref(),
            Some("a fancy fox")
        );
        assert_eq!(parsed.usage.unwrap().cost_in_usd_ticks, Some(42));
    }
}
