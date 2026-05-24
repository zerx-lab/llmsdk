//! HTTP, SSE and config utilities shared by llmsdk providers.
//!
//! Rust port of [`@ai-sdk/provider-utils`](https://github.com/vercel/ai/tree/main/packages/provider-utils).
//! M2 covers: `load_api_key`, header combining, JSON POST / GET against a
//! `reqwest::Client`, SSE parsing for streaming endpoints. Image / form-data /
//! retry / tool-name helpers are deferred to a later milestone.
//!
//! Every helper produces [`llmsdk_provider::ProviderError`] on failure; no
//! crate-local error type is introduced.
// Rust guideline compliant 2026-02-21

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod api_key;
pub mod headers;
pub mod http;
pub mod sse;

#[doc(inline)]
pub use api_key::load_api_key;
#[doc(inline)]
pub use headers::combine_headers;
#[doc(inline)]
pub use http::{HttpClient, JsonRequest, parse_json_response};
#[doc(inline)]
pub use sse::{SseEvent, sse_json_stream};
