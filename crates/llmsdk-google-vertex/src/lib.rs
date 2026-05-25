//! Google Vertex AI provider for llmsdk.
//!
//! Rust port of [`@ai-sdk/google-vertex`](https://github.com/vercel/ai/tree/main/packages/google-vertex).
//! Covers five top-level capabilities — Gemini language models, text /
//! multimodal embeddings, Imagen + Gemini image generation, Veo video
//! generation, plus three publisher-routed sub-providers (Anthropic on
//! Vertex, xAI on Vertex, MaaS open / partner models on Vertex).
//!
//! # Authentication modes
//!
//! - **Standard mode (default)**: OAuth via [`gcp_auth`]. Token providers
//!   are discovered from the ambient environment (service-account JSON,
//!   GCE metadata server, `gcloud` CLI ADC). Tokens are scoped to
//!   `https://www.googleapis.com/auth/cloud-platform`.
//! - **Express mode**: API key sent as `x-goog-api-key`. Activated by
//!   passing [`GoogleVertexBuilder::api_key`] or setting
//!   [`API_KEY_ENV_VAR`]. The endpoint URLs collapse to
//!   `https://aiplatform.googleapis.com/v1/publishers/google/...`,
//!   bypassing the regionalized project / location prefix.
//!
//! # Quick start
//!
//! ```no_run
//! # async fn doc() -> Result<(), llmsdk_provider::ProviderError> {
//! use llmsdk_google_vertex::GoogleVertex;
//! use llmsdk_provider::LanguageModel;
//! use llmsdk_provider::language_model::{CallOptions, Message};
//!
//! let provider = GoogleVertex::builder()
//!     .project("my-gcp-project")
//!     .location("us-central1")
//!     .build()
//!     .await?;
//! let model = provider.language_model("gemini-2.5-flash");
//! let _ = model
//!     .do_generate(CallOptions {
//!         prompt: vec![Message::User {
//!             content: vec![],
//!             provider_options: None,
//!         }],
//!         ..Default::default()
//!     })
//!     .await?;
//! # Ok(())
//! # }
//! ```
// Rust guideline compliant 2026-05-25

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(
    clippy::module_name_repetitions,
    clippy::needless_pass_by_value,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::option_if_let_else,
    clippy::redundant_closure_for_method_calls,
    reason = "thin wrapper over llmsdk-google / llmsdk-anthropic / llmsdk-openai \
              with a verbose option / option-bag surface for parity with ai-sdk"
)]

mod anthropic;
mod auth;
mod config;
mod embedding;
mod image;
mod language;
mod maas;
mod video;
mod xai;

pub mod tools;

pub use anthropic::{GoogleVertexAnthropic, GoogleVertexAnthropicLanguageModel};
pub use auth::{AccessTokenProvider, GcpAuthTokenProvider};
pub use config::{GoogleVertex, GoogleVertexBuilder, VertexAuthMode};
pub use embedding::GoogleVertexEmbeddingModel;
pub use image::GoogleVertexImageModel;
pub use language::GoogleVertexLanguageModel;
pub use maas::GoogleVertexMaas;
pub use video::GoogleVertexVideoModel;
pub use xai::GoogleVertexXai;

/// Default Vertex AI location used when neither builder nor env var
/// supplies one.
pub const DEFAULT_LOCATION: &str = "us-central1";

/// Express-mode base URL (apiKey auth bypasses project + location).
pub const EXPRESS_MODE_BASE_URL: &str = "https://aiplatform.googleapis.com/v1/publishers/google";

/// Env var consulted for the GCP project id.
pub const PROJECT_ENV_VAR: &str = "GOOGLE_VERTEX_PROJECT";

/// Env var consulted for the Vertex AI location / region.
pub const LOCATION_ENV_VAR: &str = "GOOGLE_VERTEX_LOCATION";

/// Env var consulted for the Express-mode API key.
pub const API_KEY_ENV_VAR: &str = "GOOGLE_VERTEX_API_KEY";

/// OAuth scope used when minting access tokens via [`gcp_auth`].
pub const CLOUD_PLATFORM_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";

/// Reported provider id for chat / language models on Vertex.
pub const PROVIDER_ID_CHAT: &str = "google.vertex.chat";

/// Reported provider id for embedding models on Vertex.
pub const PROVIDER_ID_EMBEDDING: &str = "google.vertex.embedding";

/// Reported provider id for image models on Vertex.
pub const PROVIDER_ID_IMAGE: &str = "google.vertex.image";

/// Reported provider id for video models on Vertex.
pub const PROVIDER_ID_VIDEO: &str = "google.vertex.video";

/// Reported provider id for Anthropic on Vertex.
pub const PROVIDER_ID_ANTHROPIC: &str = "google.vertex.anthropic.messages";

/// Reported provider id for xAI on Vertex.
pub const PROVIDER_ID_XAI: &str = "google.vertex.xai";

/// Reported provider id for MaaS partner / open models on Vertex.
pub const PROVIDER_ID_MAAS: &str = "google.vertex.maas";
