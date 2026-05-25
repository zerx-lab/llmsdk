//! Unified facade for llmsdk — Rust port of the [Vercel ai-sdk](https://github.com/vercel/ai)
//! provider ecosystem.
//!
//! This crate is a thin umbrella over the workspace's [`llmsdk-provider`]
//! trait crate plus one feature-gated module per concrete provider. Pick
//! the providers you need with cargo features; everything else stays out
//! of the dependency graph.
//!
//! For tighter control over what compiles, depend on the individual
//! `llmsdk-*` crates directly — they remain the unit of versioning and
//! release. This facade exists purely for ergonomics.
//!
//! # Quick start
//!
//! `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! llmsdk = { version = "0.1", features = ["openai", "anthropic"] }
//! ```
//!
//! `main.rs`:
//!
//! ```ignore
//! use llmsdk::openai::OpenAi;
//! use llmsdk::{CallOptions, LanguageModel, Message};
//!
//! let openai = OpenAi::builder().api_key("sk-...").build()?;
//! let model = openai.chat("gpt-4o-mini");
//! let result = model
//!     .do_generate(CallOptions {
//!         prompt: vec![Message::System {
//!             content: "Be concise.".into(),
//!             provider_options: None,
//!         }],
//!         ..Default::default()
//!     })
//!     .await?;
//! ```
//!
//! # Features
//!
//! Each provider is gated behind a cargo feature of the same name; all
//! providers are off by default to keep clean builds lean. The `full`
//! feature opts into every provider and the utilities crate at once.
//!
//! | Feature          | Re-exported crate          | Path                  |
//! |------------------|----------------------------|-----------------------|
//! | `openai`         | `llmsdk-openai`            | [`openai`]            |
//! | `anthropic`      | `llmsdk-anthropic`         | [`anthropic`]         |
//! | `xai`            | `llmsdk-xai`               | [`xai`]               |
//! | `mistral`        | `llmsdk-mistral`           | [`mistral`]           |
//! | `azure`          | `llmsdk-azure`             | [`azure`]             |
//! | `cohere`         | `llmsdk-cohere`            | [`cohere`]            |
//! | `google`         | `llmsdk-google`            | [`google`]            |
//! | `anthropic-aws`  | `llmsdk-anthropic-aws`     | [`anthropic_aws`]     |
//! | `amazon-bedrock` | `llmsdk-amazon-bedrock`    | [`amazon_bedrock`]    |
//! | `google-vertex`  | `llmsdk-google-vertex`     | [`google_vertex`]     |
//! | `utils`          | `llmsdk-provider-utils`    | [`utils`]             |
//! | `full`           | all of the above           | —                     |
//!
//! The provider-trait crate ([`llmsdk-provider`]) is always present and its
//! contents (traits, message types, error type, middleware, ...) are
//! re-exported at the crate root via a glob — so you write
//! `use llmsdk::LanguageModel` rather than
//! `use llmsdk::provider::LanguageModel`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[doc(inline)]
pub use llmsdk_provider::*;

#[cfg(feature = "utils")]
#[doc(inline)]
pub use llmsdk_provider_utils as utils;

#[cfg(feature = "openai")]
#[doc(inline)]
pub use llmsdk_openai as openai;

#[cfg(feature = "anthropic")]
#[doc(inline)]
pub use llmsdk_anthropic as anthropic;

#[cfg(feature = "xai")]
#[doc(inline)]
pub use llmsdk_xai as xai;

#[cfg(feature = "mistral")]
#[doc(inline)]
pub use llmsdk_mistral as mistral;

#[cfg(feature = "azure")]
#[doc(inline)]
pub use llmsdk_azure as azure;

#[cfg(feature = "cohere")]
#[doc(inline)]
pub use llmsdk_cohere as cohere;

#[cfg(feature = "google")]
#[doc(inline)]
pub use llmsdk_google as google;

#[cfg(feature = "anthropic-aws")]
#[doc(inline)]
pub use llmsdk_anthropic_aws as anthropic_aws;

#[cfg(feature = "amazon-bedrock")]
#[doc(inline)]
pub use llmsdk_amazon_bedrock as amazon_bedrock;

#[cfg(feature = "google-vertex")]
#[doc(inline)]
pub use llmsdk_google_vertex as google_vertex;
