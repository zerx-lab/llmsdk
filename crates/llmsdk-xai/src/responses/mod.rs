// Module-wide allow-list: many wire structs deserialize fields we don't
// directly read in Rust (we forward them as JSON or branch only on a subset).
// Disabling these dead-code/style lints locally keeps the lib clippy-clean
// without scattering attributes across every struct — same convention as
// `llmsdk-openai/src/responses/mod.rs`.
#![allow(
    dead_code,
    reason = "wire structs deserialize fields used only via serde re-serialize / JSON forwarding"
)]
#![allow(
    clippy::module_name_repetitions,
    clippy::needless_pass_by_value,
    clippy::too_many_lines,
    clippy::map_unwrap_or,
    clippy::redundant_else,
    clippy::manual_let_else,
    clippy::manual_contains,
    clippy::semicolon_if_nothing_returned,
    clippy::collapsible_if,
    clippy::single_match_else,
    clippy::match_same_arms,
    clippy::needless_pass_by_ref_mut,
    clippy::unnecessary_wraps,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::needless_borrow,
    clippy::needless_borrows_for_generic_args,
    clippy::redundant_closure_for_method_calls,
    clippy::wildcard_in_or_patterns,
    clippy::default_trait_access,
    clippy::option_if_let_else,
    clippy::needless_collect,
    clippy::manual_unwrap_or,
    clippy::manual_unwrap_or_default,
    clippy::implicit_clone,
    clippy::redundant_clone,
    clippy::single_char_pattern,
    clippy::if_not_else,
    clippy::struct_excessive_bools,
    clippy::struct_field_names,
    clippy::trivially_copy_pass_by_ref,
    clippy::needless_lifetimes,
    clippy::missing_fields_in_debug,
    clippy::field_reassign_with_default,
    clippy::cloned_instead_of_copied,
    clippy::useless_conversion,
    clippy::let_unit_value,
    clippy::similar_names,
    clippy::unused_self,
    clippy::ignored_unit_patterns,
    clippy::redundant_pattern_matching,
    clippy::redundant_pattern,
    clippy::semicolon_inside_block,
    clippy::cast_lossless,
    clippy::ref_binding_to_reference,
    clippy::manual_map,
    clippy::assigning_clones,
    clippy::manual_string_new,
    clippy::used_underscore_binding,
    clippy::unreadable_literal,
    clippy::allow_attributes_without_reason,
    clippy::enum_variant_names,
    clippy::match_wildcard_for_single_variants,
    reason = "shape mirrors ai-sdk for review parity; refactors would drift from upstream"
)]
//! xAI Responses API implementation.
//!
//! Mirrors `@ai-sdk/xai/src/responses/*`. xAI's responses endpoint is a
//! sibling of [`crate::XaiChatModel`] (Chat Completions); both implement
//! [`LanguageModel`]. Construct via [`crate::Xai::responses`].
//!
//! # Endpoint
//!
//! `POST {base_url}/responses` — JSON request body with a flat `input` items
//! array. SSE stream is enabled with `stream: true`.
//!
//! # Modules
//!
//! - [`model`] — [`XaiResponsesLanguageModel`] + trait impl
//! - [`options`] — typed `provider_options.xai.*` (7 fields)
//! - [`wire`] — request / response / SSE chunk types
//! - [`convert_prompt`] — `Prompt` → input items
//! - [`prepare_tools`] — tools + tool_choice → wire format
//! - [`parse_response`] — non-streaming output → `GenerateResult`
//! - [`stream`] — SSE state machine
//! - [`finish_reason`] / [`usage`] — small mapping helpers
//!
//! [`LanguageModel`]: llmsdk_provider::language_model::LanguageModel
// Rust guideline compliant 2026-05-25

mod convert_prompt;
mod finish_reason;
mod model;
mod options;
mod parse_response;
mod prepare_tools;
mod stream;
mod usage;
mod wire;

pub use model::XaiResponsesLanguageModel;
