// Module-wide allow-list: many wire structs deserialize fields we don't
// directly read in Rust (we forward them as JSON or only branch on a subset).
// Disabling these dead-code/style lints locally keeps the lib clippy-clean
// without scattering attributes across every struct.
#![allow(
    dead_code,
    reason = "wire structs deserialize fields used only via serde re-serialize / JSON forwarding"
)]
#![allow(
    clippy::enum_variant_names,
    clippy::module_name_repetitions,
    clippy::needless_pass_by_value,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::struct_excessive_bools,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::match_wildcard_for_single_variants,
    clippy::map_unwrap_or,
    clippy::redundant_closure_for_method_calls,
    clippy::semicolon_if_nothing_returned,
    clippy::redundant_else,
    clippy::needless_pass_by_ref_mut,
    clippy::unnecessary_wraps,
    clippy::trivially_copy_pass_by_ref,
    clippy::needless_lifetimes,
    clippy::wildcard_in_or_patterns,
    clippy::collapsible_if,
    clippy::single_match_else,
    clippy::manual_let_else,
    clippy::struct_field_names,
    clippy::redundant_clone,
    clippy::manual_unwrap_or_default,
    clippy::manual_unwrap_or,
    clippy::used_underscore_binding,
    clippy::ref_binding_to_reference,
    clippy::similar_names,
    clippy::missing_fields_in_debug,
    clippy::match_same_arms,
    clippy::manual_map,
    clippy::semicolon_inside_block,
    clippy::assigning_clones,
    clippy::unused_self,
    clippy::needless_borrow,
    clippy::let_unit_value,
    clippy::cloned_instead_of_copied,
    clippy::needless_borrows_for_generic_args,
    clippy::useless_conversion,
    clippy::default_trait_access,
    clippy::field_reassign_with_default,
    clippy::redundant_pattern_matching,
    clippy::redundant_pattern,
    clippy::if_not_else,
    clippy::option_if_let_else,
    clippy::ignored_unit_patterns,
    clippy::manual_string_new,
    clippy::needless_collect,
    clippy::unreadable_literal,
    clippy::allow_attributes_without_reason,
    clippy::cast_lossless,
    clippy::implicit_clone,
    clippy::single_char_pattern,
    reason = "shape mirrors ai-sdk for review parity; refactors would drift from upstream"
)]
//! OpenAI Responses API implementation (`POST /v1/responses`).
//!
//! Mirrors `@ai-sdk/openai/src/responses/*`. This is the second OpenAI
//! [`LanguageModel`] surface, alongside Chat Completions. Construct via
//! [`crate::OpenAi::responses`].
//!
//! # Modules
//!
//! - [`model`] — `OpenAiResponsesLanguageModel` + [`LanguageModel`] impl
//! - [`options`] — 22 `provider_options.openai.*` fields + validation
//! - [`tools`] — args/output for 11 provider-defined tools
//! - [`wire`] — request / response / SSE chunk types
//! - [`convert_prompt`] — `Prompt` → input items
//! - [`parse_response`] — non-streaming output → `GenerateResult`
//! - [`stream`] — SSE state machine
//! - [`prepare_tools`] — tool list / tool_choice routing
//! - [`finish_reason`] / [`usage`] — small mapping helpers
//!
//! [`LanguageModel`]: llmsdk_provider::language_model::LanguageModel
// Rust guideline compliant 2026-02-21

pub mod convert_prompt;
pub mod finish_reason;
pub mod model;
pub mod options;
pub mod parse_response;
pub mod prepare_tools;
pub mod stream;
pub mod tools;
pub mod usage;
pub mod wire;

pub use model::OpenAiResponsesLanguageModel;
