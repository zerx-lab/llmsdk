//! JSON value re-exports used across the provider surface.
//!
//! Mirrors `@ai-sdk/provider`'s `json-value` module. We re-export
//! [`serde_json::Value`] and friends so downstream crates have a single
//! import point and we can swap the underlying representation later
//! without breaking callers.
// Rust guideline compliant 2026-02-21

pub use serde_json::Value as JsonValue;

/// JSON object map (`{ string: JsonValue }`), matching `serde_json::Map<String, Value>`.
pub type JsonObject = serde_json::Map<String, JsonValue>;

/// JSON schema (currently aliased to [`JsonValue`]).
///
/// `@ai-sdk/provider` uses `JSONSchema7`. We keep an alias so providers can
/// continue accepting raw JSON schemas without committing to a `schemars`
/// dependency yet. May be replaced by a typed schema in a future major.
pub type JsonSchema = JsonValue;
