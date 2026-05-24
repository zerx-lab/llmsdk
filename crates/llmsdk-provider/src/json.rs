//! JSON value re-exports used across the provider surface.
//!
//! Mirrors `@ai-sdk/provider`'s `json-value` module. We re-export
//! [`serde_json::Value`] and friends so downstream crates have a single
//! import point, plus a typed [`JsonSchema`] alias backed by
//! [`schemars::Schema`] for the function-tool / response-format surfaces.
// Rust guideline compliant 2026-02-21

pub use serde_json::Value as JsonValue;

/// JSON object map (`{ string: JsonValue }`), matching `serde_json::Map<String, Value>`.
pub type JsonObject = serde_json::Map<String, JsonValue>;

/// JSON schema describing a tool's input or a JSON response format.
///
/// Backed by [`schemars::Schema`] — a transparent newtype over
/// [`serde_json::Value`] that implements [`serde::Serialize`] /
/// [`serde::Deserialize`] and matches `JSONSchema7` on the wire.
///
/// # Construct
///
/// - From a literal: [`schemars::json_schema!`].
/// - From a derived type: [`schemars::schema_for!`].
/// - From an existing JSON value: `serde_json::from_value(value)` (returns
///   `Result<JsonSchema, _>` because [`schemars::Schema`] validates on
///   deserialize).
///
/// # Examples
///
/// ```
/// use llmsdk_provider::json::JsonSchema;
/// use serde_json::json;
///
/// let schema: JsonSchema = serde_json::from_value(json!({
///     "type": "object",
///     "properties": { "city": { "type": "string" } }
/// })).unwrap();
/// assert_eq!(schema.as_object().unwrap().get("type").unwrap(), "object");
/// ```
pub type JsonSchema = schemars::Schema;
