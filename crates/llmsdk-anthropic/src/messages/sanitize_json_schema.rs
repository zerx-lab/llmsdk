//! Strip JSON-Schema keywords Anthropic rejects in `output_config.format.schema`.
//!
//! Mirrors `anthropic/src/sanitize-json-schema.ts`. Used only on the
//! structured-output path; the original schema is preserved for
//! client-side validation.
// Rust guideline compliant 2026-02-21

use serde_json::{Map, Value};

const SUPPORTED_STRING_FORMATS: &[&str] = &[
    "date-time",
    "time",
    "date",
    "duration",
    "email",
    "hostname",
    "uri",
    "ipv4",
    "ipv6",
    "uuid",
];

/// Keys that, if present on the input, get folded into a textual
/// description instead of being emitted as constraint keywords.
const DESCRIPTION_CONSTRAINT_KEYS: &[&str] = &[
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minProperties",
    "maxProperties",
    "not",
];

/// Strip unsupported keywords from a JSON schema and tighten objects
/// (force `additionalProperties: false`).
///
/// `Value::Bool(_)` and non-object values are returned unchanged.
pub(crate) fn sanitize_json_schema(schema: &Value) -> Value {
    sanitize_definition(schema)
}

fn sanitize_definition(definition: &Value) -> Value {
    match definition {
        Value::Object(obj) => Value::Object(sanitize_object(obj)),
        // Booleans and other non-object schemas pass through verbatim.
        other => other.clone(),
    }
}

fn sanitize_object(schema: &Map<String, Value>) -> Map<String, Value> {
    let mut result = Map::new();

    if let Some(r) = schema.get("$ref") {
        result.insert("$ref".to_owned(), r.clone());
        return result;
    }

    for key in [
        "$schema",
        "$id",
        "title",
        "description",
        "default",
        "const",
        "enum",
        "type",
    ] {
        if let Some(v) = schema.get(key) {
            result.insert(key.to_owned(), v.clone());
        }
    }

    if let Some(any_of) = schema.get("anyOf").and_then(Value::as_array) {
        result.insert(
            "anyOf".to_owned(),
            Value::Array(any_of.iter().map(sanitize_definition).collect()),
        );
    } else if let Some(one_of) = schema.get("oneOf").and_then(Value::as_array) {
        result.insert(
            "anyOf".to_owned(),
            Value::Array(one_of.iter().map(sanitize_definition).collect()),
        );
    }

    if let Some(all_of) = schema.get("allOf").and_then(Value::as_array) {
        result.insert(
            "allOf".to_owned(),
            Value::Array(all_of.iter().map(sanitize_definition).collect()),
        );
    }

    for key in ["definitions", "$defs"] {
        if let Some(defs) = schema.get(key).and_then(Value::as_object) {
            let mapped: Map<String, Value> = defs
                .iter()
                .map(|(n, d)| (n.clone(), sanitize_definition(d)))
                .collect();
            result.insert(key.to_owned(), Value::Object(mapped));
        }
    }

    let is_object_schema =
        schema.get("type").is_some_and(|t| t == "object") || schema.get("properties").is_some();
    if is_object_schema {
        if let Some(props) = schema.get("properties").and_then(Value::as_object) {
            let mapped: Map<String, Value> = props
                .iter()
                .map(|(n, d)| (n.clone(), sanitize_definition(d)))
                .collect();
            result.insert("properties".to_owned(), Value::Object(mapped));
        }
        result.insert("additionalProperties".to_owned(), Value::Bool(false));
        if let Some(req) = schema.get("required") {
            result.insert("required".to_owned(), req.clone());
        }
    }

    if let Some(items) = schema.get("items") {
        let mapped = match items {
            Value::Array(arr) => Value::Array(arr.iter().map(sanitize_definition).collect()),
            single => sanitize_definition(single),
        };
        result.insert("items".to_owned(), mapped);
    }

    if let Some(format) = schema.get("format").and_then(Value::as_str)
        && SUPPORTED_STRING_FORMATS.contains(&format)
    {
        result.insert("format".to_owned(), Value::String(format.to_owned()));
    }

    if let Some(constraint_desc) = constraint_description(schema) {
        let merged = match result.get("description").and_then(Value::as_str) {
            Some(existing) => format!("{existing}\n{constraint_desc}"),
            None => constraint_desc,
        };
        result.insert("description".to_owned(), Value::String(merged));
    }

    result
}

fn constraint_description(schema: &Map<String, Value>) -> Option<String> {
    let mut parts = Vec::new();
    for key in DESCRIPTION_CONSTRAINT_KEYS {
        let Some(v) = schema.get(*key) else { continue };
        if v.is_null() || v == &Value::Bool(false) {
            continue;
        }
        parts.push(format!(
            "{}: {}",
            format_constraint_name(key),
            format_constraint_value(v)
        ));
    }
    if let Some(format) = schema.get("format").and_then(Value::as_str)
        && !SUPPORTED_STRING_FORMATS.contains(&format)
    {
        parts.push(format!("format: {format}"));
    }
    if parts.is_empty() {
        None
    } else {
        Some(format!("{}.", parts.join("; ")))
    }
}

fn format_constraint_name(key: &str) -> String {
    let mut out = String::with_capacity(key.len() + 4);
    for c in key.chars() {
        if c.is_ascii_uppercase() {
            out.push(' ');
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

fn format_constraint_value(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn object_forces_additional_properties_false() {
        let input = json!({"type": "object", "properties": {"x": {"type": "number"}}});
        let out = sanitize_json_schema(&input);
        assert_eq!(out["additionalProperties"], Value::Bool(false));
        assert_eq!(out["properties"]["x"]["type"], "number");
    }

    #[test]
    fn one_of_renamed_to_any_of() {
        let input = json!({"oneOf": [{"type": "string"}, {"type": "number"}]});
        let out = sanitize_json_schema(&input);
        assert!(out.get("oneOf").is_none());
        assert_eq!(out["anyOf"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn unsupported_format_demoted_to_description() {
        let input = json!({"type": "string", "format": "color"});
        let out = sanitize_json_schema(&input);
        assert!(out.get("format").is_none());
        assert_eq!(out["description"], "format: color.");
    }

    #[test]
    fn supported_format_kept() {
        let input = json!({"type": "string", "format": "uuid"});
        let out = sanitize_json_schema(&input);
        assert_eq!(out["format"], "uuid");
    }

    #[test]
    fn numeric_constraints_become_description() {
        let input = json!({"type": "number", "minimum": 0, "maximum": 10});
        let out = sanitize_json_schema(&input);
        let desc = out["description"].as_str().unwrap();
        assert!(desc.contains("minimum: 0"));
        assert!(desc.contains("maximum: 10"));
        assert!(desc.ends_with('.'));
    }

    #[test]
    fn camel_case_constraint_names_lowercased() {
        let input = json!({"type": "number", "exclusiveMinimum": 5});
        let out = sanitize_json_schema(&input);
        assert!(
            out["description"]
                .as_str()
                .unwrap()
                .contains("exclusive minimum: 5")
        );
    }

    #[test]
    fn ref_short_circuits() {
        let input = json!({"$ref": "#/defs/foo", "type": "object", "additionalProperties": true});
        let out = sanitize_json_schema(&input);
        assert_eq!(out, json!({"$ref": "#/defs/foo"}));
    }

    #[test]
    fn boolean_definition_passes_through() {
        let out = sanitize_json_schema(&Value::Bool(true));
        assert_eq!(out, Value::Bool(true));
    }
}
