//! JSON Schema → OpenAPI 3.0 Schema converter.
//!
//! Mirrors `@ai-sdk/google/src/convert-json-schema-to-openapi-schema.ts`.
//! Gemini's `responseSchema` / `functionDeclarations[].parameters` follow
//! a subset of OpenAPI 3.0 Schema (no `$schema`, no
//! `additionalProperties`, etc.). This module walks a JSON Schema 7 value
//! and emits the OpenAPI-compatible subset.
//!
//! # Conversion summary
//!
//! - Empty object schemas at the root → dropped (`None`); nested empties
//!   are preserved as `{ "type": "object" }`.
//! - `type: [..., 'null']` → `anyOf` of non-null types + `nullable: true`.
//! - `anyOf` containing a `null`-typed branch → strip null, set
//!   `nullable: true`, inline the lone non-null branch when possible.
//! - `const` → singleton `enum`.
//! - Boolean schemas → `{ "type": "boolean", "properties": {} }`.
//! - Drops unsupported keywords (`$schema`, `additionalProperties`,
//!   `unevaluatedProperties`, ...).
// Rust guideline compliant 2026-05-25

use serde_json::{Map, Value};

/// Convert a JSON Schema 7 value into Gemini's OpenAPI 3.0 subset.
///
/// Returns `None` when the schema collapses to the empty-object root case
/// (matching upstream `convertJSONSchemaToOpenAPISchema`).
///
/// Kept on the crate-public surface for callers that want the full
/// root-vs-nested distinction (e.g. integration tests). Internal call sites
/// use [`convert_json_schema_to_openapi_nested`].
#[must_use]
#[allow(dead_code, reason = "exposed crate-internal helper used by tests")]
pub(crate) fn convert_json_schema_to_openapi(schema: &Value) -> Option<Value> {
    convert_inner(schema, true)
}

/// Variant that always returns a value, suitable for nested calls.
#[must_use]
pub fn convert_json_schema_to_openapi_nested(schema: &Value) -> Value {
    convert_inner(schema, false).unwrap_or_else(|| Value::Object(Map::new()))
}

fn convert_inner(schema: &Value, is_root: bool) -> Option<Value> {
    match schema {
        Value::Bool(_) => Some(Value::Object({
            let mut m = Map::new();
            m.insert("type".into(), Value::String("boolean".into()));
            m.insert("properties".into(), Value::Object(Map::new()));
            m
        })),
        Value::Object(obj) => convert_object(obj, is_root),
        Value::Null => None,
        // Non-object, non-bool input passed through untouched (no schema
        // shape applies).
        other => Some(other.clone()),
    }
}

fn is_empty_object_schema(obj: &Map<String, Value>) -> bool {
    let type_is_object = obj.get("type").and_then(Value::as_str) == Some("object");
    let no_props = obj
        .get("properties")
        .and_then(Value::as_object)
        .is_none_or(serde_json::Map::is_empty);
    let no_additional = !obj
        .get("additionalProperties")
        .is_some_and(|v| !matches!(v, Value::Bool(false)));
    type_is_object && no_props && no_additional
}

fn convert_object(obj: &Map<String, Value>, is_root: bool) -> Option<Value> {
    if is_empty_object_schema(obj) {
        if is_root {
            return None;
        }
        let mut m = Map::new();
        m.insert("type".into(), Value::String("object".into()));
        if let Some(desc) = obj.get("description").and_then(Value::as_str) {
            m.insert("description".into(), Value::String(desc.to_owned()));
        }
        return Some(Value::Object(m));
    }

    let mut result = Map::new();

    if let Some(desc) = obj.get("description") {
        result.insert("description".into(), desc.clone());
    }
    if let Some(req) = obj.get("required") {
        result.insert("required".into(), req.clone());
    }
    if let Some(fmt) = obj.get("format") {
        result.insert("format".into(), fmt.clone());
    }
    if let Some(const_val) = obj.get("const") {
        result.insert("enum".into(), Value::Array(vec![const_val.clone()]));
    }

    if let Some(ty) = obj.get("type") {
        match ty {
            Value::Array(arr) => {
                let has_null = arr.iter().any(|v| v.as_str() == Some("null"));
                let non_null: Vec<&Value> =
                    arr.iter().filter(|v| v.as_str() != Some("null")).collect();
                if non_null.is_empty() {
                    result.insert("type".into(), Value::String("null".into()));
                } else {
                    let any_of: Vec<Value> = non_null
                        .into_iter()
                        .map(|v| {
                            let mut m = Map::new();
                            m.insert("type".into(), v.clone());
                            Value::Object(m)
                        })
                        .collect();
                    result.insert("anyOf".into(), Value::Array(any_of));
                    if has_null {
                        result.insert("nullable".into(), Value::Bool(true));
                    }
                }
            }
            other => {
                result.insert("type".into(), other.clone());
            }
        }
    }

    if let Some(enum_vals) = obj.get("enum") {
        result.insert("enum".into(), enum_vals.clone());
    }

    if let Some(props) = obj.get("properties").and_then(Value::as_object) {
        let mut new_props = Map::new();
        for (k, v) in props {
            new_props.insert(k.clone(), convert_json_schema_to_openapi_nested(v));
        }
        result.insert("properties".into(), Value::Object(new_props));
    }

    if let Some(items) = obj.get("items") {
        match items {
            Value::Array(arr) => {
                let new_items: Vec<Value> = arr
                    .iter()
                    .map(convert_json_schema_to_openapi_nested)
                    .collect();
                result.insert("items".into(), Value::Array(new_items));
            }
            other => {
                result.insert("items".into(), convert_json_schema_to_openapi_nested(other));
            }
        }
    }

    if let Some(all_of) = obj.get("allOf").and_then(Value::as_array) {
        let arr: Vec<Value> = all_of
            .iter()
            .map(convert_json_schema_to_openapi_nested)
            .collect();
        result.insert("allOf".into(), Value::Array(arr));
    }

    if let Some(any_of) = obj.get("anyOf").and_then(Value::as_array) {
        let contains_null = any_of.iter().any(|s| {
            matches!(s, Value::Object(o) if o.get("type").and_then(Value::as_str) == Some("null"))
        });
        if contains_null {
            let non_null: Vec<&Value> = any_of
                .iter()
                .filter(|s| {
                    !matches!(s, Value::Object(o) if o.get("type").and_then(Value::as_str) == Some("null"))
                })
                .collect();
            if non_null.len() == 1 {
                let converted = convert_json_schema_to_openapi_nested(non_null[0]);
                result.insert("nullable".into(), Value::Bool(true));
                if let Value::Object(map) = converted {
                    for (k, v) in map {
                        result.insert(k, v);
                    }
                }
            } else {
                let arr: Vec<Value> = non_null
                    .into_iter()
                    .map(convert_json_schema_to_openapi_nested)
                    .collect();
                result.insert("anyOf".into(), Value::Array(arr));
                result.insert("nullable".into(), Value::Bool(true));
            }
        } else {
            let arr: Vec<Value> = any_of
                .iter()
                .map(convert_json_schema_to_openapi_nested)
                .collect();
            result.insert("anyOf".into(), Value::Array(arr));
        }
    }

    if let Some(one_of) = obj.get("oneOf").and_then(Value::as_array) {
        let arr: Vec<Value> = one_of
            .iter()
            .map(convert_json_schema_to_openapi_nested)
            .collect();
        result.insert("oneOf".into(), Value::Array(arr));
    }

    if let Some(min_len) = obj.get("minLength") {
        result.insert("minLength".into(), min_len.clone());
    }

    Some(Value::Object(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn empty_object_root_drops() {
        let r = convert_json_schema_to_openapi(&json!({"type":"object"}));
        assert!(r.is_none());
    }

    #[test]
    fn empty_object_nested_preserved() {
        let r = convert_json_schema_to_openapi_nested(&json!({"type":"object"}));
        assert_eq!(r, json!({"type":"object"}));
    }

    #[test]
    fn boolean_schema_wraps() {
        let r = convert_json_schema_to_openapi(&Value::Bool(true)).unwrap();
        assert_eq!(r["type"], "boolean");
    }

    #[test]
    fn nullable_type_array_split() {
        let r = convert_json_schema_to_openapi(&json!({"type":["string","null"]})).unwrap();
        assert_eq!(r["nullable"], true);
        assert!(r["anyOf"].is_array());
    }

    #[test]
    fn const_to_enum() {
        let r = convert_json_schema_to_openapi(&json!({"const":"foo"})).unwrap();
        assert_eq!(r["enum"], json!(["foo"]));
    }

    #[test]
    fn nested_properties_recurse() {
        let r = convert_json_schema_to_openapi(&json!({
            "type":"object",
            "properties":{
                "name":{"type":"string"},
                "tags":{"type":"array","items":{"type":"string"}}
            },
            "required":["name"]
        }))
        .unwrap();
        assert_eq!(r["type"], "object");
        assert_eq!(r["required"], json!(["name"]));
        assert_eq!(r["properties"]["name"]["type"], "string");
        assert_eq!(r["properties"]["tags"]["items"]["type"], "string");
    }

    #[test]
    fn anyof_with_null_collapses_single() {
        let r = convert_json_schema_to_openapi(&json!({
            "anyOf":[{"type":"string"},{"type":"null"}]
        }))
        .unwrap();
        assert_eq!(r["nullable"], true);
        assert_eq!(r["type"], "string");
    }

    #[test]
    fn anyof_with_null_multi_keeps() {
        let r = convert_json_schema_to_openapi(&json!({
            "anyOf":[{"type":"string"},{"type":"number"},{"type":"null"}]
        }))
        .unwrap();
        assert_eq!(r["nullable"], true);
        assert_eq!(r["anyOf"].as_array().unwrap().len(), 2);
    }
}
