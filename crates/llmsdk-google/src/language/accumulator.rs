//! Streaming JSON accumulator for Gemini `partialArgs` chunks.
//!
//! Mirrors `@ai-sdk/google/src/google-json-accumulator.ts`. Gemini's
//! streaming function-call protocol emits `partialArgs[]` where each entry
//! carries a `jsonPath` (e.g. `$.recipe.ingredients[0].name`) plus exactly
//! one of {`stringValue`, `numberValue`, `boolValue`, `nullValue`} and an
//! optional `willContinue` flag. The accumulator reconstructs the final
//! object **and** emits a running JSON text delta so callers can stream it
//! incrementally as `tool-input-delta` parts.
// Rust guideline compliant 2026-05-25

use serde_json::{Map, Value};

use super::wire::WirePartialArg;

#[derive(Debug, Clone)]
enum PathSeg {
    Key(String),
    Index(usize),
}

#[derive(Debug, Clone)]
struct StackEntry {
    /// Segment used to enter the container (root carries an empty string).
    #[allow(dead_code, reason = "kept for parity with upstream/debugging")]
    segment: PathSeg,
    is_array: bool,
    child_count: u32,
}

/// Incremental builder that mirrors upstream's `GoogleJSONAccumulator`.
#[derive(Debug, Default, Clone)]
pub(crate) struct GoogleJsonAccumulator {
    accumulated: Map<String, Value>,
    json_text: String,
    path_stack: Vec<StackEntry>,
    string_open: bool,
}

impl GoogleJsonAccumulator {
    /// Cheap-clone used by the stream state machine to snapshot before
    /// applying a partial-args batch.
    pub fn clone_for_state(&self) -> Self {
        self.clone()
    }
}

impl GoogleJsonAccumulator {
    /// Process a batch of partial args, returning the text delta to emit.
    pub fn process_partial_args(&mut self, partial: &[WirePartialArg]) -> String {
        let mut delta = String::new();

        for arg in partial {
            let raw_path = arg
                .json_path
                .strip_prefix("$.")
                .unwrap_or(&arg.json_path)
                .to_owned();
            if raw_path.is_empty() {
                continue;
            }
            let segments = parse_path(&raw_path);

            // Snapshot existing leaf value as `Option<Value>` so the
            // subsequent mutable borrow does not conflict.
            let existing_owned: Option<Value> = get_nested(&self.accumulated, &segments).cloned();
            let is_string_continuation = arg.string_value.is_some() && existing_owned.is_some();

            if is_string_continuation {
                let sv = arg.string_value.as_ref().unwrap();
                let escaped = json_string_inner(sv);
                if let Some(Value::String(prev)) = existing_owned {
                    set_nested(
                        &mut self.accumulated,
                        &segments,
                        Value::String(format!("{prev}{sv}")),
                    );
                } else {
                    set_nested(&mut self.accumulated, &segments, Value::String(sv.clone()));
                }
                delta.push_str(&escaped);
                continue;
            }

            let resolved = resolve_value(arg);
            let Some((value, value_json)) = resolved else {
                continue;
            };

            set_nested(&mut self.accumulated, &segments, value.clone());
            delta.push_str(&self.emit_navigation_to(&segments, arg, &value_json));
        }

        self.json_text.push_str(&delta);
        delta
    }

    /// Close all open containers and return `(final_json, closing_delta)`.
    pub fn finalize(self) -> (String, String) {
        let final_json =
            serde_json::to_string(&Value::Object(self.accumulated)).unwrap_or_default();
        let closing = final_json[self.json_text.len()..].to_owned();
        (final_json, closing)
    }

    fn emit_navigation_to(
        &mut self,
        target: &[PathSeg],
        arg: &WirePartialArg,
        value_json: &str,
    ) -> String {
        let mut out = String::new();
        if self.string_open {
            out.push('"');
            self.string_open = false;
        }
        out.push_str(&self.ensure_root());

        let target_container: Vec<PathSeg> = if target.is_empty() {
            Vec::new()
        } else {
            target[..target.len() - 1].to_vec()
        };
        let leaf = target
            .last()
            .cloned()
            .unwrap_or(PathSeg::Key(String::new()));

        let common_depth = self.find_common_stack_depth(&target_container);
        out.push_str(&self.close_down_to(common_depth));
        out.push_str(&self.open_down_to(&target_container, &leaf));
        out.push_str(&self.emit_leaf(&leaf, arg, value_json));
        out
    }

    fn ensure_root(&mut self) -> String {
        if self.path_stack.is_empty() {
            self.path_stack.push(StackEntry {
                segment: PathSeg::Key(String::new()),
                is_array: false,
                child_count: 0,
            });
            "{".into()
        } else {
            String::new()
        }
    }

    fn find_common_stack_depth(&self, target: &[PathSeg]) -> usize {
        let max_depth = std::cmp::min(self.path_stack.len().saturating_sub(1), target.len());
        let mut common = 0usize;
        for i in 0..max_depth {
            if path_seg_eq(&self.path_stack[i + 1].segment, &target[i]) {
                common += 1;
            } else {
                break;
            }
        }
        common + 1
    }

    fn close_down_to(&mut self, target_depth: usize) -> String {
        let mut out = String::new();
        while self.path_stack.len() > target_depth {
            let entry = self.path_stack.pop().expect("popped above 0");
            out.push(if entry.is_array { ']' } else { '}' });
        }
        out
    }

    fn open_down_to(&mut self, target_container: &[PathSeg], leaf: &PathSeg) -> String {
        let mut out = String::new();
        let start_idx = self.path_stack.len().saturating_sub(1);
        for i in start_idx..target_container.len() {
            let seg = &target_container[i];
            let parent_idx = self.path_stack.len() - 1;
            let parent = &mut self.path_stack[parent_idx];
            if parent.child_count > 0 {
                out.push(',');
            }
            parent.child_count += 1;
            if let PathSeg::Key(k) = seg {
                out.push_str(&json_string(k));
                out.push(':');
            }
            let next = if i + 1 < target_container.len() {
                &target_container[i + 1]
            } else {
                leaf
            };
            let is_array = matches!(next, PathSeg::Index(_));
            out.push(if is_array { '[' } else { '{' });
            self.path_stack.push(StackEntry {
                segment: seg.clone(),
                is_array,
                child_count: 0,
            });
        }
        out
    }

    fn emit_leaf(&mut self, leaf: &PathSeg, arg: &WirePartialArg, value_json: &str) -> String {
        let mut out = String::new();
        let container_idx = self.path_stack.len() - 1;
        let container = &mut self.path_stack[container_idx];
        if container.child_count > 0 {
            out.push(',');
        }
        container.child_count += 1;
        if let PathSeg::Key(k) = leaf {
            out.push_str(&json_string(k));
            out.push(':');
        }
        if arg.string_value.is_some() && arg.will_continue == Some(true) {
            // Trim trailing quote so the next chunk can append.
            out.push_str(&value_json[..value_json.len().saturating_sub(1)]);
            self.string_open = true;
        } else {
            out.push_str(value_json);
        }
        out
    }
}

fn path_seg_eq(a: &PathSeg, b: &PathSeg) -> bool {
    match (a, b) {
        (PathSeg::Key(k1), PathSeg::Key(k2)) => k1 == k2,
        (PathSeg::Index(i1), PathSeg::Index(i2)) => i1 == i2,
        _ => false,
    }
}

fn parse_path(raw: &str) -> Vec<PathSeg> {
    let mut segs = Vec::new();
    for part in raw.split('.') {
        if let Some(bracket_idx) = part.find('[') {
            if bracket_idx > 0 {
                segs.push(PathSeg::Key(part[..bracket_idx].to_owned()));
            }
            let mut rest = &part[bracket_idx..];
            while let Some(end) = rest.find(']') {
                let num_str = &rest[1..end];
                if let Ok(n) = num_str.parse::<usize>() {
                    segs.push(PathSeg::Index(n));
                }
                rest = &rest[end + 1..];
                if !rest.starts_with('[') {
                    break;
                }
            }
        } else {
            segs.push(PathSeg::Key(part.to_owned()));
        }
    }
    segs
}

fn get_nested<'a>(obj: &'a Map<String, Value>, segs: &[PathSeg]) -> Option<&'a Value> {
    let mut cur: Option<&Value> = None;
    let mut obj_ref: Option<&Map<String, Value>> = Some(obj);
    for s in segs {
        match s {
            PathSeg::Key(k) => {
                let m = obj_ref?;
                cur = m.get(k);
                obj_ref = cur.and_then(Value::as_object);
            }
            PathSeg::Index(i) => {
                let v = cur?;
                cur = v.as_array().and_then(|a| a.get(*i));
                obj_ref = cur.and_then(Value::as_object);
            }
        }
    }
    cur
}

fn set_nested(obj: &mut Map<String, Value>, segs: &[PathSeg], value: Value) {
    if segs.is_empty() {
        return;
    }
    // Wrap the root in a transient Value::Object to share the walk path.
    let mut root = Value::Object(std::mem::take(obj));
    set_nested_value(&mut root, segs, value);
    if let Value::Object(m) = root {
        *obj = m;
    }
}

fn set_nested_value(cur: &mut Value, segs: &[PathSeg], value: Value) {
    if segs.is_empty() {
        *cur = value;
        return;
    }
    let (seg, rest) = (&segs[0], &segs[1..]);
    match seg {
        PathSeg::Key(k) => {
            if !cur.is_object() {
                *cur = Value::Object(Map::new());
            }
            let map = cur.as_object_mut().expect("object");
            let next_is_array = rest.first().is_some_and(|s| matches!(s, PathSeg::Index(_)));
            if !map.contains_key(k) {
                let placeholder = if rest.is_empty() {
                    value.clone()
                } else if next_is_array {
                    Value::Array(Vec::new())
                } else {
                    Value::Object(Map::new())
                };
                map.insert(k.clone(), placeholder);
            }
            let child = map.get_mut(k).expect("just inserted");
            set_nested_value(child, rest, value);
        }
        PathSeg::Index(i) => {
            if !cur.is_array() {
                *cur = Value::Array(Vec::new());
            }
            let arr = cur.as_array_mut().expect("array");
            let next_is_array = rest.first().is_some_and(|s| matches!(s, PathSeg::Index(_)));
            while arr.len() <= *i {
                arr.push(if rest.is_empty() {
                    Value::Null
                } else if next_is_array {
                    Value::Array(Vec::new())
                } else {
                    Value::Object(Map::new())
                });
            }
            set_nested_value(&mut arr[*i], rest, value);
        }
    }
}

fn resolve_value(arg: &WirePartialArg) -> Option<(Value, String)> {
    if let Some(s) = &arg.string_value {
        return Some((Value::String(s.clone()), json_string(s)));
    }
    if let Some(n) = arg.number_value {
        let v = serde_json::Number::from_f64(n).map(Value::Number)?;
        return Some((v.clone(), serde_json::to_string(&v).ok()?));
    }
    if let Some(b) = arg.bool_value {
        return Some((Value::Bool(b), b.to_string()));
    }
    if arg.null_value.is_some() {
        return Some((Value::Null, "null".into()));
    }
    None
}

fn json_string(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into())
}

fn json_string_inner(s: &str) -> String {
    let full = json_string(s);
    // strip the surrounding quotes
    if full.len() >= 2 {
        full[1..full.len() - 1].to_owned()
    } else {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_string_field() {
        let mut acc = GoogleJsonAccumulator::default();
        let delta = acc.process_partial_args(&[WirePartialArg {
            json_path: "$.city".into(),
            string_value: Some("Boston".into()),
            ..Default::default()
        }]);
        assert_eq!(delta, "{\"city\":\"Boston\"");
        let (final_json, closing) = acc.finalize();
        assert_eq!(closing, "}");
        assert_eq!(final_json, "{\"city\":\"Boston\"}");
    }

    #[test]
    fn string_continuation() {
        let mut acc = GoogleJsonAccumulator::default();
        let _ = acc.process_partial_args(&[WirePartialArg {
            json_path: "$.s".into(),
            string_value: Some("Hel".into()),
            will_continue: Some(true),
            ..Default::default()
        }]);
        let _ = acc.process_partial_args(&[WirePartialArg {
            json_path: "$.s".into(),
            string_value: Some("lo".into()),
            ..Default::default()
        }]);
        let (final_json, _close) = acc.finalize();
        assert!(final_json.contains("Hello"));
    }
}
