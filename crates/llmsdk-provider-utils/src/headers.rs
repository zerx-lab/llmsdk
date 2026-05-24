//! Header combining helpers.
//!
//! Mirrors `@ai-sdk/provider-utils/src/combine-headers.ts`. Later layers
//! override earlier ones. A value of `None` is treated as "remove this
//! header", matching ai-sdk's `string | undefined` convention.
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;

/// Merge multiple header maps left-to-right. Later maps override earlier.
///
/// `None` values remove the header, matching the `string | undefined`
/// shape used by ai-sdk callers.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
/// use llmsdk_provider_utils::combine_headers;
///
/// let base: HashMap<String, Option<String>> = [
///     ("authorization".into(), Some("Bearer base".into())),
///     ("x-trace".into(), Some("on".into())),
/// ].into();
/// let override_: HashMap<String, Option<String>> = [
///     ("authorization".into(), Some("Bearer override".into())),
///     ("x-trace".into(), None),
/// ].into();
///
/// let merged = combine_headers([&base, &override_]);
/// assert_eq!(merged["authorization"], "Bearer override");
/// assert!(!merged.contains_key("x-trace"));
/// ```
#[must_use]
pub fn combine_headers<'a, I>(layers: I) -> HashMap<String, String>
where
    I: IntoIterator<Item = &'a HashMap<String, Option<String>>>,
{
    let mut out: HashMap<String, String> = HashMap::new();
    for layer in layers {
        for (name, value) in layer {
            match value {
                Some(v) => {
                    out.insert(name.clone(), v.clone());
                }
                None => {
                    out.remove(name);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(&str, Option<&str>)]) -> HashMap<String, Option<String>> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).into(), v.map(std::string::ToString::to_string)))
            .collect()
    }

    #[test]
    fn empty_iter_returns_empty() {
        let out = combine_headers::<[&HashMap<String, Option<String>>; 0]>([]);
        assert!(out.is_empty());
    }

    #[test]
    fn later_overrides_earlier() {
        let a = map(&[("h", Some("a"))]);
        let b = map(&[("h", Some("b"))]);
        let out = combine_headers([&a, &b]);
        assert_eq!(out["h"], "b");
    }

    #[test]
    fn none_removes_header() {
        let a = map(&[("h", Some("a"))]);
        let b = map(&[("h", None)]);
        let out = combine_headers([&a, &b]);
        assert!(!out.contains_key("h"));
    }
}
