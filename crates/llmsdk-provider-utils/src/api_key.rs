//! API key loading from explicit argument or environment.
//!
//! Mirrors `@ai-sdk/provider-utils/src/load-api-key.ts`. The TS function
//! takes 4 named args; Rust uses a builder-style struct for parity with
//! the keyword-argument call sites in provider crates.
// Rust guideline compliant 2026-02-21

use std::env;

use llmsdk_provider::ProviderError;

/// Parameters for [`load_api_key`].
///
/// `description` is used in error messages, e.g. `"OpenAI"`. `parameter_name`
/// defaults to `"apiKey"` matching ai-sdk.
#[derive(Debug, Clone)]
pub struct LoadApiKey<'a> {
    /// Caller-provided key, if any. Takes precedence over env.
    pub api_key: Option<&'a str>,
    /// Environment variable to read when `api_key` is `None`.
    pub env_var: &'a str,
    /// Provider name used in error messages.
    pub description: &'a str,
    /// Parameter name used in error messages. Defaults to `"apiKey"`.
    pub parameter_name: Option<&'a str>,
}

/// Load an API key from explicit value or environment variable.
///
/// Mirrors `loadApiKey`: explicit `api_key` wins; otherwise the named env var
/// is read; missing or blank → [`ProviderError::load_api_key`].
///
/// # Examples
///
/// ```
/// use llmsdk_provider_utils::api_key::{LoadApiKey, load_api_key};
///
/// let key = load_api_key(&LoadApiKey {
///     api_key: Some("sk-test"),
///     env_var: "DOES_NOT_MATTER",
///     description: "OpenAI",
///     parameter_name: None,
/// })
/// .unwrap();
/// assert_eq!(key, "sk-test");
/// ```
///
/// # Errors
///
/// Returns [`ProviderError::load_api_key`] when the explicit key is empty,
/// or when the env var is unset / empty.
pub fn load_api_key(params: &LoadApiKey<'_>) -> Result<String, ProviderError> {
    let LoadApiKey {
        api_key,
        env_var,
        description,
        parameter_name,
    } = *params;
    let param = parameter_name.unwrap_or("apiKey");

    if let Some(value) = api_key {
        if value.is_empty() {
            return Err(ProviderError::load_api_key(format!(
                "{description} API key must be a non-empty string."
            )));
        }
        return Ok(value.to_owned());
    }

    match env::var(env_var) {
        Ok(value) if !value.is_empty() => Ok(value),
        Ok(_) => Err(ProviderError::load_api_key(format!(
            "{description} API key must be a non-empty string. The value of the \
             {env_var} environment variable is empty."
        ))),
        Err(_) => Err(ProviderError::load_api_key(format!(
            "{description} API key is missing. Pass it using the `{param}` parameter \
             or the {env_var} environment variable."
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Use unique env var names per test to avoid cross-test interference.
    const fn params<'a>(api_key: Option<&'a str>, env_var: &'a str) -> LoadApiKey<'a> {
        LoadApiKey {
            api_key,
            env_var,
            description: "Test",
            parameter_name: None,
        }
    }

    #[test]
    fn explicit_key_wins() {
        let out = load_api_key(&params(Some("explicit"), "LLMSDK_TEST_KEY_1")).unwrap();
        assert_eq!(out, "explicit");
    }

    #[test]
    fn empty_explicit_key_errors() {
        let err = load_api_key(&params(Some(""), "LLMSDK_TEST_KEY_2")).unwrap_err();
        assert!(format!("{err}").contains("must be a non-empty string"));
    }

    #[test]
    fn missing_env_errors_with_param_name() {
        let err = load_api_key(&LoadApiKey {
            api_key: None,
            env_var: "LLMSDK_NEVER_SET_VAR_X9Z",
            description: "OpenAI",
            parameter_name: Some("api_key"),
        })
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("OpenAI"));
        assert!(msg.contains("api_key"));
        assert!(msg.contains("LLMSDK_NEVER_SET_VAR_X9Z"));
    }
}
