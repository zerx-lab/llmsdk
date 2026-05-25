//! Anthropic-on-Bedrock model handle.
// Rust guideline compliant 2026-05-25

use std::sync::Arc;

use llmsdk_anthropic::internal::{AnthropicMessagesModel, Inner as AnthropicInner};
use llmsdk_provider::ProviderError;
use serde_json::Value;

use crate::config::Inner as BedrockInner;
use crate::sigv4_auth::AnthropicAuthAdapter;

/// Anthropic-on-Bedrock model handle.
///
/// Public re-export of [`AnthropicMessagesModel`] with a Bedrock-flavored
/// [`Inner`](AnthropicInner) baked in.
pub type AmazonBedrockAnthropicModel = AnthropicMessagesModel;

impl AmazonBedrockAnthropicModelExt for AmazonBedrockAnthropicModel {}

/// Internal extension trait used to expose the cross-crate constructor as a
/// named factory (`AmazonBedrockAnthropicModel::new`) — type aliases cannot
/// carry inherent methods of their own.
pub(crate) trait AmazonBedrockAnthropicModelExt {
    /// Construct an Anthropic-on-Bedrock model handle from the parent
    /// Bedrock provider state + a Bedrock Anthropic model id (e.g.
    /// `"anthropic.claude-3-5-sonnet-20241022-v2:0"`).
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError`] when the underlying
    /// [`AnthropicInner`](llmsdk_anthropic::internal::Inner) builder fails
    /// to materialize (e.g. HTTP client init).
    #[allow(
        clippy::new_ret_no_self,
        reason = "type aliases cannot define inherent `new`; this is the closest user-facing factory"
    )]
    fn new(
        bedrock: Arc<BedrockInner>,
        model_id: String,
    ) -> Result<AmazonBedrockAnthropicModel, ProviderError> {
        let auth_adapter = Arc::new(AnthropicAuthAdapter {
            auth: bedrock.auth.clone(),
        });
        let base_url = bedrock.runtime_base_url.clone();

        let inner = AnthropicInner::builder()
            .base_url(base_url)
            .provider_name("bedrock.anthropic.messages")
            .http_client(bedrock.http.clone())
            .request_auth(auth_adapter)
            .endpoint(|base, model_id, is_streaming| {
                let suffix = if is_streaming {
                    "invoke-with-response-stream"
                } else {
                    "invoke"
                };
                format!("{base}/model/{}/{suffix}", encode_path_segment(model_id))
            })
            .body_transform(|body: &mut Value| {
                let Some(obj) = body.as_object_mut() else {
                    return;
                };
                // Bedrock injects model + version via URL / body knobs.
                obj.remove("model");
                obj.remove("stream");
                obj.insert(
                    "anthropic_version".to_owned(),
                    Value::String("bedrock-2023-05-31".to_owned()),
                );
            })
            .build()?;

        Ok(AnthropicMessagesModel::new(Arc::new(inner), model_id))
    }
}

/// Re-export of the chat path-segment encoder so the URL hook can call it
/// without depending on the chat module type.
fn encode_path_segment(input: &str) -> String {
    crate::chat::encode_path_segment(input)
}
