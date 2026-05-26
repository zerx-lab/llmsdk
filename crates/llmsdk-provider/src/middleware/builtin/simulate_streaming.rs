//! Turn a non-streaming [`do_generate`](crate::LanguageModel::do_generate)
//! response into a [`do_stream`](crate::LanguageModel::do_stream)-shaped
//! [`StreamResult`].
//!
//! Mirrors `@ai-sdk/ai/src/middleware/simulate-streaming-middleware.ts`.
//! Block-level emission (one delta per content part) — matches the ai-sdk
//! reference behaviour. Character / token simulation is *not* added because
//! it requires arbitrary policy that the caller is better positioned to
//! decide on.
// Rust guideline compliant 2026-02-21

use async_trait::async_trait;
use futures::stream;

use crate::error::Result;
use crate::language_model::{
    CallOptions, Content, LanguageModel, ResponseMetadata, StreamPart, StreamResult,
};
use crate::middleware::language_model::LanguageModelMiddleware;

/// Middleware that emits a `do_stream` derived from one `do_generate` call.
#[derive(Debug, Default, Clone, Copy)]
pub struct SimulateStreamingMiddleware;

impl SimulateStreamingMiddleware {
    /// Construct the default simulation middleware.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl LanguageModelMiddleware for SimulateStreamingMiddleware {
    async fn wrap_stream(
        &self,
        next: &dyn LanguageModel,
        params: CallOptions,
    ) -> Result<StreamResult> {
        let result = next.do_generate(params).await?;

        let mut parts: Vec<Result<StreamPart>> = Vec::new();
        parts.push(Ok(StreamPart::StreamStart {
            warnings: result.warnings.clone(),
        }));
        // Mirror upstream `simulate-streaming-middleware.ts:24` —
        // `controller.enqueue({ type: 'response-metadata', ...result.response })`
        // is unconditional; when `result.response` is absent the frame is
        // still emitted with empty fields so downstream sees a consistent
        // shape across simulated and real streams.
        let resp_metadata = result
            .response
            .as_ref()
            .map(|resp| ResponseMetadata {
                id: resp.metadata.id.clone(),
                timestamp: resp.metadata.timestamp.clone(),
                model_id: resp.metadata.model_id.clone(),
                headers: resp.metadata.headers.clone(),
            })
            .unwrap_or_default();
        parts.push(Ok(StreamPart::ResponseMetadata(resp_metadata)));

        for (idx, content) in result.content.iter().enumerate() {
            let block_id = format!("sim-{idx}");
            match content {
                Content::Text(t) => {
                    // Mirror upstream `simulate-streaming-middleware.ts:27`:
                    // empty-text content blocks are skipped — without this
                    // guard downstream consumers receive a spurious
                    // text-start / text-delta("") / text-end triple.
                    if t.text.is_empty() {
                        continue;
                    }
                    // Mirror upstream `simulate-streaming-middleware.ts:27-34`:
                    // text-start / text-end carry only `id`; text-delta carries
                    // `delta` only — `providerMetadata` is *not* threaded onto
                    // any text frame. Drop `t.provider_options` here instead of
                    // smuggling it through text-delta where downstream code
                    // wouldn't look for it.
                    parts.push(Ok(StreamPart::TextStart {
                        id: block_id.clone(),
                        provider_metadata: None,
                    }));
                    parts.push(Ok(StreamPart::TextDelta {
                        id: block_id.clone(),
                        delta: t.text.clone(),
                        provider_metadata: None,
                    }));
                    parts.push(Ok(StreamPart::TextEnd {
                        id: block_id,
                        provider_metadata: None,
                    }));
                }
                Content::Reasoning(r) => {
                    // Mirror upstream `simulate-streaming-middleware.ts:42-54`:
                    // `providerMetadata` rides on reasoning-start (not delta or
                    // end). Threading it onto delta diverges from the snapshot
                    // shape consumers built against ai-sdk expect.
                    parts.push(Ok(StreamPart::ReasoningStart {
                        id: block_id.clone(),
                        provider_metadata: r.provider_options.clone().map(into_metadata),
                    }));
                    parts.push(Ok(StreamPart::ReasoningDelta {
                        id: block_id.clone(),
                        delta: r.text.clone(),
                        provider_metadata: None,
                    }));
                    parts.push(Ok(StreamPart::ReasoningEnd {
                        id: block_id,
                        provider_metadata: None,
                    }));
                }
                Content::ToolCall(tc) => {
                    parts.push(Ok(StreamPart::ToolCall(tc.clone())));
                }
                Content::ToolResult(tr) => {
                    parts.push(Ok(StreamPart::ToolResult(tr.clone())));
                }
                Content::ToolApprovalRequest(req) => {
                    parts.push(Ok(StreamPart::ToolApprovalRequest(req.clone())));
                }
                Content::Source(s) => {
                    parts.push(Ok(StreamPart::Source(s.clone())));
                }
                Content::File(_) | Content::ReasoningFile { .. } => {
                    // No stream variant for file blocks; surface as Custom.
                    parts.push(Ok(StreamPart::Custom {
                        kind: "llmsdk.simulate.file".into(),
                        provider_metadata: None,
                    }));
                }
                Content::Custom {
                    kind,
                    provider_options,
                } => {
                    parts.push(Ok(StreamPart::Custom {
                        kind: kind.clone(),
                        provider_metadata: provider_options.clone().map(into_metadata),
                    }));
                }
            }
        }

        parts.push(Ok(StreamPart::Finish {
            usage: result.usage,
            finish_reason: result.finish_reason,
            provider_metadata: result.provider_metadata,
        }));

        Ok(StreamResult {
            stream: Box::pin(stream::iter(parts)),
            request: result.request,
            response: None,
        })
    }
}

/// `ProviderOptions` and `ProviderMetadata` share the same shape; transmute by
/// rewrap because the typedefs are identical maps.
fn into_metadata(opts: crate::shared::ProviderOptions) -> crate::shared::ProviderMetadata {
    opts
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::StreamExt;

    use super::*;
    use crate::language_model::{FinishReason, FinishReasonKind, TextPart, Usage};
    use crate::middleware::wrap_language_model;

    #[derive(Debug)]
    struct Gen {
        text: String,
    }

    #[async_trait]
    impl LanguageModel for Gen {
        fn provider(&self) -> &'static str {
            "g"
        }
        fn model_id(&self) -> &'static str {
            "g"
        }
        async fn do_generate(
            &self,
            _opts: CallOptions,
        ) -> Result<crate::language_model::GenerateResult> {
            Ok(crate::language_model::GenerateResult {
                content: vec![Content::Text(TextPart {
                    text: self.text.clone(),
                    provider_options: None,
                })],
                finish_reason: FinishReason::new(FinishReasonKind::Stop),
                usage: Usage::default(),
                provider_metadata: None,
                request: None,
                response: None,
                warnings: vec![],
            })
        }
        async fn do_stream(&self, _opts: CallOptions) -> Result<StreamResult> {
            unimplemented!("middleware should bypass do_stream")
        }
    }

    #[tokio::test]
    async fn emits_block_level_stream_from_generate() {
        let inner: Arc<dyn LanguageModel> = Arc::new(Gen {
            text: "hello".into(),
        });
        let wrapped = wrap_language_model(
            inner,
            [Arc::new(SimulateStreamingMiddleware::new()) as Arc<dyn LanguageModelMiddleware>],
        );
        let mut s = wrapped.do_stream(CallOptions::default()).await.unwrap();
        let mut tags: Vec<&'static str> = Vec::new();
        while let Some(item) = s.stream.next().await {
            tags.push(match item.unwrap() {
                StreamPart::StreamStart { .. } => "start",
                StreamPart::ResponseMetadata(_) => "response-metadata",
                StreamPart::TextStart { .. } => "text-start",
                StreamPart::TextDelta { .. } => "text-delta",
                StreamPart::TextEnd { .. } => "text-end",
                StreamPart::Finish { .. } => "finish",
                _ => "other",
            });
        }
        // `response-metadata` is unconditional (upstream spreads
        // `result.response` even when undefined) and lands between
        // `stream-start` and the content frames.
        assert_eq!(
            tags,
            vec![
                "start",
                "response-metadata",
                "text-start",
                "text-delta",
                "text-end",
                "finish"
            ]
        );
    }

    #[tokio::test]
    async fn empty_text_block_is_skipped() {
        // Mirrors upstream `simulate-streaming-middleware.ts:27` —
        // a zero-length text block must not surface as a stream segment.
        let inner: Arc<dyn LanguageModel> = Arc::new(Gen {
            text: String::new(),
        });
        let wrapped = wrap_language_model(
            inner,
            [Arc::new(SimulateStreamingMiddleware::new()) as Arc<dyn LanguageModelMiddleware>],
        );
        let mut s = wrapped.do_stream(CallOptions::default()).await.unwrap();
        let mut tags: Vec<&'static str> = Vec::new();
        while let Some(item) = s.stream.next().await {
            tags.push(match item.unwrap() {
                StreamPart::StreamStart { .. } => "start",
                StreamPart::ResponseMetadata(_) => "response-metadata",
                StreamPart::TextStart { .. } => "text-start",
                StreamPart::TextDelta { .. } => "text-delta",
                StreamPart::TextEnd { .. } => "text-end",
                StreamPart::Finish { .. } => "finish",
                _ => "other",
            });
        }
        // No text-* events for the empty block — only stream-start,
        // the unconditional response-metadata frame, and finish remain.
        assert_eq!(tags, vec!["start", "response-metadata", "finish"]);
    }

    #[tokio::test]
    async fn reasoning_provider_metadata_rides_on_start_not_delta() {
        // Mirrors upstream `simulate-streaming-middleware.ts:42-54` where
        // `providerMetadata` is attached to reasoning-start; delta carries
        // only the text. Catches the prior bug where Rust pinned the
        // metadata onto delta and left start empty.
        use crate::language_model::ReasoningPart;
        use crate::shared::ProviderOptions;

        #[derive(Debug)]
        struct ReasoningGen;

        #[async_trait]
        impl LanguageModel for ReasoningGen {
            fn provider(&self) -> &'static str {
                "r"
            }
            fn model_id(&self) -> &'static str {
                "r"
            }
            async fn do_generate(
                &self,
                _opts: CallOptions,
            ) -> Result<crate::language_model::GenerateResult> {
                let mut opts = ProviderOptions::new();
                opts.insert(
                    "anthropic".into(),
                    serde_json::json!({ "signature": "sig" })
                        .as_object()
                        .cloned()
                        .unwrap(),
                );
                Ok(crate::language_model::GenerateResult {
                    content: vec![Content::Reasoning(ReasoningPart {
                        text: "thinking…".into(),
                        provider_options: Some(opts),
                    })],
                    finish_reason: FinishReason::new(FinishReasonKind::Stop),
                    usage: Usage::default(),
                    provider_metadata: None,
                    request: None,
                    response: None,
                    warnings: vec![],
                })
            }
            async fn do_stream(&self, _opts: CallOptions) -> Result<StreamResult> {
                unimplemented!()
            }
        }

        let inner: Arc<dyn LanguageModel> = Arc::new(ReasoningGen);
        let wrapped = wrap_language_model(
            inner,
            [Arc::new(SimulateStreamingMiddleware::new()) as Arc<dyn LanguageModelMiddleware>],
        );
        let mut s = wrapped.do_stream(CallOptions::default()).await.unwrap();
        let mut start_meta: Option<crate::shared::ProviderMetadata> = None;
        let mut delta_meta: Option<crate::shared::ProviderMetadata> = None;
        while let Some(item) = s.stream.next().await {
            match item.unwrap() {
                StreamPart::ReasoningStart {
                    provider_metadata, ..
                } => start_meta = provider_metadata,
                StreamPart::ReasoningDelta {
                    provider_metadata, ..
                } => delta_meta = provider_metadata,
                _ => {}
            }
        }
        assert!(
            start_meta.is_some(),
            "reasoning-start must carry provider_metadata (upstream parity)"
        );
        assert!(
            delta_meta.is_none(),
            "reasoning-delta must NOT carry provider_metadata (upstream parity)"
        );
    }
}
