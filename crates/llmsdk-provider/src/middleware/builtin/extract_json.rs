//! Strip Markdown code fences from JSON responses.
//!
//! Mirrors `@ai-sdk/ai/src/middleware/extract-json-middleware.ts`. Many
//! models wrap JSON in ` ```json ... ``` `; this middleware unwraps the fence
//! in both `do_generate` results and `do_stream` text deltas so downstream
//! parsers see clean JSON.
//!
//! Implementation choice: the streaming path buffers the entire text segment
//! and emits a single `TextDelta` once it ends. This is simpler than ai-sdk's
//! prefix-detection state machine; the trade-off is that callers see the
//! cleaned JSON only at `TextEnd` rather than incrementally. For most JSON
//! workflows (where the caller `JSON.parse`s the whole text anyway) this is
//! fine. The aggressive-streaming variant can be added later if needed.
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;

use async_trait::async_trait;
use futures::StreamExt;

use crate::error::Result;
use crate::language_model::{
    BoxStream, CallOptions, Content, GenerateResult, LanguageModel, StreamPart, StreamResult,
    TextPart,
};
use crate::middleware::language_model::LanguageModelMiddleware;

/// Middleware that strips Markdown JSON fences from text output.
#[derive(Debug, Default, Clone, Copy)]
pub struct ExtractJsonMiddleware;

impl ExtractJsonMiddleware {
    /// Build the default middleware (strip ` ```json ` and ` ``` ` fences).
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl LanguageModelMiddleware for ExtractJsonMiddleware {
    async fn wrap_generate(
        &self,
        next: &dyn LanguageModel,
        params: CallOptions,
    ) -> Result<GenerateResult> {
        let mut result = next.do_generate(params).await?;
        for content in &mut result.content {
            if let Content::Text(part) = content {
                part.text = strip_json_fence(&part.text);
            }
        }
        Ok(result)
    }

    async fn wrap_stream(
        &self,
        next: &dyn LanguageModel,
        params: CallOptions,
    ) -> Result<StreamResult> {
        let upstream = next.do_stream(params).await?;
        let StreamResult {
            stream,
            request,
            response,
        } = upstream;

        let cleaned = transform_stream(stream);
        Ok(StreamResult {
            stream: cleaned,
            request,
            response,
        })
    }
}

fn transform_stream(stream: BoxStream<Result<StreamPart>>) -> BoxStream<Result<StreamPart>> {
    // Buffer per-block text, flush stripped on TextEnd.
    let mapped = futures::stream::unfold(
        (stream, HashMap::<String, String>::new()),
        |(mut stream, mut buffers)| async move {
            loop {
                let next = stream.next().await;
                match next {
                    None => return None,
                    Some(Err(e)) => return Some((Err(e), (stream, buffers))),
                    Some(Ok(part)) => match part {
                        StreamPart::TextStart {
                            id,
                            provider_metadata,
                        } => {
                            buffers.insert(id.clone(), String::new());
                            return Some((
                                Ok(StreamPart::TextStart {
                                    id,
                                    provider_metadata,
                                }),
                                (stream, buffers),
                            ));
                        }
                        StreamPart::TextDelta { id, delta, .. } => {
                            if let Some(buf) = buffers.get_mut(&id) {
                                buf.push_str(&delta);
                                continue; // suppress raw delta
                            }
                            // Untracked id: forward as-is.
                            return Some((
                                Ok(StreamPart::TextDelta {
                                    id,
                                    delta,
                                    provider_metadata: None,
                                }),
                                (stream, buffers),
                            ));
                        }
                        StreamPart::TextEnd {
                            id,
                            provider_metadata,
                        } => {
                            if let Some(buf) = buffers.remove(&id) {
                                let cleaned = strip_json_fence(&buf);
                                if cleaned.is_empty() {
                                    return Some((
                                        Ok(StreamPart::TextEnd {
                                            id,
                                            provider_metadata,
                                        }),
                                        (stream, buffers),
                                    ));
                                }
                                // Emit a single delta carrying the cleaned text,
                                // followed by the original TextEnd.
                                return Some((
                                    Ok(StreamPart::TextDelta {
                                        id: id.clone(),
                                        delta: cleaned,
                                        provider_metadata: None,
                                    }),
                                    (
                                        prepend(
                                            stream,
                                            StreamPart::TextEnd {
                                                id,
                                                provider_metadata,
                                            },
                                        ),
                                        buffers,
                                    ),
                                ));
                            }
                            return Some((
                                Ok(StreamPart::TextEnd {
                                    id,
                                    provider_metadata,
                                }),
                                (stream, buffers),
                            ));
                        }
                        other => return Some((Ok(other), (stream, buffers))),
                    },
                }
            }
        },
    );
    Box::pin(mapped)
}

/// Prepend one item to a stream.
fn prepend(
    stream: BoxStream<Result<StreamPart>>,
    item: StreamPart,
) -> BoxStream<Result<StreamPart>> {
    Box::pin(futures::stream::iter(std::iter::once(Ok(item))).chain(stream))
}

fn strip_json_fence(s: &str) -> String {
    let trimmed = s.trim();
    let stripped = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```JSON"))
        .or_else(|| trimmed.strip_prefix("```"))
        .unwrap_or(trimmed);
    let stripped = stripped.trim_start_matches(['\n', '\r']);
    let stripped = stripped.strip_suffix("```").unwrap_or(stripped);
    stripped.trim().to_owned()
}

// `TextPart` referenced for doc/import; suppress unused warning when only
// constructed inside tests.
#[allow(dead_code, reason = "kept for symmetry with ai-sdk imports")]
type _Unused = TextPart;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::stream;

    use super::*;
    use crate::language_model::{FinishReason, FinishReasonKind, Usage};
    use crate::middleware::wrap_language_model;

    #[derive(Debug)]
    struct Fake {
        gen_text: String,
        stream_deltas: Vec<String>,
    }

    #[async_trait]
    impl LanguageModel for Fake {
        fn provider(&self) -> &'static str {
            "fake"
        }
        fn model_id(&self) -> &'static str {
            "fake"
        }
        async fn do_generate(&self, _opts: CallOptions) -> Result<GenerateResult> {
            Ok(GenerateResult {
                content: vec![Content::Text(TextPart {
                    text: self.gen_text.clone(),
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
            let mut parts: Vec<Result<StreamPart>> = vec![Ok(StreamPart::TextStart {
                id: "b1".into(),
                provider_metadata: None,
            })];
            for d in &self.stream_deltas {
                parts.push(Ok(StreamPart::TextDelta {
                    id: "b1".into(),
                    delta: d.clone(),
                    provider_metadata: None,
                }));
            }
            parts.push(Ok(StreamPart::TextEnd {
                id: "b1".into(),
                provider_metadata: None,
            }));
            parts.push(Ok(StreamPart::Finish {
                usage: Usage::default(),
                finish_reason: FinishReason::new(FinishReasonKind::Stop),
                provider_metadata: None,
            }));
            Ok(StreamResult {
                stream: Box::pin(stream::iter(parts)),
                request: None,
                response: None,
            })
        }
    }

    #[tokio::test]
    async fn generate_strips_fence() {
        let inner: Arc<dyn LanguageModel> = Arc::new(Fake {
            gen_text: "```json\n{\"x\":1}\n```".into(),
            stream_deltas: vec![],
        });
        let wrapped = wrap_language_model(
            inner,
            [Arc::new(ExtractJsonMiddleware::new()) as Arc<dyn LanguageModelMiddleware>],
        );
        let r = wrapped
            .do_generate(CallOptions::default())
            .await
            .expect("gen");
        let Content::Text(p) = &r.content[0] else {
            panic!("text");
        };
        assert_eq!(p.text, "{\"x\":1}");
    }

    #[tokio::test]
    async fn stream_strips_fence_at_text_end() {
        let inner: Arc<dyn LanguageModel> = Arc::new(Fake {
            gen_text: String::new(),
            stream_deltas: vec!["```json\n{".into(), "\"x\":1}\n```".into()],
        });
        let wrapped = wrap_language_model(
            inner,
            [Arc::new(ExtractJsonMiddleware::new()) as Arc<dyn LanguageModelMiddleware>],
        );
        let mut s = wrapped.do_stream(CallOptions::default()).await.unwrap();
        let mut text_deltas = Vec::new();
        let mut saw_text_end = false;
        while let Some(item) = s.stream.next().await {
            match item.unwrap() {
                StreamPart::TextDelta { delta, .. } => text_deltas.push(delta),
                StreamPart::TextEnd { .. } => saw_text_end = true,
                _ => {}
            }
        }
        assert_eq!(text_deltas, vec!["{\"x\":1}".to_owned()]);
        assert!(saw_text_end);
    }
}
