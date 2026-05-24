//! Promote `<tag>...</tag>` blocks from text content into [`Content::Reasoning`].
//!
//! Mirrors `@ai-sdk/ai/src/middleware/extract-reasoning-middleware.ts`. Useful
//! when a model is prompted to emit chain-of-thought wrapped in a custom tag
//! (e.g. `<think>...</think>`); the middleware peels those blocks off the
//! visible text and surfaces them as reasoning content.
//!
//! Streaming path: buffers each text block to end-of-block and re-emits a
//! split sequence. Matches the simpler-but-coarser strategy chosen for
//! [`super::extract_json`] (see that module for rationale).
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;

use async_trait::async_trait;
use futures::StreamExt;

use crate::error::Result;
use crate::language_model::{
    BoxStream, CallOptions, Content, GenerateResult, LanguageModel, ReasoningPart, StreamPart,
    StreamResult, TextPart,
};
use crate::middleware::language_model::LanguageModelMiddleware;

/// Middleware that extracts `<tag>...</tag>` blocks from text into reasoning.
#[derive(Debug, Clone)]
pub struct ExtractReasoningMiddleware {
    tag_name: String,
    start_with_reasoning: bool,
}

impl ExtractReasoningMiddleware {
    /// Build for the given tag name (without angle brackets).
    ///
    /// If `start_with_reasoning` is `true`, content emitted *before* the
    /// first opening tag is also treated as reasoning (handy when the model
    /// always starts with chain-of-thought without an opening tag).
    #[must_use]
    pub fn new(tag_name: impl Into<String>, start_with_reasoning: bool) -> Self {
        Self {
            tag_name: tag_name.into(),
            start_with_reasoning,
        }
    }
}

/// Output of [`split_text_by_tag`]: an ordered list of text / reasoning pieces.
#[derive(Debug)]
enum Piece {
    Text(String),
    Reasoning(String),
}

fn split_text_by_tag(input: &str, tag: &str, start_with_reasoning: bool) -> Vec<Piece> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut out: Vec<Piece> = Vec::new();
    let mut rest = input;
    let mut in_reasoning = start_with_reasoning;
    while !rest.is_empty() {
        if in_reasoning {
            if let Some(idx) = rest.find(&close) {
                if idx > 0 {
                    out.push(Piece::Reasoning(rest[..idx].to_owned()));
                }
                rest = &rest[idx + close.len()..];
                in_reasoning = false;
            } else {
                out.push(Piece::Reasoning(rest.to_owned()));
                break;
            }
        } else {
            if let Some(idx) = rest.find(&open) {
                if idx > 0 {
                    out.push(Piece::Text(rest[..idx].to_owned()));
                }
                rest = &rest[idx + open.len()..];
                in_reasoning = true;
            } else {
                out.push(Piece::Text(rest.to_owned()));
                break;
            }
        }
    }
    out
}

#[async_trait]
impl LanguageModelMiddleware for ExtractReasoningMiddleware {
    async fn wrap_generate(
        &self,
        next: &dyn LanguageModel,
        params: CallOptions,
    ) -> Result<GenerateResult> {
        let mut result = next.do_generate(params).await?;
        let mut new_content: Vec<Content> = Vec::with_capacity(result.content.len());
        for c in result.content.drain(..) {
            match c {
                Content::Text(t) => {
                    let pieces =
                        split_text_by_tag(&t.text, &self.tag_name, self.start_with_reasoning);
                    for p in pieces {
                        match p {
                            Piece::Text(s) if !s.is_empty() => {
                                new_content.push(Content::Text(TextPart {
                                    text: s,
                                    provider_options: t.provider_options.clone(),
                                }));
                            }
                            Piece::Reasoning(s) if !s.is_empty() => {
                                new_content.push(Content::Reasoning(ReasoningPart {
                                    text: s,
                                    provider_options: t.provider_options.clone(),
                                }));
                            }
                            _ => {}
                        }
                    }
                }
                other => new_content.push(other),
            }
        }
        result.content = new_content;
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

        let cleaned = transform_stream(stream, self.tag_name.clone(), self.start_with_reasoning);
        Ok(StreamResult {
            stream: cleaned,
            request,
            response,
        })
    }
}

fn transform_stream(
    stream: BoxStream<Result<StreamPart>>,
    tag: String,
    start_with_reasoning: bool,
) -> BoxStream<Result<StreamPart>> {
    let mapped = futures::stream::unfold(
        (
            stream,
            HashMap::<String, String>::new(),
            tag,
            start_with_reasoning,
        ),
        |(mut stream, mut buffers, tag, start_with_reasoning)| async move {
            loop {
                match stream.next().await {
                    None => return None,
                    Some(Err(e)) => {
                        return Some((vec![Err(e)], (stream, buffers, tag, start_with_reasoning)));
                    }
                    Some(Ok(part)) => match part {
                        StreamPart::TextStart {
                            id,
                            provider_metadata,
                        } => {
                            buffers.insert(id.clone(), String::new());
                            return Some((
                                vec![Ok(StreamPart::TextStart {
                                    id,
                                    provider_metadata,
                                })],
                                (stream, buffers, tag, start_with_reasoning),
                            ));
                        }
                        StreamPart::TextDelta { id, delta, .. } => {
                            if let Some(buf) = buffers.get_mut(&id) {
                                buf.push_str(&delta);
                                continue; // swallow until end
                            }
                            return Some((
                                vec![Ok(StreamPart::TextDelta {
                                    id,
                                    delta,
                                    provider_metadata: None,
                                })],
                                (stream, buffers, tag, start_with_reasoning),
                            ));
                        }
                        StreamPart::TextEnd {
                            id,
                            provider_metadata,
                        } => {
                            let buf = buffers.remove(&id).unwrap_or_default();
                            let pieces = split_text_by_tag(&buf, &tag, start_with_reasoning);
                            let mut out: Vec<Result<StreamPart>> = Vec::new();
                            for (i, p) in pieces.into_iter().enumerate() {
                                let sub_id = format!("{id}.{i}");
                                match p {
                                    Piece::Text(s) if !s.is_empty() => {
                                        out.push(Ok(StreamPart::TextDelta {
                                            id: id.clone(),
                                            delta: s,
                                            provider_metadata: None,
                                        }));
                                    }
                                    Piece::Reasoning(s) if !s.is_empty() => {
                                        out.push(Ok(StreamPart::ReasoningStart {
                                            id: sub_id.clone(),
                                            provider_metadata: None,
                                        }));
                                        out.push(Ok(StreamPart::ReasoningDelta {
                                            id: sub_id.clone(),
                                            delta: s,
                                            provider_metadata: None,
                                        }));
                                        out.push(Ok(StreamPart::ReasoningEnd {
                                            id: sub_id,
                                            provider_metadata: None,
                                        }));
                                    }
                                    _ => {}
                                }
                            }
                            out.push(Ok(StreamPart::TextEnd {
                                id,
                                provider_metadata,
                            }));
                            return Some((out, (stream, buffers, tag, start_with_reasoning)));
                        }
                        other => {
                            return Some((
                                vec![Ok(other)],
                                (stream, buffers, tag, start_with_reasoning),
                            ));
                        }
                    },
                }
            }
        },
    )
    .flat_map(futures::stream::iter);
    Box::pin(mapped)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::stream;

    use super::*;
    use crate::language_model::{FinishReason, FinishReasonKind, Usage};
    use crate::middleware::wrap_language_model;

    #[derive(Debug)]
    struct Fake {
        text: String,
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
            let parts: Vec<Result<StreamPart>> = vec![
                Ok(StreamPart::TextStart {
                    id: "b".into(),
                    provider_metadata: None,
                }),
                Ok(StreamPart::TextDelta {
                    id: "b".into(),
                    delta: self.text.clone(),
                    provider_metadata: None,
                }),
                Ok(StreamPart::TextEnd {
                    id: "b".into(),
                    provider_metadata: None,
                }),
                Ok(StreamPart::Finish {
                    usage: Usage::default(),
                    finish_reason: FinishReason::new(FinishReasonKind::Stop),
                    provider_metadata: None,
                }),
            ];
            Ok(StreamResult {
                stream: Box::pin(stream::iter(parts)),
                request: None,
                response: None,
            })
        }
    }

    #[tokio::test]
    async fn generate_splits_reasoning_from_text() {
        let inner: Arc<dyn LanguageModel> = Arc::new(Fake {
            text: "pre <think>thought</think> post".into(),
        });
        let wrapped = wrap_language_model(
            inner,
            [Arc::new(ExtractReasoningMiddleware::new("think", false))
                as Arc<dyn LanguageModelMiddleware>],
        );
        let r = wrapped
            .do_generate(CallOptions::default())
            .await
            .expect("gen");
        assert_eq!(r.content.len(), 3);
        match (&r.content[0], &r.content[1], &r.content[2]) {
            (Content::Text(a), Content::Reasoning(b), Content::Text(c)) => {
                assert_eq!(a.text, "pre ");
                assert_eq!(b.text, "thought");
                assert_eq!(c.text, " post");
            }
            other => panic!("unexpected split: {other:?}"),
        }
    }

    #[tokio::test]
    async fn stream_splits_at_text_end() {
        let inner: Arc<dyn LanguageModel> = Arc::new(Fake {
            text: "<think>x</think>y".into(),
        });
        let wrapped = wrap_language_model(
            inner,
            [Arc::new(ExtractReasoningMiddleware::new("think", false))
                as Arc<dyn LanguageModelMiddleware>],
        );
        let mut s = wrapped.do_stream(CallOptions::default()).await.unwrap();
        let mut events: Vec<String> = Vec::new();
        while let Some(item) = s.stream.next().await {
            match item.unwrap() {
                StreamPart::TextDelta { delta, .. } => events.push(format!("text:{delta}")),
                StreamPart::ReasoningDelta { delta, .. } => events.push(format!("reason:{delta}")),
                StreamPart::TextEnd { .. } => events.push("end".into()),
                _ => {}
            }
        }
        assert!(
            events.contains(&"reason:x".to_owned()),
            "saw reason: {events:?}"
        );
        assert!(
            events.contains(&"text:y".to_owned()),
            "saw text: {events:?}"
        );
    }
}
