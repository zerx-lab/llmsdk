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
    separator: String,
}

impl ExtractReasoningMiddleware {
    /// Build for the given tag name (without angle brackets).
    ///
    /// If `start_with_reasoning` is `true`, content emitted *before* the
    /// first opening tag is also treated as reasoning (handy when the model
    /// always starts with chain-of-thought without an opening tag).
    ///
    /// Defaults to upstream `separator = "\n"`. Override with
    /// [`Self::with_separator`] to control how multiple reasoning matches
    /// and surrounding text fragments are joined.
    #[must_use]
    pub fn new(tag_name: impl Into<String>, start_with_reasoning: bool) -> Self {
        Self {
            tag_name: tag_name.into(),
            start_with_reasoning,
            separator: "\n".to_owned(),
        }
    }

    /// Override the separator used when joining multiple reasoning matches
    /// and when stitching surrounding text fragments together. Mirrors
    /// upstream `separator` option (default `"\n"`).
    #[must_use]
    pub fn with_separator(mut self, separator: impl Into<String>) -> Self {
        self.separator = separator.into();
        self
    }
}

/// Find all `<tag>captured</tag>` matches in `input`. Returns
/// `(byte_start, total_byte_len, captured_text)` for each match in source
/// order. Used by both the generate and stream paths to mirror upstream
/// `Array.from(text.matchAll(/<tag>(.*?)<\/tag>/gs))`.
fn find_matches(input: &str, tag: &str) -> Vec<(usize, usize, String)> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut out = Vec::new();
    let mut cursor = 0;
    while let Some(rel_open) = input[cursor..].find(&open) {
        let abs_open = cursor + rel_open;
        let after_open = abs_open + open.len();
        let Some(rel_close) = input[after_open..].find(&close) else {
            break;
        };
        let abs_close = after_open + rel_close;
        let captured = input[after_open..abs_close].to_owned();
        let total_len = (abs_close + close.len()) - abs_open;
        out.push((abs_open, total_len, captured));
        cursor = abs_close + close.len();
    }
    out
}

/// Extract reasoning per the upstream contract: returns
/// `(joined_reasoning, text_without_reasoning)` matching
/// `extract-reasoning-middleware.ts:30-78`. When `start_with_reasoning` is
/// `true` the input is virtually prefixed with `<tag>` (matching the
/// upstream `openingTag + part.text` line at :40).
fn extract_reasoning_join(
    input: &str,
    tag: &str,
    start_with_reasoning: bool,
    separator: &str,
) -> Option<(String, String)> {
    let owned;
    let text: &str = if start_with_reasoning {
        owned = format!("<{tag}>{input}");
        &owned
    } else {
        input
    };
    let matches = find_matches(text, tag);
    if matches.is_empty() {
        return None;
    }
    let reasoning = matches
        .iter()
        .map(|m| m.2.as_str())
        .collect::<Vec<_>>()
        .join(separator);

    // Remove matches right-to-left, splicing in `separator` whenever both
    // sides of the removed span are non-empty (mirrors :60-65).
    let mut text_without = text.to_owned();
    for (start, len, _) in matches.iter().rev() {
        let before = text_without[..*start].to_owned();
        let after = text_without[start + len..].to_owned();
        text_without = if !before.is_empty() && !after.is_empty() {
            format!("{before}{separator}{after}")
        } else {
            format!("{before}{after}")
        };
    }
    Some((reasoning, text_without))
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
                    if let Some((reasoning, text_without)) = extract_reasoning_join(
                        &t.text,
                        &self.tag_name,
                        self.start_with_reasoning,
                        &self.separator,
                    ) {
                        // Mirrors upstream `extract-reasoning-middleware.ts:67-75`:
                        // always push reasoning first, text second — even when text
                        // is empty.
                        new_content.push(Content::Reasoning(ReasoningPart {
                            text: reasoning,
                            provider_options: t.provider_options.clone(),
                        }));
                        new_content.push(Content::Text(TextPart {
                            text: text_without,
                            provider_options: t.provider_options,
                        }));
                    } else {
                        // No tag found — pass the text through untouched (matches
                        // upstream `:46-48` early return).
                        new_content.push(Content::Text(t));
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

        let cleaned = transform_stream(
            stream,
            self.tag_name.clone(),
            self.start_with_reasoning,
            self.separator.clone(),
        );
        Ok(StreamResult {
            stream: cleaned,
            request,
            response,
        })
    }
}

/// Buffer text-block deltas to end-of-block, then emit `reasoning-start /
/// reasoning-delta / reasoning-end` for the joined reasoning followed by a
/// single `text-delta` for the text remainder. Mirrors upstream behaviour
/// when consumed as a complete block (the upstream stream variant emits
/// incrementally; this coarser strategy matches what `super::extract_json`
/// uses and keeps the implementation simple).
#[allow(
    clippy::too_many_lines,
    reason = "single-pass per-block buffer state machine; extracting helpers would obscure the StreamPart match-arm flow"
)]
fn transform_stream(
    stream: BoxStream<Result<StreamPart>>,
    tag: String,
    start_with_reasoning: bool,
    separator: String,
) -> BoxStream<Result<StreamPart>> {
    let mapped = futures::stream::unfold(
        (
            stream,
            HashMap::<String, String>::new(),
            tag,
            start_with_reasoning,
            separator,
        ),
        |(mut stream, mut buffers, tag, start_with_reasoning, separator)| async move {
            loop {
                match stream.next().await {
                    None => return None,
                    Some(Err(e)) => {
                        return Some((
                            vec![Err(e)],
                            (stream, buffers, tag, start_with_reasoning, separator),
                        ));
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
                                (stream, buffers, tag, start_with_reasoning, separator),
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
                                (stream, buffers, tag, start_with_reasoning, separator),
                            ));
                        }
                        StreamPart::TextEnd {
                            id,
                            provider_metadata,
                        } => {
                            let buf = buffers.remove(&id).unwrap_or_default();
                            let mut out: Vec<Result<StreamPart>> = Vec::new();
                            if let Some((reasoning, text_without)) =
                                extract_reasoning_join(&buf, &tag, start_with_reasoning, &separator)
                            {
                                let reasoning_id = format!("{id}.reasoning");
                                out.push(Ok(StreamPart::ReasoningStart {
                                    id: reasoning_id.clone(),
                                    provider_metadata: None,
                                }));
                                out.push(Ok(StreamPart::ReasoningDelta {
                                    id: reasoning_id.clone(),
                                    delta: reasoning,
                                    provider_metadata: None,
                                }));
                                out.push(Ok(StreamPart::ReasoningEnd {
                                    id: reasoning_id,
                                    provider_metadata: None,
                                }));
                                // Always emit the text delta — even when empty,
                                // so downstream consumers see the surrounding
                                // text-start / text-end pair bracket a single
                                // text fragment (mirrors generate-side parity).
                                out.push(Ok(StreamPart::TextDelta {
                                    id: id.clone(),
                                    delta: text_without,
                                    provider_metadata: None,
                                }));
                            } else if !buf.is_empty() {
                                out.push(Ok(StreamPart::TextDelta {
                                    id: id.clone(),
                                    delta: buf,
                                    provider_metadata: None,
                                }));
                            }
                            out.push(Ok(StreamPart::TextEnd {
                                id,
                                provider_metadata,
                            }));
                            return Some((
                                out,
                                (stream, buffers, tag, start_with_reasoning, separator),
                            ));
                        }
                        other => {
                            return Some((
                                vec![Ok(other)],
                                (stream, buffers, tag, start_with_reasoning, separator),
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
    async fn generate_splits_single_think_tag() {
        // Mirrors upstream `extract-reasoning-middleware.test.ts:49-86`.
        let inner: Arc<dyn LanguageModel> = Arc::new(Fake {
            text: "<think>analyzing the request</think>Here is the response".into(),
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
        assert_eq!(r.content.len(), 2, "always reasoning + text");
        match (&r.content[0], &r.content[1]) {
            (Content::Reasoning(a), Content::Text(b)) => {
                assert_eq!(a.text, "analyzing the request");
                assert_eq!(b.text, "Here is the response");
            }
            other => panic!("unexpected split: {other:?}"),
        }
    }

    #[tokio::test]
    async fn generate_joins_multiple_think_tags_with_separator() {
        // Mirrors upstream `extract-reasoning-middleware.test.ts:128-167`:
        // multiple matches join with separator (default `\n`).
        let inner: Arc<dyn LanguageModel> = Arc::new(Fake {
            text: "<think>analyzing the request</think>Here is the response<think>thinking about the response</think>more".into(),
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
        assert_eq!(r.content.len(), 2);
        match (&r.content[0], &r.content[1]) {
            (Content::Reasoning(a), Content::Text(b)) => {
                assert_eq!(a.text, "analyzing the request\nthinking about the response");
                assert_eq!(b.text, "Here is the response\nmore");
            }
            other => panic!("unexpected split: {other:?}"),
        }
    }

    #[tokio::test]
    async fn generate_preserves_text_when_tag_absent() {
        let inner: Arc<dyn LanguageModel> = Arc::new(Fake {
            text: "no tags here".into(),
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
        assert_eq!(r.content.len(), 1);
        assert!(matches!(&r.content[0], Content::Text(t) if t.text == "no tags here"));
    }

    #[tokio::test]
    async fn generate_custom_separator_overrides_default() {
        let inner: Arc<dyn LanguageModel> = Arc::new(Fake {
            text: "<t>a</t>mid<t>b</t>".into(),
        });
        let mw = ExtractReasoningMiddleware::new("t", false).with_separator(" | ");
        let wrapped =
            wrap_language_model(inner, [Arc::new(mw) as Arc<dyn LanguageModelMiddleware>]);
        let r = wrapped
            .do_generate(CallOptions::default())
            .await
            .expect("gen");
        match (&r.content[0], &r.content[1]) {
            (Content::Reasoning(a), Content::Text(b)) => {
                assert_eq!(a.text, "a | b");
                assert_eq!(b.text, "mid");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[tokio::test]
    async fn stream_emits_reasoning_then_text() {
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
