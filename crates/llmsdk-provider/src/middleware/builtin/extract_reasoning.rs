//! Promote `<tag>...</tag>` blocks from text content into [`Content::Reasoning`].
//!
//! Mirrors `@ai-sdk/ai/src/middleware/extract-reasoning-middleware.ts`. Useful
//! when a model is prompted to emit chain-of-thought wrapped in a custom tag
//! (e.g. `<think>...</think>`); the middleware peels those blocks off the
//! visible text and surfaces them as reasoning content.
//!
//! Streaming path: incremental state machine identical to upstream — every
//! text-delta tick advances buffers and emits any `reasoning-delta` /
//! `text-delta` that can already be committed; cross-chunk partial tags are
//! held in the buffer via [`potential_start_index`].
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

/// Find where `needle` could begin within `haystack`, accepting either a
/// complete substring match or a partial suffix-of-haystack/prefix-of-needle
/// overlap. Mirrors `@ai-sdk/ai/src/util/get-potential-start-index.ts`.
fn potential_start_index(haystack: &str, needle: &str) -> Option<usize> {
    if needle.is_empty() {
        return None;
    }
    if let Some(direct) = haystack.find(needle) {
        return Some(direct);
    }
    // Look for the largest suffix of `haystack` that matches a prefix of
    // `needle`. Walk char boundaries from the end so we never split a UTF-8
    // codepoint mid-byte.
    let mut idx = haystack.len();
    for (start, _) in haystack.char_indices().rev() {
        idx = start;
        let suffix = &haystack[idx..];
        if needle.starts_with(suffix) {
            return Some(idx);
        }
    }
    let _ = idx;
    None
}

/// Per text-block extraction state, mirrors the upstream
/// `reasoningExtractions[chunk.id]` record.
#[derive(Debug)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "Mirrors upstream `reasoningExtractions[chunk.id]` shape — four independent boolean phase flags; collapsing them obscures the upstream comparison."
)]
struct Extraction {
    is_first_reasoning: bool,
    is_first_text: bool,
    after_switch: bool,
    is_reasoning: bool,
    buffer: String,
    id_counter: u32,
    text_id: String,
}

struct StreamCtx {
    stream: BoxStream<Result<StreamPart>>,
    extractions: HashMap<String, Extraction>,
    tag: String,
    start_with_reasoning: bool,
    separator: String,
    delayed_text_start: Option<StreamPart>,
}

/// Incremental re-implementation of the upstream wrap-stream transform
/// (`extract-reasoning-middleware.ts:80-247`). Each text-delta is appended
/// to a per-id buffer and processed in a loop that publishes everything up
/// to the next `<tag>` / `</tag>` boundary; partial tags straddling a chunk
/// are retained in the buffer.
fn transform_stream(
    stream: BoxStream<Result<StreamPart>>,
    tag: String,
    start_with_reasoning: bool,
    separator: String,
) -> BoxStream<Result<StreamPart>> {
    let ctx = StreamCtx {
        stream,
        extractions: HashMap::new(),
        tag,
        start_with_reasoning,
        separator,
        delayed_text_start: None,
    };
    let mapped = futures::stream::unfold(ctx, |mut ctx| async move {
        loop {
            match ctx.stream.next().await {
                None => return None,
                Some(Err(e)) => return Some((vec![Err(e)], ctx)),
                Some(Ok(part)) => {
                    let out = process_part(&mut ctx, part);
                    if !out.is_empty() {
                        return Some((out, ctx));
                    }
                    // Empty out (e.g. delayed text-start, partial-tag buffer)
                    // — keep pulling without yielding.
                }
            }
        }
    })
    .flat_map(futures::stream::iter);
    Box::pin(mapped)
}

fn process_part(ctx: &mut StreamCtx, part: StreamPart) -> Vec<Result<StreamPart>> {
    match part {
        // Defer text-start until we know the first content is plain text;
        // mirrors upstream `delayedTextStart` (vercel/ai#7774). Reasoning may
        // arrive first inside the same block, in which case text-start ends
        // up bracketed only by text-end when no plain text was published.
        StreamPart::TextStart { .. } => {
            ctx.delayed_text_start = Some(part);
            Vec::new()
        }
        StreamPart::TextDelta { id, delta, .. } => process_text_delta(ctx, &id, &delta),
        StreamPart::TextEnd {
            id,
            provider_metadata,
        } => {
            let mut out: Vec<Result<StreamPart>> = Vec::new();
            if let Some(start) = ctx.delayed_text_start.take() {
                out.push(Ok(start));
            }
            // Drop any per-block state; tag straddling end-of-block is treated
            // as plain text (upstream does the same: no flush before close).
            ctx.extractions.remove(&id);
            out.push(Ok(StreamPart::TextEnd {
                id,
                provider_metadata,
            }));
            out
        }
        other => vec![Ok(other)],
    }
}

fn process_text_delta(ctx: &mut StreamCtx, id: &str, delta: &str) -> Vec<Result<StreamPart>> {
    let opening_tag = format!("<{}>", ctx.tag);
    let closing_tag = format!("</{}>", ctx.tag);

    let extraction = ctx
        .extractions
        .entry(id.to_owned())
        .or_insert_with(|| Extraction {
            is_first_reasoning: true,
            is_first_text: true,
            after_switch: false,
            is_reasoning: ctx.start_with_reasoning,
            buffer: String::new(),
            id_counter: 0,
            text_id: id.to_owned(),
        });
    extraction.buffer.push_str(delta);

    let mut out: Vec<Result<StreamPart>> = Vec::new();
    loop {
        let next_tag: &str = if extraction.is_reasoning {
            &closing_tag
        } else {
            &opening_tag
        };

        let start_index = potential_start_index(&extraction.buffer, next_tag);
        let Some(start_idx) = start_index else {
            // No tag (full or partial) — publish whole buffer.
            let snapshot = std::mem::take(&mut extraction.buffer);
            publish(
                extraction,
                &snapshot,
                &ctx.separator,
                &mut ctx.delayed_text_start,
                &mut out,
            );
            break;
        };

        // Publish text before the (potential) tag.
        let before = extraction.buffer[..start_idx].to_owned();
        publish(
            extraction,
            &before,
            &ctx.separator,
            &mut ctx.delayed_text_start,
            &mut out,
        );

        let after_tag = start_idx + next_tag.len();
        let full_match = after_tag <= extraction.buffer.len();
        if !full_match {
            // Partial match — retain straddling bytes for next chunk.
            extraction.buffer = extraction.buffer[start_idx..].to_owned();
            break;
        }

        extraction.buffer = extraction.buffer[after_tag..].to_owned();
        if extraction.is_reasoning {
            // Closing tag — finalize current reasoning block. Emit
            // reasoning-start lazily for blocks that never published a
            // delta (e.g. `<think></think>`).
            if extraction.is_first_reasoning {
                out.push(Ok(StreamPart::ReasoningStart {
                    id: format!("reasoning-{}", extraction.id_counter),
                    provider_metadata: None,
                }));
            }
            out.push(Ok(StreamPart::ReasoningEnd {
                id: format!("reasoning-{}", extraction.id_counter),
                provider_metadata: None,
            }));
            extraction.id_counter += 1;
        }
        extraction.is_reasoning = !extraction.is_reasoning;
        extraction.after_switch = true;
    }
    out
}

fn publish(
    extraction: &mut Extraction,
    text: &str,
    separator: &str,
    delayed_text_start: &mut Option<StreamPart>,
    out: &mut Vec<Result<StreamPart>>,
) {
    if text.is_empty() {
        return;
    }
    let needs_prefix = extraction.after_switch
        && (if extraction.is_reasoning {
            !extraction.is_first_reasoning
        } else {
            !extraction.is_first_text
        });
    let payload = if needs_prefix {
        format!("{separator}{text}")
    } else {
        text.to_owned()
    };

    if extraction.is_reasoning {
        if extraction.after_switch || extraction.is_first_reasoning {
            out.push(Ok(StreamPart::ReasoningStart {
                id: format!("reasoning-{}", extraction.id_counter),
                provider_metadata: None,
            }));
        }
        out.push(Ok(StreamPart::ReasoningDelta {
            id: format!("reasoning-{}", extraction.id_counter),
            delta: payload,
            provider_metadata: None,
        }));
    } else {
        if let Some(start) = delayed_text_start.take() {
            out.push(Ok(start));
        }
        out.push(Ok(StreamPart::TextDelta {
            id: extraction.text_id.clone(),
            delta: payload,
            provider_metadata: None,
        }));
    }

    extraction.after_switch = false;
    if extraction.is_reasoning {
        extraction.is_first_reasoning = false;
    } else {
        extraction.is_first_text = false;
    }
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

    #[derive(Debug)]
    struct MultiChunkFake {
        chunks: Vec<String>,
    }

    #[async_trait]
    impl LanguageModel for MultiChunkFake {
        fn provider(&self) -> &'static str {
            "fake"
        }
        fn model_id(&self) -> &'static str {
            "fake"
        }
        async fn do_generate(&self, _opts: CallOptions) -> Result<GenerateResult> {
            unimplemented!()
        }
        async fn do_stream(&self, _opts: CallOptions) -> Result<StreamResult> {
            let mut parts: Vec<Result<StreamPart>> = vec![Ok(StreamPart::TextStart {
                id: "b".into(),
                provider_metadata: None,
            })];
            for chunk in &self.chunks {
                parts.push(Ok(StreamPart::TextDelta {
                    id: "b".into(),
                    delta: chunk.clone(),
                    provider_metadata: None,
                }));
            }
            parts.push(Ok(StreamPart::TextEnd {
                id: "b".into(),
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
    async fn stream_emits_incrementally_across_chunks() {
        // Mirrors upstream `extract-reasoning-middleware.test.ts:201-298`:
        // reasoning content arriving across multiple chunks must be emitted
        // as reasoning-delta increments rather than buffered until block
        // close. The opening / closing tags also straddle chunk boundaries.
        let inner: Arc<dyn LanguageModel> = Arc::new(MultiChunkFake {
            chunks: vec![
                "<thi".into(),
                "nk>analyzing ".into(),
                "the request</th".into(),
                "ink>Here is ".into(),
                "the response".into(),
            ],
        });
        let wrapped = wrap_language_model(
            inner,
            [Arc::new(ExtractReasoningMiddleware::new("think", false))
                as Arc<dyn LanguageModelMiddleware>],
        );
        let mut s = wrapped.do_stream(CallOptions::default()).await.unwrap();
        let mut reasoning_deltas: Vec<String> = Vec::new();
        let mut text_deltas: Vec<String> = Vec::new();
        let mut reasoning_starts = 0u32;
        let mut reasoning_ends = 0u32;
        while let Some(item) = s.stream.next().await {
            match item.unwrap() {
                StreamPart::ReasoningStart { .. } => reasoning_starts += 1,
                StreamPart::ReasoningDelta { delta, .. } => reasoning_deltas.push(delta),
                StreamPart::ReasoningEnd { .. } => reasoning_ends += 1,
                StreamPart::TextDelta { delta, .. } => text_deltas.push(delta),
                _ => {}
            }
        }
        assert_eq!(reasoning_starts, 1, "one reasoning block opened");
        assert_eq!(reasoning_ends, 1, "one reasoning block closed");
        assert!(
            reasoning_deltas.len() >= 2,
            "expected >=2 reasoning-delta ticks, got {reasoning_deltas:?}"
        );
        assert_eq!(reasoning_deltas.concat(), "analyzing the request");
        assert_eq!(text_deltas.concat(), "Here is the response");
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
