//! Strip Markdown code fences from JSON responses.
//!
//! Mirrors `@ai-sdk/ai/src/middleware/extract-json-middleware.ts` 1:1, including
//! the streaming 3-state machine (`prefix` / `streaming` / `buffering`) and
//! the `SUFFIX_BUFFER_SIZE = 12` trailing window that keeps a possible
//! closing fence intact across delta boundaries.
//!
//! The default transform strips ` ```json ` / ` ``` ` fences. Callers may
//! pass [`Self::with_transform`] to install a custom transform; the stream
//! path then routes through the `buffering` phase (full text accumulated and
//! transformed at `text-end`), matching upstream's `hasCustomTransform` gate.
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;

use async_trait::async_trait;
use futures::StreamExt;

use crate::error::Result;
use crate::language_model::{
    BoxStream, CallOptions, Content, GenerateResult, LanguageModel, StreamPart, StreamResult,
    TextPart,
};
use crate::middleware::language_model::LanguageModelMiddleware;
use crate::shared::ProviderMetadata;

/// Trailing window held back in the `streaming` phase so a closing
/// ` ```...$ ` fence cannot be split across two outbound `TextDelta` frames.
/// Mirrors upstream `SUFFIX_BUFFER_SIZE = 12`.
const SUFFIX_BUFFER_SIZE: usize = 12;

/// Shared text transform with [`Send`] + [`Sync`] bounds for cross-thread
/// reuse and cheap cloning into the stream state machine. Mirrors upstream
/// `(text: string) => string`.
type TransformFn = std::sync::Arc<dyn Fn(&str) -> String + Send + Sync>;

/// Middleware that strips Markdown JSON fences from text output.
pub struct ExtractJsonMiddleware {
    transform: Option<TransformFn>,
}

impl std::fmt::Debug for ExtractJsonMiddleware {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtractJsonMiddleware")
            .field("transform", &self.transform.is_some().then_some("<fn>"))
            .finish()
    }
}

impl Default for ExtractJsonMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl ExtractJsonMiddleware {
    /// Build the default middleware (strip ` ```json ` and ` ``` ` fences
    /// via the upstream-aligned default transform).
    #[must_use]
    pub fn new() -> Self {
        Self { transform: None }
    }

    /// Install a custom text transform. When present, the stream path
    /// switches to the `buffering` phase (accumulate the whole text block,
    /// run the transform at `text-end`). Mirrors upstream
    /// `extractJsonMiddleware({ transform })` + `hasCustomTransform` gate.
    #[must_use]
    pub fn with_transform<F>(mut self, transform: F) -> Self
    where
        F: Fn(&str) -> String + Send + Sync + 'static,
    {
        self.transform = Some(std::sync::Arc::new(transform));
        self
    }

    fn apply_transform(&self, text: &str) -> String {
        match self.transform.as_ref() {
            Some(f) => f(text),
            None => default_transform(text),
        }
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
                part.text = self.apply_transform(&part.text);
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
        let transform = self.transform.clone();
        let cleaned = transform_stream(stream, transform);
        Ok(StreamResult {
            stream: cleaned,
            request,
            response,
        })
    }
}

/// Phase of the streaming-side fence-stripping state machine.
///
/// Mirrors upstream `phase: 'prefix' | 'streaming' | 'buffering'`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Haven't yet decided whether the block starts with a fence; keep
    /// buffering until either we see a non-backtick first char (no fence), or
    /// match a full ` ```...\n ` opener.
    Prefix,
    /// Prefix resolved; forward chunks but keep the last `SUFFIX_BUFFER_SIZE`
    /// bytes in `buffer` so a trailing fence is never split.
    Streaming,
    /// Custom transform active; accumulate the whole block and transform at
    /// `text-end`.
    Buffering,
}

#[derive(Debug)]
struct BlockState {
    /// The original `TextStart` event captured so we can replay it once the
    /// prefix phase resolves.
    start_event: StreamPart,
    phase: Phase,
    buffer: String,
    /// Set to `true` when the opening ` ```... ` fence was successfully
    /// stripped, so `text-end` only needs to strip the closing fence.
    prefix_stripped: bool,
}

fn transform_stream(
    stream: BoxStream<Result<StreamPart>>,
    transform: Option<TransformFn>,
) -> BoxStream<Result<StreamPart>> {
    let has_custom_transform = transform.is_some();
    let state: HashMap<String, BlockState> = HashMap::new();
    let pending: std::collections::VecDeque<Result<StreamPart>> = std::collections::VecDeque::new();

    let init = StreamCtx {
        stream,
        state,
        pending,
        transform,
        has_custom_transform,
    };

    let mapped = futures::stream::unfold(init, |mut ctx| async move {
        loop {
            // Drain any pre-queued frames from a prior step before pulling
            // the next upstream chunk.
            if let Some(item) = ctx.pending.pop_front() {
                return Some((item, ctx));
            }
            let next = ctx.stream.next().await?;
            match next {
                Err(e) => return Some((Err(e), ctx)),
                Ok(part) => {
                    ctx.handle(part);
                    // Loop back: handle() pushes any outbound frames into
                    // `pending`. The next iteration drains them.
                }
            }
        }
    });
    Box::pin(mapped)
}

struct StreamCtx {
    stream: BoxStream<Result<StreamPart>>,
    state: HashMap<String, BlockState>,
    pending: std::collections::VecDeque<Result<StreamPart>>,
    transform: Option<TransformFn>,
    has_custom_transform: bool,
}

impl StreamCtx {
    fn apply_transform(&self, text: &str) -> String {
        match self.transform.as_ref() {
            Some(f) => f(text),
            None => default_transform(text),
        }
    }

    fn handle(&mut self, part: StreamPart) {
        match part {
            StreamPart::TextStart {
                id,
                provider_metadata,
            } => self.on_text_start(id, provider_metadata),
            StreamPart::TextDelta { id, delta, .. } => self.on_text_delta(id, delta),
            StreamPart::TextEnd {
                id,
                provider_metadata,
            } => self.on_text_end(id, provider_metadata),
            other => self.pending.push_back(Ok(other)),
        }
    }

    fn on_text_start(&mut self, id: String, provider_metadata: Option<ProviderMetadata>) {
        let start_event = StreamPart::TextStart {
            id: id.clone(),
            provider_metadata,
        };
        let phase = if self.has_custom_transform {
            Phase::Buffering
        } else {
            Phase::Prefix
        };
        self.state.insert(
            id,
            BlockState {
                start_event,
                phase,
                buffer: String::new(),
                prefix_stripped: false,
            },
        );
        // NOTE: the original TextStart is *not* forwarded here. Upstream
        // delays emission until either the prefix is resolved
        // (`controller.enqueue(block.startEvent)`) or `text-end` finds the
        // block still in prefix/buffering phase.
    }

    fn on_text_delta(&mut self, id: String, delta: String) {
        let Some(block) = self.state.get_mut(&id) else {
            // Unknown id — forward unchanged.
            self.pending.push_back(Ok(StreamPart::TextDelta {
                id,
                delta,
                provider_metadata: None,
            }));
            return;
        };
        block.buffer.push_str(&delta);

        // Custom transform: buffer everything, transform at end.
        if block.phase == Phase::Buffering {
            return;
        }

        if block.phase == Phase::Prefix {
            // Mirrors upstream `text-delta` prefix sniffing
            // (extract-json-middleware.ts:107-141).
            if !block.buffer.is_empty() && !block.buffer.starts_with('`') {
                // Not a fence — emit the deferred start and switch to streaming.
                block.phase = Phase::Streaming;
                let start = block.start_event.clone();
                self.pending.push_back(Ok(start));
            } else if block.buffer.starts_with("```") {
                if block.buffer.contains('\n') {
                    if let Some(prefix_len) = match_opening_fence_len(&block.buffer) {
                        block.buffer = block.buffer[prefix_len..].to_owned();
                        block.prefix_stripped = true;
                        block.phase = Phase::Streaming;
                        let start = block.start_event.clone();
                        self.pending.push_back(Ok(start));
                    } else {
                        // Has \n but doesn't match fence pattern.
                        block.phase = Phase::Streaming;
                        let start = block.start_event.clone();
                        self.pending.push_back(Ok(start));
                    }
                }
                // else keep buffering until we see a newline
            } else if block.buffer.len() >= 3 && !block.buffer.starts_with("```") {
                // 3+ chars but no fence opener — definitely no fence.
                block.phase = Phase::Streaming;
                let start = block.start_event.clone();
                self.pending.push_back(Ok(start));
            }
        }

        // Stream content with trailing window held back.
        if block.phase == Phase::Streaming && block.buffer.len() > SUFFIX_BUFFER_SIZE {
            // Slice on a char boundary to avoid splitting a multi-byte UTF-8
            // character. We want to keep the *last* `SUFFIX_BUFFER_SIZE`
            // bytes; walk back from the end to find a valid boundary.
            let cut = floor_char_boundary(&block.buffer, block.buffer.len() - SUFFIX_BUFFER_SIZE);
            let to_stream = block.buffer[..cut].to_owned();
            block.buffer = block.buffer[cut..].to_owned();
            if !to_stream.is_empty() {
                self.pending.push_back(Ok(StreamPart::TextDelta {
                    id: id.clone(),
                    delta: to_stream,
                    provider_metadata: None,
                }));
            }
        }
        let _ = id;
    }

    fn on_text_end(&mut self, id: String, provider_metadata: Option<ProviderMetadata>) {
        let Some(block) = self.state.remove(&id) else {
            self.pending.push_back(Ok(StreamPart::TextEnd {
                id,
                provider_metadata,
            }));
            return;
        };
        let BlockState {
            start_event,
            phase,
            buffer,
            prefix_stripped,
        } = block;

        // If the block never made it past prefix/buffering, the start event
        // has not been forwarded yet — emit it now.
        if matches!(phase, Phase::Prefix | Phase::Buffering) {
            self.pending.push_back(Ok(start_event));
        }

        let remaining = match phase {
            Phase::Buffering => self.apply_transform(&buffer),
            _ if prefix_stripped => strip_trailing_fence_replace(&buffer),
            _ => self.apply_transform(&buffer),
        };

        if !remaining.is_empty() {
            self.pending.push_back(Ok(StreamPart::TextDelta {
                id: id.clone(),
                delta: remaining,
                provider_metadata: None,
            }));
        }
        self.pending.push_back(Ok(StreamPart::TextEnd {
            id,
            provider_metadata,
        }));
    }
}

/// Default transform: strip a leading ` ```json? ` fence and a trailing
/// ` ``` ` fence, then trim. Mirrors upstream `defaultTransform` /
/// `/^```(?:json)?\s*\n?/` + `/\n?```\s*$/` + `.trim()`.
fn default_transform(text: &str) -> String {
    let after_prefix = strip_leading_fence(text);
    let after_suffix = strip_trailing_fence_replace(after_prefix);
    after_suffix.trim().to_owned()
}

/// Strip a leading ` ```json?(\s*\n?)? ` fence.
///
/// Mirrors `/^```(?:json)?\s*\n?/`: case-sensitive `json` literal, any ASCII
/// whitespace afterwards, then an optional single `\n` consumed greedily by
/// `\s*` (the trailing `\n?` in the regex is redundant since `\s*` already
/// matches newlines — replicated here for exact behavioral parity).
fn strip_leading_fence(s: &str) -> &str {
    let Some(after_fence) = s.strip_prefix("```") else {
        return s;
    };
    let after_json = after_fence.strip_prefix("json").unwrap_or(after_fence);
    // `\s*` — any ASCII whitespace (space, tab, CR, LF, FF, VT).
    let mut i = 0;
    let bytes = after_json.as_bytes();
    while i < bytes.len() && matches!(bytes[i], b' ' | b'\t' | b'\r' | b'\n' | 0x0b | 0x0c) {
        i += 1;
    }
    &after_json[i..]
}

/// Match a *complete* opening fence in a streaming buffer.
///
/// Returns the byte length of the fence (including the terminating `\n`)
/// when the buffer matches `/^```(?:json)?\s*\n/`, else `None`. Used by the
/// streaming `prefix` phase to decide whether to keep buffering or to emit
/// the start event.
fn match_opening_fence_len(buf: &str) -> Option<usize> {
    let rest = buf.strip_prefix("```")?;
    let mut consumed = 3;
    let rest = if let Some(r) = rest.strip_prefix("json") {
        consumed += 4;
        r
    } else {
        rest
    };
    let bytes = rest.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\n' => return Some(consumed + i + 1),
            b' ' | b'\t' | b'\r' | 0x0b | 0x0c => i += 1,
            _ => return None,
        }
    }
    None
}

/// Strip the trailing ` ```\s*$ ` fence (with one optional preceding `\n`),
/// then `trim_end`. Mirrors upstream
/// `remaining.replace(/\n?```\s*$/, '').trimEnd()`.
fn strip_trailing_fence_replace(s: &str) -> String {
    let bytes = s.as_bytes();
    // Find the start of trailing whitespace.
    let mut i = bytes.len();
    while i > 0 && matches!(bytes[i - 1], b' ' | b'\t' | b'\r' | b'\n' | 0x0b | 0x0c) {
        i -= 1;
    }
    let before_ws = &s[..i];
    let Some(before_fence) = before_ws.strip_suffix("```") else {
        // No fence to remove — still apply the trailing `trim_end`.
        return s.trim_end().to_owned();
    };
    // One optional preceding `\n` consumed by the regex's `\n?` group.
    let after = before_fence.strip_suffix('\n').unwrap_or(before_fence);
    after.trim_end().to_owned()
}

/// Walk back from `index` to the nearest UTF-8 char boundary `<= index`.
/// Avoids panicking when the streaming window slice would split a multi-byte
/// character.
fn floor_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    let mut i = index;
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
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

    async fn collect(stream: BoxStream<Result<StreamPart>>) -> Vec<StreamPart> {
        let mut out = Vec::new();
        let mut s = stream;
        while let Some(item) = s.next().await {
            out.push(item.unwrap());
        }
        out
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
    async fn stream_no_fence_passes_through_incrementally() {
        // Mirrors upstream behavior: a block whose first non-empty delta is
        // not a backtick switches to `streaming` phase and emits intermediate
        // text-delta frames (minus the trailing 12-byte window).
        let inner: Arc<dyn LanguageModel> = Arc::new(Fake {
            gen_text: String::new(),
            stream_deltas: vec!["hello ".into(), "world ".into(), "of streams".into()],
        });
        let wrapped = wrap_language_model(
            inner,
            [Arc::new(ExtractJsonMiddleware::new()) as Arc<dyn LanguageModelMiddleware>],
        );
        let s = wrapped.do_stream(CallOptions::default()).await.unwrap();
        let frames = collect(s.stream).await;
        let text: String = frames
            .iter()
            .filter_map(|f| match f {
                StreamPart::TextDelta { delta, .. } => Some(delta.clone()),
                _ => None,
            })
            .collect();
        // All three deltas concatenate back to the original; intermediate
        // emission is implementation-dependent but final text must be intact.
        assert_eq!(text, "hello world of streams");
        // start + at least one delta + end + finish should be present.
        assert!(matches!(frames.first(), Some(StreamPart::TextStart { .. })));
        assert!(
            frames
                .iter()
                .any(|f| matches!(f, StreamPart::TextEnd { .. }))
        );
    }

    #[tokio::test]
    async fn stream_strips_fence_split_across_deltas() {
        // The decisive test: the closing ```` sits in a *separate* delta from
        // the JSON body. Upstream's 12-byte trailing window guarantees the
        // fence can be stripped at text-end regardless of the split.
        let inner: Arc<dyn LanguageModel> = Arc::new(Fake {
            gen_text: String::new(),
            stream_deltas: vec![
                "```json\n".into(),
                "{\"city\":\"Tokyo\"}".into(),
                "\n".into(),
                "```".into(),
            ],
        });
        let wrapped = wrap_language_model(
            inner,
            [Arc::new(ExtractJsonMiddleware::new()) as Arc<dyn LanguageModelMiddleware>],
        );
        let s = wrapped.do_stream(CallOptions::default()).await.unwrap();
        let frames = collect(s.stream).await;
        let text: String = frames
            .iter()
            .filter_map(|f| match f {
                StreamPart::TextDelta { delta, .. } => Some(delta.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "{\"city\":\"Tokyo\"}");
    }

    #[tokio::test]
    async fn stream_buffering_phase_with_custom_transform() {
        // When a custom transform is registered, the streaming path must
        // accumulate the whole text block (phase = Buffering) and run the
        // transform at text-end. Mirrors upstream `hasCustomTransform` gate.
        let mw: Arc<dyn LanguageModelMiddleware> =
            Arc::new(ExtractJsonMiddleware::new().with_transform(|s| s.replace("alpha", "ALPHA")));
        let inner: Arc<dyn LanguageModel> = Arc::new(Fake {
            gen_text: String::new(),
            // Deliberately split "alpha" across two deltas so a per-delta
            // transform would miss it; only buffering catches it.
            stream_deltas: vec!["al".into(), "pha-beta".into()],
        });
        let wrapped = wrap_language_model(inner, [mw]);
        let s = wrapped.do_stream(CallOptions::default()).await.unwrap();
        let frames = collect(s.stream).await;
        let text: String = frames
            .iter()
            .filter_map(|f| match f {
                StreamPart::TextDelta { delta, .. } => Some(delta.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "ALPHA-beta");
    }

    #[tokio::test]
    async fn stream_emits_incremental_frames_past_suffix_window() {
        // The streaming phase must hand out a `text-delta` once the buffer
        // exceeds SUFFIX_BUFFER_SIZE (12). With ~40 bytes of plain JSON we
        // expect at least one mid-stream delta.
        let inner: Arc<dyn LanguageModel> = Arc::new(Fake {
            gen_text: String::new(),
            stream_deltas: vec!["{\"alpha\":\"some-long-value-that-exceeds-buffer\"}".into()],
        });
        let wrapped = wrap_language_model(
            inner,
            [Arc::new(ExtractJsonMiddleware::new()) as Arc<dyn LanguageModelMiddleware>],
        );
        let s = wrapped.do_stream(CallOptions::default()).await.unwrap();
        let frames = collect(s.stream).await;
        // Count text-delta frames; should be >= 2 (one mid-stream + final
        // tail at text-end) since the input is well over 12 bytes.
        let delta_count = frames
            .iter()
            .filter(|f| matches!(f, StreamPart::TextDelta { .. }))
            .count();
        assert!(
            delta_count >= 2,
            "expected incremental streaming (>=2 deltas), got {delta_count}: {frames:?}"
        );
    }

    #[test]
    fn default_transform_strips_lower_case_fence_only() {
        // Upstream regex `/^```(?:json)?\s*\n?/` is case-sensitive (no /i
        // flag); only lower-case `json` is consumed by the optional group.
        // An upper-case `JSON` tag is treated as the post-fence body and
        // therefore preserved.
        assert_eq!(default_transform("```json\n{\"a\":1}\n```"), "{\"a\":1}");
        assert_eq!(default_transform("```\n{\"a\":1}\n```"), "{\"a\":1}");
        // Upper-case JSON: outer ``` still stripped by the `(?:json)?\s*\n?`
        // group falling through to `\n?`, but the `JSON` label survives as
        // the first body line — matching upstream exactly.
        assert_eq!(
            default_transform("```JSON\n{\"a\":1}\n```"),
            "JSON\n{\"a\":1}"
        );
    }

    #[test]
    fn match_opening_fence_len_partial_buffer_returns_none() {
        assert_eq!(match_opening_fence_len(""), None);
        assert_eq!(match_opening_fence_len("``"), None);
        assert_eq!(match_opening_fence_len("```"), None); // no newline yet
        assert_eq!(match_opening_fence_len("```json"), None); // no newline yet
        assert_eq!(match_opening_fence_len("```json  "), None);
        assert_eq!(
            match_opening_fence_len("```json  \n"),
            Some("```json  \n".len())
        );
        assert_eq!(match_opening_fence_len("```\n"), Some(4));
        // Non-whitespace before the newline means the regex fails.
        assert_eq!(match_opening_fence_len("```xml\n"), None);
    }

    #[test]
    fn strip_trailing_fence_handles_optional_leading_newline() {
        assert_eq!(strip_trailing_fence_replace("{}\n```"), "{}");
        assert_eq!(strip_trailing_fence_replace("{}```"), "{}");
        assert_eq!(strip_trailing_fence_replace("{}```\n  "), "{}");
        // No trailing fence: still applies trimEnd.
        assert_eq!(strip_trailing_fence_replace("{}\n  "), "{}");
    }
}
