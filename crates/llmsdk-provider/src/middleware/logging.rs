//! Logging middleware that emits structured events around each model call.
//!
//! The middleware does not depend on any logging crate: callers implement
//! [`Logger`] and route events wherever they like (`tracing`, `log`, an
//! in-process channel, ...). A minimal [`StderrLogger`] is bundled as a
//! quick-start option and as a test hook.
//!
//! By default the prompt is **not** included in any event (PII / size
//! reasons); opt in with [`LoggingMiddleware::with_prompt`].
// Rust guideline compliant 2026-02-21

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;

use crate::error::{ProviderError, Result};
use crate::language_model::{
    BoxStream, CallOptions, FinishReason, GenerateResult, LanguageModel, Prompt, StreamPart,
    StreamResult, Usage,
};

use super::language_model::{CallKind, LanguageModelMiddleware};

/// Sink for [`LoggingMiddleware`] events.
///
/// Implement this trait to forward middleware events to your logging system.
/// All methods are synchronous on purpose: emitting a log line should never
/// be the bottleneck on a model call. Buffer / async-dispatch in the impl
/// if you need to.
pub trait Logger: Send + Sync + std::fmt::Debug {
    /// Called once per model call, after `transform_params` and before the
    /// inner model runs.
    fn log_call_start(&self, event: &LogCallStart<'_>);

    /// Called once per *successful* model call.
    ///
    /// For streams this fires when the stream is **opened** (not when it
    /// finishes); per-frame instrumentation is out of scope for the first
    /// iteration.
    fn log_call_end(&self, event: &LogCallEnd<'_>);

    /// Called once per *failed* model call.
    fn log_call_error(&self, event: &LogCallError<'_>);

    /// Called once per emitted [`StreamPart`] when the middleware is built
    /// with [`LoggingMiddleware::with_stream_parts`].
    ///
    /// Default no-op so existing [`Logger`] implementations keep working.
    fn log_stream_part(&self, _event: &LogStreamPart<'_>) {}
}

/// Identity + call shape, common to every event.
#[derive(Debug, Clone, Copy)]
pub struct LogContext<'a> {
    /// Wrapped model's provider id.
    pub provider: &'a str,
    /// Wrapped model's model id.
    pub model_id: &'a str,
    /// Whether this call is a generate or a stream.
    pub call_kind: CallKind,
}

/// Event emitted before the inner model runs.
#[derive(Debug, Clone, Copy)]
pub struct LogCallStart<'a> {
    /// Common identity / call shape.
    pub context: LogContext<'a>,
    /// Prompt — present only when [`LoggingMiddleware::with_prompt`] is set.
    pub prompt: Option<&'a Prompt>,
}

/// Event emitted on success.
#[derive(Debug, Clone, Copy)]
pub struct LogCallEnd<'a> {
    /// Common identity / call shape.
    pub context: LogContext<'a>,
    /// Wall-clock duration from `log_call_start` to call return.
    pub elapsed: Duration,
    /// Token usage — only meaningful for [`CallKind::Generate`] (stream
    /// totals are not available until the stream drains).
    pub usage: Option<&'a Usage>,
    /// Why the model stopped — only meaningful for [`CallKind::Generate`].
    pub finish_reason: Option<&'a FinishReason>,
}

/// Event emitted on failure.
#[derive(Debug, Clone, Copy)]
pub struct LogCallError<'a> {
    /// Common identity / call shape.
    pub context: LogContext<'a>,
    /// Wall-clock duration from `log_call_start` to error return.
    pub elapsed: Duration,
    /// The error that was returned.
    pub error: &'a ProviderError,
}

/// One per-frame event emitted while a stream is alive.
#[derive(Debug, Clone, Copy)]
pub struct LogStreamPart<'a> {
    /// Common identity / call shape.
    pub context: LogContext<'a>,
    /// Wall-clock duration since `log_call_start`.
    pub elapsed: Duration,
    /// Either the part itself (`Ok`) or the per-frame transport error (`Err`).
    pub item: std::result::Result<&'a StreamPart, &'a ProviderError>,
    /// Zero-based index of the part within the stream.
    pub index: usize,
}

/// Middleware that emits [`Logger`] events around every call.
///
/// Cheap to clone (just an `Arc` to the logger); safe to stack on top of
/// retry / cache.
#[derive(Debug, Clone)]
pub struct LoggingMiddleware {
    logger: Arc<dyn Logger>,
    log_prompt: bool,
    log_stream_parts: bool,
}

impl LoggingMiddleware {
    /// Build a middleware that forwards events to `logger`.
    #[must_use]
    pub fn new(logger: Arc<dyn Logger>) -> Self {
        Self {
            logger,
            log_prompt: false,
            log_stream_parts: false,
        }
    }

    /// Include the [`Prompt`] in [`LogCallStart`]. Off by default to avoid
    /// accidentally logging PII or large payloads.
    #[must_use]
    pub fn with_prompt(mut self, include: bool) -> Self {
        self.log_prompt = include;
        self
    }

    /// Emit a [`Logger::log_stream_part`] event for every part yielded by
    /// `do_stream`. Off by default — turning it on can be noisy.
    #[must_use]
    pub fn with_stream_parts(mut self, include: bool) -> Self {
        self.log_stream_parts = include;
        self
    }
}

#[async_trait]
impl LanguageModelMiddleware for LoggingMiddleware {
    async fn wrap_generate(
        &self,
        next: &dyn LanguageModel,
        params: CallOptions,
    ) -> Result<GenerateResult> {
        let context = LogContext {
            provider: next.provider(),
            model_id: next.model_id(),
            call_kind: CallKind::Generate,
        };
        let started = Instant::now();
        self.logger.log_call_start(&LogCallStart {
            context,
            prompt: self.log_prompt.then_some(&params.prompt),
        });
        match next.do_generate(params).await {
            Ok(result) => {
                self.logger.log_call_end(&LogCallEnd {
                    context,
                    elapsed: started.elapsed(),
                    usage: Some(&result.usage),
                    finish_reason: Some(&result.finish_reason),
                });
                Ok(result)
            }
            Err(err) => {
                self.logger.log_call_error(&LogCallError {
                    context,
                    elapsed: started.elapsed(),
                    error: &err,
                });
                Err(err)
            }
        }
    }

    async fn wrap_stream(
        &self,
        next: &dyn LanguageModel,
        params: CallOptions,
    ) -> Result<StreamResult> {
        let context = LogContext {
            provider: next.provider(),
            model_id: next.model_id(),
            call_kind: CallKind::Stream,
        };
        let started = Instant::now();
        self.logger.log_call_start(&LogCallStart {
            context,
            prompt: self.log_prompt.then_some(&params.prompt),
        });
        match next.do_stream(params).await {
            Ok(result) => {
                self.logger.log_call_end(&LogCallEnd {
                    context,
                    elapsed: started.elapsed(),
                    usage: None,
                    finish_reason: None,
                });
                if self.log_stream_parts {
                    let StreamResult {
                        stream,
                        request,
                        response,
                    } = result;
                    let provider = context.provider.to_owned();
                    let model_id = context.model_id.to_owned();
                    let wrapped = wrap_stream_with_logger(
                        stream,
                        Arc::clone(&self.logger),
                        provider,
                        model_id,
                        started,
                    );
                    return Ok(StreamResult {
                        stream: wrapped,
                        request,
                        response,
                    });
                }
                Ok(result)
            }
            Err(err) => {
                self.logger.log_call_error(&LogCallError {
                    context,
                    elapsed: started.elapsed(),
                    error: &err,
                });
                Err(err)
            }
        }
    }
}

/// Wrap `inner` so every yielded `Result<StreamPart>` triggers
/// [`Logger::log_stream_part`] before being forwarded.
fn wrap_stream_with_logger(
    inner: BoxStream<Result<StreamPart>>,
    logger: Arc<dyn Logger>,
    provider: String,
    model_id: String,
    started: Instant,
) -> BoxStream<Result<StreamPart>> {
    let stream = futures::stream::unfold(
        (inner, 0_usize, logger, provider, model_id, started),
        |(mut inner, idx, logger, provider, model_id, started)| async move {
            use futures::StreamExt as _;
            match inner.next().await {
                None => None,
                Some(item) => {
                    let ctx = LogContext {
                        provider: &provider,
                        model_id: &model_id,
                        call_kind: CallKind::Stream,
                    };
                    let event = LogStreamPart {
                        context: ctx,
                        elapsed: started.elapsed(),
                        item: item.as_ref(),
                        index: idx,
                    };
                    logger.log_stream_part(&event);
                    Some((item, (inner, idx + 1, logger, provider, model_id, started)))
                }
            }
        },
    );
    Box::pin(stream)
}

/// Minimal [`Logger`] that writes one line per event to stderr.
///
/// Useful as a quick-start and as a smoke-test hook. Not optimized for
/// production throughput; route to `tracing` / `log` for real workloads.
#[derive(Debug, Default)]
pub struct StderrLogger;

impl Logger for StderrLogger {
    fn log_call_start(&self, event: &LogCallStart<'_>) {
        eprintln!(
            "[llmsdk:start] provider={} model={} kind={:?}",
            event.context.provider, event.context.model_id, event.context.call_kind,
        );
    }

    fn log_call_end(&self, event: &LogCallEnd<'_>) {
        eprintln!(
            "[llmsdk:end]   provider={} model={} kind={:?} elapsed_ms={} finish={:?}",
            event.context.provider,
            event.context.model_id,
            event.context.call_kind,
            event.elapsed.as_millis(),
            event.finish_reason.map(|r| r.unified),
        );
    }

    fn log_call_error(&self, event: &LogCallError<'_>) {
        eprintln!(
            "[llmsdk:error] provider={} model={} kind={:?} elapsed_ms={} error={}",
            event.context.provider,
            event.context.model_id,
            event.context.call_kind,
            event.elapsed.as_millis(),
            event.error,
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use crate::language_model::FinishReasonKind;

    use super::*;

    #[derive(Debug, Default)]
    struct RecordingLogger {
        starts: Mutex<Vec<(String, String, CallKind, bool)>>,
        ends: Mutex<Vec<(String, CallKind, bool, bool)>>,
        errors: Mutex<Vec<(String, CallKind, String)>>,
        parts: Mutex<Vec<(String, usize, bool)>>,
    }

    impl Logger for RecordingLogger {
        fn log_call_start(&self, event: &LogCallStart<'_>) {
            self.starts.lock().expect("starts mutex poisoned").push((
                event.context.provider.to_owned(),
                event.context.model_id.to_owned(),
                event.context.call_kind,
                event.prompt.is_some(),
            ));
        }

        fn log_call_end(&self, event: &LogCallEnd<'_>) {
            self.ends.lock().expect("ends mutex poisoned").push((
                event.context.provider.to_owned(),
                event.context.call_kind,
                event.usage.is_some(),
                event.finish_reason.is_some(),
            ));
        }

        fn log_call_error(&self, event: &LogCallError<'_>) {
            self.errors.lock().expect("errors mutex poisoned").push((
                event.context.provider.to_owned(),
                event.context.call_kind,
                event.error.to_string(),
            ));
        }

        fn log_stream_part(&self, event: &LogStreamPart<'_>) {
            self.parts.lock().expect("parts mutex poisoned").push((
                event.context.provider.to_owned(),
                event.index,
                event.item.is_ok(),
            ));
        }
    }

    #[derive(Debug)]
    struct StubModel {
        provider: String,
        model_id: String,
        should_fail: bool,
    }

    #[async_trait]
    impl LanguageModel for StubModel {
        fn provider(&self) -> &str {
            &self.provider
        }
        fn model_id(&self) -> &str {
            &self.model_id
        }
        async fn do_generate(&self, _options: CallOptions) -> Result<GenerateResult> {
            if self.should_fail {
                return Err(ProviderError::invalid_prompt("nope"));
            }
            Ok(GenerateResult {
                content: vec![],
                finish_reason: FinishReason::new(FinishReasonKind::Stop),
                usage: Usage::default(),
                provider_metadata: None,
                request: None,
                response: None,
                warnings: vec![],
            })
        }
        async fn do_stream(&self, _options: CallOptions) -> Result<StreamResult> {
            if self.should_fail {
                return Err(ProviderError::invalid_prompt("nope"));
            }
            Ok(StreamResult {
                stream: Box::pin(futures::stream::iter(Vec::new())),
                request: None,
                response: None,
            })
        }
    }

    #[tokio::test]
    async fn success_emits_start_and_end_and_skips_prompt_by_default() {
        let logger = Arc::new(RecordingLogger::default());
        let mw = LoggingMiddleware::new(Arc::clone(&logger) as Arc<dyn Logger>);
        let model = StubModel {
            provider: "openai".to_owned(),
            model_id: "gpt-foo".to_owned(),
            should_fail: false,
        };
        mw.wrap_generate(&model, CallOptions::default())
            .await
            .expect("ok");
        let starts = logger.starts.lock().expect("starts mutex poisoned");
        assert_eq!(starts.len(), 1);
        assert_eq!(starts[0].0, "openai");
        assert_eq!(starts[0].1, "gpt-foo");
        assert_eq!(starts[0].2, CallKind::Generate);
        assert!(!starts[0].3, "prompt suppressed by default");
        let ends = logger.ends.lock().expect("ends mutex poisoned");
        assert_eq!(ends.len(), 1);
        assert!(ends[0].2, "usage attached for generate");
        assert!(ends[0].3, "finish_reason attached for generate");
        assert!(
            logger
                .errors
                .lock()
                .expect("errors mutex poisoned")
                .is_empty(),
            "no error event on success"
        );
    }

    #[tokio::test]
    async fn with_prompt_attaches_prompt_to_start_event() {
        let logger = Arc::new(RecordingLogger::default());
        let mw = LoggingMiddleware::new(Arc::clone(&logger) as Arc<dyn Logger>).with_prompt(true);
        let model = StubModel {
            provider: "openai".to_owned(),
            model_id: "gpt-foo".to_owned(),
            should_fail: false,
        };
        mw.wrap_generate(&model, CallOptions::default())
            .await
            .expect("ok");
        assert!(
            logger.starts.lock().expect("starts mutex poisoned")[0].3,
            "prompt attached when opt-in"
        );
    }

    #[tokio::test]
    async fn failure_emits_start_and_error_and_propagates() {
        let logger = Arc::new(RecordingLogger::default());
        let mw = LoggingMiddleware::new(Arc::clone(&logger) as Arc<dyn Logger>);
        let model = StubModel {
            provider: "openai".to_owned(),
            model_id: "gpt-foo".to_owned(),
            should_fail: true,
        };
        let err = mw
            .wrap_generate(&model, CallOptions::default())
            .await
            .expect_err("propagates");
        assert!(err.to_string().contains("nope"));
        assert_eq!(
            logger.errors.lock().expect("errors mutex poisoned").len(),
            1
        );
        assert!(logger.ends.lock().expect("ends mutex poisoned").is_empty());
    }

    #[derive(Debug)]
    struct ThreePartStream;

    #[async_trait]
    impl LanguageModel for ThreePartStream {
        fn provider(&self) -> &'static str {
            "openai"
        }
        fn model_id(&self) -> &'static str {
            "gpt-foo"
        }
        async fn do_generate(&self, _options: CallOptions) -> Result<GenerateResult> {
            unimplemented!()
        }
        async fn do_stream(&self, _options: CallOptions) -> Result<StreamResult> {
            let parts: Vec<Result<StreamPart>> = vec![
                Ok(StreamPart::StreamStart { warnings: vec![] }),
                Ok(StreamPart::TextStart {
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
                stream: Box::pin(futures::stream::iter(parts)),
                request: None,
                response: None,
            })
        }
    }

    #[tokio::test]
    async fn stream_parts_opt_in_emits_one_event_per_frame() {
        use futures::StreamExt as _;

        let logger = Arc::new(RecordingLogger::default());
        let mw =
            LoggingMiddleware::new(Arc::clone(&logger) as Arc<dyn Logger>).with_stream_parts(true);

        let mut result = mw
            .wrap_stream(&ThreePartStream, CallOptions::default())
            .await
            .expect("opens");
        while result.stream.next().await.is_some() {}

        let parts = logger.parts.lock().expect("parts mutex").clone();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].1, 0);
        assert_eq!(parts[2].1, 2);
        assert!(parts.iter().all(|(_, _, ok)| *ok));
    }

    #[tokio::test]
    async fn stream_success_attaches_no_usage_or_finish_reason() {
        let logger = Arc::new(RecordingLogger::default());
        let mw = LoggingMiddleware::new(Arc::clone(&logger) as Arc<dyn Logger>);
        let model = StubModel {
            provider: "openai".to_owned(),
            model_id: "gpt-foo".to_owned(),
            should_fail: false,
        };
        mw.wrap_stream(&model, CallOptions::default())
            .await
            .expect("ok");
        let ends = logger.ends.lock().expect("ends mutex poisoned");
        assert_eq!(ends[0].1, CallKind::Stream);
        assert!(!ends[0].2, "usage is None for stream");
        assert!(!ends[0].3, "finish_reason is None for stream");
    }
}
