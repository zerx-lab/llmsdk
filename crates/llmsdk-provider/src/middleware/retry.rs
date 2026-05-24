//! Retry middleware with exponential backoff.
//!
//! Retries only errors that report [`ProviderError::is_retryable`] (typically
//! HTTP 408 / 409 / 429 / 5xx). Non-retryable errors fail fast; the stream
//! variant only retries before the stream opens, never mid-stream.
//!
//! # Runtime requirement
//!
//! Uses [`tokio::time::sleep`] for backoff, so the caller must run inside a
//! tokio runtime (any flavor). No assumption is made about the wider
//! `Provider` implementation's runtime, but in practice every `llmsdk-*`
//! provider already uses tokio via `reqwest`.
// Rust guideline compliant 2026-02-21

use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;

use crate::error::{ProviderError, Result};
use crate::language_model::{CallOptions, GenerateResult, LanguageModel, StreamResult};

use super::language_model::LanguageModelMiddleware;

/// Default maximum number of attempts (initial + retries).
pub const DEFAULT_MAX_ATTEMPTS: u32 = 3;
/// Default initial backoff before the first retry.
pub const DEFAULT_INITIAL_BACKOFF: Duration = Duration::from_millis(100);
/// Default multiplicative factor applied to each successive backoff.
pub const DEFAULT_BACKOFF_MULTIPLIER: f32 = 2.0;
/// Default cap on a single backoff sleep.
pub const DEFAULT_MAX_BACKOFF: Duration = Duration::from_secs(5);
/// Default jitter ratio (no jitter).
pub const DEFAULT_JITTER_RATIO: f32 = 0.0;

/// Middleware that retries failed calls with exponential backoff.
///
/// Retry policy:
///
/// - Only errors with [`ProviderError::is_retryable`] are retried; everything
///   else propagates immediately.
/// - [`Self::wrap_generate`]: retries the full request up to
///   `max_attempts` times.
/// - [`Self::wrap_stream`]: retries opening the stream only. Once the stream
///   is open (any item, including [`crate::language_model::StreamPart::Error`],
///   has been delivered), the retry policy stops; callers decide whether to
///   re-issue the call.
///
/// Backoff is deterministic (no jitter); add jitter at the caller level if
/// you have hundreds of concurrent retriers hitting the same upstream.
///
/// # Examples
///
/// ```ignore
/// use std::sync::Arc;
/// use std::time::Duration;
/// use llmsdk_provider::{wrap_language_model, LanguageModel, LanguageModelMiddleware};
/// use llmsdk_provider::middleware::RetryMiddleware;
///
/// fn add_retry(model: Arc<dyn LanguageModel>) -> Arc<dyn LanguageModel> {
///     let retry = RetryMiddleware::builder()
///         .max_attempts(5)
///         .initial_backoff(Duration::from_millis(200))
///         .build();
///     wrap_language_model(model, [Arc::new(retry) as Arc<dyn LanguageModelMiddleware>])
/// }
/// ```
#[derive(Debug)]
pub struct RetryMiddleware {
    max_attempts: u32,
    initial_backoff: Duration,
    backoff_multiplier: f32,
    max_backoff: Duration,
    /// Full-jitter ratio in `[0.0, 1.0]`. `0.0` disables jitter. Final backoff
    /// is `base * (1 - r/2 .. 1 + r/2)` (uniform within bounds).
    jitter_ratio: f32,
    /// `SplitMix64` state seeded once from `SystemTime` nanos. `Mutex` so a
    /// `&self` retry callback can mutate it; uncontended in practice
    /// (one mutation per backoff).
    rng: Mutex<u64>,
}

impl Clone for RetryMiddleware {
    fn clone(&self) -> Self {
        Self {
            max_attempts: self.max_attempts,
            initial_backoff: self.initial_backoff,
            backoff_multiplier: self.backoff_multiplier,
            max_backoff: self.max_backoff,
            jitter_ratio: self.jitter_ratio,
            rng: Mutex::new(*self.rng.lock().expect("rng mutex poisoned")),
        }
    }
}

impl Default for RetryMiddleware {
    fn default() -> Self {
        Self {
            max_attempts: DEFAULT_MAX_ATTEMPTS,
            initial_backoff: DEFAULT_INITIAL_BACKOFF,
            backoff_multiplier: DEFAULT_BACKOFF_MULTIPLIER,
            max_backoff: DEFAULT_MAX_BACKOFF,
            jitter_ratio: DEFAULT_JITTER_RATIO,
            rng: Mutex::new(seed_from_clock()),
        }
    }
}

/// Mix the current wall-clock nanoseconds into a 64-bit seed.
#[allow(
    clippy::cast_possible_truncation,
    reason = "low 64 bits of clock are intentionally taken as PRNG seed"
)]
fn seed_from_clock() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0xDEAD_BEEF_CAFE_BABE, |d| d.as_nanos() as u64);
    // One mix step so callers that build many middlewares back-to-back don't
    // get correlated streams.
    splitmix64(&mut { nanos })
}

/// `SplitMix64` PRNG (one step).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl RetryMiddleware {
    /// Build with default policy ([`DEFAULT_MAX_ATTEMPTS`], 100ms initial,
    /// x2 multiplier, 5s cap).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Open a builder for non-default tuning.
    #[must_use]
    pub fn builder() -> RetryMiddlewareBuilder {
        RetryMiddlewareBuilder(Self::default())
    }

    /// Compute the backoff for retry index `attempt` (0-based: 0 = before
    /// first retry, 1 = before second retry, ...).
    ///
    /// When `jitter_ratio > 0`, the result is uniformly perturbed within
    /// `[base * (1 - r/2), base * (1 + r/2)]` and re-clamped to `max_backoff`.
    fn backoff_for(&self, attempt: u32) -> Duration {
        // Use `as_secs_f64` / `from_secs_f64` so the math stays in `f64`
        // throughout and `Duration` handles bounds for us.
        let exponent = i32::try_from(attempt).unwrap_or(i32::MAX);
        let factor = f64::from(self.backoff_multiplier).powi(exponent);
        let mut scaled = self.initial_backoff.as_secs_f64() * factor;
        if !scaled.is_finite() || scaled <= 0.0 {
            return self.initial_backoff;
        }
        let cap = self.max_backoff.as_secs_f64();
        scaled = scaled.min(cap);
        if self.jitter_ratio > 0.0 {
            scaled = self.apply_jitter(scaled).min(cap);
        }
        Duration::from_secs_f64(scaled.max(0.0))
    }

    /// Multiply `base` by a uniform factor in `[1 - r/2, 1 + r/2]` where
    /// `r = jitter_ratio.clamp(0.0, 1.0)`.
    #[allow(
        clippy::cast_precision_loss,
        reason = "f64 mantissa is 52 bits; raw is masked to 53 bits before the cast"
    )]
    fn apply_jitter(&self, base: f64) -> f64 {
        let r = f64::from(self.jitter_ratio.clamp(0.0, 1.0));
        // Sample uniform u in [0, 1).
        let raw = {
            let mut guard = self.rng.lock().expect("rng mutex poisoned");
            splitmix64(&mut guard)
        };
        let u = (raw >> 11) as f64 / (1u64 << 53) as f64;
        let factor = 1.0 + r * (u - 0.5);
        base * factor
    }
}

/// Builder for [`RetryMiddleware`]; create via [`RetryMiddleware::builder`].
#[derive(Debug)]
pub struct RetryMiddlewareBuilder(RetryMiddleware);

impl RetryMiddlewareBuilder {
    /// Set the maximum number of attempts (must be `>= 1`; `1` disables retry).
    #[must_use]
    pub fn max_attempts(mut self, attempts: u32) -> Self {
        self.0.max_attempts = attempts.max(1);
        self
    }

    /// Set the initial backoff applied before the first retry.
    #[must_use]
    pub fn initial_backoff(mut self, dur: Duration) -> Self {
        self.0.initial_backoff = dur;
        self
    }

    /// Set the multiplier applied between successive retries.
    #[must_use]
    pub fn backoff_multiplier(mut self, factor: f32) -> Self {
        self.0.backoff_multiplier = factor.max(1.0);
        self
    }

    /// Set the upper bound on a single backoff sleep.
    #[must_use]
    pub fn max_backoff(mut self, dur: Duration) -> Self {
        self.0.max_backoff = dur;
        self
    }

    /// Set the full-jitter ratio (clamped to `[0.0, 1.0]`).
    ///
    /// `0.0` (default) means deterministic backoff. `1.0` spreads each sleep
    /// uniformly over `[base/2, base*1.5]`. Use a non-zero value when many
    /// callers retry the same upstream simultaneously to avoid thundering-herd.
    #[must_use]
    pub fn jitter_ratio(mut self, ratio: f32) -> Self {
        self.0.jitter_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    /// Finalize the middleware.
    #[must_use]
    pub fn build(self) -> RetryMiddleware {
        self.0
    }
}

#[async_trait]
impl LanguageModelMiddleware for RetryMiddleware {
    async fn wrap_generate(
        &self,
        next: &dyn LanguageModel,
        params: CallOptions,
    ) -> Result<GenerateResult> {
        let mut attempt: u32 = 0;
        loop {
            let outcome = next.do_generate(params.clone()).await;
            match outcome {
                Ok(result) => return Ok(result),
                Err(err) => {
                    if !should_retry(&err, attempt, self.max_attempts) {
                        return Err(err);
                    }
                    tokio::time::sleep(self.backoff_for(attempt)).await;
                    attempt += 1;
                }
            }
        }
    }

    async fn wrap_stream(
        &self,
        next: &dyn LanguageModel,
        params: CallOptions,
    ) -> Result<StreamResult> {
        let mut attempt: u32 = 0;
        loop {
            let outcome = next.do_stream(params.clone()).await;
            match outcome {
                Ok(result) => return Ok(result),
                Err(err) => {
                    if !should_retry(&err, attempt, self.max_attempts) {
                        return Err(err);
                    }
                    tokio::time::sleep(self.backoff_for(attempt)).await;
                    attempt += 1;
                }
            }
        }
    }
}

/// True when `err` is retryable and we have attempts left.
///
/// `attempt` is the zero-based index of the *failed* attempt: 0 = first call
/// just failed, so we still have `max_attempts - 1` retries available.
fn should_retry(err: &ProviderError, attempt: u32, max_attempts: u32) -> bool {
    err.is_retryable() && attempt + 1 < max_attempts
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::language_model::{FinishReason, FinishReasonKind, Usage};

    use super::*;

    /// Mock model that fails the first N attempts with a configurable error,
    /// then succeeds.
    #[derive(Debug)]
    struct FlakyModel {
        provider: String,
        model_id: String,
        fail_until: u32,
        next_error: Mutex<Option<fn() -> ProviderError>>,
        call_count: AtomicUsize,
    }

    impl FlakyModel {
        fn new(fail_until: u32, err_factory: fn() -> ProviderError) -> Self {
            Self {
                provider: "test".to_owned(),
                model_id: "flaky".to_owned(),
                fail_until,
                next_error: Mutex::new(Some(err_factory)),
                call_count: AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    fn retryable_503() -> ProviderError {
        ProviderError::api_call_builder("https://api.test", "service unavailable")
            .status_code(503)
            .build()
    }

    fn non_retryable_400() -> ProviderError {
        ProviderError::api_call_builder("https://api.test", "bad request")
            .status_code(400)
            .build()
    }

    fn ok_result() -> GenerateResult {
        GenerateResult {
            content: vec![],
            finish_reason: FinishReason::new(FinishReasonKind::Stop),
            usage: Usage::default(),
            provider_metadata: None,
            request: None,
            response: None,
            warnings: vec![],
        }
    }

    #[async_trait]
    impl LanguageModel for FlakyModel {
        fn provider(&self) -> &str {
            &self.provider
        }

        fn model_id(&self) -> &str {
            &self.model_id
        }

        async fn do_generate(&self, _options: CallOptions) -> Result<GenerateResult> {
            let n = self.call_count.fetch_add(1, Ordering::SeqCst);
            if u32::try_from(n).is_ok_and(|n| n < self.fail_until) {
                let factory = self
                    .next_error
                    .lock()
                    .expect("error factory mutex poisoned")
                    .expect("error factory missing");
                return Err(factory());
            }
            Ok(ok_result())
        }

        async fn do_stream(&self, _options: CallOptions) -> Result<StreamResult> {
            let n = self.call_count.fetch_add(1, Ordering::SeqCst);
            if u32::try_from(n).is_ok_and(|n| n < self.fail_until) {
                let factory = self
                    .next_error
                    .lock()
                    .expect("error factory mutex poisoned")
                    .expect("error factory missing");
                return Err(factory());
            }
            Ok(StreamResult {
                stream: Box::pin(futures::stream::iter(Vec::new())),
                request: None,
                response: None,
            })
        }
    }

    #[tokio::test(start_paused = true)]
    async fn retries_retryable_then_succeeds() {
        let model = Arc::new(FlakyModel::new(2, retryable_503));
        let retry = RetryMiddleware::builder()
            .max_attempts(3)
            .initial_backoff(Duration::from_millis(10))
            .build();
        retry
            .wrap_generate(&*model, CallOptions::default())
            .await
            .expect("third attempt succeeds");
        assert_eq!(model.calls(), 3, "two failures + one success");
    }

    #[tokio::test(start_paused = true)]
    async fn non_retryable_fails_fast() {
        let model = Arc::new(FlakyModel::new(5, non_retryable_400));
        let retry = RetryMiddleware::builder().max_attempts(5).build();
        let err = retry
            .wrap_generate(&*model, CallOptions::default())
            .await
            .expect_err("non-retryable error propagates");
        assert!(!err.is_retryable());
        assert_eq!(model.calls(), 1, "no retries for non-retryable error");
    }

    #[tokio::test(start_paused = true)]
    async fn exhausts_attempts_and_returns_last_error() {
        let model = Arc::new(FlakyModel::new(10, retryable_503));
        let retry = RetryMiddleware::builder()
            .max_attempts(3)
            .initial_backoff(Duration::from_millis(1))
            .build();
        let err = retry
            .wrap_generate(&*model, CallOptions::default())
            .await
            .expect_err("attempts exhausted");
        assert_eq!(err.status_code(), Some(503));
        assert_eq!(model.calls(), 3, "max_attempts == 3 total calls");
    }

    #[tokio::test(start_paused = true)]
    async fn max_attempts_one_disables_retry() {
        let model = Arc::new(FlakyModel::new(5, retryable_503));
        let retry = RetryMiddleware::builder().max_attempts(1).build();
        let err = retry
            .wrap_generate(&*model, CallOptions::default())
            .await
            .expect_err("first failure propagates");
        assert!(err.is_retryable());
        assert_eq!(model.calls(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn stream_retries_open_failures() {
        let model = Arc::new(FlakyModel::new(2, retryable_503));
        let retry = RetryMiddleware::builder()
            .max_attempts(3)
            .initial_backoff(Duration::from_millis(1))
            .build();
        retry
            .wrap_stream(&*model, CallOptions::default())
            .await
            .expect("stream opens on third attempt");
        assert_eq!(model.calls(), 3);
    }

    #[test]
    fn backoff_caps_at_max() {
        let retry = RetryMiddleware::builder()
            .initial_backoff(Duration::from_millis(100))
            .backoff_multiplier(10.0)
            .max_backoff(Duration::from_secs(1))
            .build();
        // attempt 0 -> 100ms, attempt 1 -> 1000ms, attempt 2 -> capped at 1s.
        assert_eq!(retry.backoff_for(0), Duration::from_millis(100));
        assert_eq!(retry.backoff_for(1), Duration::from_secs(1));
        assert_eq!(retry.backoff_for(2), Duration::from_secs(1));
    }

    #[test]
    fn jitter_perturbs_within_expected_range() {
        let retry = RetryMiddleware::builder()
            .initial_backoff(Duration::from_millis(100))
            .backoff_multiplier(1.0) // keep base constant across attempts
            .jitter_ratio(0.5) // -25%..+25%
            .max_backoff(Duration::from_secs(10))
            .build();
        let base = 100.0;
        let lo = base * (1.0 - 0.25);
        let hi = base * (1.0 + 0.25);
        for _ in 0..32 {
            let sample = retry.backoff_for(0).as_secs_f64() * 1000.0;
            assert!(
                sample >= lo && sample <= hi,
                "jitter sample {sample}ms outside [{lo},{hi}]"
            );
        }
    }
}
