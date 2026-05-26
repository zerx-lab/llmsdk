//! Tiny time helpers — Unix epoch ⇄ RFC 3339 strings, no `chrono` dependency.
//!
//! `llmsdk_provider::shared::ResponseInfo::timestamp` is documented as an
//! ISO-8601 / RFC 3339 string ("for portability"); upstream `ai-sdk` exposes
//! a JS `Date` value which JSON-serialises to the same shape. Provider crates
//! receive Unix seconds in wire payloads (`response.created`, etc.) and
//! convert them here so every `ResponseInfo.timestamp` value carries the
//! same format regardless of provider.
// Rust guideline compliant 2026-05-26

use std::time::{SystemTime, UNIX_EPOCH};

/// Render the current wall-clock time as an RFC 3339 string.
#[must_use]
pub fn rfc3339_now() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    rfc3339_from_unix(now.as_secs(), now.subsec_nanos())
}

/// Convert a Unix-seconds integer to an RFC 3339 (UTC, millisecond
/// precision) string. Convenience wrapper around [`rfc3339_from_unix`] for
/// providers whose wire payloads only carry whole-second timestamps.
#[must_use]
pub fn rfc3339_from_unix_seconds(secs: u64) -> String {
    rfc3339_from_unix(secs, 0)
}

/// Convert a `(seconds, nanoseconds)` Unix-epoch pair to an RFC 3339 UTC
/// string with millisecond precision (`YYYY-MM-DDTHH:MM:SS.mmmZ`).
///
/// Uses Howard Hinnant's `civil_from_days` algorithm so we don't need to
/// pull in `chrono` just for this. Values stay in `u32` / `i64` ranges for
/// any plausible epoch — see lint allow-list inside.
#[must_use]
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    reason = "civil-from-days algorithm trades safety lints for arithmetic clarity; values stay within tested ranges"
)]
pub fn rfc3339_from_unix(secs: u64, nsecs: u32) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let h = (rem / 3600) as u32;
    let m = ((rem % 3600) / 60) as u32;
    let s = (rem % 60) as u32;

    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = y + i64::from(month <= 2);

    format!(
        "{year:04}-{month:02}-{d:02}T{h:02}:{m:02}:{s:02}.{ms:03}Z",
        ms = nsecs / 1_000_000
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc3339_zero_epoch() {
        assert_eq!(rfc3339_from_unix(0, 0), "1970-01-01T00:00:00.000Z");
    }

    #[test]
    fn rfc3339_known_value() {
        assert_eq!(
            rfc3339_from_unix(1_700_000_000, 0),
            "2023-11-14T22:13:20.000Z"
        );
    }

    #[test]
    fn seconds_helper_matches_pair() {
        assert_eq!(
            rfc3339_from_unix_seconds(1_700_000_000),
            "2023-11-14T22:13:20.000Z"
        );
    }
}
