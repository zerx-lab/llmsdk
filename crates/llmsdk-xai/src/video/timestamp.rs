//! Minimal ISO-8601 / RFC-3339 `now()` helper, dependency-free.
//!
//! Extracted out of `model.rs` to keep the model file under the project's
//! 400-line soft cap. We need an in-process timestamp because the
//! [`VideoResponseInfo::timestamp`] field is **required** by the trait but
//! xAI's polling endpoint does not include a job-start timestamp.
//!
//! [`VideoResponseInfo::timestamp`]: llmsdk_provider::video_model::VideoResponseInfo::timestamp
// Rust guideline compliant 2026-05-25

use std::time::{SystemTime, UNIX_EPOCH};

/// ISO-8601 / RFC-3339 representation of "now" (UTC, second precision).
pub(crate) fn now_iso8601() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    format_unix_seconds_utc(secs)
}

/// Convert UNIX seconds → `YYYY-MM-DDTHH:MM:SSZ`.
fn format_unix_seconds_utc(secs: u64) -> String {
    let days = secs / 86_400;
    let time_of_day = secs % 86_400;
    #[allow(
        clippy::cast_possible_truncation,
        reason = "time_of_day < 86_400 so each component fits in u8"
    )]
    let (hour, minute, second) = (
        (time_of_day / 3600) as u8,
        ((time_of_day % 3600) / 60) as u8,
        (time_of_day % 60) as u8,
    );
    let (y, m, d) = days_to_ymd(days);
    format!("{y:04}-{m:02}-{d:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// Convert "days since 1970-01-01" → (year, month, day) via Howard Hinnant's
/// `date.h` algorithm (public domain). All arithmetic stays in `u64` to
/// avoid sign-cast lints; valid for any non-negative day offset.
fn days_to_ymd(days: u64) -> (u32, u8, u8) {
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let (m, y) = if mp < 10 {
        (mp + 3, y)
    } else {
        (mp - 9, y + 1)
    };
    #[allow(
        clippy::cast_possible_truncation,
        reason = "year ≤ 9999 (u32), month ∈ 1..=12, day ∈ 1..=31"
    )]
    {
        (y as u32, m as u8, d as u8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_is_well_formed() {
        assert_eq!(format_unix_seconds_utc(0), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn known_vector_matches() {
        assert_eq!(
            format_unix_seconds_utc(1_700_000_000),
            "2023-11-14T22:13:20Z"
        );
    }

    #[test]
    fn now_returns_iso8601_shape() {
        let s = now_iso8601();
        // YYYY-MM-DDTHH:MM:SSZ
        assert_eq!(s.len(), 20, "got: {s}");
        assert!(s.ends_with('Z'));
        assert!(s.contains('T'));
    }
}
