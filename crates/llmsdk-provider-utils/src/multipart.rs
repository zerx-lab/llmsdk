//! Minimal `multipart/form-data` body builder (RFC 7578).
//!
//! Hand-rolled to avoid pulling in `reqwest`'s `multipart` feature. Sufficient
//! for image-edit / variation endpoints where the body is a small number of
//! parts (a few image files + a handful of text fields).
//!
//! Each call to [`Multipart::file`] / [`Multipart::text`] appends a part with
//! a CRLF-separated header and body. [`Multipart::finish`] returns the
//! boundary string (suitable for `Content-Type: multipart/form-data;
//! boundary=...`) and the assembled body bytes.
// Rust guideline compliant 2026-02-21

use std::time::{SystemTime, UNIX_EPOCH};

/// Builder for a `multipart/form-data` body.
#[derive(Debug)]
pub struct Multipart {
    boundary: String,
    body: Vec<u8>,
}

impl Multipart {
    /// Build with a fresh boundary derived from the current time and a
    /// counter (16 hex chars). Boundaries do not need to be cryptographically
    /// random — RFC 7578 only requires uniqueness within the body.
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "low 64 bits of clock are enough for boundary uniqueness within one body"
    )]
    pub fn new() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);
        let boundary = format!("----llmsdk-{nanos:016x}");
        Self {
            boundary,
            body: Vec::new(),
        }
    }

    /// Append a text field.
    pub fn text(&mut self, name: &str, value: &str) -> &mut Self {
        self.write_headers(name, None, None);
        self.body.extend_from_slice(value.as_bytes());
        self.body.extend_from_slice(b"\r\n");
        self
    }

    /// Append a file part.
    ///
    /// `filename` should match the on-wire name expected by the upstream
    /// (e.g. `"image.png"`). `content_type` defaults to
    /// `application/octet-stream` when `None`.
    pub fn file(
        &mut self,
        name: &str,
        filename: &str,
        content_type: Option<&str>,
        bytes: &[u8],
    ) -> &mut Self {
        self.write_headers(
            name,
            Some(filename),
            Some(content_type.unwrap_or("application/octet-stream")),
        );
        self.body.extend_from_slice(bytes);
        self.body.extend_from_slice(b"\r\n");
        self
    }

    fn write_headers(&mut self, name: &str, filename: Option<&str>, content_type: Option<&str>) {
        self.body.extend_from_slice(b"--");
        self.body.extend_from_slice(self.boundary.as_bytes());
        self.body.extend_from_slice(b"\r\n");
        self.body
            .extend_from_slice(b"Content-Disposition: form-data; name=\"");
        self.body.extend_from_slice(name.as_bytes());
        self.body.extend_from_slice(b"\"");
        if let Some(fname) = filename {
            self.body.extend_from_slice(b"; filename=\"");
            self.body.extend_from_slice(fname.as_bytes());
            self.body.extend_from_slice(b"\"");
        }
        self.body.extend_from_slice(b"\r\n");
        if let Some(ct) = content_type {
            self.body.extend_from_slice(b"Content-Type: ");
            self.body.extend_from_slice(ct.as_bytes());
            self.body.extend_from_slice(b"\r\n");
        }
        self.body.extend_from_slice(b"\r\n");
    }

    /// Close the body and return `(boundary, body_bytes)`. The boundary
    /// goes into the `Content-Type` header as
    /// `multipart/form-data; boundary={returned}`.
    #[must_use]
    pub fn finish(mut self) -> (String, Vec<u8>) {
        self.body.extend_from_slice(b"--");
        self.body.extend_from_slice(self.boundary.as_bytes());
        self.body.extend_from_slice(b"--\r\n");
        (self.boundary, self.body)
    }
}

impl Default for Multipart {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assembles_two_parts() {
        let mut mp = Multipart::new();
        mp.text("prompt", "draw a cat")
            .file("image", "cat.png", Some("image/png"), b"PNGDATA");
        let (boundary, body) = mp.finish();
        let body_str = String::from_utf8_lossy(&body);
        assert!(body_str.contains(&format!("--{boundary}")));
        assert!(body_str.contains("name=\"prompt\""));
        assert!(body_str.contains("draw a cat"));
        assert!(body_str.contains("filename=\"cat.png\""));
        assert!(body_str.contains("Content-Type: image/png"));
        assert!(body_str.contains("PNGDATA"));
        assert!(body_str.ends_with(&format!("--{boundary}--\r\n")));
    }
}
