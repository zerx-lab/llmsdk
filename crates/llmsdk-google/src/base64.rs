//! Tiny RFC 4648 base64 encoder/decoder.
//!
//! Inline to honor the project's "no new external dependencies" rule. The
//! Google provider needs both directions: encode (inline file bytes for
//! the wire) and decode (Imagen `bytesBase64Encoded` payloads).
// Rust guideline compliant 2026-05-25

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encode bytes to a base64 string (with padding).
#[must_use]
pub(crate) fn encode_bytes(input: &[u8]) -> String {
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0];
        let b1 = if chunk.len() > 1 { chunk[1] } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] } else { 0 };
        let i0 = (b0 >> 2) & 0x3F;
        let i1 = ((b0 << 4) | (b1 >> 4)) & 0x3F;
        let i2 = ((b1 << 2) | (b2 >> 6)) & 0x3F;
        let i3 = b2 & 0x3F;
        out.push(ALPHABET[i0 as usize] as char);
        out.push(ALPHABET[i1 as usize] as char);
        if chunk.len() > 1 {
            out.push(ALPHABET[i2 as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[i3 as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

/// Decode a base64 string into bytes.
///
/// Strict: input must be padded to a multiple of 4.
pub(crate) fn decode(input: &str) -> Result<Vec<u8>, Base64Error> {
    let bytes = input.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return Err(Base64Error::Length);
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks_exact(4) {
        let (b0, p0) = decode_byte(chunk[0])?;
        let (b1, p1) = decode_byte(chunk[1])?;
        let (b2, p2) = decode_byte(chunk[2])?;
        let (b3, p3) = decode_byte(chunk[3])?;
        if p0 || p1 {
            return Err(Base64Error::Padding);
        }
        let n =
            (u32::from(b0) << 18) | (u32::from(b1) << 12) | (u32::from(b2) << 6) | u32::from(b3);
        out.push(((n >> 16) & 0xFF) as u8);
        if !p2 {
            out.push(((n >> 8) & 0xFF) as u8);
        }
        if !p3 {
            if p2 {
                return Err(Base64Error::Padding);
            }
            out.push((n & 0xFF) as u8);
        }
    }
    Ok(out)
}

fn decode_byte(b: u8) -> Result<(u8, bool), Base64Error> {
    Ok(match b {
        b'A'..=b'Z' => (b - b'A', false),
        b'a'..=b'z' => (b - b'a' + 26, false),
        b'0'..=b'9' => (b - b'0' + 52, false),
        b'+' => (62, false),
        b'/' => (63, false),
        b'=' => (0, true),
        _ => return Err(Base64Error::Byte(b)),
    })
}

/// Base64 decode error.
#[derive(Debug)]
pub(crate) enum Base64Error {
    Length,
    Padding,
    Byte(u8),
}

impl std::fmt::Display for Base64Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Length => f.write_str("input length is not a multiple of 4"),
            Self::Padding => f.write_str("misplaced padding"),
            Self::Byte(b) => write!(f, "non-alphabet byte 0x{b:02x}"),
        }
    }
}

impl std::error::Error for Base64Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        for s in ["", "f", "fo", "foo", "foob", "fooba", "foobar"] {
            let enc = encode_bytes(s.as_bytes());
            let dec = decode(&enc).unwrap();
            assert_eq!(dec, s.as_bytes());
        }
    }
}
