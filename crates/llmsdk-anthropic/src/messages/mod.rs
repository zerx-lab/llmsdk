//! `Anthropic` Messages API implementation.
//!
//! Mirrors `@ai-sdk/anthropic/src/anthropic-language-model.ts` (subset).
//!
//! # Prompt mapping (llmsdk -> anthropic)
//!
//! ```text
//! Message::System { content }                -> top-level `system` field
//!                                                (all systems concatenated with "\n\n")
//! Message::User { content: parts }           -> { "role":"user", content:[...] }
//!   UserPart::Text                            -> { "type":"text", "text": ... }
//!   UserPart::File (image/*, url)            -> { "type":"image", "source":{"type":"url",...} }
//!   UserPart::File (image/*, base64)         -> { "type":"image", "source":{"type":"base64",...} }
//! Message::Assistant { content }             -> { "role":"assistant", content:[...] }
//!   AssistantPart::Text                       -> { "type":"text", ... }
//!   AssistantPart::ToolCall                   -> { "type":"tool_use", id, name, input }
//! Message::Tool { content }                  -> coalesced into the next user message as
//!                                                { "type":"tool_result", "tool_use_id", "content" }
//!                                                (Anthropic does not have a separate role:tool)
//! ```
//!
//! # Skipped (vs ai-sdk)
//!
//! - `thinking` / `redacted_thinking` / `signature_delta` — reasoning M7+
//! - server tools (`web_search`, `web_fetch`, `code_execution`, `mcp`,
//!   `bash`, `text_editor`, `tool_search`, `advisor`) — out of M6 scope
//! - citations / `cache_control` / `context_management` / containers
//! - non-image file parts (audio/pdf/document)
// Rust guideline compliant 2026-02-21

mod convert_prompt;
mod finish_reason;
mod model;
mod options;
mod parse_response;
mod stream;
mod stream_event;
mod usage;
mod wire;

pub use model::AnthropicMessagesModel;
