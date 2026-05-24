//! `OpenAI` Chat Completions API implementation.
//!
//! Mirrors `@ai-sdk/openai/src/chat/*`. M3 scope is `do_generate` only.
//!
//! # Prompt mapping (llmsdk -> openai)
//!
//! ```text
//! Message::System { content }                -> { "role": "system",    "content": <text> }
//! Message::User { content: parts }           -> { "role": "user",      "content": ... }
//!   UserPart::Text                            -> string when only part, else { "type": "text", ... }
//!   UserPart::File (image/*, data or url)    -> { "type": "image_url", "image_url": { "url": ... } }
//!   UserPart::File (other)                   -> warning + skipped (M3)
//! Message::Assistant { content: parts }      -> { "role": "assistant", "content": ..., "tool_calls": [...] }
//!   AssistantPart::Text                       -> joined into "content"
//!   AssistantPart::ToolCall                   -> tool_calls entry
//!   AssistantPart::Reasoning / File / Custom -> warning + skipped (M3)
//! Message::Tool { content: parts }           -> one "role: tool" message per ToolResult
//!   ToolMessagePart::ToolResult               -> { "role": "tool", "tool_call_id": ..., "content": ... }
//!   ToolMessagePart::ToolApprovalResponse    -> warning + skipped (M3)
//! ```
//!
//! # Skipped (vs ai-sdk)
//!
//! - reasoning-model special-cases (`temperature` / `top_p` stripping) — M4+
//! - search-preview model special-cases — M4+
//! - `logprobs` / `logit_bias` / `response_format=json_schema` strict — later
//! - provider-defined tools, `web_search` annotations — later
//! - parallel tool input streaming tracker — handled in M4 with `do_stream`
// Rust guideline compliant 2026-02-21

mod convert_prompt;
mod finish_reason;
mod model;
mod parse_response;
mod stream;
mod stream_chunk;
mod usage;
mod wire;

pub use model::OpenAiChatModel;

pub(crate) use convert_prompt::convert_prompt;
pub(crate) use parse_response::parse_response;
