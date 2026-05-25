//! Provider-defined tools that take no construction-time args.
//!
//! Each factory returns a [`Tool::Provider`] keyed by an `anthropic.*` id
//! that the existing `messages::model::resolve_anthropic_server_tool` table
//! recognizes and routes to the correct wire `type` + `name` + beta header.
// Rust guideline compliant 2026-02-21

use llmsdk_provider::language_model::Tool;
use serde::Serialize;

use super::common::build;

#[derive(Serialize)]
#[allow(
    clippy::empty_structs_with_brackets,
    reason = "unit struct serializes as null; we need an empty JSON object"
)]
struct Empty {}

/// `bash_20241022` — initial bash tool (beta `computer-use-2024-10-22`).
///
/// Mirrors `tool/bash_20241022.ts`.
#[must_use]
pub fn bash_20241022() -> Tool {
    build("anthropic.bash_20241022", "bash", Empty {})
}

/// `bash_20250124` — Sonnet 3.7 era bash tool (beta `computer-use-2025-01-24`).
///
/// Mirrors `tool/bash_20250124.ts`.
#[must_use]
pub fn bash_20250124() -> Tool {
    build("anthropic.bash_20250124", "bash", Empty {})
}

/// `code_execution_20250522` — initial code-execution tool.
///
/// Beta `code-execution-2025-05-22`. Mirrors `tool/code-execution_20250522.ts`.
#[must_use]
pub fn code_execution_20250522() -> Tool {
    build(
        "anthropic.code_execution_20250522",
        "code_execution",
        Empty {},
    )
}

/// `code_execution_20250825` — Python + Bash code execution.
///
/// Beta `code-execution-2025-08-25`. Mirrors `tool/code-execution_20250825.ts`.
#[must_use]
pub fn code_execution_20250825() -> Tool {
    build(
        "anthropic.code_execution_20250825",
        "code_execution",
        Empty {},
    )
}

/// `code_execution_20260120` — recommended code-execution tool.
///
/// No beta header required. Mirrors `tool/code-execution_20260120.ts`.
#[must_use]
pub fn code_execution_20260120() -> Tool {
    build(
        "anthropic.code_execution_20260120",
        "code_execution",
        Empty {},
    )
}

/// `memory_20250818` — persistent memory tool.
///
/// Beta `context-management-2025-06-27`. Mirrors `tool/memory_20250818.ts`.
#[must_use]
pub fn memory_20250818() -> Tool {
    build("anthropic.memory_20250818", "memory", Empty {})
}

/// `text_editor_20241022` — initial text editor (Sonnet 3.5).
///
/// Mirrors `tool/text-editor_20241022.ts`. Beta `computer-use-2024-10-22`.
#[must_use]
pub fn text_editor_20241022() -> Tool {
    build(
        "anthropic.text_editor_20241022",
        "str_replace_editor",
        Empty {},
    )
}

/// `text_editor_20250124` — Sonnet 3.7 text editor.
///
/// Mirrors `tool/text-editor_20250124.ts`. Beta `computer-use-2025-01-24`.
#[must_use]
pub fn text_editor_20250124() -> Tool {
    build(
        "anthropic.text_editor_20250124",
        "str_replace_editor",
        Empty {},
    )
}

/// `text_editor_20250429` — deprecated text editor (use `text_editor_20250728`).
///
/// Mirrors `tool/text-editor_20250429.ts`.
#[must_use]
#[deprecated(note = "use text_editor_20250728")]
pub fn text_editor_20250429() -> Tool {
    build(
        "anthropic.text_editor_20250429",
        "str_replace_based_edit_tool",
        Empty {},
    )
}

/// `tool_search_regex_20251119` — regex tool-search.
///
/// Mirrors `tool/tool-search-regex_20251119.ts`. Supported on Opus 4.5 / Sonnet 4.5.
#[must_use]
pub fn tool_search_regex_20251119() -> Tool {
    build(
        "anthropic.tool_search_regex_20251119",
        "tool_search_tool_regex",
        Empty {},
    )
}

/// `tool_search_bm25_20251119` — BM25 (natural-language) tool-search.
///
/// Mirrors `tool/tool-search-bm25_20251119.ts`. Supported on Opus 4.5 / Sonnet 4.5.
#[must_use]
pub fn tool_search_bm25_20251119() -> Tool {
    build(
        "anthropic.tool_search_bm25_20251119",
        "tool_search_tool_bm25",
        Empty {},
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract(tool: &Tool) -> (&str, &str, bool) {
        match tool {
            Tool::Provider(p) => (p.id.as_str(), p.name.as_str(), p.args.is_none()),
            Tool::Function(_) => panic!("expected provider tool"),
        }
    }

    #[test]
    fn zero_arg_tools_emit_no_args_field() {
        let cases: &[(Tool, &str, &str)] = &[
            (bash_20241022(), "anthropic.bash_20241022", "bash"),
            (
                code_execution_20260120(),
                "anthropic.code_execution_20260120",
                "code_execution",
            ),
            (memory_20250818(), "anthropic.memory_20250818", "memory"),
            (
                text_editor_20250124(),
                "anthropic.text_editor_20250124",
                "str_replace_editor",
            ),
            (
                tool_search_regex_20251119(),
                "anthropic.tool_search_regex_20251119",
                "tool_search_tool_regex",
            ),
        ];
        for (tool, want_id, want_name) in cases {
            let (id, name, args_empty) = extract(tool);
            assert_eq!(id, *want_id);
            assert_eq!(name, *want_name);
            assert!(args_empty, "expected args=None for {want_id}");
        }
    }
}
