//! `text_editor_20250728` — text editor with optional `max_characters`.
//!
//! Mirrors `tool/text-editor_20250728.ts`. No beta header required.
// Rust guideline compliant 2026-02-21

use llmsdk_provider::language_model::Tool;
use serde::Serialize;

use super::common::build;

/// Construction-time args for [`text_editor_20250728`].
#[derive(Debug, Clone, Copy, Serialize, Default)]
pub struct TextEditor20250728Args {
    /// Optional view-command truncation limit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_characters: Option<u32>,
}

/// `text_editor_20250728` — view / create / `str_replace` / insert.
///
/// Note: this version does not support `undo_edit`. Supported models:
/// Sonnet 4, Opus 4, Opus 4.1.
#[must_use]
pub fn text_editor_20250728(args: TextEditor20250728Args) -> Tool {
    build(
        "anthropic.text_editor_20250728",
        "str_replace_based_edit_tool",
        args,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_args_omits_field() {
        let t = text_editor_20250728(TextEditor20250728Args::default());
        match t {
            Tool::Provider(p) => assert!(p.args.is_none()),
            Tool::Function(_) => panic!("expected provider tool"),
        }
    }

    #[test]
    fn max_characters_snake_case() {
        let t = text_editor_20250728(TextEditor20250728Args {
            max_characters: Some(50_000),
        });
        match t {
            Tool::Provider(p) => {
                let args = p.args.unwrap();
                assert_eq!(
                    args.get("max_characters").unwrap(),
                    &serde_json::json!(50_000)
                );
            }
            Tool::Function(_) => panic!("expected provider tool"),
        }
    }
}
