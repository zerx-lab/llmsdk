//! `computer_*` — desktop control tools.
//!
//! Mirrors `tool/computer_2024*.ts` / `_2025*.ts`. Beta headers per version:
//! - `20241022` → `computer-use-2024-10-22`
//! - `20250124` → `computer-use-2025-01-24`
//! - `20251124` → `computer-use-2025-11-24` (adds `enable_zoom`)
// Rust guideline compliant 2026-02-21

use llmsdk_provider::language_model::Tool;
use serde::Serialize;

use super::common::build;

/// Construction-time args for [`computer_20241022`] / [`computer_20250124`].
#[derive(Debug, Clone, Copy, Serialize, Default)]
pub struct ComputerArgs {
    /// Display width in pixels.
    pub display_width_px: u32,
    /// Display height in pixels.
    pub display_height_px: u32,
    /// Optional X11 display number.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_number: Option<u32>,
}

/// Construction-time args for [`computer_20251124`] (adds `enable_zoom`).
#[derive(Debug, Clone, Copy, Serialize, Default)]
pub struct ComputerArgsWithZoom {
    /// Display width in pixels.
    pub display_width_px: u32,
    /// Display height in pixels.
    pub display_height_px: u32,
    /// Optional X11 display number.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_number: Option<u32>,
    /// Enable the `zoom` action.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_zoom: Option<bool>,
}

/// `computer_20241022` — initial computer-use tool.
#[must_use]
pub fn computer_20241022(args: ComputerArgs) -> Tool {
    build("anthropic.computer_20241022", "computer", args)
}

/// `computer_20250124` — Sonnet 3.7 era computer tool.
#[must_use]
pub fn computer_20250124(args: ComputerArgs) -> Tool {
    build("anthropic.computer_20250124", "computer", args)
}

/// `computer_20251124` — Opus 4.5 computer tool with optional `enable_zoom`.
#[must_use]
pub fn computer_20251124(args: ComputerArgsWithZoom) -> Tool {
    build("anthropic.computer_20251124", "computer", args)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args_of(tool: &Tool) -> serde_json::Value {
        match tool {
            Tool::Provider(p) => serde_json::Value::Object(p.args.clone().unwrap_or_default()),
            Tool::Function(_) => panic!("expected provider tool"),
        }
    }

    #[test]
    fn computer_args_snake_case_on_wire() {
        let t = computer_20250124(ComputerArgs {
            display_width_px: 1024,
            display_height_px: 768,
            display_number: Some(1),
        });
        let args = args_of(&t);
        assert_eq!(args["display_width_px"], 1024);
        assert_eq!(args["display_height_px"], 768);
        assert_eq!(args["display_number"], 1);
    }

    #[test]
    fn computer_v3_supports_zoom() {
        let t = computer_20251124(ComputerArgsWithZoom {
            display_width_px: 1920,
            display_height_px: 1080,
            display_number: None,
            enable_zoom: Some(true),
        });
        let args = args_of(&t);
        assert_eq!(args["enable_zoom"], true);
        assert!(args.get("display_number").is_none());
    }
}
