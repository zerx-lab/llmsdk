//! `Anthropic` provider 全功能可用性 smoke 测试。
//!
//! 覆盖 `llmsdk-anthropic` 当前实现的所有公开能力：
//! - Messages：基础生成、流式、多轮 function tool use、JSON `response_format`、
//!   多模态 vision (image URL)、extended thinking
//! - 服务端 provider tool：`web_search_20260209`
//! - Files API：上传文本文件 + 打印 reference / metadata
//! - Skills API：上传单文件 skill 包 + 打印 id / version
//!
//! # Run
//!
//! ```bash
//! # 推荐：复制并填写 .env（.gitignore 已忽略），无需手动 export
//! cp .env.example .env
//! cargo run -p llmsdk-anthropic --example anthropic_smoke
//!
//! # 或者直接 inline
//! ANTHROPIC_API_KEY=sk-ant-... cargo run -p llmsdk-anthropic --example anthropic_smoke
//!
//! # 只跑一个 demo
//! cargo run -p llmsdk-anthropic --example anthropic_smoke -- chat
//! ```
//!
//! `.env` 加载顺序（先到先得，实环境变量永远优先）：
//! 1. `$CWD/.env`                                — 仓库根
//! 2. `crates/llmsdk-anthropic/.env`             — crate 内覆盖
//! 3. workspace 根（通过 `CARGO_MANIFEST_DIR/../..` 编译期解析）
//! 4. `crates/llmsdk-anthropic/examples/.env`
//!
//! # 可选环境变量
//!
//! | 变量                          | 默认                            | 说明                            |
//! |-------------------------------|---------------------------------|---------------------------------|
//! | `ANTHROPIC_API_KEY`           | (二选一)                        | `x-api-key`                     |
//! | `ANTHROPIC_AUTH_TOKEN`        | (二选一)                        | `Authorization: Bearer`         |
//! | `ANTHROPIC_BASE_URL`          | `https://api.anthropic.com/v1`  | 自建网关 / 代理                 |
//! | `ANTHROPIC_VERSION`           | `2023-06-01`                    | `anthropic-version` header      |
//! | `ANTHROPIC_CHAT_MODEL`        | `claude-3-5-sonnet-latest`      | chat/stream/tools/json          |
//! | `ANTHROPIC_VISION_MODEL`      | `claude-3-5-sonnet-latest`      | vision                          |
//! | `ANTHROPIC_THINKING_MODEL`    | `claude-3-7-sonnet-latest`      | thinking                        |
//! | `ANTHROPIC_WEB_SEARCH_MODEL`  | `claude-3-5-sonnet-latest`      | web-search                      |
//! | `ANTHROPIC_VISION_IMAGE_URL`  | wikipedia 一张 320px JPEG       | vision 输入图片                 |
//!
//! # CLI 参数（位置 1）
//!
//! `all` (默认) | `chat` | `stream` | `tools` | `json` | `vision` |
//! `thinking` | `web-search` | `files` | `skills`
//!
//! 未识别的名字会被报错；任何单个 demo 失败不会终止其它 demo。

use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::process::ExitCode;
use std::sync::OnceLock;

use futures::StreamExt;
use llmsdk_anthropic::Anthropic;
use llmsdk_anthropic::tools::{WebSearchArgs, web_search_20260209};
use llmsdk_provider::language_model::{
    AssistantPart, CallOptions, Content, FilePart, FunctionTool, Message, ResponseFormat,
    StreamPart, TextPart, Tool, ToolChoice, ToolMessagePart, ToolResultOutput, ToolResultPart,
    UserPart,
};
use llmsdk_provider::shared::{FileData, ProviderOptions};
use llmsdk_provider::{
    FilesModel, LanguageModel, SkillFile, SkillsModel, UploadFileData, UploadFileOptions,
    UploadSkillOptions,
};
use serde_json::{Map as JsonMap, Value as JsonValue, json};

type DynErr = Box<dyn Error + Send + Sync + 'static>;

const DEFAULT_VISION_URL: &str =
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/Tomato_je.jpg/320px-Tomato_je.jpg";

#[tokio::main]
async fn main() -> ExitCode {
    report_dotenv_sources();
    let demo = env::args().nth(1).unwrap_or_else(|| "all".to_owned());
    match run(&demo).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("\n✗ smoke aborted: {err}");
            let mut src = err.source();
            while let Some(s) = src {
                eprintln!("  caused by: {s}");
                src = s.source();
            }
            ExitCode::FAILURE
        }
    }
}

async fn run(demo: &str) -> Result<(), DynErr> {
    let provider = build_provider()?;

    let demos: Vec<&str> = if demo == "all" {
        vec![
            "chat",
            "stream",
            "tools",
            "json",
            "vision",
            "thinking",
            "web-search",
            "files",
            "skills",
        ]
    } else {
        vec![demo]
    };

    let mut fail = 0u32;
    for name in &demos {
        let result = match *name {
            "chat" => demo_chat(&provider).await,
            "stream" => demo_stream(&provider).await,
            "tools" => demo_tools(&provider).await,
            "json" => demo_json(&provider).await,
            "vision" => demo_vision(&provider).await,
            "thinking" => demo_thinking(&provider).await,
            "web-search" => demo_web_search(&provider).await,
            "files" => demo_files(&provider).await,
            "skills" => demo_skills(&provider).await,
            other => Err::<(), DynErr>(format!("unknown demo: {other}").into()),
        };
        if let Err(e) = result {
            fail += 1;
            eprintln!("✗ [{name}] failed: {e}");
            let mut src = e.source();
            while let Some(s) = src {
                eprintln!("    caused by: {s}");
                src = s.source();
            }
        }
    }

    println!();
    println!(
        "═════ smoke 结束：{} 通过 / {} 失败 ═════",
        demos.len() - fail as usize,
        fail
    );
    if fail > 0 {
        return Err(format!("{fail} demo(s) failed").into());
    }
    Ok(())
}

// ─── .env 加载（零依赖、零 unsafe） ──────────────────────────────
//
// Rust 2024 edition 把 `std::env::set_var` 标记为 unsafe（项目禁止
// unsafe），所以这里不去修改进程 env，而是把 .env 解析到一个静态
// `HashMap`，所有 env 读取统一走 [`opt_env`] / [`env_or`]：先看真实
// 环境变量，再 fallback 到 .env。
//
// 加载顺序（按出现顺序合并，先到先得）：
// 1. `$CWD/.env`                                — 用户在 workspace 根运行
// 2. `$CARGO_MANIFEST_DIR/.env`                 — crate-local 覆盖
// 3. `$CARGO_MANIFEST_DIR/../../.env`           — workspace 根（编译期路径）
// 4. `$CARGO_MANIFEST_DIR/examples/.env`        — example 同目录

static DOTENV: OnceLock<DotenvBundle> = OnceLock::new();

#[derive(Debug, Default)]
struct DotenvBundle {
    values: HashMap<String, String>,
    loaded: Vec<String>,
}

fn dotenv_bundle() -> &'static DotenvBundle {
    DOTENV.get_or_init(load_dotenv)
}

fn load_dotenv() -> DotenvBundle {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let candidates = [
        ".env".to_owned(),
        format!("{manifest}/.env"),
        format!("{manifest}/../../.env"),
        format!("{manifest}/examples/.env"),
    ];

    let mut values: HashMap<String, String> = HashMap::new();
    let mut loaded: Vec<String> = Vec::new();
    let mut seen_canonical: Vec<std::path::PathBuf> = Vec::new();

    for path in &candidates {
        let p = Path::new(path);
        let Ok(canon) = p.canonicalize() else {
            continue;
        };
        if seen_canonical.contains(&canon) {
            continue;
        }
        let Ok(text) = fs::read_to_string(&canon) else {
            continue;
        };
        seen_canonical.push(canon.clone());
        loaded.push(canon.display().to_string());
        parse_dotenv_into(&text, &mut values);
    }

    DotenvBundle { values, loaded }
}

fn parse_dotenv_into(text: &str, into: &mut HashMap<String, String>) {
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let key = k.trim();
        if key.is_empty() {
            continue;
        }
        let mut value = v.trim();
        if (value.starts_with('"') && value.ends_with('"') && value.len() >= 2)
            || (value.starts_with('\'') && value.ends_with('\'') && value.len() >= 2)
        {
            value = &value[1..value.len() - 1];
        }
        into.entry(key.to_owned())
            .or_insert_with(|| value.to_owned());
    }
}

fn report_dotenv_sources() {
    let bundle = dotenv_bundle();
    if bundle.loaded.is_empty() {
        eprintln!("(没有找到 .env 文件，仅使用进程环境变量)");
    } else {
        eprintln!("加载到 {} 个 .env 文件:", bundle.loaded.len());
        for p in &bundle.loaded {
            eprintln!("  • {p}");
        }
    }
}

fn opt_env(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .or_else(|| dotenv_bundle().values.get(name).cloned())
}

fn env_or(name: &str, default: &str) -> String {
    opt_env(name).unwrap_or_else(|| default.to_owned())
}

// ─── provider 构建 ───────────────────────────────────────────────

fn build_provider() -> Result<Anthropic, DynErr> {
    let mut builder = Anthropic::builder();
    // api_key 与 auth_token 互斥；若两个都给，构造期会自动报错。
    if let Some(key) = opt_env("ANTHROPIC_API_KEY") {
        builder = builder.api_key(key);
    } else if let Some(token) = opt_env("ANTHROPIC_AUTH_TOKEN") {
        builder = builder.auth_token(token);
    }
    if let Some(base) = opt_env("ANTHROPIC_BASE_URL") {
        builder = builder.base_url(base);
    }
    if let Some(version) = opt_env("ANTHROPIC_VERSION") {
        builder = builder.version(version);
    }
    Ok(builder.build()?)
}

fn header(title: &str) {
    println!();
    println!("──────────────────────────────────────────────");
    println!("▶ {title}");
    println!("──────────────────────────────────────────────");
}

fn system(text: &str) -> Message {
    Message::System {
        content: text.to_owned(),
        provider_options: None,
    }
}

fn user_text(text: &str) -> Message {
    Message::User {
        content: vec![UserPart::Text(TextPart {
            text: text.to_owned(),
            provider_options: None,
        })],
        provider_options: None,
    }
}

fn print_text_content(content: &[Content]) {
    for part in content {
        if let Content::Text(t) = part {
            println!("{}", t.text);
        }
    }
}

// ─── 1. Chat 基础生成 ────────────────────────────────────────────

async fn demo_chat(provider: &Anthropic) -> Result<(), DynErr> {
    let model_id = env_or("ANTHROPIC_CHAT_MODEL", "claude-3-5-sonnet-latest");
    header(&format!("messages · do_generate · {model_id}"));

    let model = provider.messages(model_id);
    let result = model
        .do_generate(CallOptions {
            prompt: vec![
                system("你是一个简洁的助手，回答控制在两句话以内。"),
                user_text("用一句话告诉我什么是 Rust 的所有权（ownership）。"),
            ],
            max_output_tokens: Some(200),
            temperature: Some(0.2),
            ..Default::default()
        })
        .await?;

    print_text_content(&result.content);
    println!(
        "[finish={:?} usage={:?}]",
        result.finish_reason.unified, result.usage
    );
    for w in &result.warnings {
        println!("[warning] {w:?}");
    }
    Ok(())
}

// ─── 2. Chat 流式 ────────────────────────────────────────────────

async fn demo_stream(provider: &Anthropic) -> Result<(), DynErr> {
    use std::io::Write;

    let model_id = env_or("ANTHROPIC_CHAT_MODEL", "claude-3-5-sonnet-latest");
    header(&format!("messages · do_stream · {model_id}"));

    let model = provider.messages(model_id);
    let mut stream = model
        .do_stream(CallOptions {
            prompt: vec![user_text("用 3 个要点说明 actor 模型。简短。")],
            max_output_tokens: Some(300),
            ..Default::default()
        })
        .await?
        .stream;

    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    let mut final_usage = None;
    let mut final_reason = None;
    while let Some(item) = stream.next().await {
        match item? {
            StreamPart::TextDelta { delta, .. } => {
                handle.write_all(delta.as_bytes())?;
                handle.flush()?;
            }
            StreamPart::Finish {
                usage,
                finish_reason,
                ..
            } => {
                final_usage = Some(usage);
                final_reason = Some(finish_reason);
            }
            StreamPart::Error { error } => {
                eprintln!("\n[stream error] {error}");
            }
            _ => {}
        }
    }
    println!();
    println!("[finish={final_reason:?} usage={final_usage:?}]");
    Ok(())
}

// ─── 3. 多轮 function tool use ───────────────────────────────────

async fn demo_tools(provider: &Anthropic) -> Result<(), DynErr> {
    let model_id = env_or("ANTHROPIC_CHAT_MODEL", "claude-3-5-sonnet-latest");
    header(&format!(
        "messages · function tool use (multi-turn) · {model_id}"
    ));

    let weather_schema = serde_json::from_value(json!({
        "type": "object",
        "properties": {
            "city":    { "type": "string", "description": "city name" },
            "unit":    { "type": "string", "enum": ["celsius", "fahrenheit"] }
        },
        "required": ["city"],
        "additionalProperties": false
    }))?;

    let tool = Tool::Function(FunctionTool {
        name: "get_weather".to_owned(),
        description: Some("Look up the current weather for a city.".to_owned()),
        input_schema: weather_schema,
        input_examples: None,
        strict: None,
        provider_options: None,
    });

    let model = provider.messages(model_id);

    // ── round 1：让模型决定调用工具
    let round1 = model
        .do_generate(CallOptions {
            prompt: vec![
                system("当用户问天气时请调用 get_weather 工具。"),
                user_text("北京现在多少度？用摄氏度回答。"),
            ],
            tools: Some(vec![tool.clone()]),
            tool_choice: Some(ToolChoice::Auto),
            max_output_tokens: Some(400),
            ..Default::default()
        })
        .await?;

    let tool_call = round1.content.iter().find_map(|c| match c {
        Content::ToolCall(tc) => Some(tc.clone()),
        _ => None,
    });

    let Some(call) = tool_call else {
        println!("(模型没有调用工具，原始输出如下)");
        print_text_content(&round1.content);
        return Ok(());
    };

    println!(
        "→ 模型请求 tool {} (id={}) input={}",
        call.tool_name, call.tool_call_id, call.input
    );

    // ── 本地"执行"工具：返回固定 JSON
    let fake_result = json!({
        "city": call.input.get("city").cloned().unwrap_or(json!("北京")),
        "tempC": 7,
        "condition": "晴",
        "humidity": 0.32
    });

    // ── round 2：把 tool_result 喂回模型
    let round2 = model
        .do_generate(CallOptions {
            prompt: vec![
                system("根据 get_weather 的结果用中文回答用户。"),
                user_text("北京现在多少度？用摄氏度回答。"),
                Message::Assistant {
                    content: vec![AssistantPart::ToolCall(call.clone())],
                    provider_options: None,
                },
                Message::Tool {
                    content: vec![ToolMessagePart::ToolResult(ToolResultPart {
                        tool_call_id: call.tool_call_id.clone(),
                        tool_name: call.tool_name.clone(),
                        output: ToolResultOutput::Json {
                            value: fake_result,
                            provider_options: None,
                        },
                        provider_options: None,
                    })],
                    provider_options: None,
                },
            ],
            tools: Some(vec![tool]),
            tool_choice: Some(ToolChoice::Auto),
            max_output_tokens: Some(400),
            ..Default::default()
        })
        .await?;

    println!("→ 最终回复:");
    print_text_content(&round2.content);
    println!(
        "[finish={:?} usage={:?}]",
        round2.finish_reason.unified, round2.usage
    );
    Ok(())
}

// ─── 4. JSON response_format ─────────────────────────────────────

async fn demo_json(provider: &Anthropic) -> Result<(), DynErr> {
    let model_id = env_or("ANTHROPIC_CHAT_MODEL", "claude-3-5-sonnet-latest");
    header(&format!("messages · response_format=json · {model_id}"));

    let schema = serde_json::from_value(json!({
        "type": "object",
        "properties": {
            "language": { "type": "string" },
            "year":     { "type": "integer" },
            "creator":  { "type": "string" },
            "paradigms":{ "type": "array", "items": { "type": "string" } }
        },
        "required": ["language", "year", "creator", "paradigms"],
        "additionalProperties": false
    }))?;

    // 启用 structuredOutputMode = outputFormat 走 wire 的 output_config.format。
    let mut anthropic_opts = JsonMap::new();
    anthropic_opts.insert(
        "structuredOutputMode".to_owned(),
        JsonValue::String("outputFormat".to_owned()),
    );
    let mut po = ProviderOptions::new();
    po.insert("anthropic".to_owned(), anthropic_opts);

    let model = provider.messages(model_id);
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("用 JSON 描述 Rust 编程语言的基本信息。")],
            response_format: Some(ResponseFormat::Json {
                schema: Some(schema),
                name: Some("LanguageInfo".to_owned()),
                description: Some("基本元信息".to_owned()),
            }),
            max_output_tokens: Some(400),
            provider_options: Some(po),
            ..Default::default()
        })
        .await?;

    print_text_content(&result.content);
    println!(
        "[finish={:?} usage={:?}]",
        result.finish_reason.unified, result.usage
    );
    for w in &result.warnings {
        println!("[warning] {w:?}");
    }
    Ok(())
}

// ─── 5. Vision：图片 URL 输入 ────────────────────────────────────

async fn demo_vision(provider: &Anthropic) -> Result<(), DynErr> {
    let model_id = env_or("ANTHROPIC_VISION_MODEL", "claude-3-5-sonnet-latest");
    let image_url = env_or("ANTHROPIC_VISION_IMAGE_URL", DEFAULT_VISION_URL);
    header(&format!("messages · vision (image URL) · {model_id}"));
    println!("image: {image_url}");

    let model = provider.messages(model_id);
    let result = model
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![
                    UserPart::Text(TextPart {
                        text: "这是什么？用一句话描述。".to_owned(),
                        provider_options: None,
                    }),
                    UserPart::File(FilePart {
                        filename: None,
                        data: FileData::Url { url: image_url },
                        media_type: "image/jpeg".to_owned(),
                        provider_options: None,
                    }),
                ],
                provider_options: None,
            }],
            max_output_tokens: Some(200),
            ..Default::default()
        })
        .await?;

    print_text_content(&result.content);
    println!(
        "[finish={:?} usage={:?}]",
        result.finish_reason.unified, result.usage
    );
    Ok(())
}

// ─── 6. Extended thinking ────────────────────────────────────────

async fn demo_thinking(provider: &Anthropic) -> Result<(), DynErr> {
    let model_id = env_or("ANTHROPIC_THINKING_MODEL", "claude-3-7-sonnet-latest");
    header(&format!("messages · extended thinking · {model_id}"));

    // 通过 provider_options.anthropic.thinking 开启 extended thinking。
    let mut anthropic_opts = JsonMap::new();
    anthropic_opts.insert(
        "thinking".to_owned(),
        json!({
            "type": "enabled",
            "budgetTokens": 2048
        }),
    );
    let mut po = ProviderOptions::new();
    po.insert("anthropic".to_owned(), anthropic_opts);

    let model = provider.messages(model_id);
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text(
                "如果一辆车 2 小时跑 130 公里，平均时速是多少？给出最终答案。",
            )],
            // budgetTokens=2048 时 SDK 会自动确保 max_tokens > budget，这里给足。
            max_output_tokens: Some(4096),
            provider_options: Some(po),
            ..Default::default()
        })
        .await?;

    let mut had_reasoning = false;
    for part in &result.content {
        match part {
            Content::Reasoning(r) => {
                had_reasoning = true;
                let snippet: String = r.text.chars().take(200).collect();
                println!("(thinking, 截断 200 字) {snippet}…");
            }
            Content::Text(t) => println!("→ answer: {}", t.text),
            _ => {}
        }
    }
    if !had_reasoning {
        println!("(模型未返回 thinking 块——该模型可能不支持 extended thinking)");
    }
    println!(
        "[finish={:?} usage={:?}]",
        result.finish_reason.unified, result.usage
    );
    for w in &result.warnings {
        println!("[warning] {w:?}");
    }
    Ok(())
}

// ─── 7. Server-side web_search 工具 ──────────────────────────────

async fn demo_web_search(provider: &Anthropic) -> Result<(), DynErr> {
    let model_id = env_or("ANTHROPIC_WEB_SEARCH_MODEL", "claude-3-5-sonnet-latest");
    header(&format!(
        "messages · provider tool anthropic.web_search_20260209 · {model_id}"
    ));

    let web_search = web_search_20260209(WebSearchArgs {
        max_uses: Some(3),
        ..Default::default()
    });

    let model = provider.messages(model_id);
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text(
                "请告诉我 Rust 编程语言的最新稳定版版本号（如有需要可联网搜索）。一句话。",
            )],
            tools: Some(vec![web_search]),
            tool_choice: Some(ToolChoice::Auto),
            max_output_tokens: Some(600),
            ..Default::default()
        })
        .await?;

    let mut printed_text = false;
    for part in &result.content {
        match part {
            Content::Text(t) => {
                println!("{}", t.text);
                printed_text = true;
            }
            Content::Source(src) => println!("• source: {src:?}"),
            Content::ToolCall(tc) => println!("• tool_call: {} input={}", tc.tool_name, tc.input),
            Content::ToolResult(tr) => println!("• tool_result: {} ", tr.tool_name),
            _ => {}
        }
    }
    if !printed_text {
        println!("(模型没有给出最终文本——账户可能未开通 web_search beta)");
    }
    println!(
        "[finish={:?} usage={:?}]",
        result.finish_reason.unified, result.usage
    );
    for w in &result.warnings {
        println!("[warning] {w:?}");
    }
    Ok(())
}

// ─── 8. Files API：上传 ──────────────────────────────────────────

async fn demo_files(provider: &Anthropic) -> Result<(), DynErr> {
    header("files · upload_file (text/plain)");

    let files = provider.files();
    let result = files
        .upload_file(UploadFileOptions {
            data: UploadFileData::Text {
                text: "Hello from llmsdk anthropic smoke test.\nLine 2.\n".to_owned(),
            },
            media_type: "text/plain".to_owned(),
            filename: Some("llmsdk_smoke_hello.txt".to_owned()),
            provider_options: None,
        })
        .await?;

    println!(
        "→ provider_reference: {:?}",
        result.provider_reference.get("anthropic")
    );
    println!(
        "  filename={:?} media_type={:?}",
        result.filename, result.media_type
    );
    if let Some(meta) = &result.provider_metadata
        && let Some(anthropic_meta) = meta.get("anthropic")
    {
        println!("  metadata.anthropic = {anthropic_meta:?}");
    }
    for w in &result.warnings {
        println!("[warning] {w:?}");
    }
    Ok(())
}

// ─── 9. Skills API：上传 ─────────────────────────────────────────

async fn demo_skills(provider: &Anthropic) -> Result<(), DynErr> {
    header("skills · upload_skill (single-file bundle)");

    // 最小 SKILL.md：name / description 头 + 一段 markdown。
    let skill_md = "---\nname: llmsdk-smoke-greeter\ndescription: Tiny smoke-test skill that says hi.\n---\n\n# Greeter\n\nAlways respond with a friendly greeting in the user's language.\n";

    let skills = provider.skills();
    let result = skills
        .upload_skill(UploadSkillOptions {
            files: vec![SkillFile {
                path: "SKILL.md".to_owned(),
                data: UploadFileData::Text {
                    text: skill_md.to_owned(),
                },
            }],
            display_title: Some("llmsdk smoke greeter".to_owned()),
            provider_options: None,
        })
        .await?;

    println!(
        "→ provider_reference: {:?}",
        result.provider_reference.get("anthropic")
    );
    println!(
        "  display_title={:?} name={:?} version={:?}",
        result.display_title, result.name, result.latest_version
    );
    if let Some(desc) = &result.description {
        println!("  description={desc}");
    }
    if let Some(meta) = &result.provider_metadata
        && let Some(anthropic_meta) = meta.get("anthropic")
    {
        println!("  metadata.anthropic = {anthropic_meta:?}");
    }
    for w in &result.warnings {
        println!("[warning] {w:?}");
    }
    Ok(())
}
