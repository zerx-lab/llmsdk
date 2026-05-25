# 0005 — M12 Anthropic Full API Parity

> Status: in progress
> Upstream reference: `vercel/ai` @ `packages/anthropic/src/**` + `packages/provider/src/{files,skills}/v4/**`
> Prereqs: `0001-trait-design.md`、`0002-middleware-design.md`、`0003-m10-design.md`、`0004-m11-responses-design.md`

## Goal

把 ai-sdk Anthropic provider 剩余三条 API 路径全部接入 llmsdk，达成 100% feature parity：

1. **Files API**（`POST /v1/files`）—— ai-sdk `FilesV4` 接口的 Rust 等价物
2. **Skills API**（`POST /v1/skills` + `GET /v1/skills/{id}/versions/{v}`）—— ai-sdk `SkillsV4` 接口的 Rust 等价物
3. **`anthropic.tools.*` 顶层 typed tool factory**（20 个 typed helper）—— 与上游 `anthropic-tools.ts` 完全对等的开发者体验

外加把 Messages API 端点剩余的"响应元数据深度解析"项（iterations / container / context_management.applied_edits）补齐。

按 CLAUDE.md 强制规则"启动新阶段前必须列出全部范围"——本文档即范围 ground truth；不允许中途静默推迟。开始前已与用户对齐 3 处范围决策（见末尾 "Open Questions Resolved"）。

## 范围（全部纳入）

### A. 新增 trait（`llmsdk-provider`）

#### A.1 `FilesModel` trait

新模块 `crates/llmsdk-provider/src/files_model/`：

```rust
#[async_trait::async_trait]
pub trait FilesModel: Send + Sync + std::fmt::Debug {
    fn provider(&self) -> &str;
    fn specification_version(&self) -> &'static str { "v4" }

    async fn upload_file(&self, options: UploadFileOptions) -> Result<UploadFileResult, ProviderError>;
}

pub struct UploadFileOptions {
    pub data: FileData,                          // Bytes | Text
    pub media_type: String,                      // IANA media type
    pub filename: Option<String>,
    pub provider_options: Option<ProviderOptions>,
}

pub enum FileData {
    Bytes(bytes::Bytes),
    Text(String),
}

pub struct UploadFileResult {
    pub provider_reference: ProviderReference,   // { "anthropic": "file-..." }
    pub media_type: Option<String>,
    pub filename: Option<String>,
    pub provider_metadata: Option<ProviderMetadata>,
    pub warnings: Vec<CallWarning>,
}
```

`ProviderReference` 类型新建 —— `HashMap<String, String>`（与上游 `SharedV4ProviderReference` 一致）。

#### A.2 `SkillsModel` trait

新模块 `crates/llmsdk-provider/src/skills_model/`：

```rust
#[async_trait::async_trait]
pub trait SkillsModel: Send + Sync + std::fmt::Debug {
    fn provider(&self) -> &str;
    fn specification_version(&self) -> &'static str { "v4" }

    async fn upload_skill(&self, options: UploadSkillOptions) -> Result<UploadSkillResult, ProviderError>;
}

pub struct UploadSkillOptions {
    pub files: Vec<SkillFile>,                   // 至少一个文件
    pub display_title: Option<String>,
    pub provider_options: Option<ProviderOptions>,
}

pub struct SkillFile {
    pub path: String,                            // skill 根目录下的相对路径
    pub data: FileData,                          // 复用 FilesModel 的 FileData
}

pub struct UploadSkillResult {
    pub provider_reference: ProviderReference,
    pub display_title: Option<String>,
    pub name: Option<String>,
    pub description: Option<String>,
    pub latest_version: Option<String>,
    pub provider_metadata: Option<ProviderMetadata>,
    pub warnings: Vec<CallWarning>,
}
```

#### A.3 不动 `Provider` trait

按用户已决定：trait 集成方式 = **新增独立 trait，不动 Provider trait**。Anthropic struct 直接 inherent `.files()` / `.skills()` 返回拥有 trait 的 model handle；下游用户拿到的就是 trait object 或具体类型。

`llmsdk-provider/src/lib.rs` 增加：
```rust
pub mod files_model;
pub mod skills_model;
pub use files_model::{FilesModel, UploadFileOptions, UploadFileResult, FileData};
pub use skills_model::{SkillsModel, UploadSkillOptions, UploadSkillResult, SkillFile};
```

### B. AnthropicFiles 实现（`llmsdk-anthropic`）

新模块 `crates/llmsdk-anthropic/src/files/`：

| 文件 | 职责 |
|---|---|
| `mod.rs` | re-export |
| `model.rs` | `AnthropicFiles` struct + `FilesModel` impl |
| `wire.rs` | `WireUploadResponse` (id / type / filename / mime_type / size_bytes / created_at / downloadable) |

**端点**：`POST {base_url}/files`
**Header**：`anthropic-beta: files-api-2025-04-14` + `x-api-key` + `anthropic-version`
**Body**：multipart/form-data (复用 `llmsdk-provider-utils::multipart`)
- `file`：FileData → 单 part；media_type 进 Content-Type；filename 进 disposition

**响应映射**：
```
WireUploadResponse {
  id, filename, mime_type, size_bytes, created_at, downloadable?
} →
UploadFileResult {
  provider_reference: { "anthropic" → id },
  media_type: response.mime_type or original,
  filename: response.filename or original,
  provider_metadata: {
    "anthropic": {
      "filename": ..., "mimeType": ..., "sizeBytes": ..., "createdAt": ...,
      "downloadable": ... (仅当存在)
    }
  },
  warnings: []
}
```

### C. AnthropicSkills 实现（`llmsdk-anthropic`）

新模块 `crates/llmsdk-anthropic/src/skills/`：

| 文件 | 职责 |
|---|---|
| `mod.rs` | re-export |
| `model.rs` | `AnthropicSkills` struct + `SkillsModel` impl |
| `wire.rs` | `WireSkillResponse` + `WireSkillVersionResponse` |

**端点**：
1. `POST {base_url}/skills` —— multipart：`display_title` (optional) + 多个 `files[]` part（filename = path）
2. `GET {base_url}/skills/{skill_id}/versions/{version}` —— JSON 响应：`{ type, skill_id, name?, description? }`

**Header**：`anthropic-beta: skills-2025-10-02` + `x-api-key` + `anthropic-version`

**响应映射**：上游行为是 upload 后若 `latest_version != null` 则二次拉版本元信息（name + description 用版本数据优先回填，回退到 upload 响应字段）。Rust 侧完全复刻：

```rust
let response = post_skills(...);
let version_meta = if let Some(v) = &response.latest_version {
    Some(get_version_meta(&response.id, v).await?)
} else { None };

UploadSkillResult {
    provider_reference: { "anthropic" → response.id },
    display_title: response.display_title,
    name: version_meta.as_ref().and_then(|v| v.name.clone()).or(response.name),
    description: version_meta.as_ref().and_then(|v| v.description.clone()).or(response.description),
    latest_version: response.latest_version,
    provider_metadata: {
        "anthropic": {
            "source": response.source,
            "createdAt": response.created_at,
            "updatedAt": response.updated_at,
        }
    },
    warnings: []
}
```

### D. 20 个 typed tool factory（`llmsdk-anthropic`）

新模块 `crates/llmsdk-anthropic/src/tools/`：

每个 factory 是一个 `fn(args: TypedArgs) -> Tool::Provider { id, name, args }`，返回值是 `llmsdk-provider::language_model::tool::Tool::Provider` variant，直接可放进 `CallOptions.tools`。

| factory 名 | id（注入到 Tool::Provider.id） | name 字段 | args struct（typed + validated） |
|---|---|---|---|
| `advisor_20260301` | `anthropic.advisor_20260301` | `"advisor"` | `{ model: String, max_uses?: u32, caching?: bool }` |
| `bash_20241022` | `anthropic.bash_20241022` | `"bash"` | `{}` |
| `bash_20250124` | `anthropic.bash_20250124` | `"bash"` | `{}` |
| `code_execution_20250522` | `anthropic.code_execution_20250522` | `"code_execution"` | `{}` |
| `code_execution_20250825` | `anthropic.code_execution_20250825` | `"code_execution"` | `{}` |
| `code_execution_20260120` | `anthropic.code_execution_20260120` | `"code_execution"` | `{}` |
| `computer_20241022` | `anthropic.computer_20241022` | `"computer"` | `{ display_width_px: u32, display_height_px: u32, display_number?: u32 }` |
| `computer_20250124` | `anthropic.computer_20250124` | `"computer"` | `{ display_width_px, display_height_px, display_number? }` |
| `computer_20251124` | `anthropic.computer_20251124` | `"computer"` | `{ display_width_px, display_height_px, display_number?, enable_zoom?: bool }` |
| `memory_20250818` | `anthropic.memory_20250818` | `"memory"` | `{}` |
| `text_editor_20241022` | `anthropic.text_editor_20241022` | `"str_replace_editor"` | `{}` |
| `text_editor_20250124` | `anthropic.text_editor_20250124` | `"str_replace_editor"` | `{}` |
| `text_editor_20250429` | `anthropic.text_editor_20250429` | `"str_replace_based_edit_tool"` | `{}` |
| `text_editor_20250728` | `anthropic.text_editor_20250728` | `"str_replace_based_edit_tool"` | `{ max_characters?: u32 }` |
| `web_fetch_20250910` | `anthropic.web_fetch_20250910` | `"web_fetch"` | `{ max_uses?, allowed_domains?, blocked_domains?, citations?: CitationsConfig, max_content_tokens? }` |
| `web_fetch_20260209` | `anthropic.web_fetch_20260209` | `"web_fetch"` | 同上 |
| `web_search_20250305` | `anthropic.web_search_20250305` | `"web_search"` | `{ max_uses?, allowed_domains?, blocked_domains?, user_location?: UserLocation }` |
| `web_search_20260209` | `anthropic.web_search_20260209` | `"web_search"` | 同上 |
| `tool_search_regex_20251119` | `anthropic.tool_search_regex_20251119` | `"tool_search_tool_regex"` | `{}` |
| `tool_search_bm25_20251119` | `anthropic.tool_search_bm25_20251119` | `"tool_search_tool_bm25"` | `{}` |

辅助 struct：
- `CitationsConfig { enabled: bool }`
- `UserLocation { type: "approximate", city?, region?, country?, timezone? }`

每个 factory 内部把 args struct `serde_json::to_value` 后塞进 `Tool::Provider { id, name, args }` 的 `args` 字段。args struct 用 `#[derive(Serialize)]` + `#[serde(rename_all = "snake_case")]`（与 Anthropic wire schema 对齐）。

**注意**：现有 `messages/model.rs::resolve_anthropic_server_tool()` 路由表（M10.5 加的）保持不变；本阶段只是新增 typed factory 入口，路由层不动。

### E. `AnthropicBuilder` 配置扩展

`config.rs` 增量：

| 新增 builder 方法 | 字段 | 行为 |
|---|---|---|
| `.auth_token(token)` | `auth_token: Option<String>` | 与 `api_key` 互斥；同时设置时 `build()` 返回 `ProviderError::invalid_argument("Both api_key and auth_token provided; use one")`；设置后用 `Authorization: Bearer {token}` 替代 `x-api-key` |
| `.name(name)` | `provider_name: Option<String>` | 自定义 provider 名（用于 `Provider::provider()` 返回值）；默认 `"anthropic.messages"` / `"anthropic.files"` / `"anthropic.skills"` |
| `.generate_id(fn)` | `generate_id: Option<Arc<dyn Fn() -> String + Send + Sync>>` | 可选；目前 Anthropic 实现中无强制使用点，预留与上游对齐 |
| `.chat(id)` | inherent fn | 等价 `messages(id)` 别名 |
| `.language_model(id)` | inherent fn | 等价 `messages(id)` 别名 |
| `.files()` | inherent fn | 返回 `AnthropicFiles { inner: Arc<Inner> }` |
| `.skills()` | inherent fn | 返回 `AnthropicSkills { inner: Arc<Inner> }` |

`auth_token` 加载逻辑：`load_optional_setting(self.auth_token, "ANTHROPIC_AUTH_TOKEN")`（与上游一致）。

`build()` 校验：
```rust
match (self.api_key.is_some(), self.auth_token.is_some()) {
    (true, true) => return Err(ProviderError::invalid_argument(...)),
    (false, false) => /* try ANTHROPIC_AUTH_TOKEN; else load ANTHROPIC_API_KEY */,
    (true, false) => use api_key,
    (false, true) => use auth_token,
}
```

### F. Messages 响应元数据深度解析

#### F.1 Usage iterations

`messages/usage.rs` 新增 typed enum：

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UsageIteration {
    Compaction {
        input_tokens: u64,
        output_tokens: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_creation_input_tokens: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_read_input_tokens: Option<u64>,
    },
    Message {
        input_tokens: u64,
        output_tokens: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_creation_input_tokens: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_read_input_tokens: Option<u64>,
    },
    AdvisorMessage {
        model: String,
        input_tokens: u64,
        output_tokens: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_creation_input_tokens: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_read_input_tokens: Option<u64>,
    },
}
```

非流式 `do_generate` 响应 → 写入 `provider_metadata.anthropic.usageIterations: [...]`。
流式 `do_stream` `message_delta` 的最终 usage 帧同样处理（如果包含 iterations）。

#### F.2 Container metadata

`messages/parse_response.rs` 新增解析：

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerMetadata {
    pub expires_at: String,                          // RFC 3339
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skills: Option<Vec<ContainerSkill>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerSkill {
    #[serde(rename = "type")]
    pub kind: String,                                // "user"
    pub skill_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}
```

响应顶级若有 `container: {...}` → 写入 `provider_metadata.anthropic.container`。

#### F.3 Context management applied edits

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AppliedContextEdit {
    #[serde(rename = "clear_tool_uses_20250919")]
    ClearToolUses {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cleared_tool_uses: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cleared_input_tokens: Option<u64>,
    },
    #[serde(rename = "clear_thinking_20251015")]
    ClearThinking {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cleared_thinking_turns: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cleared_input_tokens: Option<u64>,
    },
    #[serde(rename = "compact_20260112")]
    Compact {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cleared_input_tokens: Option<u64>,
    },
}
```

响应顶级若有 `context_management.applied_edits: [...]` → 写入 `provider_metadata.anthropic.contextManagement.appliedEdits`。

### G. 测试覆盖

新增契约测试：

1. `crates/llmsdk-anthropic/tests/contract_files.rs`
   - upload 字节数据；upload base64；media_type 透传；filename 自定义；error 路径；header 验证
2. `crates/llmsdk-anthropic/tests/contract_skills.rs`
   - upload skill 单文件；upload skill 多文件；带 display_title；version 元信息回填；不带 latest_version 跳过 GET；error 路径
3. `crates/llmsdk-anthropic/tests/contract_tools_typed.rs`
   - 20 个 factory 各自 wire 输出对照（id / name / args 序列化）
4. 扩展 `contract_messages_options.rs`
   - iterations 三 variant 解析；container metadata 解析；context_management.applied_edits 三 variant 解析

新增单元测试（src/**/tests）：
- `tools/*::tests` —— args struct 序列化往返
- `files::wire::tests` —— WireUploadResponse 反序列化
- `skills::wire::tests` —— WireSkillResponse + WireSkillVersionResponse
- `messages::usage::tests` —— UsageIteration 三 variant 往返
- `messages::parse_response::tests` —— ContainerMetadata / AppliedContextEdit

### H. 文档与最终检查

1. CLAUDE.md 标 M12 完成段；列出 trait 改动数（+2 新 trait + ProviderReference 类型）
2. todo.md 把 "Anthropic Files API endpoint" 移出推迟段
3. `cargo fmt --check` + `cargo clippy --workspace --all-targets -- -D warnings` 通过
4. `cargo nextest run --workspace` 全绿（M11 已 321 测试；本阶段预期 +40~+60）
5. 启动 Explore subagent 对照上游审核 PASS

## 范围外（明确推迟到 M13+）

- **Gemini provider**（用户两次推迟，本阶段保持不动；M12 完成后转入）
- **fileIdPrefixes 用户可配置化**（M11 todo）
- **新增 middleware 抽象**（M10.5 已定）
- **trait 层 ProviderTrait::files() / skills() 工厂方法**（用户明确决定不进 Provider trait）
- **CacheMiddleware 分布式 store reference impl**

## Open Questions Resolved

| Q | 决议 |
|---|---|
| Trait 集成方式？ | 新增独立 FilesModel / SkillsModel trait，不动 Provider trait |
| 20 typed tool factory 范围？ | 全 20 个都做 typed factory + args 校验 |
| 元数据深度解析？ | 全做（iterations + container + context_management.applied_edits 全部 typed enum） |

## Trait 改动汇总（截至 M12）

M1–M12 累计 trait 改动 11 处：
- M8 `ImageResult.warnings`
- M10 `JsonSchema = schemars::Schema` / `ImageOptions.files+mask` / `ImageResult.usage` / `ImageUsage` / `ImageUsageInputDetails`
- M10.5 `StreamPart::File` / `StreamPart::ReasoningFile` / `Tool::Provider` wire tag 改为 `provider`
- M11 `ToolCallPart.dynamic`
- M12 新增 `FilesModel` trait + `SkillsModel` trait + 关联 `FileData` / `ProviderReference` / `SkillFile` / `UploadFileOptions` / `UploadFileResult` / `UploadSkillOptions` / `UploadSkillResult` 类型（**非破坏性，纯新增**）

## 实施顺序

1. 写本设计文档（task #1） ✓
2. 新增 FilesModel trait → subagent 审核 → 通过则继续（task #2）
3. 新增 SkillsModel trait → subagent 审核 → 通过则继续（task #3）
4. 实现 AnthropicFiles + 契约测试（task #4）
5. 实现 AnthropicSkills + 契约测试（task #5）
6. 实现 20 typed tool factory + 单元测试（task #6）
7. 扩展 AnthropicBuilder（auth_token / name / chat / language_model / files / skills aliases）（task #7）
8. 响应元数据深度解析（task #8）
9. 全量 fmt + clippy + nextest（task #9）+ 最终 subagent 审核
10. 更新 CLAUDE.md + todo.md（task #10）

每个 trait + provider capability 完成后按 CLAUDE.md "Checkpoint 规则" 启动 subagent 审核。

