# 0006 — M13 First-Tier Provider Parity

> Status: in progress
> Upstream reference: `vercel/ai` @ `packages/{xai,mistral,azure,cohere,google,anthropic-aws,amazon-bedrock,google-vertex}/src/**`
> Prereqs: `0001-trait-design.md`、`0002-middleware-design.md`、`0003-m10-design.md`、`0004-m11-responses-design.md`、`0005-m12-anthropic-full-design.md`

## Goal

把 ai-sdk 全部"一线大厂"provider 全量接入 llmsdk，达成 100% feature parity，
覆盖 8 个 provider 包：xAI / Mistral / Azure OpenAI / Cohere / Google Gemini /
Anthropic on AWS / Amazon Bedrock / Google Vertex AI。

同时引入两类此前未覆盖的 trait（VideoModel + RerankingModel），让 llmsdk-provider
trait 表面对齐 ai-sdk v4 全部 6 类 model（LanguageModel / EmbeddingModel /
ImageModel / FilesModel / SkillsModel / **VideoModel** / **RerankingModel**）
+ SkillsModel + FilesModel；SpeechModel / TranscriptionModel 留到下一阶段
（M14 — TTS / STT）。

按 CLAUDE.md 强制规则"启动新阶段前必须列出全部范围"——本文档即范围 ground
truth；不允许中途静默推迟。开始前已与用户对齐 4 处范围决策（见末尾
"Open Questions Resolved"）。

## 范围（全部纳入）

### A. 新增 trait（`llmsdk-provider`）

#### A.1 `VideoModel` trait

新模块 `crates/llmsdk-provider/src/video_model/`：

```rust
#[async_trait::async_trait]
pub trait VideoModel: Send + Sync + std::fmt::Debug {
    fn provider(&self) -> &str;
    fn model_id(&self) -> &str;
    fn specification_version(&self) -> &'static str { "v4" }

    /// Most video models only generate 1 video per call due to cost.
    /// Returning `None` means use a global default (1).
    async fn max_videos_per_call(&self) -> Option<u32> { Some(1) }

    async fn do_generate(&self, options: VideoOptions) -> Result<VideoResult, ProviderError>;
}

pub struct VideoOptions {
    pub prompt: Option<String>,
    pub n: u32,                                  // default 1
    pub aspect_ratio: Option<String>,            // "16:9" / "9:16" / "1:1" / ...
    pub resolution: Option<String>,              // "1280x720" / "1920x1080"
    pub duration_seconds: Option<f64>,
    pub fps: Option<u32>,
    pub seed: Option<u64>,
    pub image: Option<VideoFile>,                // image-to-video starting frame
    pub provider_options: Option<ProviderOptions>,
    pub headers: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum VideoFile {
    File {
        media_type: String,                       // "video/mp4" / "image/png" / ...
        data: VideoFileData,                      // Bytes | Base64String
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_options: Option<ProviderOptions>,
    },
    Url {
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_options: Option<ProviderOptions>,
    },
}

#[derive(Debug, Clone)]
pub enum VideoFileData {
    Bytes(bytes::Bytes),
    Base64(String),
}

pub struct VideoResult {
    pub videos: Vec<VideoData>,                  // URL / base64 / binary
    pub warnings: Vec<CallWarning>,
    pub provider_metadata: Option<ProviderMetadata>,
    pub response: ResponseInfo,                  // timestamp + model_id + headers
}

#[derive(Debug, Clone)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum VideoData {
    Url { url: String, media_type: String },
    Base64 { data: String, media_type: String },
    Binary { data: bytes::Bytes, media_type: String },
}
```

依据：`packages/provider/src/video-model/v4/*`。  
注意 ai-sdk 标记 `Experimental_VideoModelV4`（仍 experimental，但 v4 已稳定足以引入）。
我们 trait 名 = `VideoModel`（不加 Experimental 前缀，与其它 5 类 model 表面一致）。
若上游升 v5，再随主 trait 一起升。

#### A.2 `RerankingModel` trait

新模块 `crates/llmsdk-provider/src/reranking_model/`：

```rust
#[async_trait::async_trait]
pub trait RerankingModel: Send + Sync + std::fmt::Debug {
    fn provider(&self) -> &str;
    fn model_id(&self) -> &str;
    fn specification_version(&self) -> &'static str { "v4" }

    async fn do_rerank(&self, options: RerankingOptions) -> Result<RerankingResult, ProviderError>;
}

pub struct RerankingOptions {
    pub documents: RerankingDocuments,            // 二态枚举
    pub query: String,
    pub top_n: Option<u32>,
    pub provider_options: Option<ProviderOptions>,
    pub headers: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum RerankingDocuments {
    Text { values: Vec<String> },
    Object { values: Vec<serde_json::Map<String, serde_json::Value>> },
}

pub struct RerankingResult {
    pub ranking: Vec<RankingEntry>,               // sorted by relevance_score desc
    pub warnings: Vec<CallWarning>,
    pub provider_metadata: Option<ProviderMetadata>,
    pub response: Option<ResponseInfo>,
}

pub struct RankingEntry {
    pub index: u32,                               // index in original list
    pub relevance_score: f64,
}
```

依据：`packages/provider/src/reranking-model/v4/*`。

#### A.3 Middleware 表面

按 M10 立的规矩（EmbeddingModelMiddleware / ImageModelMiddleware 已立），
本轮新增：

- `VideoModelMiddleware` trait + 默认 no-op
- `RerankingModelMiddleware` trait + 默认 no-op
- `wrap_video_model(model, middlewares)` / `wrap_reranking_model(model, middlewares)`
- 不向 `ProviderMiddlewareSet` 加 video / reranking 字段（避免破坏现有 API）；
  下游需要批量包装就显式列 video/reranking 列表传 `wrap_*`。

#### A.4 trait 改动汇总

- 纯新增：`VideoModel` / `RerankingModel` trait + 关联类型 + 两个 middleware trait + 两个 wrap_ 函数
- **零破坏性**（与 M12 同款策略）

### B. Provider 列表（全部纳入）

| # | Crate | 端点 | trait 用到 | 新依赖 |
|---|---|---|---|---|
| B.1 | `llmsdk-xai` | Chat / Image / Video / Responses / Files | LM + Image + VideoModel + FilesModel | 无 |
| B.2 | `llmsdk-mistral` | Chat / Embedding | LM + Embed | 无 |
| B.3 | `llmsdk-azure` | OpenAI Chat / Responses / Embed / Image (deployment URL 方言) | LM + Embed + Image（复用 llmsdk-openai） | 无 |
| B.4 | `llmsdk-cohere` | Chat / Embed / Rerank | LM + Embed + RerankingModel | 无 |
| B.5 | `llmsdk-google` | Language / Embed / Image / Video / Files | LM + Embed + Image + VideoModel + FilesModel | 无 |
| B.6 | `llmsdk-anthropic-aws` | Anthropic Messages + Files + Skills 经 SigV4 | LM + FilesModel + SkillsModel（复用 llmsdk-anthropic） | `aws-sigv4` |
| B.7 | `llmsdk-amazon-bedrock` | Converse / ConverseStream / Embed / Image / Anthropic / Rerank | LM + Embed + Image + RerankingModel | `aws-sigv4` + `aws-smithy-eventstream` |
| B.8 | `llmsdk-google-vertex` | Vertex (Gemini / Anthropic / xAI / MaaS) Language + Embed + Image + Video | LM + Embed + Image + VideoModel | `gcp_auth` |

### C. 共享基础设施（`llmsdk-provider-utils`）

#### C.1 AWS SigV4 fetch wrapper

新模块 `crates/llmsdk-provider-utils/src/aws_sigv4.rs`：

- 依赖 `aws-sigv4` crate（仅此一项，不引入 `aws-config` / `aws-sdk-*`）
- 提供 `sign_request(req, credentials, region, service)` 同步函数（基于 `aws-sigv4::http_request::sign`）
- 提供 `AwsCredentials { access_key_id, secret_access_key, session_token: Option<String> }`
  + `AwsCredentialsProvider` trait（静态 / env / 自定义闭包三种实现）
- 不引入 STS / IMDS（v1/v2）/ AssumeRole；下游想要就自己实现 trait

#### C.2 AWS EventStream binary frame decoder

新模块 `crates/llmsdk-provider-utils/src/aws_eventstream.rs`：

- 依赖 `aws-smithy-eventstream` crate
- 提供 `decode_event_stream(byte_stream) -> impl Stream<Item = Result<EventStreamMessage, ProviderError>>`
- `EventStreamMessage { headers: HashMap<String, EventStreamValue>, payload: bytes::Bytes }`
- 不引入 `aws-smithy-http`；headers 类型自封装一个最小 enum（与 `HeaderValue` 解耦）

#### C.3 GCP OAuth token provider

新模块 `crates/llmsdk-provider-utils/src/gcp_auth.rs`：

- 依赖 `gcp_auth` crate（pure-Rust，覆盖 SA JSON / ADC / metadata server / gcloud CLI 全部 fallback）
- 提供 `GcpTokenProvider` trait（异步 `get_access_token() -> Result<String, ProviderError>`）
- 默认实现 = wrapper around `gcp_auth::provider()`（自动 fallback）
- 提供 builder 显式指定 SA JSON path
- 不缓存 token —— `gcp_auth` 内部已经做了

#### C.4 不复用 ai-sdk `aws4fetch` 路径

ai-sdk 用 `aws4fetch`（小型 JS 库）；Rust 侧选择 `aws-sigv4` 官方实现以
保证算法正确性（SigV4 + CanonicalRequest 极易跑偏）。已与用户对齐。

### D. Provider 详细范围

#### D.1 `llmsdk-xai`

依据 `packages/xai/src/**`，对照实现：

- **Chat** (`POST /v1/chat/completions`)：
  - OpenAI 兼容 wire；额外 provider options：
    - `reasoningEffort`（low/high）
    - `searchParameters`（mode/maxSearchResults/sources）
    - `webSearchOptions`（searchContextSize）
  - 自带 `livesearch` 内置工具（route 到 `xai.live_search`）
  - 解析 `citations[]` → `Content::Source { url_citation }`
  - 解析 `reasoning_content` → `Content::Reasoning`
- **Responses** (`POST /v1/responses`)：
  - xAI 自有 responses API（与 OpenAI Responses 不同 wire）
  - 单独 `XaiResponsesLanguageModel`；不与 `OpenAiResponsesLanguageModel` 共代码
- **Image** (`POST /v1/images/generations`)：
  - `grok-2-image*` 系列；prompt + n + size 透传
- **Video** (`POST /v1/videos/generations` + `/edits` + `/extensions`)：
  - 4 种模式：
    - 默认（text-to-video）
    - `edit-video` (+ `videoUrl`)
    - `extend-video` (+ `videoUrl`)
    - `reference-to-video` (+ `referenceImageUrls[]`)
  - 异步任务 = POST 拿 jobId → 轮询 `GET /v1/videos/jobs/{id}` 直到 `succeeded` / `failed`
  - `pollIntervalMs` / `pollTimeoutMs` provider option
  - 首个 `VideoModel` 实现，trait 验证基准
- **Files** (`POST /v1/files`)：
  - multipart 上传，类似 OpenAI Files
  - `purpose` provider option
- **不实现**：Embedding（xAI 不提供）；TTS（不提供）

#### D.2 `llmsdk-mistral`

依据 `packages/mistral/src/**`：

- **Chat** (`POST /v1/chat/completions`)：
  - OpenAI 兼容但有差异：
    - `prefix` 续写模式（assistant 最后一条消息加 `prefix: true`）
    - `safe_prompt` provider option
    - `random_seed` 替代 `seed`
    - `document_image_url` 文档图像
    - `document_url` PDF 文档
  - Function calling 与 OpenAI 兼容
  - 解析 `reasoning_content`（magistral）→ `Content::Reasoning`
- **Embedding** (`POST /v1/embeddings`)：
  - `mistral-embed` / `codestral-embed`
  - `output_dtype` / `output_dimension` provider option

#### D.3 `llmsdk-azure`

依据 `packages/azure/src/**`：

- 复用 `llmsdk-openai` 的 Chat / Responses / Embed / Image / Completion 全部实现
- 自定义 URL 模板：
  - `{baseURL}/openai/deployments/{deploymentId}/{endpoint}?api-version={apiVersion}`
  - 默认 `apiVersion` = `'preview'`（与 ai-sdk 上游一致）
- 双认证：
  - `api-key` header（默认，从 `AZURE_API_KEY` env）
  - `Authorization: Bearer <AAD token>`（用户传 `Resolvable<string>` token 提供者）
- 复用 `OpenAiChatLanguageModel` 内核 + 自定义 `Config { build_url, headers }`
- Azure 自有 `azureOpenaiTools` = OpenAI tools 别名（pass-through）
- `useDeploymentBasedUrls` provider option（false 时走 `/openai/v1/` 而非 `/openai/deployments/...`）

#### D.4 `llmsdk-cohere`

依据 `packages/cohere/src/**`：

- **Chat** (`POST /v1/chat`)：
  - Cohere 自有 wire，与 OpenAI 不兼容
  - `messages[]` 含 system/user/assistant/tool 四角色 + `tool_plan` 字段
  - `documents[]` RAG 文档
  - `connectors[]` Cohere connectors
  - `citation_options`（mode = fast/accurate/off）
  - `safety_mode` provider option
  - 解析 `tool_plan` → `Content::Reasoning`
  - 解析 `citations[]` → `Content::Source`
- **Embedding** (`POST /v2/embed`)：
  - `embed-english-v3.0` 等
  - `input_type`（search_query / search_document / classification / clustering）
  - `embedding_types` provider option（float/int8/uint8/binary/ubinary）
- **Reranking** (`POST /v2/rerank`)：
  - `rerank-english-v3.0` 等
  - 首个 `RerankingModel` 实现，trait 验证基准
  - `documents` 同时支持 text 与 object（按 trait `RerankingDocuments` 二态）
  - `max_chunks_per_doc` provider option

#### D.5 `llmsdk-google`

依据 `packages/google/src/**`：

- **Language** (`POST {model}:generateContent` + `:streamGenerateContent?alt=sse`)：
  - Gemini wire 格式：`contents[]` (role: user/model) + `parts[]`
  - `parts[]` 类型：`text` / `inlineData` (base64 image/audio/pdf) / `fileData` (URI) / `functionCall` / `functionResponse` / `thought` (thinking) / `executableCode` / `codeExecutionResult`
  - `tools[]` 类型：`functionDeclarations[]` / `googleSearch{}` / `googleSearchRetrieval{}` / `codeExecution{}` / `urlContext{}` / `enterpriseWebSearch{}` / `computerUse{}`
  - `toolConfig.functionCallingConfig.mode` = ANY/AUTO/NONE/VALIDATED
  - `safetySettings[]` provider option
  - `generationConfig`：temperature / topK / topP / maxOutputTokens / responseMimeType / responseSchema / responseModalities / mediaResolution / seed / thinkingConfig (includeThoughts + thinkingBudget) / speechConfig
  - JSON Schema → OpenAPI 3.0 Schema 转换（移除 `$schema`/`additionalProperties` 等不支持字段）
  - 解析 `groundingMetadata` / `urlContextMetadata` → provider_metadata + Content::Source
  - 解析 `usageMetadata.cachedContentTokenCount` → Usage.cached_input_tokens
- **Embedding** (`POST {model}:embedContent` + `:batchEmbedContents`)：
  - `text-embedding-004` / `gemini-embedding-001`
  - `task_type` / `output_dimensionality` / `title`
- **Image** (`POST {model}:predict`)：
  - `imagen-3.0-*` / `imagen-4.0-*`
  - `aspectRatio` / `safetyFilterLevel` / `personGeneration` provider option
- **Video** (`POST {model}:predictLongRunning` + `GET /operations/{name}`)：
  - `veo-2.0-generate-001` 等
  - 异步 LRO（Long-Running Operation）：POST 拿 operation name → 轮询直到 `done: true`
  - `pollIntervalMs` / `pollTimeoutMs` provider option
- **Files** (`POST /upload/v1beta/files?uploadType=multipart`)：
  - Gemini Files API（24h TTL）
  - `displayName` / `mimeType` provider option
- 路径：`/v1beta/models/{model}:method`（默认）；`/v1/...` 可切换

#### D.6 `llmsdk-anthropic-aws`

依据 `packages/anthropic-aws/src/**`：

- 直连 Anthropic on AWS（**非** Bedrock 路径；走 Anthropic 自己的 AWS 部署）
- URL = `https://api.anthropic.aws/v1/messages`（region 拼接）
- 双认证模式：
  - **SigV4**（默认，从 `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` env）
  - **API Key** mode（`ANTHROPIC_AWS_API_KEY`，覆盖 SigV4，用 `x-api-key` header）
- 强制 header：`anthropic-workspace-id: {workspace_id}`（从 `ANTHROPIC_AWS_WORKSPACE_ID` env）
- 复用 `llmsdk-anthropic` 的 Messages 协议 100%（共享 prompt 转换、响应解析、流式）
- 同时复用 `llmsdk-anthropic` 的 Files + Skills（也走 SigV4）
- 不复用 anthropicTools（直接 re-export `llmsdk_anthropic::tools`）
- 实现策略：`AnthropicAws::new(config)` → 构造 `llmsdk_anthropic::Anthropic`（自定义
  base_url + 自定义 http_client（SigV4 wrapper））→ 暴露 `.language_model() / .files() / .skills()`

#### D.7 `llmsdk-amazon-bedrock`

依据 `packages/amazon-bedrock/src/**`：

- **Converse API**（默认 chat 入口）：
  - URL = `{base}/model/{modelId}/converse` + `:converse-stream`（流式）
  - Bedrock Converse 通用消息格式：`messages[]` (role: user/assistant) + `content[]`（text/image/document/toolUse/toolResult/reasoningContent/guardContent/cachePoint）
  - `inferenceConfig`：maxTokens / temperature / topP / stopSequences
  - `additionalModelRequestFields` 透传 model-specific 选项（如 anthropic 的 thinking）
  - `toolConfig.toolChoice` = `{auto:{}}` / `{any:{}}` / `{tool: {name}}`
  - `guardrailConfig` provider option
  - 流式 = AWS EventStream binary frame（用 C.2 decoder）
  - 解析 `usage.cacheReadInputTokens` / `cacheWriteInputTokens` → Usage cached
  - 解析 `reasoningContent.reasoningText` → Content::Reasoning（含 signature）
- **Embedding** (`POST /model/{modelId}/invoke`)：
  - Titan Embed (`amazon.titan-embed-text-v2:0`)
  - Cohere Embed on Bedrock (`cohere.embed-english-v3`)
  - 各 model family 不同 wire（按 modelId 前缀分发）
- **Image** (`POST /model/{modelId}/invoke`)：
  - Stable Diffusion / SDXL / Titan Image / Nova Canvas
  - 各 family 不同 wire（按前缀分发）
- **Anthropic on Bedrock**（**非** anthropic-aws 路径；走 Bedrock InvokeModel）：
  - URL = `{base}/model/anthropic.claude-*/invoke`
  - 复用 `llmsdk-anthropic` 的请求 body 构造，但响应/流走 Bedrock 包装
- **Reranking** (`POST /rerank`)：
  - `cohere.rerank-v3-5:0` / `amazon.rerank-v1:0`
  - 走通用 reranking API（POST `/rerank`，非 `/model/.../invoke`）
- 认证 = SigV4（C.1）；service = `bedrock`
- `additional-model-id-attributes` URL query 透传（如 region routing）

#### D.8 `llmsdk-google-vertex`

依据 `packages/google-vertex/src/**`：

- 认证 = GCP OAuth（C.3，gcp_auth crate），scope = `cloud-platform`
- **Express Mode**（基于 API Key）：URL = `https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:method` + header `x-goog-api-key`
- **Standard Mode**（基于 OAuth）：URL = `https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:method`
  - `location: 'global'` 时不带 region 前缀
- **Language**：复用 `llmsdk-google` 的 GoogleLanguageModel 内核 + 自定义 Config
- **Embedding** (`/publishers/google/models/{model}:predict`)：
  - Vertex 自有 wire（instances[] + parameters）
- **Image** (`/publishers/google/models/{model}:predict`)：
  - Vertex Imagen，与 Express 路径有差异
- **Video** (`{model}:predictLongRunning`)：
  - Veo on Vertex
- **Anthropic on Vertex** (`/publishers/anthropic/models/{model}:rawPredict`)：
  - 子模块 `vertex/anthropic`；复用 `llmsdk-anthropic`
- **xAI on Vertex** (`/publishers/x-ai/models/{model}`)：
  - 子模块 `vertex/xai`；复用 `llmsdk-xai` Chat
- **MaaS (Model-as-a-Service)** (`/publishers/{publisher}/models/{model}:rawPredict`)：
  - 子模块 `vertex/maas`；OpenAI-compatible 包装（DeepSeek / Llama / Mistral on Vertex）
  - 复用 `llmsdk-openai` Chat 内核（通过 OpenAI-compatible 切入）
- **Edge**（无 Node.js 环境的版本）：Rust 侧不区分 edge / node，统一一个 crate

### E. 测试要求

每个 provider 至少 5 个契约测试文件：
- `contract_<feature>_basic.rs`（do_generate 单轮）
- `contract_<feature>_stream.rs`（do_stream）
- `contract_<feature>_tools.rs`（tool calling）
- `contract_<feature>_options.rs`（provider options 透传）
- `contract_<feature>_advanced.rs`（特殊路径：reasoning / citations / multimodal 等）

外加：
- `llmsdk-xai/tests/contract_video.rs`（首个 VideoModel impl 必备）
- `llmsdk-cohere/tests/contract_rerank.rs`（首个 RerankingModel impl 必备）
- `llmsdk-amazon-bedrock/tests/contract_event_stream.rs`（SigV4 + EventStream 端到端）
- `llmsdk-google-vertex/tests/contract_express_mode.rs`（API key fallback path）

新 trait 单元测试：
- `llmsdk-provider/tests/video_model_trait.rs`
- `llmsdk-provider/tests/reranking_model_trait.rs`

### F. 依赖增量（已对齐用户）

| Crate | 用途 | 加在哪 |
|---|---|---|
| `aws-sigv4` | SigV4 签名算法 | `llmsdk-provider-utils` (optional feature `aws-sigv4`) |
| `aws-smithy-eventstream` | EventStream binary 帧解码 | `llmsdk-provider-utils` (optional feature `aws-event-stream`) |
| `gcp_auth` | GCP OAuth token 自交换 + ADC fallback | `llmsdk-provider-utils` (optional feature `gcp-auth`) |

所有三个依赖均 gated 在 `llmsdk-provider-utils` 的 optional feature 后面，
默认 feature 不包含。下游 crate（`llmsdk-anthropic-aws` / `llmsdk-amazon-bedrock`
/ `llmsdk-google-vertex`）通过 `llmsdk-provider-utils = { features = [...] }`
开启所需 feature。

### G. 不在本里程碑

- **SpeechModel trait + TTS 三家**（elevenlabs / hume / lmnt）→ M14
- **TranscriptionModel trait + STT 四家**（deepgram / assemblyai / gladia / revai）→ M14
- 其它 28 个非一线 provider（perplexity / groq / cerebras / fireworks /
  togetherai / deepinfra / baseten / huggingface / replicate / alibaba /
  bytedance / moonshotai / deepseek / openai-compatible / open-responses /
  gateway / vercel / voyage / black-forest-labs / fal / prodia / luma /
  klingai / mcp / quiverai）→ M15+
- IMDS v1/v2 / STS / AssumeRole 三种 AWS 凭据提供器 → 下游用户自实现 trait

## Open Questions Resolved

1. **范围切分**：用户选 "单一 M13 一次性落地"，按 CLAUDE.md 强制规则一阶段
   列全部范围。8 个 provider + 2 trait 全部纳入本里程碑。
2. **AWS SigV4**：用户选 "官方 aws-sigv4 + aws-smithy-eventstream"，引入
   两个 crate。算法实现交给上游官方包，零自实现 SigV4。
3. **GCP OAuth**：用户选 "gcp_auth crate"，覆盖 ADC / SA JSON / metadata
   server / gcloud CLI 全部 fallback。零手写 JWT。
4. **Reranking trait**：用户选 "新增 RerankingModel trait"，与上游 v4 对齐；
   不走 provider-specific 方法。

## 实施顺序

1. 起草本文档（task #1）✓
2. trait 新增：VideoModel + RerankingModel + 两个 middleware + wrap_*（task #2 + #3）
3. subagent 审核 trait 设计（task #4）
4. provider 实现按依赖顺序：
   - **B.1 xAI**（task #5，首个 VideoModel impl）
   - **B.2 Mistral**（task #6，纯 HTTP）
   - **B.3 Azure**（task #7，复用 OpenAI）
   - **B.4 Cohere**（task #8，首个 RerankingModel impl）
   - **B.5 Google Gemini**（task #9，最大 provider）
   - C.1 SigV4 wrapper（task #10）
   - **B.6 Anthropic on AWS**（task #11）
   - **B.7 Amazon Bedrock**（task #12，含 C.2 EventStream）
   - **B.8 Google Vertex**（task #13，含 C.3 gcp_auth）
5. M13 收口（task #14）

每完成一个 provider → 跑契约测试 + subagent 审核；都通过再继续下一个。
