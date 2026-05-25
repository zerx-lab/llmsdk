# TODO

> 待办与未对齐项汇总。与 `CLAUDE.md` 中里程碑段保持引用一致：里程碑写"已完成
> 范围"，本文件写"尚未对齐范围"。

## 未对齐的 ai-sdk 提供商（对照 `/home/zero/Desktop/code/github/ai/packages/`）

参考：当前 llmsdk 已实现 10 个 provider：openai / anthropic / xai / mistral /
azure / cohere / google / anthropic-aws / amazon-bedrock / google-vertex
（共 10 / 41 个 provider 包）。

下表中"trait 复用"列说明该 provider 是否能直接复用现有 `LanguageModel /
EmbeddingModel / ImageModel / VideoModel / RerankingModel / FilesModel /
SkillsModel` 七个 trait，或需要扩新的 trait（TTS / STT 等）。

### 一线大厂模型（M13 已完成 ✓）
| Provider | ai-sdk 包名 | 端点形态 | trait 复用 | 状态 |
|---|---|---|---|---|
| Google Gemini | `google` | `generateContent` / `streamGenerateContent` + Embedding + Image + Video + Files | ✓ 复用 | ✓ M13 |
| Google Vertex AI | `google-vertex` | Vertex（含 Gemini + Anthropic + xAI + MaaS） | ✓ 复用 | ✓ M13 |
| Anthropic on AWS | `anthropic-aws` | Anthropic 自有 AWS 部署 + SigV4 | ✓ 复用 | ✓ M13 |
| Amazon Bedrock | `amazon-bedrock` | Bedrock Converse + Embedding + Image + Anthropic + Rerank | ✓ 复用 | ✓ M13 |
| Azure OpenAI | `azure` | OpenAI 协议方言 | ✓ 复用 | ✓ M13 |
| Mistral | `mistral` | Chat + Embed | ✓ 复用 | ✓ M13 |
| Cohere | `cohere` | Chat + Embed + Rerank | ✓ 复用 | ✓ M13 |
| xAI Grok | `xai` | Chat + Image + Video + Responses + Files | ✓ 复用 | ✓ M13 |

### 高速推理 / 开源托管（待 M15）
| Provider | ai-sdk 包名 | trait 复用 |
|---|---|---|
| Groq | `groq` | ✓ |
| Cerebras | `cerebras` | ✓ |
| Fireworks | `fireworks` | ✓ |
| Together AI | `togetherai` | ✓ |
| DeepInfra | `deepinfra` | ✓ |
| Baseten | `baseten` | ✓ |
| HuggingFace | `huggingface` | ✓ |
| Replicate | `replicate` | ✓ |

### 国内模型（待 M15）
| Provider | ai-sdk 包名 | trait 复用 |
|---|---|---|
| Alibaba 通义 / Qwen | `alibaba` | ✓ |
| ByteDance 豆包 / Doubao | `bytedance` | ✓ |
| Moonshot Kimi | `moonshotai` | ✓ |
| DeepSeek | `deepseek` | ✓ |

### 搜索增强 / 通用 / 网关（待 M16）
| Provider | ai-sdk 包名 | trait 复用 |
|---|---|---|
| Perplexity | `perplexity` | ✓ |
| OpenAI-Compatible | `openai-compatible` | ✓ |
| Open-Responses | `open-responses` | ✓ |
| Vercel AI Gateway | `gateway` | ✓ |
| Vercel | `vercel` | ✓ |

### Embedding 专项（待 M16）
| Provider | ai-sdk 包名 | trait 复用 |
|---|---|---|
| Voyage | `voyage` | ✓（仅 Embedding） |

### 图像 / 视频生成（待 M16）
| Provider | ai-sdk 包名 | trait 复用 |
|---|---|---|
| Black Forest Labs (Flux) | `black-forest-labs` | ✓ Image |
| Fal | `fal` | ✓ Image + Video |
| Prodia | `prodia` | ✓ Image |
| Luma | `luma` | ✓ Video |
| Kling AI | `klingai` | ✓ Video |

### 语音合成 (TTS) — 需新增 `SpeechModel` trait（M14）
- `elevenlabs`
- `hume`
- `lmnt`

### 语音转写 (STT) — 需新增 `TranscriptionModel` trait（M14）
- `deepgram`
- `assemblyai`
- `gladia`
- `revai`

### 其它（待 M16+）
| Provider | ai-sdk 包名 | trait 复用 |
|---|---|---|
| MCP client provider | `mcp` | ⚠ 需要 MCP 客户端基础设施 |
| QuiverAI | `quiverai` | ✓ |

## 概览
- 41 个 provider 包 → 已对齐 **10 个**（M1–M13 累计），剩 **31 个**。
- 27 个可直接复用现有 7 个 trait。
- 7 个需要扩 trait（M14 范围）：
  - 7 个 TTS/STT（`SpeechModel` + `TranscriptionModel`）

## M14+ 推迟项
> 历史里程碑（M1–M9 阶段曾有"推迟到下一阶段"的特性，自 M10 起此做法已禁止；
> 此段保留作为事实记录）。M10 / M10.5 / M11 / M12 / M13 已将累计推迟项全部纳入并完成，
> 暂无残留推迟项。
