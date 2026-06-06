# MiniCPM5-1B.axera

> `openbmb/MiniCPM5-1B` 在 `AX650 / AX650N` 上的复现工程。

本仓库的目标是帮助用户完成两类工作：

- 复现板端运行与精度验证
- 重新编译 `MiniCPM5-1B` 的 LLM 产物

本仓库面向需要完整复现实验过程、重新编译模型或核对精度的用户。

> 当前仓库只保存复现所需的脚本和 tokenizer/config，不提交 `.axmodel`、embedding、ONNX、safetensors 等编译或推理产物。如果你希望直接体验面向用户的实际 Demo，请参考 Hugging Face 发布页：<https://huggingface.co/AXERA-TECH/MiniCPM5-1B>。

## 适用范围

- 平台：`AX650 / AX650N`
- 支持的板端能力：
  - 文本对话
  - 多轮文本对话
  - `enable_thinking=true/false` 请求级切换
  - thinking 模式下返回显式 `<think>...</think>` 标记
- `MiniCPM5-1B` 是纯文本模型，不支持图片、视频或音频输入
- 当前仓库不提供 Python 端 `.axmodel` 推理脚本；端到端验证以 Hugging Face 发布包中的 `axllm serve` 为准

## 仓库职责

```text
.
├── python/         # tokenizer/config 相关文件
└── model_convert/  # LLM 编译脚本和转换说明
```

根目录 `README.md` 负责说明“如何准备运行目录并在板端复现结果”。  
如果你需要重新编译 LLM axmodel，请阅读 [model_convert/README.md](./model_convert/README.md)。

## 运行前准备

### 1. 准备运行目录

在执行板端命令前，请确认以下文件已经准备好：

```text
python/
├── MiniCPM5-1B/          # tokenizer / config 相关文件
└── MiniCPM5-1B_axmodel/  # 用户本地编译得到的 LLM 运行目录，不提交仓库
```

如果你只希望直接运行模型，请使用 Hugging Face 发布包：

```text
AXERA-TECH/MiniCPM5-1B
```

发布包已经包含 `bin/axllm`、LLM axmodel、embedding 和 runtime config，可以直接执行 `axllm serve .`。

### 2. 安装板端依赖

如果只使用发布包中的 `axllm serve`，不需要额外 Python 推理依赖。

如果你需要自行编译模型，需要准备 AXERA NPU 开发环境，并保证 `pulsar2 llm_build` 可用。

### 3. Thinking 模式说明

发布包的 runtime config 默认设置：

```json
{
  "enable_thinking": true
}
```

默认请求会进入 thinking 模式，并在 OpenAI API 返回内容中显式包含 `<think>...</think>`，便于前端区分 reasoning 和最终回答。  
如果请求中设置 `"enable_thinking": false`，则按 no-thinking 模式生成。

## 板端复现

以下命令推荐在 Hugging Face 发布包目录执行。发布包布局和本 `.axera` 仓库不同，发布包可以直接作为 `axllm` 的运行目录。

### `axllm serve`

```bash
cd /path/to/MiniCPM5-1B
chmod +x ./bin/axllm
./bin/axllm serve .
```

默认端口为 `8000`。健康检查：

```bash
curl http://127.0.0.1:8000/health
```

### 默认 thinking 请求

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "AXERA-TECH/MiniCPM5-1B-AX650-C128-P1152-CTX2047",
    "messages": [
      {"role": "user", "content": "1+1等于几？请简短回答。"}
    ],
    "max_tokens": 128,
    "temperature": 0
  }'
```

### 关闭 thinking

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "AXERA-TECH/MiniCPM5-1B-AX650-C128-P1152-CTX2047",
    "messages": [
      {"role": "user", "content": "中国的首都是哪里？请只回答城市名。"}
    ],
    "enable_thinking": false,
    "max_tokens": 32,
    "temperature": 0
  }'
```

## 模型转换

本仓库提供 LLM 编译脚本：

```bash
cd model_convert
./llm_build_ax650.sh
```

脚本默认输出到：

```text
python/MiniCPM5-1B_axmodel/
```

`*_axmodel/`、`.axmodel`、embedding `.bin`、ONNX 和 safetensors 均已被 `.gitignore` 忽略，不应提交到本仓库。

如果你需要重新执行编译，请阅读 [model_convert/README.md](./model_convert/README.md)。
