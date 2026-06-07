# MiniCPM5-1B.axera

> `openbmb/MiniCPM5-1B` 在 `AX650 / AX650N` 上的复现工程。

本仓库的目标是帮助开发者完成三类工作：

- 复现板端运行与精度验证
- 重新编译 `MiniCPM5-1B` 的 LLM 产物
- 使用 Python 脚本做 Torch / AxModel 文本调试

本仓库面向需要重新编译模型、核对文本效果或排查板端问题的开发者。

> 当前仓库只保存开发侧所需的脚本和 tokenizer/config，不提交 `.axmodel`、embedding、ONNX、safetensors 等编译或推理产物。如果你希望直接体验面向用户的实际 Demo，请参考 Hugging Face 发布页：<https://huggingface.co/AXERA-TECH/MiniCPM5-1B>。

## 适用范围

- 平台：`AX650 / AX650N`
- 支持的板端能力：
  - 文本对话
  - 多轮文本对话
  - `enable_thinking=true/false` 请求级切换
  - thinking 模式下返回显式 `<think>...</think>` 标记
- `MiniCPM5-1B` 是纯文本模型，不支持图片、视频或音频输入
- 发布包默认关闭 thinking；如需 reasoning 输出，需要在请求中显式设置 `enable_thinking=true`
- 当前仓库提供开发侧 Python 文本调试脚本；面向最终用户的平铺运行目录仍以 Hugging Face 发布包为准

## 仓库职责

```text
.
├── python/         # tokenizer/config、文本推理脚本和 Gradio Demo
└── model_convert/  # LLM 编译脚本和转换说明
```

这个仓库是开发侧 staging 目录，不是最终发布包。

最终平铺发布包请使用 Hugging Face 仓库：

- `AXERA-TECH/MiniCPM5-1B`

如果你需要重新编译 LLM axmodel，请阅读 [model_convert/README.md](./model_convert/README.md)。

## 运行前准备

### 1. 准备运行目录

在执行板端命令前，请确认以下文件已经准备好：

```text
python/
├── MiniCPM5-1B/          # tokenizer / config 相关文件
├── MiniCPM5-1B_axmodel/  # 用户本地编译得到的 LLM 运行目录，不提交仓库
├── infer_torch.py
├── infer_axmodel.py
├── gradio_demo.py
└── utils/
```

如果你只希望直接运行模型，请使用 Hugging Face 发布包：

```text
AXERA-TECH/MiniCPM5-1B
```

发布包已经包含 `bin/axllm`、LLM axmodel、embedding 和 runtime config，可以直接执行 `axllm serve .`。

### 2. 安装板端依赖

如果只使用发布包中的 `axllm serve`，不需要额外 Python 推理依赖。

如果你需要自行编译模型，需要准备 AXERA NPU 编译环境，并保证 `pulsar2 llm_build` 可用。

### 3. Thinking 开关说明

发布包的 runtime config 默认设置：

```json
{
  "enable_thinking": false
}
```

默认请求会按 no-thinking 模式生成。
如果请求中设置 `"enable_thinking": true`，则进入 thinking 模式，并在 OpenAI API 返回内容中显式包含 `<think>...</think>`，便于前端区分 reasoning 和最终回答。

## 本地脚本

以下命令默认在 `python/` 目录执行。

### x86 Torch 文本参考

```bash
cd python

python3 infer_torch.py \
  --hf_model /path/to/original/MiniCPM5-1B \
  --prompt "1+1等于几？请直接回答。"
```

如果要显式开启 thinking：

```bash
python3 infer_torch.py \
  --hf_model /path/to/original/MiniCPM5-1B \
  --prompt "中国的首都是哪里？请先思考，再给出简短答案。" \
  --enable-thinking
```

### 板端 AxModel 文本调试

```bash
cd python

python3 infer_axmodel.py \
  --hf_model ./MiniCPM5-1B \
  --axmodel_path ./MiniCPM5-1B_axmodel \
  --prompt "1+1等于几？请直接回答。"
```

如果要显式开启 thinking：

```bash
python3 infer_axmodel.py \
  --hf_model ./MiniCPM5-1B \
  --axmodel_path ./MiniCPM5-1B_axmodel \
  --prompt "中国的首都是哪里？请先思考，再给出简短答案。" \
  --enable-thinking
```

### Gradio 调试界面

```bash
cd python
python3 gradio_demo.py \
  --hf_model ./MiniCPM5-1B \
  --axmodel_path ./MiniCPM5-1B_axmodel \
  --port 7860
```

## 发布包运行

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

### 默认 no-thinking 请求

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

### 显式开启 thinking

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "AXERA-TECH/MiniCPM5-1B-AX650-C128-P1152-CTX2047",
    "messages": [
      {"role": "user", "content": "中国的首都是哪里？请先思考，再给出简短答案。"}
    ],
    "enable_thinking": true,
    "max_tokens": 128,
    "temperature": 0
  }'
```

## 模型转换

本仓库提供 LLM 编译脚本：

```bash
cd model_convert
export INPUT_PATH=/path/to/original/MiniCPM5-1B
./llm_build_ax650.sh
```

脚本默认输出到：

```text
python/MiniCPM5-1B_axmodel/
```

`*_axmodel/`、`.axmodel`、embedding `.bin`、ONNX 和 safetensors 均已被 `.gitignore` 忽略，不应提交到本仓库。

如果你需要重新执行编译，请阅读 [model_convert/README.md](./model_convert/README.md)。
