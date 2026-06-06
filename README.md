# MiniCPM5-1B.axera

> `openbmb/MiniCPM5-1B` 在 `AX650 / AX650N` 上的复现工程。

本仓库面向需要重新编译模型、核对运行配置或理解 AXERA 适配流程的开发者。  
如果你只需要直接在板端运行，请优先使用 Hugging Face 发布包：<https://huggingface.co/AXERA-TECH/MiniCPM5-1B>。

## 适用范围

- 平台：`AX650 / AX650N`
- Runtime：`axllm serve` / `axllm run`
- 模型类型：纯文本 LLM
- 编译类型：`model_type=llama`，`hidden_state_type=bf16`
- 上下文配置：`prefill_len=128`，`kv_cache_len=2047`，`prefill_max_token_num=1280`
- 默认生成模式：`enable_thinking=true`

当前发布包已经验证：

- 文本对话
- 多轮文本对话
- `enable_thinking=true/false` 请求级切换
- thinking 模式下 OpenAI API 输出显式 `<think>...</think>` 标记

MiniCPM5-1B 是纯文本模型；当前包不支持图片、视频或音频输入。

本仓库不提交任何 `.axmodel`、embedding、ONNX、safetensors 等编译或推理产物。

## 仓库职责

```text
.
├── model_convert/   # LLM 编译脚本和转换说明
├── python/          # tokenizer/config 元数据
└── README.md
```

根目录 `README.md` 说明仓库定位、运行能力和复现方式。  
重新执行 `pulsar2 llm_build` 请阅读 [model_convert/README.md](./model_convert/README.md)。

## 与发布包的关系

本仓库是开发侧复现仓库，包含转换脚本和小型元数据。  
面向最终用户的可直接运行包是 Hugging Face 仓库：

```text
AXERA-TECH/MiniCPM5-1B
```

发布包采用根目录直接 `axllm serve .` 的布局；本 `.axera` 仓库保留 `python/` 与 `model_convert/` 结构，便于复现编译过程和核对产物。

## 环境准备

### 编译环境

`pulsar2 llm_build` 需要 AXERA NPU 开发环境。示例：

```bash
export CODEBASE_ROOT=/path/to/npu-codebase
export DEPLOY_ROOT=/path/to/auto_model_deployment
export CONDA_SH=/path/to/conda.sh
export CONDA_ENV=npu
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$CODEBASE_ROOT"
source script/npu_dev
```

本仓库的脚本默认使用内部验证路径；外部用户应通过环境变量覆盖：

```bash
export CODEBASE_ROOT=/path/to/npu-codebase
export DEPLOY_ROOT=/path/to/auto_model_deployment
export INPUT_PATH=/path/to/openbmb/MiniCPM5-1B
export CONDA_SH=/path/to/conda.sh
export CONDA_ENV=npu
```

### 板端运行环境

`.axmodel` 只能在 AX650 板端运行，不要在 x86 服务器上执行。

如果只是使用最终发布包的 `axllm serve`，不需要 Python 调试依赖。

## 重新编译 LLM

在仓库根目录执行：

```bash
cd model_convert
./llm_build_ax650.sh
```

默认输出到本地工作区：

```text
python/MiniCPM5-1B_axmodel/
```

`*_axmodel/` 已被 `.gitignore` 忽略，不应提交到本仓库。

也可以显式指定输出目录：

```bash
./llm_build_ax650.sh /path/to/output_axmodel
```

脚本默认启用：

```bash
FLOAT_MATMUL_USE_CONV_EU=1
```

该选项在 AX650 上可以明显改善 TTFT。

## 发布包验证

推荐在最终发布包目录执行：

```bash
cd /path/to/MiniCPM5-1B
chmod +x ./bin/axllm
./bin/axllm serve .
```

默认端口为 `8000`。健康检查：

```bash
curl http://127.0.0.1:8000/health
```

默认 thinking 请求：

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

关闭 thinking：

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

## 本仓库内容

本仓库只保留复现所需的脚本和小型元数据：

```text
python/
└── MiniCPM5-1B/          # tokenizer / config 相关文件，不包含原始权重
```

说明：

- `python/MiniCPM5-1B_axmodel/` 是本地编译输出目录，不提交仓库
- 发布包中的 runtime `config.json` 默认设置 `enable_thinking=true`
- 原始 Hugging Face `safetensors` 权重不随本仓库发布
- 最终用户部署请使用 Hugging Face 发布包，而不是直接把本仓库当作运行包
