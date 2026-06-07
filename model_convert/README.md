# MiniCPM5-1B 模型转换与编译

本文档描述 `openbmb/MiniCPM5-1B` 在 AXERA 平台上的开发侧工作流，覆盖以下内容：

- LLM `pulsar2 llm_build` 编译
- 编译输出目录约定
- 编译产物的板端加载检查
- 与 Hugging Face 发布包的产物同步关系

本文档默认面向开发者使用，所有命令默认在 `model_convert/` 目录下执行。

> 当前仓库不提交 `.axmodel`、embedding、ONNX、safetensors 等编译或推理产物。编译输出目录已通过 `.gitignore` 忽略。

## 目录说明

```text
model_convert/
├── README.md
└── llm_build_ax650.sh      # AX650 LLM 编译脚本
```

`MiniCPM5-1B` 是纯文本模型，本目录不涉及 Vision、Audio 或 Video encoder 的导出与编译。

## 环境准备

`pulsar2 llm_build` 相关命令需要 AXERA NPU 编译环境。请先准备：

- 可直接执行 `pulsar2 llm_build` 的 shell 环境
- 原始 Hugging Face 模型目录：`openbmb/MiniCPM5-1B`
- 已安装 `pulsar2 llm_build` 依赖的 Python/conda 环境

脚本通过环境变量接收路径：

```bash
export INPUT_PATH=/path/to/openbmb/MiniCPM5-1B

# 如果当前 shell 还没有进入编译环境，可以额外设置：
# export CONDA_SH=/path/to/conda.sh
# export CONDA_ENV=npu
```

其中：

- `INPUT_PATH`：原始 Hugging Face 模型目录
- `CONDA_SH` / `CONDA_ENV`：可选，用于激活编译环境；如果当前 shell 已经在正确环境中，可以不设置

下文所有脚本默认以 `model_convert/` 为当前工作目录。

## 推荐顺序

如果你的目标是复现当前 LLM 编译流程，建议按下面顺序执行：

1. 准备原始 Hugging Face 权重目录
2. 确认当前 shell 可以执行 `pulsar2 llm_build`
3. 设置 `INPUT_PATH`，必要时设置 `CONDA_SH`、`CONDA_ENV`
4. 执行 `./llm_build_ax650.sh`
5. 在 AX650 板端用 `ax_run_model` 检查单个子图是否能加载
6. 将编译输出整理到 Hugging Face 发布包布局
7. 使用发布包中的 `axllm serve` 做端到端验证

## 已验证配置

当前已验证的 LLM 编译配置：

| Item | Value |
|---|---|
| `model_type` | `llama` |
| `hidden_state_type` | `bf16` |
| `prefill_len` | `128` |
| `kv_cache_len` | `2047` |
| `last_kv_cache_len` | `128, 256, 384, 512, 640, 768, 896, 1024, 1152` |
| `chip` | `AX650` |
| `parallel` | `32` |
| MatMul 优化 | `FLOAT_MATMUL_USE_CONV_EU=1` |

`FLOAT_MATMUL_USE_CONV_EU=1` 是当前 AX650 验证中使用的配置，可明显改善 TTFT。

## 准备原始模型

从 Hugging Face 下载原始模型权重，并用 `$INPUT_PATH` 指向该目录：

```bash
git clone https://huggingface.co/openbmb/MiniCPM5-1B /path/to/original/MiniCPM5-1B
export INPUT_PATH=/path/to/original/MiniCPM5-1B
```

原始权重不提交到本仓库。

## 编译 LLM axmodel

在 `model_convert/` 目录执行：

```bash
./llm_build_ax650.sh
```

默认输出到：

```text
../python/MiniCPM5-1B_axmodel
```

也可以显式指定输出目录：

```bash
./llm_build_ax650.sh /path/to/output_axmodel
```

等价核心命令如下：

```bash
FLOAT_MATMUL_USE_CONV_EU=1 pulsar2 llm_build \
  --input_path "$INPUT_PATH" \
  --output_path "$OUTPUT_PATH" \
  --model_type llama \
  --hidden_state_type bf16 \
  --prefill_len 128 \
  --kv_cache_len 2047 \
  --last_kv_cache_len 128 \
  --last_kv_cache_len 256 \
  --last_kv_cache_len 384 \
  --last_kv_cache_len 512 \
  --last_kv_cache_len 640 \
  --last_kv_cache_len 768 \
  --last_kv_cache_len 896 \
  --last_kv_cache_len 1024 \
  --last_kv_cache_len 1152 \
  --chip AX650 \
  -c 0 \
  --parallel 32
```

## 输出目录说明

编译完成后，输出目录通常包含：

```text
MiniCPM5-1B_axmodel/
├── llama_p128_l0_together.axmodel
├── ...
├── llama_p128_l23_together.axmodel
├── llama_post.axmodel
├── model.embed_tokens.weight.bfloat16.bin
├── minicpm5_tokenizer.txt
├── config.json
└── post_config.json
```

这些文件属于编译产物，不提交到 `.axera` 仓库。  
如果需要发布，请整理到 Hugging Face 发布包的最终布局中。

## 板端加载检查

`.axmodel` 只能在 AX650 板端执行。可以先检查单个子图能否加载：

```bash
cd /path/to/MiniCPM5-1B_axmodel
/opt/bin/ax_run_model -m llama_p128_l0_together.axmodel -g 0 --skip-running
```

端到端验证建议使用 Hugging Face 发布包：

```bash
cd /path/to/MiniCPM5-1B
./bin/axllm serve .
```

发布包的 runtime config 默认关闭 `enable_thinking`。如果需要 thinking 行为，请在 OpenAI 请求中设置 `"enable_thinking": true`。
