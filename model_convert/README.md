# MiniCPM5-1B 模型转换

本文档说明 `openbmb/MiniCPM5-1B` 在 AX650 上的 LLM 编译方式。

## 当前状态

- 已验证 `pulsar2 llm_build`
- `model_type=llama`
- `hidden_state_type=bf16`
- `prefill_len=128`
- `kv_cache_len=2047`
- 额外 prefill group 覆盖到 `1152`
- 推荐启用 `FLOAT_MATMUL_USE_CONV_EU=1`

MiniCPM5-1B 是纯文本模型，不涉及 VIT、音频或视频编码器编译。

本仓库不提交 `.axmodel`、embedding、ONNX、safetensors 等编译或推理产物。

## 环境变量

脚本默认值适配内部验证环境。外部用户应按自己的目录覆盖：

```bash
export CODEBASE_ROOT=/path/to/npu-codebase
export DEPLOY_ROOT=/path/to/auto_model_deployment
export INPUT_PATH=/path/to/openbmb/MiniCPM5-1B
export CONDA_SH=/path/to/conda.sh
export CONDA_ENV=npu
```

其中：

- `CODEBASE_ROOT`：包含 `script/npu_dev` 和 `pulsar2 llm_build` 集成代码的 npu-codebase
- `DEPLOY_ROOT`：模型部署工作区根目录
- `INPUT_PATH`：原始 Hugging Face 模型目录
- `CONDA_SH` / `CONDA_ENV`：用于激活编译环境

## 编译命令

在本目录执行：

```bash
./llm_build_ax650.sh
```

默认输出到本地工作区：

```text
../python/MiniCPM5-1B_axmodel
```

`*_axmodel/` 已被 `.gitignore` 忽略，不应提交到本仓库。

也可以指定输出目录：

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

## 验证建议

编译完成后，先在 AX650 板端用 `ax_run_model` 检查单个子图是否能加载，再使用最终发布包中的 `axllm serve` 做端到端验证。

最终发布包验证示例：

```bash
cd /path/to/MiniCPM5-1B
./bin/axllm serve .
```
