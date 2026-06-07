import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def resolve_dtype(name: str):
    if name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def main():
    parser = argparse.ArgumentParser(description="MiniCPM5-1B official PyTorch text inference")
    parser.add_argument(
        "--hf_model",
        required=True,
        help="Path to the full HuggingFace MiniCPM5-1B model directory with weights",
    )
    parser.add_argument("--prompt", default="1+1等于几？请直接回答。")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--do-sample", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    load_kwargs = {
        "device_map": device,
        "trust_remote_code": True,
    }
    if dtype != "auto":
        load_kwargs["torch_dtype"] = dtype

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.hf_model, **load_kwargs).eval()

    messages = []
    if args.system_prompt.strip():
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": args.prompt})

    render_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if args.enable_thinking:
        render_kwargs["enable_thinking"] = True
    else:
        render_kwargs["enable_thinking"] = False

    text = tokenizer.apply_chat_template(messages, **render_kwargs)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print("input_ids.shape:", tuple(inputs.input_ids.shape))
    print("enable_thinking:", args.enable_thinking)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
        )

    answer_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(answer)


if __name__ == "__main__":
    main()
