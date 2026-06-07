import argparse

import numpy as np
from ml_dtypes import bfloat16
from transformers import AutoConfig, AutoTokenizer

from utils.infer_func import InferManager
from utils.runtime import default_axmodel_path
from utils.runtime import default_hf_model
from utils.runtime import load_text_embeddings


def build_prompt(tokenizer, prompt: str, system_prompt: str, enable_thinking: bool) -> str:
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def main():
    parser = argparse.ArgumentParser(description="MiniCPM5-1B axmodel text inference")
    parser.add_argument(
        "--hf_model",
        default=default_hf_model(__file__),
        help="Path to the tokenizer/config directory",
    )
    parser.add_argument(
        "--axmodel_path",
        default=default_axmodel_path(__file__),
        help="Path to the compiled axmodel directory",
    )
    parser.add_argument("--prompt", default="1+1等于几？请直接回答。")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.hf_model, trust_remote_code=True)
    embeds = load_text_embeddings(args.axmodel_path, config)

    text = build_prompt(tokenizer, args.prompt, args.system_prompt, args.enable_thinking)
    inputs = tokenizer(text, return_tensors="np")
    token_ids = inputs["input_ids"][0].tolist()
    prefill_data = np.take(embeds, token_ids, axis=0).astype(bfloat16)

    eos_token_id = config.eos_token_id if isinstance(config.eos_token_id, list) else None

    runner = InferManager(config, args.axmodel_path, max_seq_len=2047)
    token_ids = runner.prefill(
        tokenizer,
        token_ids,
        prefill_data,
        slice_len=128,
    )
    runner.decode(
        tokenizer,
        token_ids,
        embeds,
        prefill_len=128,
        eos_token_id=eos_token_id,
        max_new_tokens=args.max_new_tokens,
    )
    print("\n")


if __name__ == "__main__":
    main()
