import argparse
import socket
import time
from typing import Generator

import gradio as gr
import numpy as np
from ml_dtypes import bfloat16
from transformers import AutoConfig, AutoTokenizer

from utils.infer_func import InferManager
from utils.runtime import default_axmodel_path
from utils.runtime import default_hf_model
from utils.runtime import load_text_embeddings


def _list_host_ips():
    ips = set()
    try:
        hostname = socket.gethostname()
        infos = socket.getaddrinfo(hostname, None, family=socket.AF_INET)
        for info in infos:
            ip = info[4][0]
            if ip and not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass
    if not ips:
        ips.add("127.0.0.1")
    return sorted(ips)


class MiniCPM5GradioDemo:
    def __init__(self, hf_model: str, axmodel_path: str, max_seq_len: int = 2047):
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(hf_model, trust_remote_code=True)
        self.embeds = load_text_embeddings(axmodel_path, self.config)
        self.axmodel_path = axmodel_path
        self.max_seq_len = max_seq_len
        self.slice_len = 128

    def _eos_token_id(self):
        if isinstance(self.config.eos_token_id, list):
            return self.config.eos_token_id
        return None

    def _render_prompt(self, messages, enable_thinking: bool) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def generate_stream(
        self,
        messages,
        max_new_tokens: int = 512,
        enable_thinking: bool = False,
    ) -> Generator[str, None, None]:
        text = self._render_prompt(messages, enable_thinking=enable_thinking)
        inputs = self.tokenizer(text, return_tensors="np")
        token_ids = inputs["input_ids"][0].tolist()
        prefill_data = np.take(self.embeds, token_ids, axis=0).astype(bfloat16)

        runner = InferManager(self.config, self.axmodel_path, max_seq_len=self.max_seq_len)

        t0 = time.perf_counter()
        token_ids = runner.prefill(
            self.tokenizer,
            token_ids,
            prefill_data,
            slice_len=self.slice_len,
        )
        ttft_ms = (time.perf_counter() - t0) * 1000

        yield f"[TTFT: {ttft_ms:.1f} ms] "

        decode_start = time.perf_counter()
        decoded_count = 0
        for text_so_far in runner.decode_stream(
            self.tokenizer,
            token_ids,
            self.embeds,
            prefill_len=self.slice_len,
            eos_token_id=self._eos_token_id(),
            max_new_tokens=max_new_tokens,
        ):
            decoded_count += 1
            elapsed = time.perf_counter() - decode_start
            speed = decoded_count / elapsed if elapsed > 0 else 0
            yield f"[TTFT: {ttft_ms:.1f} ms | {speed:.1f} tok/s] {text_so_far}"

    def chat(self, message, history, system_prompt, enable_thinking):
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        for user_msg, bot_msg in history:
            messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})
        messages.append({"role": "user", "content": message})

        response = ""
        for chunk in self.generate_stream(messages, enable_thinking=enable_thinking):
            response = chunk
            yield response


def main():
    parser = argparse.ArgumentParser(description="MiniCPM5-1B Gradio Demo")
    parser.add_argument("--hf_model", default=default_hf_model(__file__))
    parser.add_argument("--axmodel_path", default=default_axmodel_path(__file__))
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo_engine = MiniCPM5GradioDemo(args.hf_model, args.axmodel_path)

    with gr.Blocks(title="MiniCPM5-1B on AX NPU") as demo:
        gr.Markdown("# MiniCPM5-1B on AX NPU")
        gr.Markdown("Developer-side text demo for MiniCPM5-1B.")

        system_prompt = gr.Textbox(label="System Prompt", value="")
        enable_thinking = gr.Checkbox(label="Enable Thinking", value=False)
        gr.ChatInterface(
            fn=demo_engine.chat,
            additional_inputs=[system_prompt, enable_thinking],
            title=None,
        )

    print(f"Starting Gradio server on port {args.port}")
    for ip in _list_host_ips():
        print(f"  http://{ip}:{args.port}")
    demo.launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
