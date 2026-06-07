import os
import re
from typing import Tuple

import numpy as np
from axengine import InferenceSession
from ml_dtypes import bfloat16
from tqdm import tqdm


def _find_axmodel_files(base_dir: str, expected_layers: int = None):
    files = os.listdir(base_dir)
    layer_pattern = re.compile(r"^(?P<prefix>.*)_p(?P<prefill>\d+)_l(?P<idx>\d+)_together\.axmodel$")
    post_pattern = re.compile(r"^(?P<prefix>.*)_post\.axmodel$")

    prefix_map = {}
    for fname in files:
        matched = layer_pattern.match(fname)
        if matched:
            prefix = matched.group("prefix")
            idx = int(matched.group("idx"))
            prefix_map.setdefault(prefix, []).append((idx, fname))

    if not prefix_map:
        raise FileNotFoundError(f"No decoder axmodel files found in {base_dir}")

    prefix = max(prefix_map.items(), key=lambda kv: len(kv[1]))[0]
    layer_files = sorted(prefix_map[prefix], key=lambda item: item[0])

    if expected_layers is not None and len(layer_files) != expected_layers:
        print(
            f"[WARN] Detected {len(layer_files)} decoder layers for prefix {prefix}, "
            f"but config expects {expected_layers}."
        )

    post_file = None
    for fname in files:
        matched = post_pattern.match(fname)
        if matched and matched.group("prefix") == prefix:
            post_file = fname
            break

    if post_file is None:
        raise FileNotFoundError(f"Cannot find post axmodel for prefix {prefix} in {base_dir}")

    return layer_files, post_file, prefix


class InferManager:
    def __init__(self, config, model_dir, max_seq_len=2047):
        self.config = config
        self.max_seq_len = max_seq_len
        self.sub_dim = config.hidden_size // config.num_attention_heads if not config.head_dim else config.head_dim
        self.kv_dim = self.sub_dim * config.num_key_value_heads

        self.k_caches = [
            np.zeros((1, self.max_seq_len, self.kv_dim), dtype=bfloat16)
            for _ in range(config.num_hidden_layers)
        ]
        self.v_caches = [
            np.zeros((1, self.max_seq_len, self.kv_dim), dtype=bfloat16)
            for _ in range(config.num_hidden_layers)
        ]

        layer_files, post_file, prefix = _find_axmodel_files(model_dir, config.num_hidden_layers)
        print(f"[INFO] Detected decoder prefix: {prefix}")

        self.decoder_sessions = []
        for _, fname in tqdm(layer_files, desc="Init InferenceSession"):
            self.decoder_sessions.append(InferenceSession(os.path.join(model_dir, fname)))

        self.post_process_session = InferenceSession(os.path.join(model_dir, post_file))
        print("Model loaded successfully!")

    def _input_meta(self, session, shape_group: int):
        return {item.name: tuple(item.shape) for item in session.get_inputs(shape_group=shape_group)}

    def _decoder_output_names(self, shape_group: int) -> Tuple[str, str, str]:
        # The current AX650 MiniCPM5 exports keep stable output names across all
        # shape groups, so do not synthesize group-specific suffixes here.
        return ("K_cache_out", "V_cache_out", "output")

    def _run_decoder(self, session, input_feed, shape_group: int):
        names = self._decoder_output_names(shape_group)
        outputs = None

        try:
            outputs = session.run(list(names), input_feed, shape_group=shape_group)
        except (TypeError, ValueError):
            try:
                outputs = session.run(list(names), input_feed, shape_group)
            except (TypeError, ValueError):
                outputs = session.run(None, input_feed, shape_group=shape_group)

        if isinstance(outputs, dict):
            return outputs[names[0]], outputs[names[1]], outputs[names[2]]

        if isinstance(outputs, (list, tuple)):
            if len(outputs) == 3:
                return outputs[0], outputs[1], outputs[2]
            offset = shape_group * 3
            if len(outputs) >= offset + 3:
                return outputs[offset], outputs[offset + 1], outputs[offset + 2]
            return outputs[0], outputs[1], outputs[2]

        return outputs[0], outputs[1], outputs[2]

    @staticmethod
    def _top_p(probs: np.ndarray, p: float) -> np.ndarray:
        sorted_indices = np.argsort(probs)
        filtered = probs.copy()
        cumulative = 0.0
        for idx in sorted_indices[::-1]:
            if cumulative >= p:
                filtered[idx] = 0
            cumulative += filtered[idx]
        return filtered / cumulative

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - logits.max()
        exp_logits = np.exp(logits)
        return (exp_logits / np.sum(exp_logits)).astype(np.float64)

    def post_process(
        self,
        logits,
        top_k=1,
        top_p=0.95,
        temperature=0.7,
        repetition_penalty=1.0,
        token_ids=None,
    ):
        logits = logits.astype(np.float32).flatten()
        if repetition_penalty is not None and repetition_penalty != 1.0 and token_ids:
            for token_id in set(token_ids):
                if 0 <= token_id < logits.size:
                    if logits[token_id] < 0:
                        logits[token_id] *= repetition_penalty
                    else:
                        logits[token_id] /= repetition_penalty

        top_k = max(1, min(int(top_k), logits.size))
        temperature = max(float(temperature), 1e-6)
        top_p = min(max(float(top_p), 1e-6), 1.0)

        candidate_indices = np.argpartition(logits, -top_k)[-top_k:]
        candidate_logits = logits[candidate_indices] / temperature
        candidate_probs = self._softmax(candidate_logits)
        candidate_probs = self._top_p(candidate_probs, top_p)
        candidate_probs = candidate_probs.astype(np.float64) / candidate_probs.sum()
        chosen_idx = np.random.multinomial(1, candidate_probs).argmax()
        next_token = candidate_indices[chosen_idx]
        return next_token

    def prefill(
        self,
        tokenizer,
        token_ids,
        embed_data,
        slice_len=128,
        top_k=1,
        top_p=0.95,
        temperature=0.7,
        repetition_penalty=1.0,
    ):
        seq_len = len(token_ids)
        slice_indices = [i for i in range(seq_len // slice_len + 1)]
        total_prefill_len = slice_len * (slice_indices[-1] + 1)
        if total_prefill_len <= 0:
            return token_ids

        for slice_idx in slice_indices:
            indices = np.arange(
                slice_idx * slice_len,
                (slice_idx + 1) * slice_len,
                dtype=np.uint32,
            ).reshape(1, -1)

            mask = np.zeros((1, slice_len, slice_len * (slice_idx + 1))) - 65536
            data = np.zeros((1, slice_len, self.config.hidden_size)).astype(bfloat16)

            for i, token_pos in enumerate(range(slice_idx * slice_len, (slice_idx + 1) * slice_len)):
                if token_pos < len(token_ids):
                    mask[:, i, : slice_idx * slice_len + i + 1] = 0
                    data[:, i : i + 1, :] = embed_data[token_pos].reshape((1, 1, self.config.hidden_size)).astype(
                        bfloat16
                    )

            remain_len = seq_len - slice_idx * slice_len if slice_idx == slice_indices[-1] else slice_len
            mask = mask.astype(bfloat16)

            for layer_idx in range(self.config.num_hidden_layers):
                input_meta = self._input_meta(self.decoder_sessions[layer_idx], slice_idx + 1)
                input_feed = {
                    "K_cache": (
                        self.k_caches[layer_idx][:, 0 : slice_len * slice_idx, :]
                        if slice_idx
                        else np.zeros(input_meta["K_cache"], dtype=bfloat16)
                    ),
                    "V_cache": (
                        self.v_caches[layer_idx][:, 0 : slice_len * slice_idx, :]
                        if slice_idx
                        else np.zeros(input_meta["V_cache"], dtype=bfloat16)
                    ),
                    "indices": indices,
                    "input": data,
                    "mask": mask,
                }
                k_out, v_out, data = self._run_decoder(
                    self.decoder_sessions[layer_idx],
                    input_feed,
                    shape_group=slice_idx + 1,
                )
                self.k_caches[layer_idx][:, slice_idx * slice_len : slice_idx * slice_len + remain_len, :] = k_out[
                    :, :remain_len, :
                ]
                self.v_caches[layer_idx][:, slice_idx * slice_len : slice_idx * slice_len + remain_len, :] = v_out[
                    :, :remain_len, :
                ]

        post_out = self.post_process_session.run(
            None,
            {"input": data[:, seq_len - (len(slice_indices) - 1) * slice_len - 1, None, :]},
        )[0]
        next_token = self.post_process(
            post_out,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            token_ids=token_ids,
        )
        token_ids.append(next_token)
        return token_ids

    def decode(
        self,
        tokenizer,
        token_ids,
        embed_matrix,
        prefill_len=128,
        eos_token_id=None,
        stream=True,
        top_k=1,
        top_p=0.95,
        temperature=0.7,
        repetition_penalty=1.0,
        max_new_tokens=None,
        stream_callback=None,
    ):
        decoded_text = tokenizer.decode(token_ids[-1], skip_special_tokens=True)
        if stream:
            print("answer >>", decoded_text, end="", flush=True)
        if stream_callback is not None:
            stream_callback(decoded_text)

        mask = np.zeros((1, 1, self.max_seq_len + 1), dtype=np.float32).astype(bfloat16)
        mask[:, :, : self.max_seq_len] -= 65536
        seq_len = len(token_ids) - 1
        if prefill_len > 0:
            mask[:, :, :seq_len] = 0

        max_new_tokens = self.max_seq_len if max_new_tokens is None else int(max_new_tokens)
        generated = 0

        for step_idx in range(self.max_seq_len):
            if prefill_len > 0 and step_idx < seq_len:
                continue

            cur_token = token_ids[step_idx]
            indices = np.array([step_idx], np.uint32).reshape((1, 1))
            data = embed_matrix[cur_token, :].reshape((1, 1, self.config.hidden_size)).astype(bfloat16)

            for layer_idx in range(self.config.num_hidden_layers):
                input_feed = {
                    "K_cache": self.k_caches[layer_idx],
                    "V_cache": self.v_caches[layer_idx],
                    "indices": indices,
                    "input": data,
                    "mask": mask,
                }
                k_out, v_out, data = self._run_decoder(self.decoder_sessions[layer_idx], input_feed, shape_group=0)
                self.k_caches[layer_idx][:, step_idx, :] = k_out[:, :, :]
                self.v_caches[layer_idx][:, step_idx, :] = v_out[:, :, :]

            mask[..., step_idx] = 0
            if step_idx < seq_len - 1:
                continue

            post_out = self.post_process_session.run(None, {"input": data})[0]
            next_token = self.post_process(
                post_out,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                token_ids=token_ids,
            )

            if eos_token_id is not None and next_token in eos_token_id:
                break
            if next_token == tokenizer.eos_token_id:
                break

            token_ids.append(next_token)
            generated += 1
            if generated >= max_new_tokens:
                break

            decoded_piece = tokenizer.decode(next_token, skip_special_tokens=True)
            decoded_text += decoded_piece
            if stream:
                print(decoded_piece, end="", flush=True)
            if stream_callback is not None:
                stream_callback(decoded_text)

        return decoded_text

    def decode_stream(
        self,
        tokenizer,
        token_ids,
        embed_matrix,
        prefill_len=128,
        eos_token_id=None,
        top_k=1,
        top_p=0.95,
        temperature=0.7,
        repetition_penalty=1.0,
        max_new_tokens=None,
    ):
        decoded_text = tokenizer.decode(token_ids[-1], skip_special_tokens=True)
        yield decoded_text

        mask = np.zeros((1, 1, self.max_seq_len + 1), dtype=np.float32).astype(bfloat16)
        mask[:, :, : self.max_seq_len] -= 65536
        seq_len = len(token_ids) - 1
        if prefill_len > 0:
            mask[:, :, :seq_len] = 0

        max_new_tokens = self.max_seq_len if max_new_tokens is None else int(max_new_tokens)
        generated = 0

        for step_idx in range(self.max_seq_len):
            if prefill_len > 0 and step_idx < seq_len:
                continue

            cur_token = token_ids[step_idx]
            indices = np.array([step_idx], np.uint32).reshape((1, 1))
            data = embed_matrix[cur_token, :].reshape((1, 1, self.config.hidden_size)).astype(bfloat16)

            for layer_idx in range(self.config.num_hidden_layers):
                input_feed = {
                    "K_cache": self.k_caches[layer_idx],
                    "V_cache": self.v_caches[layer_idx],
                    "indices": indices,
                    "input": data,
                    "mask": mask,
                }
                k_out, v_out, data = self._run_decoder(self.decoder_sessions[layer_idx], input_feed, shape_group=0)
                self.k_caches[layer_idx][:, step_idx, :] = k_out[:, :, :]
                self.v_caches[layer_idx][:, step_idx, :] = v_out[:, :, :]

            mask[..., step_idx] = 0
            if step_idx < seq_len - 1:
                continue

            post_out = self.post_process_session.run(None, {"input": data})[0]
            next_token = self.post_process(
                post_out,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                token_ids=token_ids,
            )

            if eos_token_id is not None and next_token in eos_token_id:
                break
            if next_token == tokenizer.eos_token_id:
                break

            token_ids.append(next_token)
            generated += 1
            if generated >= max_new_tokens:
                break

            decoded_text += tokenizer.decode(next_token, skip_special_tokens=True)
            yield decoded_text
