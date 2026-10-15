#!/usr/bin/env python3
# remote qwen3-4b text encoder for z-image-turbo and flux.2-klein-4b (cpu)
#
# POST /v1/encode {"prompt": "...", "mode": "zimage" | "klein"}
#   returns safetensors with one bf16 tensor "prompt_embeds"
#   zimage: (n_tokens, 2560)  hidden_states[-2], thinking on, no padding
#   klein:  (512, 7680)       hidden_states 9/18/27 stacked, thinking off, padded to 512
# GET /health
#
# usage: python encode-server.py --model /path/to/model --threads 6
#   model dir needs text_encoder/ and tokenizer/ (same files in both model repos)

import argparse
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from safetensors.torch import save
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger("imgenc")
KLEIN_LAYERS = (9, 18, 27)
MODES = ("zimage", "klein")
CACHE_MAX_BYTES = 512 << 20  # klein embeds are 7.5mb each


class Encoder:
    def __init__(self, path, threads, cache_size):
        torch.set_num_threads(threads)
        self.tok = AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
        model = AutoModelForCausalLM.from_pretrained(os.path.join(path, "text_encoder"),
                                                     dtype=torch.bfloat16, low_cpu_mem_usage=True)
        self.base = model.model.eval()
        del model.lm_head  # not needed, only hidden states

        # weights stay bf16, matmuls run in fp32 (bf16 is emulated on cpus without avx512-bf16)
        def fp32_linear(mod, x):
            b = None if mod.bias is None else mod.bias.float()
            return torch.nn.functional.linear(x, mod.weight.float(), b)

        for m in self.base.modules():
            if isinstance(m, torch.nn.Linear):
                m.forward = fp32_linear.__get__(m)
            elif m is not self.base and not isinstance(m, torch.nn.Embedding) and not list(m.children()):
                m.float()  # norms etc, tiny
        self.base.embed_tokens.register_forward_hook(lambda mod, inp, out: out.float())

        self.lock = threading.Lock()
        self.cache = OrderedDict()
        self.cache_size = cache_size

    @torch.inference_mode()
    def _encode(self, prompt, mode):
        klein = mode == "klein"
        text = self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=not klein)
        if not klein:
            ids = self.tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids
            hs = self.base(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
            return hs[-2][0].to(torch.bfloat16).contiguous()

        # klein: pad to 512 and keep the pads, the dit sees them
        t = self.tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        hs = self.base(input_ids=t.input_ids, attention_mask=t.attention_mask,
                       output_hidden_states=True, use_cache=False).hidden_states
        out = torch.stack([hs[k][0] for k in KLEIN_LAYERS], dim=0)  # (3, 512, 2560)
        return out.permute(1, 0, 2).reshape(512, -1).to(torch.bfloat16).contiguous()

    def encode(self, prompt, mode="zimage"):
        key = (mode, prompt)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key], 0.0, True
            t0 = time.perf_counter()
            emb = self._encode(prompt, mode)
            dt = time.perf_counter() - t0
            self.cache[key] = emb
            while (len(self.cache) > self.cache_size
                   or sum(e.nbytes for e in self.cache.values()) > CACHE_MAX_BYTES):
                self.cache.popitem(last=False)
            return emb, dt, False


def make_handler(enc):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json", headers=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                body = {"status": "ok", "cached_prompts": len(enc.cache), "threads": torch.get_num_threads()}
                self._send(200, json.dumps(body).encode())
            else:
                self._send(404, b'{"error": "not found"}')

        def do_POST(self):
            if self.path != "/v1/encode":
                return self._send(404, b'{"error": "not found"}')
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                prompt = req["prompt"]
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("prompt must be a non-empty string")
                mode = req.get("mode", "zimage")
                if mode not in MODES:
                    raise ValueError(f"mode must be one of {MODES}")
            except Exception as e:
                return self._send(400, json.dumps({"error": str(e)}).encode())

            emb, dt, hit = enc.encode(prompt, mode)
            log.info("encode mode=%s tokens=%d %.2fs cache=%s", mode, emb.shape[0], dt, "hit" if hit else "miss")
            self._send(200, save({"prompt_embeds": emb}), "application/octet-stream",
                       {"X-Encode-Seconds": f"{dt:.3f}", "X-Tokens": str(emb.shape[0]),
                        "X-Cache": "hit" if hit else "miss"})

        def log_message(self, fmt, *args):
            log.debug(fmt, *args)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--cache", type=int, default=256, help="max prompts in cache")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    t0 = time.perf_counter()
    enc = Encoder(args.model, args.threads, args.cache)
    enc.encode("warmup")  # load the mmapped weights before first request
    enc.cache.clear()
    log.info("ready in %.1fs, listening on %s:%d", time.perf_counter() - t0, args.host, args.port)
    ThreadingHTTPServer((args.host, args.port), make_handler(enc)).serve_forever()


if __name__ == "__main__":
    main()
