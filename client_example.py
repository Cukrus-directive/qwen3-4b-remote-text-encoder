# example client - gets prompt embeds from encode-server.py, runs the pipeline on the gpu
#
#   python client_example.py zimage "a red fox standing in fresh snow" -o fox.png
#   python client_example.py klein "make the car red" --image car.png -o red.png

import argparse
import json
import os
import urllib.request

import torch
from safetensors.torch import load

ENCODER_URL = os.environ.get("ENCODER_URL", "http://localhost:8090/v1/encode")
MODELS = {
    "zimage": "unsloth/Z-Image-Turbo-unsloth-bnb-4bit",
    "klein": "cukrus/FLUX.2-klein-4B-bnb-nf4",
}


def get_embeds(prompt, mode):
    body = json.dumps({"prompt": prompt, "mode": mode}).encode()
    req = urllib.request.Request(ENCODER_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        print("encode:", r.headers["X-Encode-Seconds"], "s, cache", r.headers["X-Cache"])
        return load(r.read())["prompt_embeds"].to("cuda")


ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=MODELS)
ap.add_argument("prompt")
ap.add_argument("--image", action="append")  # klein only, up to 3
ap.add_argument("--seed", type=int, default=4200)
ap.add_argument("-o", "--out", default="out.png")
args = ap.parse_args()

emb = get_embeds(args.prompt, args.mode)
gen = torch.Generator("cuda").manual_seed(args.seed)

if args.mode == "zimage":
    from diffusers import ZImagePipeline
    pipe = ZImagePipeline.from_pretrained(MODELS["zimage"], text_encoder=None, tokenizer=None,
                                          torch_dtype=torch.bfloat16).to("cuda")
    # list with one (n, 2560) tensor per prompt
    img = pipe(prompt_embeds=[emb], num_inference_steps=9, guidance_scale=0.0, generator=gen).images[0]
else:
    from diffusers import Flux2KleinPipeline
    from diffusers.utils import load_image
    pipe = Flux2KleinPipeline.from_pretrained(MODELS["klein"], text_encoder=None, tokenizer=None,
                                              torch_dtype=torch.bfloat16).to("cuda")
    refs = [load_image(p) for p in args.image] if args.image else None
    # (512, 7680) -> (1, 512, 7680)
    img = pipe(prompt_embeds=emb[None], image=refs, num_inference_steps=4, guidance_scale=1.0,
               generator=gen).images[0]

img.save(args.out)
print("saved", args.out)
