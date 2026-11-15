"""A LoRA training checkpoint (DCP dir) -> a plain .pt of the base architecture with the adapters merged in
(W = W0 + alpha / r * B @ A), loadable by the servers with the non-LoRA experiment name.

    python scripts/merge_lora_dcp.py <.../checkpoints/iter_000002000> <out.pt> [--alpha 32 --rank 32]
"""
import argparse
import os
import tempfile

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="checkpoints/iter_<n> (its model/ subdir is read)")
    ap.add_argument("out")
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--rank", type=int, default=32)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(a.out))) as d:
        tmp = os.path.join(d, "raw.pt")
        dcp_to_torch_save(os.path.join(a.ckpt, "model"), tmp)
        sd = torch.load(tmp, map_location="cpu", weights_only=False)
    if "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    scale = a.alpha / a.rank
    plain_keys = {k.replace("base_model.model.", "").replace("base_layer.", ""): k for k in sd}
    out, merged = {}, 0
    for k, v in sd.items():
        if "lora_A" in k or "lora_B" in k:
            continue
        plain = k.replace("base_model.model.", "").replace("base_layer.", "")
        pre = plain[: -len("weight")] if plain.endswith(".weight") else None
        if pre is not None and (pre + "lora_A.default.weight") in plain_keys:
            A = sd[plain_keys[pre + "lora_A.default.weight"]].float()
            B = sd[plain_keys[pre + "lora_B.default.weight"]].float()
            v = (v.float() + scale * (B @ A)).to(v.dtype)
            merged += 1
        out[plain] = v
    torch.save(out, a.out)
    print(f"merged {merged} LoRA layers, {len(out)} keys -> {a.out}")


if __name__ == "__main__":
    main()
