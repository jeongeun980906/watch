"""Episode text embeddings for the same-source gate of the matcher (02_ep_lang_act.py emb_text, verbatim):
T5-11B encoder, fp16, attention-masked mean pool, L2-normalized, max_length 64, on each episode's raw
task_description (zarr.json attributes of the episode). Needs one GPU with ~24 GB; ~2 min for 6.7K unique texts.

usage (watch venv: torch + transformers): t5_text.py --root <v10-state episodes> --out temb.npz
"""
import argparse, json, os, time

import numpy as np
import torch
from transformers import T5EncoderModel, T5TokenizerFast


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="google-t5/t5-11b")
    a = ap.parse_args()
    eps = sorted(os.listdir(a.root)); texts = []
    for e in eps:
        attrs = json.load(open(f"{a.root}/{e}/zarr.json")).get("attributes", {})
        texts.append(str(attrs.get("task_description", "")))
    t0 = time.time()
    tok = T5TokenizerFast.from_pretrained(a.model)
    enc = T5EncoderModel.from_pretrained(a.model, torch_dtype=torch.float16).to("cuda").eval()
    ut = sorted(set(texts)); out = []
    for i in range(0, len(ut), 64):
        b = tok(ut[i:i + 64], padding=True, truncation=True, max_length=64, return_tensors="pt").to("cuda")
        with torch.no_grad(): o = enc(**b)
        m = b["attention_mask"].unsqueeze(-1).half(); v = (o.last_hidden_state * m).sum(1) / m.sum(1)
        out.append(torch.nn.functional.normalize(v.float(), dim=1).cpu().numpy())
    U = np.concatenate(out); ix = {t: k for k, t in enumerate(ut)}
    np.savez(a.out, ep=np.array(eps), emb=np.stack([U[ix[t]] for t in texts]).astype(np.float32), text=np.array(texts))
    print(f"{len(eps)} episodes, {len(ut)} unique texts, {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
