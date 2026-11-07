# SPDX-License-Identifier: Apache-2.0
"""Precompute T5 text embeddings for the Tabletop-Sim post-train manifest.

Reads ``<manifest>/tasks.json`` (written by ``convert_tabletop_to_egoverse57.py``) and
embeds every instruction it lists, PLUS the strings the simulator emits at eval time that
the released dataset never contains.

That second set is the point of this script. ``TabletopSimShortStateDataset`` keys T5 by
the instruction, and the base loader falls back to ``zeros(512, 1024)`` on a miss — i.e.
a missing embedding does not crash, it silently drops language conditioning. RoboCasa hit
the same class of bug from the other direction (see docs/robocasa_all24_t5.md §0), so the
union is precomputed once here and the dataset's ``require_t5`` guard checks coverage at
construction.

Two mismatches exist between the dataset and the sim source:

* ``aloha_lift_box`` — dataset says "Put ball top of the box and lift the box" but
  ``LiftBox.instruction`` in the sim is "Lift the box with the front facing the camera".
  An eval that reads ``env.task.get_instruction()`` would produce an unseen string.
* ``aloha_shoes_table_new`` — "brown shoes" where training has "black shoes".

``aloha_dish_drainer_new`` and ``aloha_handover_box_new`` reuse their base task's string
exactly, so they need nothing extra.

Usage::

    uv run --extra cu128 --group robocasa --python 3.10 \
        -m cosmos_policy.datasets.save_tabletop_sim_t5_text_embeddings \
        --manifest_dir /mnt/ddn/dataset/tabletop_ego57/manifest
"""
from __future__ import annotations

import argparse
import json
import os

from cosmos_policy.datasets.t5_embedding_utils import generate_t5_embeddings, save_embeddings

# Sourced from tabletop/aloha_env.py in jellyho/Tabletop-Sim @1d4b895:
#   LiftBox:256 / LiftBoxNew:354, ShoesTableNew:392
OOD_INSTRUCTIONS = [
    "Lift the box with the front facing the camera",
    "Pick up the brown shoes and put them side by side on the purple table",
]

# TwinVLA (arXiv 2511.05275) 의 EASY 프로토콜: "Test scenes match the training data, but the
# instructions are novel." 논문은 사용한 문장을 공개하지 않으므로 여기서 새로 쓴다. 의미는
# 학습 문장과 동일하고 표현만 다르며, 학습 데이터에 등장하지 않는다(=novel).
# 학습 데이터의 고유 instruction 은 task 당 정확히 하나뿐이라(52,043 윈도우 / 10 문장)
# 데이터 안에서 가져올 수 있는 paraphrase 가 없다.
PARAPHRASE_INSTRUCTIONS = [
    # aloha_dish_drainer  <- "Pick up the dish and put on to the drainer"
    "Place the plate onto the dish rack",
    # aloha_handover_box  <- "Handover the box and place into the pink basket"
    "Pass the box from one arm to the other and drop it into the pink basket",
    # aloha_shoes_table   <- "Pick up the black shoes and put them side by side on the purple table"
    "Arrange the black shoes next to each other on the purple table",
    # aloha_lift_box      <- "Put ball top of the box and lift the box"
    #   (sim 이 내는 "Lift the box with the front facing the camera" 는 위 OOD 목록에 이미 있다)
    "Lift the box up with the ball resting on top",
    # aloha_box_into_pot_easy <- "Put the {color} box into the {color} pot" x6
    "Place the brown box in the blue pot",
    "Place the brown box in the green pot",
    "Place the white box in the blue pot",
    "Place the white box in the green pot",
    "Place the yellow box in the blue pot",
    "Place the yellow box in the green pot",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute T5 embeddings for Tabletop-Sim.")
    p.add_argument("--manifest_dir", type=str,
                   default="/mnt/ddn/dataset/tabletop_ego57/manifest")
    p.add_argument("--out_dir", type=str, default=None,
                   help="where to write t5_embeddings.pkl (default: manifest_dir)")
    p.add_argument("--no_ood", action="store_true",
                   help="embed only the instructions present in the training data")
    p.add_argument("--no_paraphrase", action="store_true",
                   help="skip the EASY-protocol paraphrases (see PARAPHRASE_INSTRUCTIONS)")
    p.add_argument("--extra", nargs="*", default=[],
                   help="additional instruction strings to embed")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(os.path.join(args.manifest_dir, "tasks.json")) as fp:
        tasks = json.load(fp)

    train = sorted({s for v in tasks.values() for s in v})
    ood = [] if args.no_ood else [s for s in OOD_INSTRUCTIONS if s not in train]
    para = ([] if args.no_paraphrase
            else [s for s in PARAPHRASE_INSTRUCTIONS if s not in train and s not in ood])
    extra = [s for s in args.extra if s not in train and s not in ood and s not in para]
    commands = sorted(set(train) | set(ood) | set(para) | set(extra))

    print(f"{len(train)} instruction(s) from {len(tasks)} training task(s):")
    for t, v in tasks.items():
        for s in v:
            print(f"   [{t}] {s!r}")
    if ood:
        print(f"{len(ood)} eval-only (OOD) instruction(s):")
        for s in ood:
            print(f"   [ood] {s!r}")
    if para:
        print(f"{len(para)} EASY paraphrase(s):")
        for s_ in para:
            print(f"   [para] {s_!r}")
    if extra:
        print(f"{len(extra)} extra instruction(s): {extra}")
    print(f"-> embedding {len(commands)} unique strings")

    emb = generate_t5_embeddings(commands)
    out_dir = args.out_dir or args.manifest_dir
    os.makedirs(out_dir, exist_ok=True)
    path = save_embeddings(emb, out_dir)

    missing = [c for c in commands if c not in emb]
    shape = tuple(next(iter(emb.values())).shape)
    print(f"\n[done] {len(emb)} embeddings {shape} -> {path}  (missing: {missing})")


if __name__ == "__main__":
    main()
