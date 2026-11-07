"""Humanoid-Everyday (HE) offline-eval dataset — mid-V3 full-frame, STATE-CHUNK.

HE 는 EgoVerse 매니페스트 포맷(windows.npz + pairs.npz)을 그대로 쓰고 zarr 루트도
``EgoVerseRetDataset._grp`` 이 ``emb == "he"`` 로 이미 HE_ROOT 로 라우팅한다. 그래서
소스 플러밍을 새로 짤 필요가 없고, ``EgoVerseRetV3StateDataset`` 에서 바뀌는 것은
경로 기본값과 **텍스트 조건 alias** 뿐이다.

EgoDex eval(``EgoDexRetV3StateDataset``)과 동일한 규약:
  * ``emb = "he"``   — HE 네이티브 robust state57 통계로 정규화 (held-out embodiment)
  * ``cls = "other"`` -> "manipulating objects [he]" -> 학습된 ``[human]`` 임베딩으로 alias.
    (v10 t5_embeddings.pkl 에 ``[he]`` 키가 없으므로 alias 가 없으면 T5 가 0 이 된다 —
    cotrain 이 그랬다. 평가에서는 egodex 와 맞춰 human 텍스트를 준다.)

페어는 ``egoverse_pair_builder/168_he_match.py`` 산출 (태스크명 정확 일치 게이트 +
banded-DTW rerank, 같은 에피소드 제외). 매니페스트는 ``169_he_manifest.py`` 가 만든다.
"""
from __future__ import annotations

from cosmos_policy.datasets.egoverse_ret_dataset import CLASS_TO_COMMAND
from cosmos_policy.datasets.egoverse_ret_v3_state_dataset import EgoVerseRetV3StateDataset

HE_MANIFEST = "/mnt/ddn/dataset/egoverse_pairs_he"
HE_T5_EMB = "/mnt/ddn/dataset/egoverse_pairs_he/t5_embeddings.pkl"
HE_ACT_STATS = "/mnt/ddn/dataset/egoverse_pairs_he/action57_stats.json"

HE_EMB = "he"        # 정규화 키 (state57_stats_staterobust.json 의 'he')
TEXT_EMB = "human"   # 텍스트 조건용 학습된 태그 (HE ~ 일반 human)


class HeRetV3StateDataset(EgoVerseRetV3StateDataset):
    """HE eval in the v3 full-frame STATE-chunk layout — see module doc."""

    def __init__(
        self,
        manifest_dir: str = HE_MANIFEST,
        t5_text_embeddings_path: str = HE_T5_EMB,
        action_stats_path: str = HE_ACT_STATS,
        use_image_aug: bool = False,          # eval: no aug
        use_stronger_image_aug: bool = False,
        retrieval_dropout_prob: float = 0.0,
        **kw,
    ):
        super().__init__(
            manifest_dir=manifest_dir,
            t5_text_embeddings_path=t5_text_embeddings_path,
            action_stats_path=action_stats_path,
            use_image_aug=use_image_aug,
            use_stronger_image_aug=use_stronger_image_aug,
            retrieval_dropout_prob=retrieval_dropout_prob,
            **kw,
        )
        # "<cmd> [he]" -> 학습된 "<cmd> [human]" 임베딩 (T5 재계산 불필요)
        n_alias = 0
        for cmd in CLASS_TO_COMMAND.values():
            human_key = f"{cmd} [{TEXT_EMB}]"
            if human_key in self.t5_text_embeddings:
                self.t5_text_embeddings[f"{cmd} [{HE_EMB}]"] = self.t5_text_embeddings[human_key]
                n_alias += 1
        print(f"[HeRetV3State] windows={len(self.wep)} queries={self.N} "
              f"t5_alias[{HE_EMB}]<-[{TEXT_EMB}]={n_alias} "
              f"state_stats_keys={list(self.state_stats.keys())}", flush=True)


if __name__ == "__main__":
    ds = HeRetV3StateDataset(num_frames=92, final_image_size=448)
    print("len", len(ds))
    s = ds[0]
    print("video", s["video"].shape, "actions(states)", s["actions"].shape,
          "ret_actions(states)", s["retrieved_actions"].shape,
          "proprio(past chunk)", s["proprio"].shape)
    print("T", len(s["slot_roles"]), "ncond", s["num_conditional_frames"],
          "emb", s["embodiment"], "ret", s["retrieved_embodiment"],
          "t5", tuple(s["t5_text_embeddings"].shape),
          "t5_nonzero", float(s["t5_text_embeddings"].abs().sum()) > 0)
