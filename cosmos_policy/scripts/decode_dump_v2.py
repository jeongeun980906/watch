"""Stage 1 (.venv) for the mid-v2 model (compact layout, slot_roles + t_positions):
for a checkpoint x dataset, generate + decode GEN/GT query video + gen action, FK the gen
action to 3D head-frame wrists, and dump ONE compressed npz per query window — SAME npz
contract as decode_dump.py (gen/gt JPEG-bytes, h, w, fkL, fkR, emb, ret, idx), so stage 2
(/mnt/ddn/tmp/wilor_metric.py, wilor_env) runs unmodified.

v2 deltas vs decode_dump.py:
  * per-sample geometry — k demo latents / g query latents from the sample itself
    (k = num_conditional_frames-3, g = future_video_end - future_video_start);
    pixel packing [1 | 4 | 4k | 4 | 4 | 4g] -> query clip = pixels [13+4k, 13+4k+4g);
    action latent = 3+k; generated action chunk is NATIVE length 4g (92 human / 60 MD).
  * image aug: the v2 dataset hardcodes aug ON in __getitem__ — disabled here via a
    module-level preprocess_image wrap (eval must be aug-free, like the old eval datasets).
  * demo choice: v2 samples a random demo among the row's embodiment-locked matches;
    seeded per row here so shards/reruns are reproducible.
  * --egodex swaps in EgoDexRetV2Dataset (no egodex v2 experiment config exists;
    the model config stays cosmos_predict2p5_2b_egoverse_ret_v2).
"""
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, "/mnt/ddn/cosmos_policy")
from cosmos_policy.scripts.eval_egoverse_ret import _fk_skeleton, _vid_uint8, _load_stats, _to_cuda, ACTION_DIM
from cosmos_policy.datasets.state_action_convert import (
    derived_action57_from_states, fk_skeleton_state as _fk_skeleton_state)
from cosmos_policy._src.imaginaire.lazy_config import instantiate
from cosmos_policy._src.imaginaire.utils import distributed
from cosmos_policy._src.imaginaire.utils.context_managers import distributed_init
from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint
from cosmos_policy.models.policy_text2world_model import extract_action_chunk_from_latent
from cosmos_policy.scripts.decode_dump import jpg

# eval: NO image aug (v2 __getitem__ hardcodes aug=True for training; do not modify it)
import cosmos_policy.datasets.egoverse_ret_v2_dataset as _v2mod
_orig_pp = _v2mod.preprocess_image
_v2mod.preprocess_image = lambda imgs, **kw: _orig_pp(
    imgs, **{**kw, "use_image_aug": False, "stronger_image_aug": False})


def sample_geometry(s):
    """(k, g, query-pixel slice, action latent idx, native action len) from a v2 sample."""
    k = int(s["num_conditional_frames"]) - 3
    g = int(s["future_video_end_latent_idx"]) - int(s["future_video_start_latent_idx"])
    assert s["actions"].shape[0] == 4 * g, (s["actions"].shape, k, g)
    return k, g, slice(13 + 4 * k, 13 + 4 * k + 4 * g), 3 + k, 4 * g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--weights_pt", required=True)
    ap.add_argument("--act_stats", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--per-episode", dest="per_episode", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=0); ap.add_argument("--num_steps", type=int, default=35)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--num-shards", dest="num_shards", type=int, default=1)
    ap.add_argument("--egodex", action="store_true", help="EgoDexRetV2Dataset instead of the config dataset")
    ap.add_argument("--state_stats", default=None,
                    help="state57 stats json to use instead of the dataset's (eval on the TEST manifest "
                         "must keep the TRAIN split's state normalization: egoverse_pairs_v10/state57_stats.json)")
    ap.add_argument("opts", nargs=argparse.REMAINDER)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    with distributed_init():
        distributed.init()
    _opts = [o for o in a.opts if o != "--"]
    _exp = next((o.split("=", 1)[1] for o in _opts if o.startswith("experiment=")), None)
    _eo = [o for o in _opts if not o.startswith("experiment=")]
    model, config = load_model_from_checkpoint(experiment_name=_exp, s3_checkpoint_dir=a.weights_pt,
                                               config_file=a.config, enable_fsdp=False, load_ema_to_reg=False, experiment_opts=_eo)
    model = model.cuda().eval(); tk = model.tensor_kwargs
    if a.egodex:
        from cosmos_policy.datasets.egodex_ret_v2_dataset import EgoDexRetV2Dataset
        ds = EgoDexRetV2Dataset()
    else:
        ds = instantiate(config.dataloader_train.dataset)
    if a.state_stats:
        ds.state_stats = ds._load_or_compute_state_stats(a.state_stats)
    stats = _load_stats(a.act_stats)
    # STATE-chunk runs (experiment name contains "_state"): chunks are ABSOLUTE state57;
    # FK composes per frame (no delta integration) with the dataset's (train) STATE stats,
    # and the npz gen_a/gt_a keys carry the DERIVED delta action57 (integrator inverse) so
    # action_metrics_v2 scores the SAME convention as delta runs.
    _state = "_state" in config.job.name
    _sst = (lambda e: (lambda st: (st["lo"], st["hi"]))(
        ds.state_stats.get(str(e), ds.state_stats["_global"]))) if _state else None

    # enumerate query rows: per-episode subsample + episode cap + shard (EXACT mirror of decode_dump)
    from collections import OrderedDict
    N = len(ds.query_idx); ep_of_row = ds.wep[ds.query_idx]
    by_ep = OrderedDict()
    for r in range(N):
        by_ep.setdefault(str(ep_of_row[r]), []).append(r)
    K = int(a.per_episode); rows = []
    for ridx in by_ep.values():
        if K <= 0 or K >= len(ridx):
            rows.extend(ridx)
        else:
            st = len(ridx) / float(K); rows.extend(ridx[int(j * st)] for j in range(K))
    if a.episodes > 0 and a.episodes < len(rows):
        step = len(rows) / float(a.episodes); rows = [rows[int(i * step)] for i in range(a.episodes)]
    rows = rows[a.shard::a.num_shards]
    print(f"shard {a.shard}/{a.num_shards}: {len(rows)} query windows (v2 compact, N={N})", flush=True)

    for j, r in enumerate(rows):
        idx = int(r)
        np.random.seed((1000003 * (idx + 1)) % (2**31 - 1))     # reproducible demo pick
        s = ds[idx]; db = _to_cuda(s, tk); emb = s["embodiment"]; ret = s["retrieved_embodiment"]
        k, g, qry, ai, nf = sample_geometry(s)
        with torch.no_grad(), torch.autocast("cuda", dtype=tk.get("dtype", torch.bfloat16)):
            raw, x0, _ = model.get_data_and_condition(db)
            lat = model.generate_samples_from_batch(db, guidance=1.5, state_shape=x0.shape[1:], n_sample=x0.shape[0],
                                                    num_steps=a.num_steps, shift=getattr(model.config, "shift", 1.0))
            gv = model.decode(lat)
            gen_a = extract_action_chunk_from_latent(lat[:, :, ai], (nf, ACTION_DIM), decoder=getattr(model, "action_decoder", None))[0].float().cpu().numpy()
        gt_q = _vid_uint8(raw[0])[qry]; gen_q = _vid_uint8(gv[0])[qry]
        # metric correctness: single-cam embodiments carry content only in the TL
        # quadrant of the 2x2 grid canvas — crop it and resize back to full size so
        # WiLoR sees native-scale hands and image metrics exclude the black quadrants.
        if not any(g in str(emb) for g in ("mimicdroid", "gr1")):
            def _tl(v):
                h, w = v.shape[1] // 2, v.shape[2] // 2
                import torch.nn.functional as _F
                import torch as _t
                x = _t.from_numpy(v[:, :h, :w].copy()).permute(0, 3, 1, 2).float()
                x = _F.interpolate(x, size=(v.shape[1], v.shape[2]), mode="bilinear", align_corners=False)
                return x.permute(0, 2, 3, 1).clamp(0, 255).byte().numpy()
            gt_q = _tl(gt_q); gen_q = _tl(gen_q)
        _q = int(ds.query_idx[idx % ds.N]); state0 = np.asarray(ds._grp(_q).state("state57")[int(ds.ww0[_q])], np.float32)
        gt_a = np.asarray(s["actions"], np.float32)[:nf]
        extra = {}
        if _state:
            slo, shi = _sst(emb)
            fk = _fk_skeleton_state(gen_a, slo, shi, state0)
            fk_gt = _fk_skeleton_state(gt_a, slo, shi, state0)
            _den = lambda x: (x + 1.0) / 2.0 * (shi - slo) + slo          # noqa: E731
            gen_s, gt_s = _den(gen_a), _den(gt_a)                          # physical states
            extra = dict(gen_norm=gen_a.astype(np.float32), gt_norm=gt_a.astype(np.float32),
                         gen_state=gen_s.astype(np.float32), gt_state=gt_s.astype(np.float32),
                         state0=state0)
            gen_a = derived_action57_from_states(gen_s, state0).astype(np.float32)
            gt_a = derived_action57_from_states(gt_s, state0).astype(np.float32)
        else:
            lo, hi = stats.get(str(emb), stats["_global"])
            fk = _fk_skeleton(gen_a, lo, hi, state0)
            # GT-action FK wrists (same stats/state0/frame as gen) for action-space metrics
            fk_gt = _fk_skeleton(gt_a, lo, hi, state0)
        HP, HR = fk["head_p"], fk["head_R"]
        wl = np.einsum("tji,tj->ti", HR, fk["L_P"] - HP); wr = np.einsum("tji,tj->ti", HR, fk["R_P"] - HP)
        wl_gt = np.einsum("tji,tj->ti", fk_gt["head_R"], fk_gt["L_P"] - fk_gt["head_p"])
        wr_gt = np.einsum("tji,tj->ti", fk_gt["head_R"], fk_gt["R_P"] - fk_gt["head_p"])
        np.savez(f"{a.out_dir}/s{a.shard}_{j:06d}.npz", gen=jpg(gen_q), gt=jpg(gt_q), h=gen_q.shape[1], w=gen_q.shape[2],
                 fkL=wl.astype(np.float32), fkR=wr.astype(np.float32), emb=str(emb), ret=str(ret), idx=idx, k=k, g=g,
                 gen_a=gen_a.astype(np.float32), gt_a=gt_a,
                 fkL_gt=wl_gt.astype(np.float32), fkR_gt=wr_gt.astype(np.float32), **extra)
        if j % 20 == 0: print(f"  {j}/{len(rows)} (idx={idx} k={k} g={g} {ret}->{emb})", flush=True)
    print(f"shard {a.shard} DONE {len(rows)} -> {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
