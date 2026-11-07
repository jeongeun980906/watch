"""Stage 1 (.venv): for a checkpoint × dataset, generate + decode GEN/GT query video + gen action,
FK the gen action to 3D head-frame wrists, and dump ONE compressed npz per query window
(frames stored as JPEG bytes → ~1.5MB/sample instead of ~28MB raw). Sharded + per-episode subsample.
Mirrors the batch-eval query enumeration. WiLoR/image metrics run in stage 2 (wilor_env)."""
import os, sys, argparse, numpy as np, torch, cv2
sys.path.insert(0, "/mnt/ddn/cosmos_policy")
from cosmos_policy.scripts.eval_egoverse_ret import _fk_skeleton, _vid_uint8, _load_stats, _to_cuda, ACTION_DIM
from cosmos_policy._src.imaginaire.lazy_config import instantiate
from cosmos_policy._src.imaginaire.utils import distributed
from cosmos_policy._src.imaginaire.utils.context_managers import distributed_init
from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint
from cosmos_policy.models.policy_text2world_model import extract_action_chunk_from_latent


import io
from PIL import Image


def jpg(frames, q=88):                                         # (.venv cv2 lacks imencode) -> PIL
    out = []
    for f in frames:
        b = io.BytesIO(); Image.fromarray(f).save(b, "JPEG", quality=q); out.append(b.getvalue())
    return np.array(out, object)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--weights_pt", required=True)
    ap.add_argument("--act_stats", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--per-episode", dest="per_episode", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=0); ap.add_argument("--num_steps", type=int, default=35)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--num-shards", dest="num_shards", type=int, default=1)
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
    ds = instantiate(config.dataloader_train.dataset); R = ds.R; nf = ds.num_frames
    stats = _load_stats(a.act_stats); qry = slice(4 * R + 9, 8 * R + 9)

    # enumerate query rows: per-episode subsample + episode cap + shard (EXACT mirror of batch eval)
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
    print(f"shard {a.shard}/{a.num_shards}: {len(rows)} query windows (R={R} nf={nf})", flush=True)

    for k, r in enumerate(rows):
        idx = int(r); s = ds[idx]; db = _to_cuda(s, tk); emb = s["embodiment"]; ret = s["retrieved_embodiment"]
        with torch.no_grad(), torch.autocast("cuda", dtype=tk.get("dtype", torch.bfloat16)):
            raw, x0, _ = model.get_data_and_condition(db)
            lat = model.generate_samples_from_batch(db, guidance=1.5, state_shape=x0.shape[1:], n_sample=x0.shape[0],
                                                    num_steps=a.num_steps, shift=getattr(model.config, "shift", 1.0))
            ai = int(db["action_latent_idx"][0]) if torch.is_tensor(db["action_latent_idx"]) else int(db["action_latent_idx"])
            # 액션 latent 를 GT(검은 프레임) latent 로 교체한 뒤 영상 디코드. causal 토크나이저가
            # future 블록 앞 latent 를 컨텍스트로 쓰는데 그 자리는 이미지가 아니라 액션 latent 라
            # 그대로 두면 future 첫 ~8 프레임이 무너진다 (eval_egoverse_ret_batch.py 의 같은 주석 참조).
            _lat_v = lat.clone(); _lat_v[:, :, ai] = x0[:, :, ai].to(_lat_v.dtype)
            gv = model.decode(_lat_v)
            gen_a = extract_action_chunk_from_latent(lat[:, :, ai], (nf, ACTION_DIM), decoder=getattr(model, "action_decoder", None))[0].float().cpu().numpy()
        gt_q = _vid_uint8(raw[0])[qry]; gen_q = _vid_uint8(gv[0])[qry]
        _q = int(ds.query_idx[idx % ds.N]); state0 = np.asarray(ds._grp(_q).state("state57")[int(ds.ww0[_q])], np.float32)
        lo, hi = stats.get(str(emb), stats["_global"]); fk = _fk_skeleton(gen_a, lo, hi, state0)
        HP, HR = fk["head_p"], fk["head_R"]
        wl = np.einsum("tji,tj->ti", HR, fk["L_P"] - HP); wr = np.einsum("tji,tj->ti", HR, fk["R_P"] - HP)
        np.savez(f"{a.out_dir}/s{a.shard}_{k:06d}.npz", gen=jpg(gen_q), gt=jpg(gt_q), h=gen_q.shape[1], w=gen_q.shape[2],
                 fkL=wl.astype(np.float32), fkR=wr.astype(np.float32), emb=str(emb), ret=str(ret), idx=idx)
        if k % 20 == 0: print(f"  {k}/{len(rows)}", flush=True)
    print(f"shard {a.shard} DONE {len(rows)} -> {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
