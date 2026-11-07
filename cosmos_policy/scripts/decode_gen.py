"""Stage 1 (.venv, NO extra deps): decode GEN query video + gen action for a few samples,
FK the gen action to 3D wrist trajectories, save an npz per sample for the WiLoR stage."""
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, "/mnt/ddn/cosmos_policy")
from cosmos_policy.scripts.eval_egoverse_ret import _fk_skeleton, _vid_uint8, _load_stats, _to_cuda, ACTION_DIM
from cosmos_policy._src.imaginaire.lazy_config import instantiate
from cosmos_policy._src.imaginaire.utils import distributed
from cosmos_policy._src.imaginaire.utils.context_managers import distributed_init
from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint
from cosmos_policy.models.policy_text2world_model import extract_action_chunk_from_latent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--weights_pt", required=True)
    ap.add_argument("--act_stats", required=True); ap.add_argument("--out_dir", default="/mnt/ddn/tmp/gen_dump")
    ap.add_argument("--idxs", default="2000 8000"); ap.add_argument("--num_steps", type=int, default=35)
    ap.add_argument("opts", nargs=argparse.REMAINDER)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    with distributed_init():
        distributed.init()
    _opts = [o for o in a.opts if o != "--"]
    _exp = next((o.split("=", 1)[1] for o in _opts if o.startswith("experiment=")), None)
    _eo = [o for o in _opts if not o.startswith("experiment=")]
    model, config = load_model_from_checkpoint(experiment_name=_exp, s3_checkpoint_dir=a.weights_pt,
                                               config_file=a.config, enable_fsdp=False, load_ema_to_reg=False,
                                               experiment_opts=_eo)
    model = model.cuda().eval(); tk = model.tensor_kwargs
    ds = instantiate(config.dataloader_train.dataset); R = ds.R; nf = ds.num_frames
    stats = _load_stats(a.act_stats); qry = slice(4 * R + 9, 8 * R + 9)
    for idx in [int(x) for x in a.idxs.split()]:
        s = ds[idx]; db = _to_cuda(s, tk); emb = s["embodiment"]; ret = s["retrieved_embodiment"]
        with torch.no_grad(), torch.autocast("cuda", dtype=tk.get("dtype", torch.bfloat16)):
            raw, x0, _ = model.get_data_and_condition(db)
            lat = model.generate_samples_from_batch(db, guidance=1.5, state_shape=x0.shape[1:],
                                                    n_sample=x0.shape[0], num_steps=a.num_steps,
                                                    shift=getattr(model.config, "shift", 1.0))
            gv = model.decode(lat)
            ai = int(db["action_latent_idx"][0]) if torch.is_tensor(db["action_latent_idx"]) else int(db["action_latent_idx"])
            gen_a = extract_action_chunk_from_latent(lat[:, :, ai], (nf, ACTION_DIM), decoder=getattr(model, "action_decoder", None))[0].float().cpu().numpy()
        gt_v = _vid_uint8(raw[0]); gen_v = _vid_uint8(gv[0]); gt_q = gt_v[qry]; gen_q = gen_v[qry]
        _q = int(ds.query_idx[idx % ds.N]); state0 = np.asarray(ds._grp(_q).state("state57")[int(ds.ww0[_q])], np.float32)
        lo, hi = stats.get(str(emb), stats["_global"])
        fk = _fk_skeleton(gen_a, lo, hi, state0)
        HP, HR = fk["head_p"], fk["head_R"]
        hw = {sd: np.einsum("tji,tj->ti", HR, fk[sd + "_P"] - HP) for sd in ("L", "R")}   # head-frame wrist
        np.savez(f"{a.out_dir}/idx{idx}_{ret}2{emb}.npz",
                 gen_q=gen_q, gt_q=gt_q, gen_action=gen_a, fk_wrist_L=hw["L"], fk_wrist_R=hw["R"],
                 fk_wrist_L_world=fk["L_P"], fk_wrist_R_world=fk["R_P"], emb=emb, ret=ret, idx=idx)
        print(f"saved idx{idx} {ret}2{emb} gen_q{gen_q.shape}", flush=True)
    print("DONE ->", a.out_dir, flush=True)


if __name__ == "__main__":
    main()
