"""2x-pool one-shot eval server — a thin wrapper around eval_gr1_server.py that
pins the 2x-sped taskdemos and matches them in the no-velocity descriptor space.
The base file is NOT modified; everything is a subclass + a runtime rebind.

Two changes vs the base server, both required for the 2x post-train
(experiment=cosmos_predict2p5_2b_egoverse_gr1_ret_v6_2x_448_state):

  1. NO-VEL retrieval descriptor (gr1_online_retrieval_2x.install()) — the 2x demos
     were re-matched with pos+rot+grip only, so the serve-time within-episode matching
     must drop velocity too (a 2x demo has doubled frame-to-frame velocity).
  2. One-shot pins the 2x taskdemos: `oneshot_task=<T>` -> taskdemo_<T>_<T>_D_demo_{1,2,3}__2x.

Everything else (model, conditioning build, action decode, HTTP API, CLI) is inherited
verbatim from eval_gr1_server.

⚠️ pinned-advance for 2x: keep the default `match` (unconstrained no-vel re-match — no
frame constants). The `mono`/`lockstep` pointer policies use FRAME-unit constants
(--mono-floor 8/16, hardcoded band +48, lockstep +chunk_size=32) calibrated for 1x
demos; a 2x demo is HALF as long, so those advance the pointer ~2x too fast and run off
the demo end. If you must use mono on 2x, HALVE the constants (floor 4/8, band +24) —
which needs a code change; prefer `match`.

Run (ICL / one-shot, 2x):
  uv run --extra cu128 --group egoverse python -m cosmos_policy.scripts.eval_gr1_server_2x \\
      --checkpoint <2x_post_train_ckpt_dir> --port 8700
  # client is UNCHANGED — the standard one-shot-pool3 client hits this server:
  #   ... eval_gr1_client.py --server http://127.0.0.1:8700 --oneshot-pool3 \\
  #       --task PnPSinkToCabinet --demo-only demo_1 --episodes 10 ...
"""
from __future__ import annotations

import os
import sys

from cosmos_policy.scripts import eval_gr1_server as base
from cosmos_policy.scripts import gr1_online_retrieval_2x as _nv

_2X_SUFFIX = "__2x"


class GR1Policy2x(base.GR1Policy):
    """GR1 one-shot policy that pins the 2x taskdemos (no-vel matching via _nv.install)."""

    def __init__(self, args) -> None:
        super().__init__(args)
        # HALF-RATE pointer for the half-length 2x demos: without this the pinned pointer
        # advances at the 1x pace (+chunk_size / mono band +48) and reaches the 2x demo's
        # END by ~t=160, freezing conditioning for the rest of a long episode (PnP place
        # phase). Halved so the pointer tracks the 2x demo's compressed timeline.
        # Env-overridable for sweeping: MONO_BAND_2X (default 24), POINTER_STEP_2X (default chunk_size//2).
        self._mono_band = int(os.environ.get("MONO_BAND_2X", "24"))
        self._pointer_step = int(os.environ.get("POINTER_STEP_2X", str(max(1, self.chunk_size // 2))))
        print(f"[2x] half-rate pointer: mono_band={self._mono_band} pointer_step={self._pointer_step}",
              flush=True)

    def reset(self, req: dict | None = None) -> None:
        req = dict(req or {})
        ot = req.get("oneshot_task")
        # oneshot-pool3 path: rewrite oneshot_task -> explicit 2x taskdemo triple so the
        # base reset pins/within-matches those (episode_windows uses the no-vel descriptor).
        if ot and not req.get("oneshot_demo"):
            eps = [f"taskdemo_{ot}_{ot}_D_demo_{i}{_2X_SUFFIX}" for i in (1, 2, 3)]
            eps = [e for e in eps if os.path.isdir(os.path.join(base.HUM_ROOT, e))]
            if not eps:
                raise FileNotFoundError(f"oneshot_task has no 2x taskdemos: {ot}")
            req.pop("oneshot_task", None)
            req["oneshot_demo"] = {"ep": eps[0], "w0": 0, "_all": eps}
        else:
            # explicit oneshot_demo: point it (and its _all list) at the 2x episodes.
            od = req.get("oneshot_demo")
            if isinstance(od, dict):
                od = dict(od)

                def _to2x(e: str) -> str:
                    e = str(e)
                    if e.endswith(_2X_SUFFIX):
                        return e
                    cand = e + _2X_SUFFIX
                    return cand if os.path.isdir(os.path.join(base.HUM_ROOT, cand)) else e
                if "ep" in od:
                    od["ep"] = _to2x(od["ep"])
                if isinstance(od.get("_all"), (list, tuple)):
                    od["_all"] = [_to2x(e) for e in od["_all"]]
                req["oneshot_demo"] = od
        return super().reset(req)


def main() -> None:
    _nv.install()                      # no-vel descriptors (before any policy/query call)
    base.GR1Policy = GR1Policy2x       # base.main() instantiates GR1Policy -> our subclass

    # 2x-experiment defaults (only if the caller didn't set them). Stats live in the 2x
    # manifest (query/target identical to 1x, copied); model arch is the same config.
    argstr = " ".join(sys.argv)
    if "--experiment" not in argstr:
        sys.argv += ["--experiment", "cosmos_predict2p5_2b_egoverse_gr1_ret_v6_2x_448_state"]
    if "--pairs-dir" not in argstr:
        sys.argv += ["--pairs-dir", "/mnt/ddn/dataset/egoverse_pairs_gr1_v6_2x"]
    base.main()


if __name__ == "__main__":
    main()
