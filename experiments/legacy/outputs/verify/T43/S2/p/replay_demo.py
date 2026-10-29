"""T43 S2: open-loop replay of the RECORDED DATASET ACTIONS in the policy client's MuJoCo scene (verifier, CPU, no server).

Client  : the T42-1 client snapshot ${AG3S_PROJECT_ROOT}-t42 (pi05_TO_hybrid 98e07d6 = the client that recorded the T42-1 E0/E3b
          rollouts), pi05_infer.main() run in-process with exactly the E0 arguments of outputs/verify/T42/p/run_client.sh
          (--model rby1_randomized_pick_place_16d --episode-index EP --max-steps 600 --headless --warmup-steps 1000
          --record-frames RUN --record RUN/third_person_front.mp4 --view front --trajectory-out RUN/traj.npz
          --obstacle-stop-distance 0), except: no --policy-seed (there is no model), and load_remote_policy is replaced by
          DatasetPolicy.  So the reset (reset_randomized_scene from the dataset record, actuator hold, 1000 warmup steps),
          apply_action('rby1_16d': 16-D absolute joint targets + 2 gripper openings in [0, 1] x RBY1_GRIPPER_OPEN), the
          steps per action (1 / (15 Hz * timestep)) and the recording are the client's own code.
Policy  : DatasetPolicy.infer(obs) returns the next 8 rows (OPEN_LOOP_HORIZON) of the episode's parquet 'action' column:
          control step t gets action[t] (t = 0 .. len-1); after the last frame the last action is repeated (hold) up to
          step 600.  The observation is ignored (open loop).
usage: replay_demo.py EP OUTDIR   -> OUTDIR/DEMO_ep<EP>/ (frames.jsonl, manifest.json, traj.npz, third_person_front.mp4),
       OUTDIR/DEMO_ep<EP>.replay.json (n frames, held steps, action source)"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, sys, pathlib
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, pandas as pd

SNAP = pathlib.Path("${AG3S_PROJECT_ROOT}-t42")
DS = pathlib.Path(_ag3s_workspace_path('legacy/pi05_TO_hybrid/data/rby1_randomized_pick_place_16d_v1'))
sys.path.insert(0, str(SNAP)); sys.path.insert(0, str(SNAP / "integration" / "rby1" / "bringup"))
os.chdir(SNAP / "pi05_TO_hybrid")
import pi05_infer as P  # noqa: E402
assert P.__file__.startswith(str(SNAP)), P.__file__


class DatasetPolicy:
    def __init__(self, A):
        self.A = np.asarray(A, np.float64); self.k = 0; self.calls = 0; self.held = 0

    def infer(self, obs):
        H = P.OPEN_LOOP_HORIZON
        idx = np.arange(self.k, self.k + H); self.held += int((idx >= len(self.A)).sum())
        rows = self.A[np.minimum(idx, len(self.A) - 1)]
        self.k += H; self.calls += 1
        return {"actions": rows}


def main(ep, outdir):
    ep = int(ep); out = pathlib.Path(outdir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DS / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet", columns=["action", "frame_index"])
    A = np.stack(df["action"].to_numpy()).astype(np.float64)
    assert (df["frame_index"].to_numpy() == np.arange(len(df))).all()
    pol = DatasetPolicy(A)
    P.load_remote_policy = lambda remote: pol
    run = out / f"DEMO_ep{ep}"
    sys.argv = ["pi05_infer.py", "--model", "rby1_randomized_pick_place_16d", "--remote", "localhost:1", "--episode-index", str(ep),
                "--max-steps", "600", "--headless", "--warmup-steps", "1000", "--record-frames", str(run),
                "--record", str(run / "third_person_front.mp4"), "--view", "front", "--trajectory-out", str(run / "traj.npz"),
                "--obstacle-stop-distance", "0"]
    P.main()
    rep = {"episode": ep, "n_dataset_frames": len(A), "policy_calls": pol.calls, "held_steps_after_last_frame": pol.held,
           "argv": sys.argv, "client": P.__file__, "action_source": str(DS / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet")}
    (out / f"DEMO_ep{ep}.replay.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


if __name__ == "__main__":
    main(*sys.argv[1:3])
