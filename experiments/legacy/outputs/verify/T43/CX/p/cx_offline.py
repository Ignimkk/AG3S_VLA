"""T43 CX — open-loop re-solve of recorded V11AB / V11ABC chunks with C toggled (verifier, CPU only; benchmark code unchanged, read from the
snapshot ${AG3S_PROJECT_ROOT}-t43v11 via PYTHONPATH). Wraps AG3S.trajopt.experiments.ta_offline (T43 TA harness): the recorded scene, policy chunk,
q_now, continuity, B / C history are held fixed; variant overrides on top of the run's own server config (server_proc.txt):
  base   = the run's server config as recorded (V11ABC: A + B + C; V11AB: A + B)
  noC    = base with limits.servo_accel = 'off'         (for V11ABC records)
  addC   = base with relaxed 1 deg, weight 0.01 (= V11ABC C flags)   (for V11AB records)
usage: cx_offline.py --index ANALYSIS_JSON --key KEY --out OUT --variants base noC --first-chunk N --max-chunks M"""
import sys
import AG3S.trajopt.experiments.ta_offline as T
T.VARIANTS["noC"] = {"limits": {"servo_accel": "off"}}
T.VARIANTS["addC"] = {"limits": T._c("relaxed", 1.0, 0.01)}
if __name__ == "__main__":
    sys.exit(T.main(sys.argv[1:]))
