"""T39 S1 — 정책 RNG seed 입구와 고정 warmup.

π0.5 의 정책 RNG 는 서버 시작 때 `jax.random.key(0)` 이고 다시 놓이지 않는다 (`policy.py:65, 75`).
O1 은 `(episode, seed)` 가 run 을 정하게 하려고 seed 를 명시한다. 이 파일이 박는 것:

1. **wire** — seed 를 안 주면 요청이 T39 전과 **바이트까지** 같다 (msgpack 골든). 주면 실제 전송을
   지나 `ag3s/policy_seed` 로 돌아오고, 정책 관측에는 없다. 틀린 seed 는 보내기 전에 죽는다.
2. **감싸개** — `SeededPolicy` (`--no-safe`) 는 맨 키를 벗기고 `_rng` 를 다시 놓고 회신한다. 키가 없으면
   obs 를 **같은 객체**로 넘긴다. `_rng` 를 못 찾으면 크게 실패한다.
3. **SafePolicy · ToOnlyPolicy** — reset 뒤 · `policy.infer` 바로 앞에서 re-key, 응답 회신, 기록.
4. **client** — `SafeRemoteClient.infer(policy_seed=)` 가 싣고, 회신이 없으면 (옛 서버) 죽는다.
5. **JAX 의미** (jax 가 있을 때) — openpi `Policy` 와 같은 split 규약에서 seed 0 = 새 서버의 순서,
   같은 seed = 같은 noise, 다른 seed = 다른 noise. openpi 소스의 `_rng` 두 줄이 그대로인지.
6. **pi05_infer.py** — `run_warmup_steps` 가 정확히 N step, flag · 첫 요청에만 · manifest (소스).
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import pathlib
import re

import numpy as np
import pytest

from AG3S.trajopt import policy_seed as ps
from AG3S.trajopt import wire
from AG3S.trajopt.client import SafeRemoteClient
from tests.trajopt.test_safe_client import DOF, H, _client, _Server

ROOT = pathlib.Path(__file__).resolve().parents[2]
PI05_INFER = ROOT / "pi05_TO_hybrid" / "rby1_bringup" / "pi05_infer.py"
OPENPI_POLICY = ROOT / "pi05_TO_hybrid" / "openpi" / "src" / "openpi" / "policies" / "policy.py"
SERVE_SAFE = ROOT / "benchmark" / "trajopt" / "serve_safe.py"

SEED = 18071


def _fake_key(seed):
    return ("key", int(seed))


# =============================================================================================
# 1. wire
# =============================================================================================

#: T39 전 `pack_request` 의 키 순서 (아래 `_golden_request`, 카메라 둘). 2026-10-02 에 HEAD 의
#: `wire.py` 로 만들었다.
PRE_T39_KEYS = [
    "state", "prompt", "images", "ag3s/exec_feedback", "ag3s/stamp_mode", "ag3s/cameras",
    "ag3s/phase", "ag3s/active_manipulators", "ag3s/reset", "ag3s/seq",
    "ag3s/depth/zed_left", "ag3s/K/zed_left", "ag3s/T_base_cam/zed_left",
    "ag3s/robot_state/zed_left", "ag3s/stamp/zed_left", "ag3s/render_stamp/zed_left",
    "ag3s/depth/wrist_cam_l", "ag3s/K/wrist_cam_l", "ag3s/T_base_cam/wrist_cam_l",
    "ag3s/robot_state/wrist_cam_l", "ag3s/stamp/wrist_cam_l", "ag3s/render_stamp/wrist_cam_l"]
#: 같은 요청의 msgpack_numpy 바이트 sha256 (1959 B) — **T39 전 `wire.py` (benchmark main 563dd4f 의 `git show HEAD:trajopt/wire.py`)
#: 로 만든 값.** seed 를 안 준 요청이 이 바이트와 다르면 seed 입구가 옛 동작을 바꾼 것이다.
PRE_T39_SHA256 = "098cc281dfe1fd4c335f210b255918cfb135d691e3da7e5a5ff641814e81b0ca"


def _golden_request(**kw):
    obs = {"state": (np.arange(16, dtype=np.float32) / 10), "prompt": "pick",
           "images": {"cam_high": np.arange(48, dtype=np.uint8).reshape(4, 4, 3)}}
    cams = ("zed_left", "wrist_cam_l")
    return wire.pack_request(
        obs, cameras=cams,
        depth={c: np.linspace(0.3, 2.0, 20).reshape(4, 5) + i for i, c in enumerate(cams)},
        intrinsics={c: np.eye(3) * (i + 1) for i, c in enumerate(cams)},
        extrinsics={c: np.eye(4) for c in cams},
        robot_state={c: np.arange(24, dtype=np.float64) * 0.01 * (i + 1)
                     for i, c in enumerate(cams)},
        stamps={c: 100.25 for c in cams}, phase="approach", active_manipulators=("left",),
        reset=True, seq=1, exec_feedback=wire.no_exec_feedback("first chunk"),
        render_stamps={c: 100.5 + i for i, c in enumerate(cams)}, stamp_mode="sim_frozen", **kw)


def _msgpack_numpy():
    pytest.importorskip("msgpack")
    path = (ROOT / "pi05_TO_hybrid" / "openpi" / "packages" / "openpi-client" / "src" /
            "openpi_client" / "msgpack_numpy.py")
    if not path.exists():  # pragma: no cover
        pytest.skip(f"{path} not found")
    spec = importlib.util.spec_from_file_location("_t39_msgpack_numpy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_seed_keeps_the_pre_t39_key_set_and_order():
    assert list(_golden_request()) == PRE_T39_KEYS
    assert list(_golden_request(policy_seed=None)) == PRE_T39_KEYS
    assert not any("policy_seed" in k for k in _golden_request())


def test_no_seed_request_is_byte_identical_to_pre_t39():
    mp = _msgpack_numpy()
    for request in (_golden_request(), _golden_request(policy_seed=None)):
        packed = mp.packb(request)
        assert len(packed) == 1959
        assert hashlib.sha256(packed).hexdigest() == PRE_T39_SHA256


def test_seed_round_trips_through_the_real_transport_and_never_reaches_the_policy():
    mp = _msgpack_numpy()
    request = mp.unpackb(mp.packb(_golden_request(policy_seed=np.int64(SEED))))
    policy_obs, scene = wire.strip_request(request)

    assert wire.unpack_policy_seed(scene) == SEED
    assert type(wire.unpack_policy_seed(scene)) is int
    assert set(policy_obs) == {"state", "prompt", "images"}
    # 키 하나만 더해졌다 — 나머지는 seed 없는 요청과 같다.
    assert [k for k in request if k != wire.PREFIX + wire.POLICY_SEED] == PRE_T39_KEYS


def test_seed_round_trips_without_msgpack():
    policy_obs, scene = wire.strip_request(_golden_request(policy_seed=7))
    assert wire.unpack_policy_seed(scene) == 7
    assert wire.POLICY_SEED not in policy_obs
    assert wire.unpack_policy_seed(wire.strip_request(_golden_request())[1]) is None


@pytest.mark.parametrize("bad, exc", [(True, TypeError), (1.0, TypeError), ("3", TypeError),
                                      (None, TypeError), (-1, ValueError),
                                      (2 ** 32, ValueError)])
def test_a_bad_seed_dies_before_it_is_sent(bad, exc):
    with pytest.raises(exc):
        wire.check_policy_seed(bad)
    if bad is not None:
        with pytest.raises(exc):
            _golden_request(policy_seed=bad)
        with pytest.raises(exc):
            wire.unpack_policy_seed({wire.POLICY_SEED: bad})


def test_seed_range_edges_and_echo():
    assert wire.check_policy_seed(0) == 0
    assert wire.check_policy_seed(np.uint32(2 ** 32 - 1)) == 2 ** 32 - 1
    assert wire.unpack_policy_seed_echo({}) is None
    assert wire.unpack_policy_seed_echo({wire.POLICY_SEED: np.int64(5)}) == 5


# =============================================================================================
# 2. 감싸개 (`--no-safe`) · rng_owner
# =============================================================================================


class _RngPolicy:
    """openpi `Policy` 의 RNG 규약만: `_rng` 를 갖고, `infer` 마다 '쪼갠다' (여기선 기록만)."""

    metadata = {"policy": "fake"}

    def __init__(self):
        self._rng = ("key", 0)
        self.seen = []           # (obs 객체, infer 시점의 _rng)
        self.resets = 0

    def infer(self, obs, **_):
        self.seen.append((obs, self._rng))
        self._rng = ("split", self._rng)
        return {"actions": np.zeros((H, DOF)), "policy_extra": 7}

    def reset(self):
        self.resets += 1


class _Wrap:
    def __init__(self, inner):
        self._policy = inner


def test_rng_owner_walks_the_wrapper_chain():
    inner = _RngPolicy()
    assert ps.rng_owner(_Wrap(_Wrap(inner))) is inner
    assert ps.rng_owner(inner) is inner


def test_rng_owner_fails_loudly_when_there_is_no_rng():
    class _NoRng:
        def infer(self, obs):
            return {}

    with pytest.raises(ps.PolicySeedError, match=r"_Wrap -> _NoRng"):
        ps.rng_owner(_Wrap(_NoRng()))
    assert "UNAVAILABLE" in ps.describe_rng_owner(_Wrap(_NoRng()))
    assert "re-keys _RngPolicy._rng" in ps.describe_rng_owner(_Wrap(_RngPolicy()))


def test_rng_owner_refuses_a_pytorch_policy():
    torch_like = _RngPolicy()
    torch_like._is_pytorch_model = True
    with pytest.raises(ps.PolicySeedError, match="PyTorch"):
        ps.rng_owner(torch_like)


def test_seeded_policy_without_the_key_is_a_pure_pass_through():
    inner = _RngPolicy()
    sp = ps.SeededPolicy(inner, key_fn=_fake_key)
    obs = {"state": np.zeros(DOF), "prompt": "pick"}
    out = sp.infer(obs)

    assert inner.seen[0][0] is obs                       # 같은 객체 — 사본도 아니다
    assert inner.seen[0][1] == ("key", 0)                 # RNG 를 건드리지 않았다
    assert wire.POLICY_SEED not in out
    assert sp.metadata is inner.metadata                  # 접속 때 보내는 바이트가 같다


def test_seeded_policy_rekeys_strips_and_echoes():
    inner = _RngPolicy()
    sp = ps.SeededPolicy(inner, key_fn=_fake_key)
    sp.infer({"state": np.zeros(DOF)})                    # RNG 를 한 번 진행시킨다
    obs = {"state": np.zeros(DOF), "prompt": "pick", wire.POLICY_SEED: SEED}
    out = sp.infer(obs)

    seen_obs, seen_rng = inner.seen[-1]
    assert seen_rng == ("key", SEED)                      # infer 직전에 다시 놓였다
    assert wire.POLICY_SEED not in seen_obs               # 정책에는 새지 않는다
    assert set(seen_obs) == {"state", "prompt"}
    assert wire.POLICY_SEED in obs                        # 호출자의 dict 는 그대로
    assert out[wire.POLICY_SEED] == SEED and out["policy_extra"] == 7
    assert (sp.last_seed, sp.last_seed_request) == (SEED, 2)

    sp.infer({"state": np.zeros(DOF)})                    # 다음 요청: 이어서 split
    assert inner.seen[-1][1] == ("split", ("key", SEED))


def test_seeded_policy_refuses_a_bad_seed_before_inferring():
    inner = _RngPolicy()
    sp = ps.SeededPolicy(inner, key_fn=_fake_key)
    with pytest.raises(TypeError):
        sp.infer({"state": np.zeros(DOF), wire.POLICY_SEED: 1.5})
    assert inner.seen == [] and inner._rng == ("key", 0)


def test_apply_request_seed_does_nothing_without_a_seed():
    inner = _RngPolicy()
    assert ps.apply_request_seed(_Wrap(inner), {"seq": 3}, key_fn=_fake_key) is None
    assert inner._rng == ("key", 0)
    assert ps.apply_request_seed(_Wrap(inner), {wire.POLICY_SEED: 9}, key_fn=_fake_key) == 9
    assert inner._rng == ("key", 9)


def test_serve_safe_wraps_the_bare_policy_and_logs_the_entry():
    src = SERVE_SAFE.read_text()
    assert "served = SeededPolicy(policy)" in src
    assert 'logging.info("%s", describe_rng_owner(served))' in src
    assert 'describe_xla_determinism(os.environ.get("XLA_FLAGS"))' in src


def test_xla_determinism_line_says_how_far_a_seed_reproduces():
    on = ps.describe_xla_determinism("--xla_gpu_enable_command_buffer=")
    assert "only WITHIN this server process" in on
    assert "only WITHIN" in ps.describe_xla_determinism(None)
    off = ps.describe_xla_determinism(
        "--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0")
    assert "across server processes" in off


# =============================================================================================
# 3. SafePolicy · ToOnlyPolicy
# =============================================================================================


@pytest.fixture
def fake_jax_key(monkeypatch):
    """CPU shim 에는 jax 가 없다 — 키 생성만 바꾼다 (그 의미는 5 절이 jax 로 본다)."""
    monkeypatch.setattr(ps, "_jax_key", _fake_key)


class _ResettingRngPolicy(_RngPolicy):
    """reset 이 RNG 를 건드린다면 — seed 는 그 **뒤**에 놓여야 한다."""

    def reset(self):
        super().reset()
        self._rng = ("reset-clobbered",)


def _sp_request(seq, *, reset=False, seed=None):
    req = {"state": np.zeros(DOF), wire.PREFIX + "cameras": [], wire.PREFIX + "seq": seq,
           wire.PREFIX + "reset": reset}
    if seed is not None:
        req[wire.PREFIX + wire.POLICY_SEED] = seed
    return req


def test_safe_policy_rekeys_right_before_infer_and_records_it(fake_jax_key):
    from tests.trajopt.test_safe_policy import _safe_policy

    inner = _ResettingRngPolicy()
    sp = _safe_policy(inner)

    out = sp.infer(_sp_request(1, reset=True, seed=SEED))
    assert inner.resets == 1
    assert inner.seen[-1][1] == ("key", SEED)             # reset 뒤, infer 직전
    assert not any("policy_seed" in k or k.startswith(wire.PREFIX) for k in inner.seen[-1][0])
    assert out[wire.POLICY_SEED] == SEED
    assert sp._exec_summary()["policy_seed"] == {"applied": SEED, "episode": SEED}

    out = sp.infer(_sp_request(2))
    assert wire.POLICY_SEED not in out                    # 적용한 요청에만 회신
    assert inner.seen[-1][1] == ("split", ("key", SEED))  # 이어서 split
    assert sp._exec_summary()["policy_seed"] == {"applied": None, "episode": SEED}

    sp.infer(_sp_request(1, reset=True))                  # seed 없는 새 에피소드
    assert sp._exec_summary()["policy_seed"] == {"applied": None, "episode": None}


def test_safe_policy_without_a_seed_never_touches_the_rng(fake_jax_key):
    from tests.trajopt.test_safe_policy import _safe_policy

    inner = _RngPolicy()
    sp = _safe_policy(inner)
    out = sp.infer(_sp_request(1))
    assert inner.seen[-1][1] == ("key", 0)
    assert wire.POLICY_SEED not in out


def test_safe_policy_fails_loudly_when_the_policy_has_no_rng(fake_jax_key):
    from tests.trajopt.test_safe_policy import _safe_policy, _StubPolicy

    sp = _safe_policy(_StubPolicy())
    sp.infer(_sp_request(1))                              # seed 없으면 그대로 돈다
    with pytest.raises(ps.PolicySeedError):
        sp.infer(_sp_request(2, seed=SEED))


def test_to_only_policy_rekeys_and_echoes(fake_jax_key):
    from AG3S.trajopt.config import TrajOptConfig
    from AG3S.ag3s.robot_models import load_rby1
    from AG3S.trajopt.to_only_policy import ToOnlyPolicy
    from tests.trajopt.test_to_only_policy import SERVER_CFG, _request

    class _ChunkRng(_RngPolicy):
        def infer(self, obs, **kw):
            super().infer(obs, **kw)
            return {"actions": np.zeros((H, wire.ACTION_WIDTH))}

    inner = _ChunkRng()
    pol = ToOnlyPolicy(inner, robot_model=load_rby1(),
                       to_config=TrajOptConfig.from_dict(SERVER_CFG))
    request = _request(1, reset=True)
    request[wire.PREFIX + wire.POLICY_SEED] = SEED
    out = pol.infer(request)
    assert inner.seen[-1][1] == ("key", SEED)
    assert out[wire.POLICY_SEED] == SEED
    assert wire.POLICY_SEED not in pol.infer(_request(2))


# =============================================================================================
# 4. client
# =============================================================================================


class _EchoServer(_Server):
    def __init__(self, *, echo=True, **kw):
        super().__init__(**kw)
        self.echo = echo

    def infer(self, request):
        out = super().infer(request)
        key = wire.PREFIX + wire.POLICY_SEED
        if self.echo and key in request:
            out[wire.POLICY_SEED] = request[key]
        return out


def test_client_sends_the_seed_only_when_asked():
    server = _EchoServer()
    client = _client(server)
    client.infer({"state": np.zeros(DOF)}, reset=True, policy_seed=SEED)
    client.infer({"state": np.zeros(DOF)})

    assert server.requests[0][wire.PREFIX + wire.POLICY_SEED] == SEED
    assert wire.PREFIX + wire.POLICY_SEED not in server.requests[1]
    assert client.last_policy_seed is None and client.last_safe is True


def test_client_dies_when_an_old_server_ignores_the_seed():
    client = _client(_EchoServer(echo=False))
    with pytest.raises(RuntimeError, match="does not apply policy seeds"):
        client.infer({"state": np.zeros(DOF)}, reset=True, policy_seed=SEED)
    # seed 를 안 보내면 옛 서버와도 그대로 돈다.
    out = _client(_EchoServer(echo=False)).infer({"state": np.zeros(DOF)}, reset=True)
    assert out["actions"].shape == (H, DOF)


def test_client_refuses_a_bad_seed():
    with pytest.raises(ValueError):
        _client(_EchoServer()).infer({"state": np.zeros(DOF)}, policy_seed=-3)


# =============================================================================================
# 5. JAX 의미 — openpi `Policy` 의 split 규약 그대로
# =============================================================================================


def test_openpi_policy_still_keeps_its_key_where_the_wrapper_looks():
    """openpi 가 `_rng` 를 옮기면 감싸개는 시작 로그에서 UNAVAILABLE, seed 요청에서 예외를 낸다.
    여기서 먼저 알린다 (서브모듈 sync 뒤)."""
    if not OPENPI_POLICY.exists():  # pragma: no cover
        pytest.skip(f"not present: {OPENPI_POLICY}")
    src = OPENPI_POLICY.read_text()
    assert "self._rng = rng or jax.random.key(0)" in src
    assert "self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)" in src


class _JaxLikePolicy:
    """openpi `Policy.infer` 의 RNG 두 줄과 같은 규약 (`policy.py:65, 75`)."""

    def __init__(self):
        import jax

        self._jax = jax
        self._rng = jax.random.key(0)

    def infer(self, obs, **_):
        self._rng, sub = self._jax.random.split(self._rng)
        return {"actions": np.asarray(self._jax.random.normal(sub, (4, 3)))}


def _run(policy, n, seed=None):
    out = []
    for i in range(n):
        obs = {"state": np.zeros(3)}
        if seed is not None and i == 0:
            obs[wire.POLICY_SEED] = seed
        out.append(policy.infer(obs)["actions"])
    return np.stack(out)


def test_jax_seed_zero_reproduces_a_fresh_server_and_seeds_pair_runs():
    pytest.importorskip("jax")
    fresh = _run(ps.SeededPolicy(_JaxLikePolicy()), 6)    # 새 서버, seed 없음 = T39 전 순서

    server = ps.SeededPolicy(_JaxLikePolicy())
    _run(server, 6)                                       # 앞 run 이 RNG 를 진행시켰다
    old_second_run = _run(server, 6)                      # T39 전: 두 번째 run 은 다른 noise
    assert not np.array_equal(old_second_run, fresh)

    assert np.array_equal(_run(server, 6, seed=0), fresh)            # seed 0 = 새 서버
    a1, a2 = _run(server, 6, seed=SEED), _run(server, 6, seed=SEED)
    assert np.array_equal(a1, a2)                                     # 같은 seed = 비트 동일
    b = _run(server, 6, seed=SEED + 1)
    assert not np.any([np.array_equal(x, y) for x, y in zip(a1, b)])  # 다른 seed = 청크마다 다름


# =============================================================================================
# 6. pi05_infer.py — warmup · flag · 첫 요청에만 · manifest
# =============================================================================================


@pytest.fixture(scope="module")
def pi05_src():
    if not PI05_INFER.exists():  # pragma: no cover
        pytest.skip(f"not present: {PI05_INFER}")
    return PI05_INFER.read_text()


@pytest.fixture(scope="module")
def run_warmup_steps(pi05_src):
    """모듈 전체를 import 하면 mujoco viewer · rby1_manipulation 이 필요하다 — 함수 하나만 꺼낸다."""
    tree = ast.parse(pi05_src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "run_warmup_steps")
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(PI05_INFER), "exec"), ns)
    return ns["run_warmup_steps"]


def test_warmup_steps_exactly_n(run_warmup_steps):
    for n in (0, 1, 1000):
        count = {"step": 0, "sync": 0}
        done = run_warmup_steps(n, lambda: count.__setitem__("step", count["step"] + 1),
                                sync=lambda: count.__setitem__("sync", count["sync"] + 1))
        assert done == n and count == {"step": n, "sync": n}


def test_warmup_stops_when_the_viewer_closes_and_refuses_negative(run_warmup_steps):
    steps = []
    done = run_warmup_steps(10, lambda: steps.append(1), is_running=lambda: len(steps) < 4)
    assert done is None and len(steps) == 4
    with pytest.raises(ValueError):
        run_warmup_steps(-1, lambda: None)


def test_warmup_does_not_read_the_wall_clock(run_warmup_steps, pi05_src):
    tree = ast.parse(pi05_src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "run_warmup_steps")
    body = ast.unparse(fn)
    assert "time." not in body and "sleep" not in body


def test_pi05_infer_flags_and_first_request_only(pi05_src):
    assert '"--warmup-steps", type=int, default=None' in pi05_src
    assert '"--policy-seed", type=int, default=None' in pi05_src
    assert "if args.warmup_steps is not None:" in pi05_src
    # safe: 첫 요청에만
    assert re.search(r"policy_seed=\(args\.policy_seed\s+if t_step == 0 else None\)", pi05_src)
    # E0: 첫 요청에만, 사본에 싣고, 회신을 검사한다
    assert "elif args.policy_seed is not None and t_step == 0:" in pi05_src
    assert "policy.infer({**obs, POLICY_SEED_KEY: int(args.policy_seed)})" in pi05_src
    assert "if result.get(POLICY_SEED_KEY) != args.policy_seed:" in pi05_src
    # 기록
    assert 'manifest["policy_seed"] = args.policy_seed' in pi05_src
    assert '"warmup_steps": args.warmup_steps' in pi05_src
    # 로컬 PC 사본은 benchmark 를 import 하지 않는다 — 상수가 같아야 한다.
    assert 'POLICY_SEED_KEY = "policy_seed"' in pi05_src and wire.POLICY_SEED == "policy_seed"
    assert "POLICY_SEED_LIMIT = 2 ** 32" in pi05_src and wire.POLICY_SEED_LIMIT == 2 ** 32
