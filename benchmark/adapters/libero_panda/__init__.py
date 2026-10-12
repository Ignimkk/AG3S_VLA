"""LIBERO Panda -> V11AB camera adapter (SL1). Python 3.8 compatible, numpy only at import time.

``camera``    robosuite/LIBERO observation + ``env.sim`` -> per-camera metric depth (OpenCV rows), K,
              T_base_cam, robot_state, stamp; policy-image <-> OpenCV pixel maps.
``wire_pack`` those per-camera frames -> the V11AB wire keys/dtypes of ``AG3S.trajopt.wire``
              (a 3.8 replica; equivalence is enforced by a cross-env test, not by importing AG3S).
"""
