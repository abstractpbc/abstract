"""I2RT YAM bimanual arms and the ``/act`` policies that drive them.

The registry serves three of these components by name:

* embodiment ``yam_arms`` — ``YAMEmbodiment``
* policy ``molmoact2`` — ``MolmoAct2Policy``
* policy ``gr00t`` — ``gr00t_policy``

``run_preflight`` verifies a policy and embodiment are compatible before any motion.
"""

from __future__ import annotations

from abstract_harness.robot.yam.config import ActServerConfig, MolmoActConfig, YamConfig
from abstract_harness.robot.yam.embodiment import YAMEmbodiment
from abstract_harness.robot.yam.operator import OperatorIO
from abstract_harness.robot.yam.packing import STATE_KEY, TOTAL_DIM, pack, split
from abstract_harness.robot.yam.policy import ActServerPolicy, MolmoAct2Policy, gr00t_policy
from abstract_harness.robot.yam.preflight import build, run_preflight

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("abstract-harness")
except PackageNotFoundError:  # pragma: no cover - only hit in a non-installed tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "STATE_KEY",
    "TOTAL_DIM",
    "ActServerConfig",
    "ActServerPolicy",
    "MolmoAct2Policy",
    "MolmoActConfig",
    "OperatorIO",
    "YAMEmbodiment",
    "YamConfig",
    "build",
    "gr00t_policy",
    "pack",
    "run_preflight",
    "split",
]
