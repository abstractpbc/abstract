"""A robot's model file, read for what a preset needs: joints, limits and frames.

URDF and MJCF are parsed with the standard library; nothing is simulated or compiled.
From a URDF: every joint (type, parent and child link, position, velocity and effort
limits) and every link as a frame. From an MJCF: every joint with its range (degrees
converted to radians unless ``<compiler angle="radian">``, joint ``<default>`` classes
applied), bodies, sites and cameras as frames, and the actuators' joint order, which is
the order MuJoCo-trained policies expect. ``<include>`` files are followed.

The parser trusts its input the way a robot's own description is trusted; it does not
resolve meshes or validate dynamics.
"""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

#: Joint types that can be commanded (a position or a velocity per joint).
MOVABLE = frozenset({"revolute", "continuous", "prismatic", "hinge", "slide"})
_MAX_INCLUDE_DEPTH = 8


class ModelError(ValueError):
    """A model file that cannot be read as URDF or MJCF."""


@dataclass(frozen=True)
class ModelJoint:
    """One joint as the model declares it. Limits are in SI units (rad or m, rad/s or
    m/s, N*m or N); ``None`` where the model gives none."""

    name: str
    type: str
    lower: float | None = None
    upper: float | None = None
    velocity: float | None = None
    effort: float | None = None
    parent: str | None = None
    child: str | None = None

    @property
    def movable(self) -> bool:
        return self.type in MOVABLE

    @property
    def unit(self) -> str:
        return "m" if self.type in ("prismatic", "slide") else "rad"


@dataclass(frozen=True)
class RobotModel:
    """What a preset reads from a model file."""

    name: str
    format: str  # "urdf" | "mjcf"
    path: Path
    joints: tuple[ModelJoint, ...] = ()
    frames: frozenset[str] = frozenset()
    # MJCF: joints in actuator order (what ``data.ctrl`` drives); empty for URDF.
    actuated: tuple[str, ...] = ()
    digest: str = ""
    frame_kinds: dict[str, str] = field(default_factory=dict)

    def joint(self, name: str) -> ModelJoint | None:
        for joint in self.joints:
            if joint.name == name:
                return joint
        return None

    @property
    def movable(self) -> tuple[ModelJoint, ...]:
        """Commandable joints: in actuator order for an MJCF with actuators, else in
        document order."""
        if self.actuated:
            by_name = {joint.name: joint for joint in self.joints}
            return tuple(by_name[name] for name in self.actuated if name in by_name)
        return tuple(joint for joint in self.joints if joint.movable)


def _float(text: str | None, what: str) -> float | None:
    if text is None or not text.strip():
        return None
    try:
        value = float(text)
    except ValueError:
        raise ModelError(f"{what}: {text!r} is not a number") from None
    if not math.isfinite(value):
        raise ModelError(f"{what}: {text!r} is not finite")
    return value


def _floats(text: str | None, count: int, what: str) -> tuple[float, ...] | None:
    if text is None or not text.strip():
        return None
    parts = text.split()
    if len(parts) != count:
        raise ModelError(f"{what}: expected {count} numbers, got {text!r}")
    return tuple(_float(part, what) or 0.0 for part in parts)


def _parse(path: Path) -> ET.Element:
    try:
        return ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ModelError(f"{path}: not well-formed XML ({exc})") from None
    except OSError as exc:
        raise ModelError(f"{path}: cannot be read ({exc.strerror or exc})") from None


def load_model(path: Path) -> RobotModel:
    """Read a URDF (``<robot>``) or MJCF (``<mujoco>``) file."""
    root = _parse(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if root.tag == "robot":
        return _urdf(root, path, digest)
    if root.tag == "mujoco":
        return _mjcf(root, path, digest)
    raise ModelError(f"{path}: expected a URDF <robot> or an MJCF <mujoco> root, got <{root.tag}>")


def _urdf(root: ET.Element, path: Path, digest: str) -> RobotModel:
    links = [link.get("name") for link in root.findall("link")]
    joints: list[ModelJoint] = []
    seen: set[str] = set()
    for element in root.findall("joint"):
        name = element.get("name")
        kind = element.get("type")
        if not name or not kind:
            raise ModelError(f"{path}: a <joint> needs a name and a type")
        if name in seen:
            raise ModelError(f"{path}: joint {name!r} is declared twice")
        seen.add(name)
        limit = element.find("limit")
        lower = upper = velocity = effort = None
        if limit is not None:
            what = f"{path}: joint {name} limit"
            lower = _float(limit.get("lower"), what)
            upper = _float(limit.get("upper"), what)
            velocity = _float(limit.get("velocity"), what)
            effort = _float(limit.get("effort"), what)
            if kind in ("revolute", "prismatic") and lower is None and upper is None:
                lower = upper = 0.0  # URDF's defaults for an unspecified range
        if kind == "continuous":
            lower = upper = None  # no position limit, whatever the file says
        parent = element.find("parent")
        child = element.find("child")
        joints.append(
            ModelJoint(
                name=name,
                type=kind,
                lower=lower,
                upper=upper,
                velocity=velocity,
                effort=effort,
                parent=parent.get("link") if parent is not None else None,
                child=child.get("link") if child is not None else None,
            )
        )
    frames = frozenset(name for name in links if name)
    return RobotModel(
        name=root.get("name") or path.stem,
        format="urdf",
        path=path,
        joints=tuple(joints),
        frames=frames,
        digest=digest,
        frame_kinds={name: "link" for name in frames},
    )


def _expand_includes(root: ET.Element, base: Path, depth: int = 0) -> None:
    """Replace each ``<include file=...>`` with the included document's children. As in
    MuJoCo, every include path is relative to the main model file's directory."""
    if depth > _MAX_INCLUDE_DEPTH:
        raise ModelError(f"{base}: <include> nested more than {_MAX_INCLUDE_DEPTH} deep")
    for parent in list(root.iter()):
        for index, child in enumerate(list(parent)):
            if child.tag != "include":
                continue
            file = child.get("file")
            if not file:
                raise ModelError(f"{base}: <include> needs a file")
            included = _parse(base / file)
            _expand_includes(included, base, depth + 1)
            parent.remove(child)
            for offset, grandchild in enumerate(list(included)):
                parent.insert(index + offset, grandchild)


def _mjcf_defaults(root: ET.Element) -> dict[str, dict[str, str]]:
    """Joint attributes per default class, inherited down the ``<default>`` tree."""
    classes: dict[str, dict[str, str]] = {}

    def walk(element: ET.Element, inherited: dict[str, str]) -> None:
        name = element.get("class", "main")
        joint = element.find("joint")
        attributes = {**inherited, **(joint.attrib if joint is not None else {})}
        classes[name] = attributes
        for child in element.findall("default"):
            walk(child, attributes)

    for top in root.findall("default"):
        walk(top, {})
    return classes


def _mjcf(root: ET.Element, path: Path, digest: str) -> RobotModel:
    _expand_includes(root, path.parent)
    compiler = root.find("compiler")
    degrees = compiler is None or compiler.get("angle", "degree") != "radian"
    defaults = _mjcf_defaults(root)
    joints: list[ModelJoint] = []
    frames: dict[str, str] = {}
    seen: set[str] = set()

    def visit(body: ET.Element, body_class: str, parent_body: str | None) -> None:
        name = body.get("name")
        if name:
            frames[name] = "body"
        child_class = body.get("childclass", body_class)
        for site in body.findall("site"):
            if site.get("name"):
                frames[site.get("name") or ""] = "site"
        for camera in body.findall("camera"):
            if camera.get("name"):
                frames[camera.get("name") or ""] = "camera"
        for element in [*body.findall("joint"), *body.findall("freejoint")]:
            joint_class = element.get("class", child_class)
            attributes = {**defaults.get("main", {}), **defaults.get(joint_class, {})}
            attributes.update(element.attrib)
            kind = "free" if element.tag == "freejoint" else attributes.get("type", "hinge")
            joint_name = element.get("name")
            if not joint_name:
                if kind == "free":
                    continue  # an unnamed floating base: nothing a preset can address
                raise ModelError(f"{path}: a <joint> in body {name!r} needs a name")
            if joint_name in seen:
                raise ModelError(f"{path}: joint {joint_name!r} is declared twice")
            seen.add(joint_name)
            what = f"{path}: joint {joint_name} range"
            bounds = _floats(attributes.get("range"), 2, what)
            limited = attributes.get("limited", "auto")
            lower = upper = None
            if bounds is not None and limited != "false":
                lower, upper = bounds
                if kind == "hinge" and degrees:
                    lower, upper = math.radians(lower), math.radians(upper)
            effort = None
            force = _floats(attributes.get("actuatorfrcrange"), 2, f"{path}: {joint_name}")
            if force is not None:
                effort = max(abs(force[0]), abs(force[1]))
            joints.append(
                ModelJoint(
                    name=joint_name,
                    type=kind,
                    lower=lower,
                    upper=upper,
                    effort=effort,
                    parent=parent_body,
                    child=name,
                )
            )
        for child in body.findall("body"):
            visit(child, child_class, name)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ModelError(f"{path}: an MJCF model needs a <worldbody>")
    for site in worldbody.findall("site"):
        if site.get("name"):
            frames[site.get("name") or ""] = "site"
    for camera in worldbody.findall("camera"):
        if camera.get("name"):
            frames[camera.get("name") or ""] = "camera"
    for body in worldbody.findall("body"):
        visit(body, "main", None)
    actuated: list[str] = []
    actuator = root.find("actuator")
    if actuator is not None:
        for element in actuator:
            joint_name = element.get("joint")
            if joint_name and joint_name not in actuated:
                actuated.append(joint_name)
    return RobotModel(
        name=root.get("model") or path.stem,
        format="mjcf",
        path=path,
        joints=tuple(joints),
        frames=frozenset(frames),
        actuated=tuple(actuated),
        digest=digest,
        frame_kinds=frames,
    )
