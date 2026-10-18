"""Run formats: a finished run read as robot-neutral episodes and written as LeRobot v3.0,
for robots that are not a YAM (a 7-joint arm with a gripper, and the cubepick sim)."""

import json

import pytest

from abstract_harness.robot import episode as cards
from abstract_harness.robot import formats
from abstract_harness.robot.formats import run as runs

STAMP = "20261012_190000_0a1b2c3d"
LABELS = [f"joint_{i}" for i in range(7)] + ["gripper"]
CAMERAS = {"wrist": (96, 128), "scene": (120, 160)}
TRIALS = (("pick the cube", 6), ("open the drawer", 4))


def make_run(root, *, spaces=True, state=True, frames=True):
    """A finished run as the episode leaves it: two trials of a 7-joint arm with a gripper
    and two cameras of different sizes, each trial with the frame stored after its last
    action."""
    np = pytest.importorskip("numpy")
    run = root / "20261012-190000"
    (run / "actions" / STAMP).mkdir(parents=True)
    (run / "frames" / STAMP).mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows_x, cols_x = np.mgrid[0:160, 0:160]
    samples = []
    for index, (task, length) in enumerate(TRIALS):
        scene = f"scene-{index}"
        rows = [
            {
                "kind": "header",
                "run_id": STAMP,
                "scene_id": scene,
                "epoch": 0,
                "action_dim": 8,
                "labels": LABELS,
            }
        ]
        for t in range(length + 1):
            if frames:
                for camera, (height, width) in CAMERAS.items():
                    image = np.stack([(cols_x + 9 * t) % 256, rows_x % 256, cols_x * 0 + 60], 2)
                    image = image[:height, :width].astype(np.uint8)
                    np.save(run / "frames" / STAMP / f"{scene}-e0_{camera}_{t:06d}.npy", image)
            if t == length:
                break
            row = {"t": t, "action": rng.normal(size=8).round(4).tolist()}
            if state:
                row["state"] = {
                    "joint_pos": rng.normal(size=7).round(4).tolist(),
                    "gripper": [round(float(rng.random()), 4)],
                }
            rows.append(row)
        path = run / "actions" / STAMP / f"{scene}-e0.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        samples.append(
            {
                "instruction": task,
                "trial_metadata": [{"actions": f"actions/{STAMP}/{scene}-e0.jsonl"}],
            }
        )
    info = {"control_hz": 15.0}
    if spaces:
        info["spaces"] = {
            "action": {"shape": [8], "labels": LABELS, "control_mode": "joint_pos"},
            "cameras": [
                {"name": name, "height": height, "width": width, "channels": 3}
                for name, (height, width) in CAMERAS.items()
            ],
            "state": [
                {"key": "joint_pos", "shape": [7], "unit": "rad", "dtype": "float64"},
                {"key": "gripper", "shape": [1], "unit": "normalized", "dtype": "float64"},
            ],
        }
    log = {"eval": {"embodiment": "franka_panda", "embodiment_info": info}, "samples": samples}
    (run / "adhoc_1.json").write_text(json.dumps(log))
    (run / "adhoc_1.rrd").write_bytes(b"RRF2")
    return run


def test_read_takes_the_layout_from_the_run(tmp_path):
    read = runs.read(make_run(tmp_path))
    assert (read.embodiment, read.fps, read.action_labels) == ("franka_panda", 15.0, LABELS)
    assert [(f.key, f.size) for f in read.state_fields] == [("joint_pos", 7), ("gripper", 1)]
    assert [(c.name, c.height, c.width) for c in read.cameras] == [
        ("wrist", 96, 128),
        ("scene", 120, 160),
    ]
    assert [(e.task, e.length) for e in read.episodes] == list(TRIALS)
    # One frame per step; the frame stored after the last action has no row.
    assert [len(e.frames["scene"]) for e in read.episodes] == [6, 4]


def test_read_names_what_a_format_cannot_do_without(tmp_path):
    with pytest.raises(runs.RunError, match="no embodiment spaces"):
        runs.read(make_run(tmp_path / "a", spaces=False))
    with pytest.raises(runs.RunError, match="has no state"):
        runs.read(make_run(tmp_path / "b", state=False))
    with pytest.raises(runs.RunError, match="frames missing"):
        runs.read(make_run(tmp_path / "c", frames=False))


def test_a_run_from_before_formats_still_gets_its_card(tmp_path):
    # No spaces in the eval log and no state in the action log: the card reads it as before.
    run = make_run(tmp_path, spaces=False, state=False)
    assert cards.card(run, "yam-1")["instruction"] == "pick the cube"


def lerobot_deps():
    for name in ("numpy", "pyarrow", "av"):
        pytest.importorskip(name)


def test_lerobot_writes_a_v3_dataset_from_the_run(tmp_path):
    lerobot_deps()
    import av
    import numpy as np
    import pyarrow.parquet as pq

    run = make_run(tmp_path)
    results = formats.write(run, ["lerobot"])
    assert results["lerobot"]["ok"], results
    assert results["lerobot"]["episodes"] == 2 and results["lerobot"]["frames"] == 10
    # Not chosen, and lerobot succeeded: the recording is not kept for upload.
    assert results["rrd"] == {"ok": True, "removed": ["adhoc_1.rrd"]}
    assert not list(run.glob("*.rrd"))
    root = run / "lerobot"
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()) == [
        ".complete.json",
        "data/chunk-000/file-000.parquet",
        "meta/episodes/chunk-000/file-000.parquet",
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.parquet",
        "videos/observation.images.scene/chunk-000/file-000.mp4",
        "videos/observation.images.wrist/chunk-000/file-000.mp4",
    ]
    info = json.loads((root / "meta/info.json").read_text())
    assert (info["codebase_version"], info["robot_type"], info["fps"]) == (
        "v3.0",
        "franka_panda",
        15,
    )
    assert (info["total_episodes"], info["total_frames"], info["total_tasks"]) == (2, 10, 2)
    features = info["features"]
    assert features["action"]["names"] == LABELS
    assert features["observation.state"]["names"] == [f"joint_pos.{i}" for i in range(7)] + [
        "gripper"
    ]
    assert features["observation.images.scene"]["shape"] == [120, 160, 3]
    assert features["observation.images.wrist"]["info"]["video.codec"] == "av1"

    data = pq.read_table(root / "data/chunk-000/file-000.parquet")
    assert data.column_names == [
        "action",
        "observation.state",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    assert str(data.schema.field("action").type) == "fixed_size_list<element: float>[8]"
    source = [
        json.loads(line)
        for path in sorted((run / "actions" / STAMP).glob("*.jsonl"))
        for line in path.read_text().splitlines()[1:]
    ]
    np.testing.assert_allclose(
        data.column("action").to_pylist(), [row["action"] for row in source], rtol=1e-6
    )
    np.testing.assert_allclose(
        data.column("observation.state").to_pylist()[7],
        source[7]["state"]["joint_pos"] + source[7]["state"]["gripper"],
        rtol=1e-6,
    )
    assert data.column("episode_index").to_pylist() == [0] * 6 + [1] * 4
    assert data.column("frame_index").to_pylist() == [*range(6), *range(4)]
    assert data.column("task_index").to_pylist() == [0] * 6 + [1] * 4

    episodes = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    assert [(e["episode_index"], e["length"], e["tasks"]) for e in episodes] == [
        (0, 6, ["pick the cube"]),
        (1, 4, ["open the drawer"]),
    ]
    assert [(e["dataset_from_index"], e["dataset_to_index"]) for e in episodes] == [
        (0, 6),
        (6, 10),
    ]
    key = "videos/observation.images.wrist"
    assert [(e[f"{key}/from_timestamp"], e[f"{key}/to_timestamp"]) for e in episodes] == [
        (0.0, 6 / 15),
        (6 / 15, 10 / 15),
    ]
    assert len(episodes[0]["stats/observation.images.wrist/mean"]) == 3
    tasks = pq.read_table(root / "meta/tasks.parquet")
    assert tasks.to_pylist() == [
        {"task_index": 0, "task": "pick the cube"},
        {"task_index": 1, "task": "open the drawer"},
    ]
    assert json.loads(tasks.schema.metadata[b"pandas"])["index_columns"] == ["task"]
    stats = json.loads((root / "meta/stats.json").read_text())
    assert len(stats["action"]["q99"]) == 8 and stats["action"]["count"] == [10]
    assert np.asarray(stats["observation.images.scene"]["mean"]).shape == (3, 1, 1)

    for camera, (height, width) in CAMERAS.items():
        video = root / f"videos/observation.images.{camera}/chunk-000/file-000.mp4"
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            decoded = list(container.decode(stream))
        assert stream.codec_context.name in ("libdav1d", "av1", "libaom-av1")
        assert (stream.width, stream.height, len(decoded)) == (width, height, 10)
        assert [round(float(frame.time) * 15) for frame in decoded] == list(range(10))


def test_both_formats_keep_the_recording(tmp_path):
    lerobot_deps()
    run = make_run(tmp_path)
    results = formats.write(run, ["rrd", "lerobot"])
    assert (
        results["rrd"] == {"ok": True, "files": ["adhoc_1.rrd"], "verified": False}
        and results["lerobot"]["ok"]
    )
    assert (run / "adhoc_1.rrd").is_file() and (run / "lerobot/meta/info.json").is_file()


def test_a_failed_format_never_raises_and_keeps_the_recording(tmp_path):
    run = make_run(tmp_path, spaces=False)
    results = formats.write(run, ["lerobot", "hdf5"])
    assert not results["lerobot"]["ok"] and "no embodiment spaces" in results["lerobot"]["error"]
    assert not results["hdf5"]["ok"] and "unknown format 'hdf5'" in results["hdf5"]["error"]
    assert (run / "adhoc_1.rrd").is_file() and not (run / "lerobot").exists()
    assert "rrd" not in results
    assert formats.write(run, ["rrd"]) == {
        "rrd": {"ok": True, "files": ["adhoc_1.rrd"], "verified": False}
    }


def test_a_cubepick_episode_becomes_a_lerobot_dataset(tmp_path):
    """The real core end to end on a robot that is not a YAM: one camera of 32x32, a 2-D
    action, two state keys without a StateSpec, 10 Hz."""
    lerobot_deps()
    import pyarrow.parquet as pq

    from abstract_harness.robot.core import eval
    from abstract_harness.robot.core.mock import CubePickEmbodiment, ScriptedPolicy
    from abstract_harness.robot.core.scene import Scene
    from abstract_harness.robot.core.scorer import success_at_end
    from abstract_harness.robot.core.task import Task

    task = Task(
        name="reach",
        scenes=[Scene(id="s0", instruction="reach the cube", init_seed=0)],
        scorer=success_at_end(),
        max_steps=12,
    )
    eval(task, ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path), store_frames=True)
    results = formats.write(tmp_path, ["lerobot"])
    assert results["lerobot"]["ok"], results
    info = json.loads((tmp_path / "lerobot/meta/info.json").read_text())
    assert (info["robot_type"], info["fps"]) == ("cubepick", 10)
    assert info["features"]["action"]["names"] == ["dx", "dy"]
    assert info["features"]["observation.state"]["names"] == [
        "cube_pos.0",
        "cube_pos.1",
        "eef_pos.0",
        "eef_pos.1",
    ]
    assert info["features"]["observation.images.top"]["shape"] == [32, 32, 3]
    data = pq.read_table(tmp_path / "lerobot/data/chunk-000/file-000.parquet")
    assert data.num_rows == info["total_frames"] > 0


def test_lerobot_itself_loads_the_dataset(tmp_path):
    """Upstream reader compatibility; exercised by the manual release-validation job."""
    lerobot_deps()
    datasets = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    import numpy as np

    run = make_run(tmp_path)
    assert formats.write(run, ["lerobot"])["lerobot"]["ok"]
    dataset = datasets.LeRobotDataset("local/franka", root=run / "lerobot")
    assert (len(dataset), dataset.meta.robot_type, dataset.fps) == (10, "franka_panda", 15)
    item = dataset[7]
    assert item["task"] == "open the drawer" and int(item["episode_index"]) == 1
    assert tuple(item["observation.images.scene"].shape) == (3, 120, 160)
    source = json.loads((run / "actions" / STAMP / "scene-1-e0.jsonl").read_text().splitlines()[2])
    np.testing.assert_allclose(item["action"].numpy(), source["action"], rtol=1e-6)


def test_dataset_omits_unexecuted_transitions_and_uses_joint_command(tmp_path):
    run = make_run(tmp_path)
    path = run / "actions" / STAMP / "scene-1-e0.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["action_executed"] = False
    rows[2]["command"] = {"joint_pos": [0.25] * 8}
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    episode = formats.run.read(run).episodes[1]
    assert episode.length == len(rows) - 2
    assert episode.actions[0] == [0.25] * 8
    assert episode.frames["scene"][0].name.endswith("_000001.npy")
