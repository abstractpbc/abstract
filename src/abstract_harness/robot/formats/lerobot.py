"""A run as a LeRobot v3.0 dataset, without the ``lerobot`` package (no torch on the rig).

One dataset per run, one episode per trial, laid out as ``lerobot`` 0.6.1 writes it:

    lerobot/
      meta/info.json                         features, fps, robot_type, paths
      meta/stats.json                        min, max, mean, std, count, q01..q99
      meta/tasks.parquet                     task_index per instruction, indexed by task
      meta/episodes/chunk-000/file-000.parquet   length, rows, video spans, stats per episode
      data/chunk-000/file-000.parquet        one row per step
      videos/observation.images.<camera>/chunk-000/file-000.mp4   AV1, episodes back to back

Every feature comes from the run (``run.py``): ``action`` is the embodiment's action
with its own labels, ``observation.state`` concatenates its state fields in their
declared order (``<key>.<i>`` names), and each camera is ``observation.images.<name>``
at its own size. ``robot_type`` is the embodiment's name and ``fps`` its control rate,
which must be a whole number of steps per second. Videos are encoded as ``lerobot``
encodes them by default (SVT-AV1, CRF 30, preset 12, a keyframe every 2 frames for
random access in training), from the raw frames. Image stats sample frames and shrink
them the way ``lerobot`` does. tests/test_robot_formats.py loads the result with the real
``lerobot`` package where that is installed.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from abstract_harness.robot.formats.run import Episode, Run, RunError

CODEBASE_VERSION = "v3.0"
CHUNKS_SIZE = 1000
DATA_FILES_SIZE_IN_MB = 100
VIDEO_FILES_SIZE_IN_MB = 200
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH = "meta/episodes/chunk-000/file-000.parquet"
QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}
# lerobot's defaults (configs/video.py).
ENCODER, CRF, PRESET, GOP, FAST_DECODE = "libsvtav1", 30, 12, 2, 0
DEFAULTS = ("timestamp", "frame_index", "episode_index", "index", "task_index")


def _image_key(camera: str) -> str:
    return f"observation.images.{camera}"


def _features(run: Run, fps: int, crf: int, preset: int, gop: int) -> dict[str, Any]:
    features: dict[str, Any] = {
        "action": {
            "dtype": "float32",
            "shape": [len(run.action_labels)],
            "names": list(run.action_labels),
        }
    }
    if run.state_fields:
        names = [
            field.key if field.size == 1 else f"{field.key}.{i}"
            for field in run.state_fields
            for i in range(field.size)
        ]
        features["observation.state"] = {"dtype": "float32", "shape": [len(names)], "names": names}
    for camera in run.cameras:
        features[_image_key(camera.name)] = {
            "dtype": "video",
            "shape": [camera.height, camera.width, camera.channels],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": camera.height,
                "video.width": camera.width,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.fps": fps,
                "video.channels": camera.channels,
                "has_audio": False,
                "video.g": gop,
                "video.crf": crf,
                "video.preset": preset,
                "video.fast_decode": FAST_DECODE,
                "video.video_backend": "pyav",
                "video.extra_options": {},
                "is_depth_map": False,
            },
        }
    for name in DEFAULTS:
        dtype = "float32" if name == "timestamp" else "int64"
        features[name] = {"dtype": dtype, "shape": [1], "names": None}
    return features


def _vector_stats(np: Any, values: Any) -> dict[str, Any]:
    """Stats over axis 0 of an (N, D) array, as lists (``meta/stats.json``)."""
    values = np.asarray(values, dtype=np.float64)
    stats = {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "count": np.array([len(values)]),
    }
    for key, q in QUANTILES.items():
        stats[key] = np.quantile(values, q, axis=0)
    return {key: value.tolist() for key, value in stats.items()}


def _sample_indices(np: Any, length: int) -> list[int]:
    """lerobot's compute_stats.sample_indices: 100 to 10,000 frames, length ** 0.75."""
    least = min(100, length)
    count = max(least, min(int(length**0.75), 10_000))
    return np.round(np.linspace(0, length - 1, count)).astype(int).tolist()


def _image_samples(np: Any, paths: list[Path]) -> Any:
    """Sampled frames as (N, C, H, W) floats in [0, 1], shrunk like lerobot's
    auto_downsample_height_width (to about 150 px when a side exceeds 300)."""
    frames = []
    for index in _sample_indices(np, len(paths)):
        image = np.load(paths[index], allow_pickle=False)
        height, width = image.shape[:2]
        if max(height, width) >= 300:
            factor = int((width if width > height else height) / 150)
            image = image[::factor, ::factor]
        frames.append(np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0)
    return np.stack(frames)


def _image_stats(np: Any, samples: Any) -> dict[str, Any]:
    """Per-channel stats as (C, 1, 1) nested lists, the shape lerobot keeps for images."""
    axes = (0, 2, 3)
    stats = {
        "min": samples.min(axis=axes, keepdims=True)[0],
        "max": samples.max(axis=axes, keepdims=True)[0],
        "mean": samples.mean(axis=axes, keepdims=True)[0],
        "std": samples.std(axis=axes, keepdims=True)[0],
        "count": np.array([len(samples)]),
    }
    for key, q in QUANTILES.items():
        stats[key] = np.quantile(samples, q, axis=axes, keepdims=True)[0]
    return {key: value.tolist() for key, value in stats.items()}


def _encode(av: Any, np: Any, out: Path, paths: list[Path], fps: int, options: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(out), "w", options={"movflags": "faststart"}) as container:
        first = np.load(paths[0], allow_pickle=False)
        stream = container.add_stream(ENCODER, fps, options=options)
        stream.pix_fmt = "yuv420p"
        stream.height, stream.width = first.shape[:2]
        for path in paths:
            image = np.load(path, allow_pickle=False)
            if image.shape != first.shape or image.dtype != np.uint8:
                raise RunError(f"{path.name}: {image.dtype} {image.shape} after {first.shape}")
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _state_rows(np: Any, run: Run, episode: Episode) -> Any:
    columns = [
        np.array(
            [[np.nan if x is None else x for x in row] for row in episode.states[field.key]],
            dtype=np.float32,
        )
        for field in run.state_fields
    ]
    return np.concatenate(columns, axis=1)


def write(
    run: Run, out: Path, *, crf: int = CRF, preset: int = PRESET, gop: int = GOP
) -> dict[str, Any]:
    """Write ``run`` as a LeRobot dataset at ``out`` (replacing what is there); a summary.
    The dataset appears at ``out`` only when complete."""
    try:
        import av
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RunError(f"the robot extra is missing: {exc}") from None
    fps = round(run.fps)
    if abs(run.fps - fps) > 1e-6:
        raise RunError(f"LeRobot needs whole frames per second; the control rate is {run.fps}")
    episodes = [episode for episode in run.episodes if episode.length]
    if not episodes:
        raise RunError("no trial recorded a step")
    part = out.with_name(out.name + ".part")
    shutil.rmtree(part, ignore_errors=True)
    # SVT-AV1 reads its log level once, from the environment: errors only, not a page of
    # its configuration per camera after every run.
    os.environ.setdefault("SVT_LOG", "1")
    features = _features(run, fps, crf, preset, gop)
    tasks = list(dict.fromkeys(episode.task for episode in episodes))
    options = {
        "g": str(gop),
        "crf": str(crf),
        "preset": str(preset),
        "svtav1-params": f"fast-decode={FAST_DECODE}",
    }
    try:
        columns: dict[str, list[Any]] = {name: [] for name in features if name in DEFAULTS}
        actions, states = [], []
        rows, start = [], 0
        for index, episode in enumerate(episodes):
            length = episode.length
            action = np.asarray(episode.actions, dtype=np.float32)
            state = _state_rows(np, run, episode) if run.state_fields else None
            timestamp = (np.arange(length) / fps).astype(np.float32)
            frame_index = np.arange(length)
            per_episode = {
                "timestamp": timestamp,
                "frame_index": frame_index,
                "episode_index": np.full(length, index),
                "index": np.arange(start, start + length),
                "task_index": np.full(length, tasks.index(episode.task)),
            }
            for name, values in per_episode.items():
                columns[name].append(values)
            actions.append(action)
            stats = {"action": _vector_stats(np, action)}
            if state is not None:
                states.append(state)
                stats["observation.state"] = _vector_stats(np, state)
            for camera in run.cameras:
                stats[_image_key(camera.name)] = _image_stats(
                    np, _image_samples(np, episode.frames[camera.name])
                )
            for name, values in per_episode.items():
                stats[name] = _vector_stats(np, values.reshape(-1, 1))
            row: dict[str, Any] = {
                "episode_index": index,
                "tasks": [episode.task],
                "length": length,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": start,
                "dataset_to_index": start + length,
            }
            from_frame = start
            for camera in run.cameras:
                key = _image_key(camera.name)
                row[f"videos/{key}/chunk_index"] = 0
                row[f"videos/{key}/file_index"] = 0
                row[f"videos/{key}/from_timestamp"] = from_frame / fps
                row[f"videos/{key}/to_timestamp"] = (from_frame + length) / fps
            for name in features:
                for stat, value in stats[name].items():
                    row[f"stats/{name}/{stat}"] = value
            row["meta/episodes/chunk_index"] = 0
            row["meta/episodes/file_index"] = 0
            rows.append(row)
            start += length

        # Steps, in the feature order of info.json.
        table: dict[str, Any] = {}
        all_actions = np.concatenate(actions)
        table["action"] = pa.FixedSizeListArray.from_arrays(
            pa.array(all_actions.ravel(), pa.float32()), all_actions.shape[1]
        )
        if states:
            all_states = np.concatenate(states)
            table["observation.state"] = pa.FixedSizeListArray.from_arrays(
                pa.array(all_states.ravel(), pa.float32()), all_states.shape[1]
            )
        for name in DEFAULTS:
            dtype = pa.float32() if name == "timestamp" else pa.int64()
            table[name] = pa.array(np.concatenate(columns[name]), dtype)
        hf_features: dict[str, Any] = {}
        for name, feature in features.items():
            if feature["dtype"] == "video":
                continue
            value = {"dtype": feature["dtype"], "_type": "Value"}
            hf_features[name] = (
                value
                if name in DEFAULTS
                else {"feature": value, "length": feature["shape"][0], "_type": "List"}
            )
        data = pa.table(table).replace_schema_metadata(
            {"huggingface": json.dumps({"info": {"features": hf_features}})}
        )
        (part / "data/chunk-000").mkdir(parents=True)
        pq.write_table(data, part / DATA_PATH.format(chunk_index=0, file_index=0))

        # Videos: each camera's episodes back to back, at the timestamps rows give them.
        for camera in run.cameras:
            paths = [path for episode in episodes for path in episode.frames[camera.name]]
            key = _image_key(camera.name)
            video = part / VIDEO_PATH.format(video_key=key, chunk_index=0, file_index=0)
            _encode(av, np, video, paths, fps, options)

        (part / "meta/episodes/chunk-000").mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), part / EPISODES_PATH)
        pandas = {
            "index_columns": ["task"],
            "column_indexes": [
                {
                    "name": None,
                    "field_name": None,
                    "pandas_type": "unicode",
                    "numpy_type": "object",
                    "metadata": {"encoding": "UTF-8"},
                }
            ],
            "columns": [
                {
                    "name": "task_index",
                    "field_name": "task_index",
                    "pandas_type": "int64",
                    "numpy_type": "int64",
                    "metadata": None,
                },
                {
                    "name": "task",
                    "field_name": "task",
                    "pandas_type": "unicode",
                    "numpy_type": "object",
                    "metadata": None,
                },
            ],
            "attributes": {},
            "creator": {"library": "pyarrow", "version": pa.__version__},
            "pandas_version": "2.3.3",
        }
        task_table = pa.table(
            {"task_index": pa.array(range(len(tasks)), pa.int64()), "task": pa.array(tasks)}
        ).replace_schema_metadata({"pandas": json.dumps(pandas)})
        pq.write_table(task_table, part / "meta/tasks.parquet")

        totals: dict[str, Any] = {"action": _vector_stats(np, all_actions)}
        if states:
            totals["observation.state"] = _vector_stats(np, all_states)
        for camera in run.cameras:
            samples = np.concatenate(
                [_image_samples(np, episode.frames[camera.name]) for episode in episodes]
            )
            totals[_image_key(camera.name)] = _image_stats(np, samples)
        for name in DEFAULTS:
            totals[name] = _vector_stats(np, np.concatenate(columns[name]).reshape(-1, 1))
        (part / "meta/stats.json").write_text(json.dumps(totals, indent=4))
        info = {
            "codebase_version": CODEBASE_VERSION,
            "fps": fps,
            "features": features,
            "total_episodes": len(episodes),
            "total_frames": start,
            "total_tasks": len(tasks),
            "chunks_size": CHUNKS_SIZE,
            "data_files_size_in_mb": DATA_FILES_SIZE_IN_MB,
            "video_files_size_in_mb": VIDEO_FILES_SIZE_IN_MB,
            "data_path": DATA_PATH,
            "video_path": VIDEO_PATH if run.cameras else None,
            "robot_type": run.embodiment,
            "splits": {"train": f"0:{len(episodes)}"},
        }
        (part / "meta/info.json").write_text(json.dumps(info, indent=4))
        shutil.rmtree(out, ignore_errors=True)
        part.rename(out)
    finally:
        shutil.rmtree(part, ignore_errors=True)
    files = [path for path in out.rglob("*") if path.is_file()]
    return {
        "dir": out.name,
        "episodes": len(episodes),
        "frames": start,
        "cameras": [camera.name for camera in run.cameras],
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
    }
