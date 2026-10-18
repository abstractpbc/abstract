# Physical YAM rig: bring-up and reference

`abstract robot` runs an OpenAI model or an action policy on real arms; see the
[README](../README.md) section "Running on Physical Hardware" for what each part of `robot/` owns.
This page is for running it on your own rig: the bring-up order, the configuration, where
runs go and the safety notes. Every step marked **moves** needs an operator at the rig,
hands clear, ready to stop the run (Ctrl-C parks the arms; a second Ctrl-C during the park
leaves them limp).

## The rig

The commands assume a bimanual YAM setup: two I2RT YAM arms with LINEAR_4310 grippers,
each on its own CAN interface (a gs_usb adapter, or a CANable2 with slcan firmware), a
RealSense D435i over the table and a D405 on each wrist, and a Linux host for the arms and
cameras with the `yam` extra (`uv sync --extra yam --group yam-driver`; for a non-YAM robot, install the
embodiment-agnostic `uv sync --extra robot`). A VLA runs on the Abstract Gateway, on its
own GPU server (e.g. a Jetson Thor next to the rig), or in-process.

One TOML holds the whole setup. Start from an example and copy it to `*.local.toml`
(gitignored) before filling in your rig's values:

- [`rigs/yam-abstract.toml`](../rigs/yam-abstract.toml): a hosted or local `/v1` policy
  (`DreamZero-Astra`, `π0.5`, `MolmoAct2`, `Isaac 0.5`);
- [`rigs/yam-agent.toml`](../rigs/yam-agent.toml): an OpenAI model behind the spend meter;
- [`rigs/yam-vla.toml`](../rigs/yam-vla.toml): MolmoAct2 (or GR00T) behind an `/act`
  server.

At least the camera serials, the CAN channels and, for a VLA, `server_url` are yours to
set.

## Before any step

- Both CAN interfaces must be up at 1 Mbit/s. A gs_usb adapter is a kernel CAN device
  (`sudo ip link set can0 up type can bitrate 1000000`). A CANable2 with slcan firmware
  needs slcand after every boot or replug,
  `sudo slcand -o -c -s8 /dev/ttyACM0 can1 && sudo ip link set can1 up`, unless you
  install the unit below.
- Run everything that moves inside tmux. SIGHUP/SIGTERM request Python teardown, but
  dropped sessions, repeated signals and native faults can prevent completion; keep a hardware stop.
- Connecting an arm always runs i2rt's gripper calibration: both grippers sweep to their
  hard stops for a few seconds. Start with the arms folded and resting.

## A CANable2 arm as a systemd unit

[`rigs/canable-slcan/`](../rigs/canable-slcan/) has an example unit and udev rule for an
arm on a CANable2 with slcan firmware, as `can1`. `can1-slcan.service` runs
`slcand -F -o -c -s8` on the adapter's stable `/dev/serial/by-id/` path and brings `can1`
up; it stops when the adapter is unplugged. `70-canable2-can1.rules` starts it whenever
the adapter (USB `16d0:117e`) appears, and keeps ModemManager from probing it: the
adapter enumerates as an AT-command CDC-ACM modem, and ModemManager would race slcand
for it.

Both files name the adapter by its USB serial, a placeholder (`0123456789AB`) in the
examples. Find yours with `ls /dev/serial/by-id/` (the serial is in the adapter's name
there) or `udevadm info -a -n /dev/ttyACM0 | grep -m1 'ATTRS{serial}'`, and put it in the
rule's `ATTRS{serial}` and in the unit's device path. `BindsTo=` and `After=` take that
path as a systemd device unit name, which
`systemd-escape --path --suffix=device /dev/serial/by-id/<name>` prints. Then, with both
arms parked and no run in progress (this drops `can1` for a few seconds), from the
harness checkout:

```sh
sudo pkill -x slcand    # the hand-started can1, if any; can1 disappears
sudo install -m 644 rigs/canable-slcan/can1-slcan.service /etc/systemd/system/
sudo install -m 644 rigs/canable-slcan/70-canable2-can1.rules /etc/udev/rules.d/
sudo systemctl daemon-reload
sudo udevadm control --reload
```

Then unplug and replug the CANable2 (or `sudo udevadm trigger --action=add
/dev/serial/by-id/usb-Openlight_Labs_CANable2_*-if00 && sudo systemctl start
can1-slcan.service`). `systemctl status can1-slcan.service` should be active and
`ip link show can1` UP; replug once more and reboot once to confirm both paths, then
`abstract robot check` reports `can:can1` up. To remove: delete both files, then
`daemon-reload` and `udevadm control --reload`.

## Order

1. **Readiness, motion-free and free.** `uv run abstract robot check --config rigs/yam-agent.toml`
   checks the config, both CAN interfaces, that every motor's CAN loss-of-comms
   failsafe (`TIMEOUT` register, read only) is armed, that every `*_depth_serial` in the
   config is a connected RealSense (librealsense device list, no
   streaming; a camera on a USB 2.x link passes with a `warning`), the API key, that the
   model id exists (`GET /v1/models/<id>`), and runs the doctor, which builds `yam_arms`
   from the config without connecting the driver; any doctor error fails. (In `eef_pos` mode
   `yam_arms` declares a 14-D `joint_pos` and a 14-D `eef_state`; the doctor, like the
   agent, takes `eef_state` as the reference.) Last, a rehearsal runs the agent policy through
   the meter on the simulated `cubepick` embodiment against a local stand-in for the
   Responses API: the real request and response path, with no arms, key
   or spend. It runs twice: once calling `done`, and once with a budget too small for one
   call, which must end on the meter's 402 with a non-zero exit, no upstream call and a
   `blocked` ledger entry.
2. **Camera sides, motion-free.** `[rig.embodiment]` assigns each RealSense serial to a
   slot (`top_depth_serial`, `left_depth_serial`, `right_depth_serial`). When a serial is
   missing, `check` lists the ones librealsense sees. Confirm the wrist sides by covering
   the left gripper's camera and checking which frame goes dark.
3. **Hold check, moves, one arm at a time.**
   `uv run abstract robot holdcheck --config rigs/yam-agent.toml right --zero-gravity false`,
   then `left` (`right` and `left` are the config's `right_channel` and `left_channel`).
   The examples run with `zero_gravity_mode = false`; it must pass. The check releases
   torque when it ends, so support the arm or start it folded.
4. **Smoke run, moves, cents.** A copy of your agent config with `budget_usd = 0.25` and
   `max_calls = 3`, instruction `"Do not move. Call done."`. Exercises connect, homing to
   `yam_arms`' EEF home pose, one model call through the meter, and the park. Read
   `meter-summary.json` and the agent transcript afterwards.
5. **First task, moves.** Cup and box on the table in view of the top camera:
   `uv run abstract robot run --config rigs/yam-agent.toml --allow-paid "put the cup in the box"`.
   The operator grades the episode when it ends.

## VLA runs

[`rigs/yam-vla.toml`](../rigs/yam-vla.toml) points at a MolmoAct2-BimanualYAM `/act`
server: `examples/yam/host_server_yam.py --port 8202 --cuda-graph` in a MolmoAct2
checkout, on a GPU machine (`server_url = "http://<policy-host>:8202"`, or
`http://127.0.0.1:8202` on the same host). `policy = "gr00t"` drives a GR00T `/act`
server through the same client instead (default port 8203). `check` runs one real
inference against the server and fails if it is down, the reply is not a finite N×14
chunk, or the compatibility check finds the policy incompatible with the config. First
runs: keep `max_steps` low, stay at the rig, and watch the first chunk; MolmoAct2 streams
30 actions per inference rather than one deliberate move per call. Joint mode is where
the collision guardrail can work once the arm bases are measured.

**Reference settings.** The config follows AllenAI's reference inference code for
MolmoAct2-BimanualYAM ([`williamtsai726/YAM`](https://github.com/williamtsai726/YAM),
`gello_software/experiments/launch_yaml_eval_molmoact.py` and
`gello/cameras/realsense_camera.py`): RGB at 640×360 (the full 16:9 sensor view), 30 Hz,
start at all-zero joints with grippers open, every action of each 30-step chunk played,
raw i2rt joint positions as state. The YAM client's defaults differ in frame size
(224×224, squashed) and rate (10 Hz); colour (RGB), state and action convention match.
Over USB 2 the D435i has no 640×360 mode; the example shows the keys that capture
1280×720 there (same view) and resize.

**Out-of-distribution scenes hold still.** On a table unlike its training data,
MolmoAct2 tends to hold the arms within a fraction of a radian of home while the server
is healthy: the model card's own training sample (`assets/sample_*_rgb.png`, its state
and task) through the same server moves the arms, and swapping the sample's images,
instruction or state for the rig's shows the rig's images are what keep it still. The
checkpoint's 34 tasks (`allenai/MolmoAct2-BimanualYAM-Dataset`, `meta/tasks.parquet`)
are blocks (spell words, towers, lift/rotate/flip), packing, cloth folding, plates,
barcode scanning, scooping, waste sorting into a dishwasher rack and cable untangling,
all recorded in AllenAI's lab. Try an in-distribution task (e.g. "Fold the cloth" with a
cloth spread flat between both arms, as in the training demos) before concluding
anything about your rig, or fine-tune on demonstrations from it.

Not the cause when it holds still: the action convention (the checkpoint's
`norm_stats.json` has matching action and state means, so absolute targets, as the `/act`
client assumes); the chunk `dt_ms` (the server reports inference time and the client
labels it a rate, but the episode treats a chunk's rate as advisory and nothing reads
it); the start pose (the reference starts at all zeros). Check that joint commands reach
the motors: a wrist camera shows its own wrist's commanded motion.

## Run formats (`[data]`)

A finished run is kept in the formats the robot TOML names, the Rerun recording by
default:

```toml
[data]
formats = ["rrd"]               # the Rerun recording (default)
# formats = ["lerobot"]         # a LeRobot v3.0 dataset instead
# formats = ["rrd", "lerobot"]  # both
```

The formats (`robot/formats/`) run after the episode and before any upload. Verified offload may delete
the run and its raw frames, and they put their output in the run directory, so the upload
sends it like any other file. Nothing in them names a robot. The episode records the
embodiment's own layout in the eval log (`eval.embodiment_info.spaces`: the action's shape
and labels, each camera's name and size, the state fields in their declared order) and,
on every action log row, the state that action was chosen from. A YAM, a Franka or a UR
goes through the same code.

- `rrd`: the recording the episode wrote live. The format checks it is there.
- `lerobot`: `lerobot/` in the run, a LeRobot v3.0 dataset laid out as `lerobot` 0.6.1
  writes one, one episode per trial. `action` carries the embodiment's labels,
  `observation.state` concatenates the state fields (`<key>.<i>`), and each camera is
  `observation.images.<camera>`, AV1 video from the raw frames with lerobot's defaults
  (CRF 30, preset 12, a keyframe every 2 frames). `robot_type` is the embodiment's name
  and `fps` its control rate, which must be a whole number. It is written with pyarrow and
  PyAV, no torch, and `lerobot` 0.6.1 loads it (`tests/test_robot_formats.py`). A
  781-step run with three 640x360 cameras converts in about 6 s and comes to 18 MB,
  against 41 MB for its `.rrd`.

When `rrd` is not chosen, the live recording is removed once every chosen format has
succeeded; if one fails, the recording stays. A failed format makes the run command return nonzero; upload can still preserve available
artifacts, while raw frames remain local. Its error is in the summary under `formats`. Runs from before the formats have no
spaces in their eval log and no state in their action logs, so they cannot be converted.

## Where runs go (`[store]`)

A run is written under `[rig] log_root` (default `$XDG_DATA_HOME/abstract/robot/runs`, else
`~/.local/share/abstract/robot/runs`). Without `[store]`, runs stay there. With `[store]`
in the robot TOML, selected artifacts are uploaded after the episode and attempted teardown. Then `run` sends every file of the run but the raw camera frames (`frames/`,
`.npy`, 99% of a run's bytes), which the Rerun recording carries as JPEG images, checks
each one where it landed, and only then marks the run complete; the run's labels go with
it (below). The run directory is deleted only if selected exports have matching completion
receipts. Missing or degraded recordings and failed conversions retain raw frames locally,
with a `.uploaded.json` receipt to prevent repeated automatic uploads. Repair and explicitly
upload that run again when ready. One line per run stays in `<log_root>/runs.jsonl`, including
`local_retained` and its reason when applicable. A failed upload keeps the run and does not change the run's exit
code. `[rig] name` (default: the hostname, lower case) is the robot's name in the store.

```toml
[rig]
name = "yam-1"

[store]
route = "abstract"   # or "bucket", with bucket (required), prefix (default: the rig's name), env_file
```

Two routes, one interface (`robot/store.py`):

- `route = "abstract"`: the Abstract runs API, under the account of the API key, the
  user's own; nothing else names the account. The key is
  `ABSTRACT_API_KEY`, else the one `abstract login` saved (`credentials.json` in
  `$ABSTRACT_CONFIG_DIR`, else `$XDG_CONFIG_HOME/abstract`, else `~/.config/abstract`). The
  API base is `[store] api_url`, else the one saved with the key, else
  `https://api.abstract.inc/v1`. Each file goes up with its SHA-256, which the API checks
  (above 64 MiB in 32 MiB parts); a file already stored with that digest is skipped. Then
  `complete` lists every file and the run's catalog row: the API checks each file is there
  with its size and that nothing else is, and writes the run's manifest. Runs are read back
  through the API: `GET /v1/robot/runs?robot=<name>` lists the completed runs with their
  labels, `GET /v1/robot/runs/<name>/<run>` is a run's manifest, and
  `.../files/<path>` one of its files. The API reference is at
  [abstract.inc/docs](https://abstract.inc/docs).
- `route = "bucket"`: a Cloudflare R2 bucket you own, which the rig writes itself over its
  S3 API (`cloud.py`), with R2 credentials in `env_file`. `bucket` has no default: name
  yours. Any R2 bucket works; the variable names below are only how the client reads its
  credentials. Each object's size and MD5 ETag is checked there; `manifest.json` goes
  after the run's other objects and lists every file with its SHA-256, so a prefix
  without it is an incomplete upload, and the run's line goes into the bucket's catalog:

```text
<bucket>/
  <prefix>/episodes.jsonl      the catalog: one line of labels per run, sorted by run
  <prefix>/<YYYY>/<MM>/<DD>/<run>/
    episode.json               labels and timeline markers (below)
    adhoc_<id>.json            eval log (operator score, termination)
    adhoc_<id>.rrd             Rerun recording: cameras (JPEG q75), joint state, actions
    actions/<trial>/scene-0-e0.jsonl   every commanded action
    setup.json                 components and arguments, credential fields redacted
    run-summary.json | meter-summary.json, meter.jsonl, transcripts/, wire/ (agent runs)
    manifest.json              written last; lists every object
```

`<prefix>` is `[store] prefix`, else the rig's name, and `<run>` the run's timestamp
(`YYYYMMDD-HHMMSS`).

The `.rrd` is the one file to look at: `rerun adhoc_<id>.rrd` after downloading it. It is
the only record of joint state, which the eval log does not keep, and the only copy of the
cameras: one JPEG (quality 75) per camera per step. Its frames can thin out when the
recorder falls behind (the run prints a `RerunSink dropped N camera frame(s)` warning; the
control loop never waits for it); the manifest's `frames` counts the raw frames the run
stored, to check against.

```sh
uv run abstract robot upload --config rigs/yam-vla.toml            # every finished run
uv run abstract robot upload --config rigs/yam-vla.toml <run>      # one run
```

`upload` skips runs still being written (no summary and a file changed in the last 10
minutes) and files the store already holds byte for byte, so it is safe to repeat.
`check`'s `store` line proves the credentials: on the API route it lists one run and names
the API and the key's safe prefix (`abs_live_<8>…<6>`, never the key); on the bucket route
it writes and reads back `<prefix>/.check.json`, which also proves the clock.

For the bucket route, create the bucket and a token once, in the Cloudflare dashboard:
R2 → Create bucket (automatic location, Standard class, no public access), then R2 →
Manage API tokens → Create API token: Object Read & Write, applied to that bucket only.
On the rig, `~/.config/abstract/r2.env` (`chmod 600`), or the file `env_file` names:

```sh
R2_ACCOUNT_ID=<32-character account ID>
R2_RUNS_ACCESS_KEY_ID=<access key ID>
R2_RUNS_SECRET_ACCESS_KEY=<secret access key>
```

The same three variables in the environment take precedence over the file.

### Episode labels and markers

Each run's `episode.json` (`robot/episode.py`) is built from its eval log, summary,
action log and, for an agent, its transcript:

- what was asked: `instruction`, `task` (the instruction as a slug);
- who acted: `policy` (the checkpoint or model, e.g. `isaac-0.5`, `molmoact2`,
  `gpt-6.1-sol`), `kind` (`vla` or `agent`), `client` (the registered policy client),
  `server_url` or `effort`, `config` (the TOML file name);
- how it ended: `outcome` (`success`, `partial`, `failure`, `skipped` from the operator's
  verdict; `error`; `unscored`), `score`, `verdict`, `note`, `termination`, `error`;
- what each part of the robot was commanded to do: `parts`, `control_mode`,
  `control_interface` (`joints` or `eef_pos`), `arms_moved`, `displacement` (per part, the
  farthest any position axis was commanded from where it started, in `displacement_unit`:
  `rad` in joint control, `m` on x, y, z in cartesian control; per-step changes are summed
  and velocities integrated at `control_hz`), `gripper_closes`; measured joint state is in
  the `.rrd`. The parts are the embodiment's own (`ActionSemantics.dim_parts`: `left` and
  `right` on the YAM), else the sides of a symmetric `<side>_<name>` layout, else one part,
  `robot` (a Franka, a UR). `arms_moved`, `displacement` and `gripper_closes` keep their
  names from the YAM-only cards and are keyed by part; runs from before the embodiment's
  spaces are read by their action labels;
- timing, cost and versions: `started_at`, `duration_s`, `steps`, `control_hz`,
  `inference_latency_s`, `spent_usd`, `model_calls`, `hindsight`, `git_commit`,
  `harness_version`;
- `tags` (from `run --tag` and `label --tag`) and `overrides` (from `label`);
- `markers`, each at a step `t`: `operator_message` (a typed steering line),
  `tool_call` (an agent's tool, its note and arguments, `until` the next call: one
  segment of the task in the agent's words), `gripper_close` and `gripper_open` per part
  (`part`, and `arm` as before),
  and `end` (outcome, termination, error, note). A step indexes the action log and the
  `.rrd`'s `step` timeline, where the camera images are.

The catalog line is the card without the markers (it counts them), plus where the run is.
On the API route the listing carries it (its main fields: run, task, policy, kind, outcome,
verdict, tags, steps, duration_s, started_at; the manifest has the whole row); on the bucket
route it is the bucket's catalog. Read either from any machine and filter it:

```sh
curl -s -H "Authorization: Bearer $(abstract token)" \
  "https://api.abstract.inc/v1/robot/runs?robot=yam-1&limit=1000" \
  | jq -c '.runs[] | select(.episode.outcome == "success") | {run, policy: .episode.policy}'
CLOUDFLARE_ACCOUNT_ID=<account id> npx wrangler r2 object get \
  my-robot-runs/yam-1/episodes.jsonl --remote --pipe \
  | jq -c 'select(.outcome == "success" and .kind == "vla") | {run, policy, task, prefix}'
```

Label at run time with `--tag` (repeatable; lower case, no spaces), or afterwards:

```sh
uv run abstract robot run --config rigs/yam-vla.toml --tag cloth --tag demo "Fold the cloth"
uv run abstract robot label --config rigs/yam-vla.toml <run> --tag good-demo
uv run abstract robot label --config rigs/yam-vla.toml <run> --policy isaac-0.5
uv run abstract robot label --config rigs/yam-vla.toml <run> --outcome partial
uv run abstract robot label --config rigs/yam-vla.toml --all    # rebuild every card
```

`run --no-upload` ends the run at its summary and keeps its directory: its `{"upload"}`
line says it was skipped, and `abstract robot upload` sends it later (`serve` does, in the
background).

`run --no-verdict` skips the verdict and grader-notes prompts at the end of the episode
(demos, `serve --no-verdict`): the run is `unscored` unless `/y`, `/n` or `/p` was typed
at its console, and an embodiment's own success or failure is not adopted as the verdict
either. `label --outcome` sets it later. The home and start gates still ask: they are
safety, not labeling.

`label` rebuilds each card from the run's uploaded files, so a run uploaded before
labels existed gets one, and keeps the tags and overrides the card had; `--policy` and
`--outcome` are stored as overrides and survive later rebuilds. It rewrites
`episode.json` and the run's manifest and catalog line (on the API route by completing the
run again), and nothing else.

## HTTP control (`serve`)

The full API for programs that drive the rig is in [RPC.md](RPC.md).

`abstract robot serve` starts and steers runs over HTTP and streams what they do. Each run
is `abstract robot run` itself, unchanged (same refusals, meter, summary and upload), on a
pseudo-terminal the server owns. The server follows the run through the events the run
writes to a pipe it is given as `--events-fd` (below), and types answers into its
terminal as an operator would. The protocol is Pi's RPC mode
(pi.dev/docs/latest/rpc) over HTTP instead of stdin/stdout. A client POSTs one command,
`{"id", "type", ...}`, to `/rpc` and gets `{"id", "type": "response", "command", "success",
"data" | "error"}` back. Everything after acceptance arrives on `/events`. Prompts that
block a run are `extension_ui_request` events, answered with `extension_ui_response`, as
Pi does for extension dialogs.

```sh
# on the rig, in tmux (the server parks a running episode before it exits)
uv run abstract robot serve --config rigs/yam-vla.toml                         # 127.0.0.1:8780
uv run abstract robot serve --config rigs/yam-vla.toml --host <private-address>
uv run abstract robot serve --config rigs/yam-vla.toml --no-verdict  # demo: no grading
uv run abstract robot serve --config rigs/yam-vla.toml --no-verdict --auto-start \
  --host <private-address>                                # live demo: no clicks per run
# from another machine: ssh -L 8780:127.0.0.1:8780 <rig>, then
# http://localhost:8780/ : cameras, terminal, run controls, gate and verdict dialogs
```

Bind a non-loopback `--host` only on a network you trust (a VPN or a private LAN); an
SSH tunnel to the default `127.0.0.1` needs nothing else.

Demo mode, `serve --no-verdict`: the server as the robot's console. Type a prompt, the run
starts and shows live on the page, and when it ends the server waits for the next prompt.
Every run it starts gets `run --no-verdict`, so none asks for a verdict or grader notes
(they stay `unscored` unless ended with a verdict, `stop {verdict}` or `/y /n /p`).
`get_state` says `noVerdict: true`, and the page shows "verdicts off". The home and start
gates still ask before every run, unless serve also has `--auto-start`.

Live demo, `serve --no-verdict --auto-start`: runs started by a relay (for example posts
read from a social feed and sent as `prompt`) with nobody clicking. Every run gets
`-E auto_start=true`, yam_arms' option that drops the stand-clear and scene-ready gates:
homing starts when the prompt arrives and the episode right after it. Esc, End episode and
Ctrl-C (Abort) still stop a run and park the arms; `unattended` is never used, because it
also drops the end-of-episode poll. The page shows a red "gates off" pill. `hold`
(`{"type": "hold", "on": true}`, the page's "Pause new runs") refuses new prompts while
the scene is reset; a running episode goes on, and `get_state.held` says so. With the gates
off, the person at the rig is the only check: someone stays there with the e-stop for the
whole demo, pauses new runs before reaching into the workspace, and the relay keeps the
RPC token on a machine you control, never anywhere public.

Uploads in the background: with a `[store]`, every run serve starts gets
`run --no-upload`, so it ends at its summary and keeps its directory. The phase goes back to
`idle` as soon as the run's process exits (arms parked, devices released), and Start run
works again at once. Serve then uploads the run with `abstract robot upload --config <cfg>
<run>`, one run at a time in a queue, while the next run goes on: `upload_queued`,
`upload_start` and `upload_end` (`run`, `ok`, `error`, `bytes`) events, `uploads` in
`get_state` (`queued`, `current`, `last`), an "uploading <run>" pill on the page and the
result in its terminal pane. A failed upload leaves the run on disk, says so, and the next
`abstract robot upload` retries it. On shutdown with no run going, serve waits for the
upload under way and prints the runs it leaves on disk. When Start run is disabled, the page
says why (a run in progress, its data and summary being written, a check).

| Command | Effect |
|---|---|
| `prompt` `{message, tags?}` | start `robot run` with that instruction, each tag as a `--tag` episode label (`data.disposition: "started"`) |
| `steer` `{message}` | one line to the policy, as typed at the operator console |
| `stop` `{verdict?: y\|n\|partial, note?}` | end the episode from the console (`/stop`, `/y`, `/n`, `/p`); the arms park |
| `abort` | Ctrl-C: stop now and park |
| `extension_ui_response` `{id, confirmed\|value\|cancelled}` | answer a gate (Enter; cancel is Ctrl-C), the verdict (`y n partial skip`) or the grader notes |
| `hold` `{on}` | pause or resume new runs |
| `check` | `abstract robot check`, results as `check_result` events |
| `get_state`, `get_output {lines}`, `get_commands` | phase, run, step, pending dialogs, status line, last run; recent terminal lines |

Events (`GET /events`, server-sent; `?since=<seq>` replays from the buffer, and a
reconnecting stream resumes after its `Last-Event-ID`; the page also reads the state again
on every reconnect): `run_start`, `run_log_dir`, `output` (terminal lines), `status` (the sticky
line: homing, settling, `Running.`, parking), `extension_ui_request` /
`extension_ui_resolved` (`by` rpc, terminal or exit, and how it was answered), `progress`
(control steps, at most 4 a second), `transcript` (agent messages), `trial_end`,
`episode_end` (exit code; the arms are parked), `run_end` (exit code, summary, upload
results, `events`, and the run's refusal as `runError`), `hold`, `upload_queued`,
`upload_start`, `upload_end`, `check_*`, `preview`.

Run events. `abstract robot run --events-fd N` also writes the run as JSON lines,
`{"t", "type", ...}`, to the inherited pipe N; without the flag it writes nothing new, and
the terminal, stdout lines and prompts are the same either way. A writer thread owns the
pipe, so a reader that stalls loses events (a `dropped` count) but never blocks the
episode. The source is `robot/events.py`:

| Event | Fields | Sent by |
|---|---|---|
| `hello` | `version`, `pid` | the channel, first |
| `log_dir` | `path` | `run` |
| `dialog`, `dialog_resolved` | `id`, `kind` (gate, verdict, notes), `title`, `options`; `answer` | the operator session's gates, verdict and notes prompts |
| `status` | `text` (null when the line closes) | the operator session |
| `progress` | `trial`, `step` | `run_task` (a log sink) |
| `frame` | `trial`, `camera`, `step`, `path` of the whole `.npy` | the frame store |
| `transcript` | `trial`, `step`, `message` | `run_task` |
| `trial_end` | `trial`, `status`, `termination_reason`, `verdict` | `run_task` |
| `episode_end` | `exit_code` | the episode, after the park |
| `summary`, `upload` | the `{"summary"}` and `{"upload"}` line objects | `run` |
| `error`, `dropped` | `message`; `count` | the channel |

A dialog's `answer` is `confirmed`, the verdict as typed (or `unrecognized`), `text` or
`empty` for the notes (never the note itself), `interrupted` (Ctrl-C) or `failed`.

Feed: `GET /feed/<camera>` is an MJPEG-style stream that an `<img>` plays, and
`/feed/<camera>.png` is one frame. During a run it is the newest frame the run stores, as
its `frame` events name them (a full-size JPEG; PNG without OpenCV). Between runs it is a
RealSense preview from the config's `*_depth_serial` cameras (JPEG, 1280x720 where the
link allows it, else 640x480) that starts while someone watches and stops before a run or
check opens the cameras.

Safety: no client passes `-E`/`-P` and nothing sets `unattended`, so the YAM gates block
every start unless the operator started serve with `--auto-start` (above). Confirming one
over HTTP is pressing Enter at the rig: do it only when you can see the arms, through the
feed or in person, and someone can reach the e-stop. `--allow-paid` on `serve` is
required before clients can start an `[agent]` config. Every endpoint except `/` and
`/health` needs the token from `~/.config/abstract/rpc.env` (created on first start, 0600)
as `Authorization: Bearer` or `?token=`. The serving terminal mirrors the run and passes
keys through: Esc, Enter and Ctrl-C work there as in `robot run`, and Ctrl-C with no run
stops the server.

```sh
TOKEN=$(sed -n 's/^ABSTRACT_RPC_TOKEN=//p' ~/.config/abstract/rpc.env)
curl -s localhost:8780/rpc -H "Authorization: Bearer $TOKEN" -d '{"type":"get_state"}'
curl -s localhost:8780/rpc -H "Authorization: Bearer $TOKEN" \
  -d '{"id":"1","type":"prompt","message":"fold the cloth"}'
curl -sN "localhost:8780/events?since=0&token=$TOKEN"
```

## Safety notes

- A MolmoAct2 `/act` server (`host_server_yam.py`) listens without authentication, so
  anyone on its network can drive inference (not the arms). Bind it to a private address
  or firewall it before using a shared network.
- Every motor needs its CAN loss-of-comms failsafe: a motor with `TIMEOUT = 0` keeps
  holding torque, executing its last command, after its controller exits. `check` fails if
  any motor reads 0 and names the fix, i2rt's
  `motor_config_tool/set_timeout.py --channel <can> --timeout <value>`; set every motor
  of both arms to the same nonzero value.
- A D435i on a USB 2 link has less bandwidth: the examples run depth at 15 fps, and the
  VLA example shows the capture sizes that work there. `check` warns about a USB 2 link;
  a USB 3 port is better.
- The collision guardrail is off in the examples: it needs the arm bases and table height
  measured (`collision_*_base_pos`), and `yam_arms` skips it in `eef_pos` mode anyway.
  Until then nothing stops the two arms meeting except the per-arm workspace bounds and the
  operator.
- `yam_arms`' EEF home pose is provisional. Watch the first homing ramp.
- i2rt is pinned to Git commit `ac096928d6899ddf852a71c5e8fbaa6055cd9745`
  in the separate `yam-driver` dependency group
  (`enable_auto_recovery=False`: a motor error fails the episode instead of being cleared
  inside the control loop), the one `robot/yam` was validated on. i2rt's own `close()`
  closes the CAN socket without joining its control loop; `abstract robot run`
  (`yam_arms`) and `holdcheck` close through `close_robot_safely`, which joins the loop
  first.
- An agent run prints a `live:` hint suggesting `abstract robot view ... --host 0.0.0.0`.
  Nothing is served unless someone runs it; on a shared network bind the viewer to
  `127.0.0.1` or a private address instead.
- Model id and prices come from the operator. `check` confirms the id; prices are not
  checked against the provider. Configure a provider-side spend limit as well.
- OpenAI bills requests above 272K input tokens at long-context rates (the agent
  example's `[agent.long_context_pricing]`: $4 input, $0.20 cached, $5 cache write, $15
  output per million). The meter refuses any request whose input estimate exceeds
  `long_context_threshold` (default 272000); set `[agent.long_context_pricing]` so a call
  reported above it anyway is settled at those rates instead of blocking the rest of the
  run.

For cancellation, inference deadlines, advisory claims, measured references, lifecycle
collision checks and the limits of parking, see [SAFETY.md](SAFETY.md).
