# Robot RPC: `abstract robot serve`

`abstract robot serve` lets another program run the rig over HTTP. A client sends a
prompt and serve starts `abstract robot run` with it as the instruction, the same run an
operator types, with the same refusals, guardrails, meter, summary and upload. The client
follows the run on an event stream, can steer or stop it, and watches the cameras. The
rig runs one episode at a time. The protocol is Pi's RPC mode (pi.dev/docs/latest/rpc)
over HTTP: one command per POST, everything after it on server-sent events. The code is
`src/abstract_harness/robot/rpc.py`.

## Starting the server

On the rig, inside tmux, to keep the supervising terminal stable:

```sh
uv run abstract robot serve --config rigs/yam-vla.toml --host 127.0.0.1
```

| Flag | Effect |
| --- | --- |
| `--config` | The robot TOML. It fixes the policy, the rig and the store for every run. |
| `--host`, `--port` | Bind address, default `127.0.0.1:8780`. |
| `--allow-paid` | Lets clients start runs of an `[agent]` config, which call a paid model. |
| `--no-verdict` | Demo mode: runs end without the verdict and grader-notes prompts, and are stored unscored. |
| `--auto-start` | Live demo: every run gets `-E auto_start=true` and skips the two readiness gates (see Safety). |
| `--token-file` | Where the token is kept, default `~/.config/abstract/rpc.env`. |
| `--env-file` | The file with the model's API key, for `[agent]` configs. |
| `--allow-no-tmux` | Serve outside tmux. Not for real arms. |

Ctrl-C in serve's terminal stops it when no run is active. During a run, Ctrl-C reaches
the episode, which parks the arms. Keys typed in that terminal go to the run (Enter, Esc,
Ctrl-C), as in `robot run`.

## Authentication

Every endpoint except `/` and `/health` needs the token, `ABSTRACT_RPC_TOKEN` in
`~/.config/abstract/rpc.env`. serve creates that file (mode 0600) on its first start.
Send the token as `Authorization: Bearer <token>`, or as `?token=<token>` where a header
can't be set (an `<img>`, an `EventSource`). A wrong or missing token gets 401
`{"error": "missing or wrong token"}`. Whoever holds the token can move the arms: keep it
on trusted machines and serve on localhost or a private VPN.

## Endpoints

| Method | Path | Token | Returns |
| --- | --- | --- | --- |
| GET | `/` | no | The control page (cameras, terminal, run controls, dialogs). |
| GET | `/health` | no | `{"ok": true, "phase": "idle"}` |
| POST | `/rpc` | yes | One command, one response (below). |
| GET | `/events` | yes | Server-sent events (below). |
| GET | `/feed/<camera>` | yes | `multipart/x-mixed-replace; boundary=frame`. An `<img>` plays it. `?fps=` 1 to 15 (default 5); `?width=` shrinks wider JPEG frames to that width. |
| GET | `/feed/<camera>.png` (or `.jpg`) | yes | One frame. 503 if none arrives within 8 s. |

Camera names are in `get_state.cameras` (`top_cam`, `left_cam`, `right_cam`, plus any
other camera the running episode stores). During a run, a feed is the newest frame the run
stored, as a full-size JPEG (640x360 for YAM policy frames). Between runs it is a
RealSense preview (JPEG, 1280x720 where the camera's link allows it, else 640x480). The preview starts while someone watches and stops before a run or check opens
the cameras. 404 for an unknown camera or path.

## Commands: `POST /rpc`

The body is one JSON object of at most 64 KiB: `{"type": "<command>", "id": "<optional,
echoed back>", ...fields}`. The answer is always HTTP 200 for a command that parsed:

```json
{"id": "1", "type": "response", "command": "prompt", "success": true, "data": {"disposition": "started", "run": "run-1a2b3c4d"}}
{"id": "2", "type": "response", "command": "prompt", "success": false, "error": "the rig is running; steer, stop or abort the run first"}
```

Other failures: invalid JSON gets 400, with command `parse` and `success: false`; a
missing body or one over 64 KiB gets 413. `success: true` means accepted. What follows
arrives as events.

| Command | Fields | `data` | Refused when |
| --- | --- | --- | --- |
| `get_state` | none | The state (next section). | never |
| `get_commands` | none | `{"commands": [...]}` | never |
| `get_output` | `lines` (1 to 1000, default 200) | `{"lines": [...]}`: the last terminal lines of the current or last run | `lines` is out of range |
| `prompt` | `message` (string, required), `tags` (list of strings), `streamingBehavior` | `{"disposition": "started", "run": "run-…"}` | see below |
| `steer` | `message` | `{"disposition": "sent"}` | no run is active; the message starts with `/` |
| `stop` | `verdict` (`y`, `n` or `partial`, optional), `note` (≤ 500 chars, optional) | none | no run is active |
| `abort` | none | none | no run is active |
| `extension_ui_response` | `id`, plus `confirmed: true`, `cancelled: true` or `value` | none | no such pending dialog, or a wrong answer for its kind |
| `hold` | `on` (true or false) | `{"held": true}` | `on` is not a boolean |
| `check` | none | `{"disposition": "started"}` | the phase is not `idle` |

**prompt.**
- **Message:** the instruction for the run, one line of 1 to 2000 characters with no
  control characters. Collapse newlines before sending.
- **Tags:** episode labels, lowercased; each must be 1 to 64 of `a-z 0-9 . _ : = + -`,
  with no spaces.
- **Refused when:**
  - the operator paused new runs: `held: the operator paused new runs (hold off to start one)`;
  - the phase is not `idle`: `the rig is running; …`, or `finishing` or `checking`;
  - the config calls a paid model and serve has no `--allow-paid`;
  - the message or a tag is invalid.
- **During a run:** a prompt with `"streamingBehavior": "steer"` is sent as `steer`.

**steer** types one line to the policy, as the operator console does.

**stop** ends the episode from the console: `/stop`, or `/y`, `/n`, `/p` when a verdict is
given (the note follows it). The arms park as at any episode end.

**abort** requests Ctrl-C cancellation. The owning process attempts teardown and normally
exits with code 130. This is a cooperative software control, not an immediate actuator stop.

**extension_ui_response** answers a dialog the run opened (`extension_ui_request`):

| Dialog | `method` | Answer | `cancelled: true` |
| --- | --- | --- | --- |
| Gate: stand clear, then position the scene | `confirm` | `confirmed: true` (Enter) | Ctrl-C: the run ends and parks |
| Verdict, without `--no-verdict` | `select` | `value`: `y`, `n`, `partial` or `skip` | `skip` |
| Grader notes, without `--no-verdict` | `input` | `value`: ≤ 500 chars, or empty | empty |

Confirm a gate only when someone can see the arms and reach the e-stop: it is pressing
Enter at the rig.

**hold** with `on: true` refuses new prompts until `on: false`. A running episode goes on.
The operator uses it to reset the scene between runs.

**check** runs `abstract robot check`, which is motion-free. Its results arrive as events.

## Phases and state

`idle` → `prompt` → `running` → `episode_end` (teardown returned) → `finishing` (the run writes
its data and summary) → run process exits → `idle` + `run_end`. During `check` the phase
is `checking`, then back to `idle`. A prompt is accepted only when the phase is `idle` and
nothing is held. With a `[store]`, runs end without uploading and serve uploads them
afterwards, one at a time, so an upload never blocks the next prompt.

`get_state` data:

| Field | Meaning |
| --- | --- |
| `phase` | `idle`, `running`, `finishing` or `checking` |
| `policy` | `agent:<model>` or the VLA policy name |
| `config` | The robot TOML's path. |
| `allowPaid`, `noVerdict`, `autoStart` | serve's flags |
| `held` | New runs are paused. |
| `run` | `null`, or the active run: `id`, `instruction`, `tags`, `startedAt`, `logDir`, `pid`, `step` |
| `pendingDialogs` | Open dialogs: `id`, `method`, `title`, `options` (verdict only) |
| `status` | The run's sticky status line (homing, settling, `Running. Max ~120s.`, parking), or `""` |
| `lastRun` | The last finished run: the `run` fields plus the `run_end` fields |
| `uploads` | `{"queued": [run, …], "current": run or null, "last": {run, ok, error, bytes} or null}` |
| `cameras` | Feed names. |
| `preview` | `{"on": bool, "viewers": n}` |
| `eventSeq` | The last event's number: pass it as `since` to read only what comes next. |

## Events: `GET /events`

Server-sent events. Each one is `id: <seq>` then `data: <json>`. The JSON always has
`type`, `seq` (increasing) and `time`. A comment line (`: keepalive`) comes every 15 s
with no events.
- **Where it starts:** `?since=<seq>` replays everything after that number. Without it,
  only new events arrive.
- **Reconnects:** a reconnecting client's `Last-Event-ID` header wins over `since`, so a
  standard `EventSource` resumes where it stopped.
- **Buffer:** the last 5000 events are kept.

| Event | Fields | When |
| --- | --- | --- |
| `run_start` | `id`, `instruction`, `tags`, `startedAt`, `logDir`, `pid`, `step` | A prompt started a run. |
| `run_log_dir` | `id`, `logDir` | The run named its log directory. |
| `output` | `text` | A terminal line (ANSI removed, at most 4000 chars). |
| `status` | `text` | The status line changed; `""` when it closes. |
| `extension_ui_request` | `id`, `method`, `title`, `options` | A dialog waits for an answer. |
| `extension_ui_resolved` | `id`, `by` (`rpc`, `terminal` or `exit`), `answer` | It was answered: `confirmed`, the verdict (or `unrecognized`), `text`, `empty`, `interrupted` or `failed`; none when `by` is `exit`. |
| `progress` | `id`, `step` | Control steps, at most 4 a second. |
| `transcript` | `id`, `message` | An agent policy's message. |
| `trial_end` | `id`, `status`, `termination_reason`, `verdict` | The trial ended. |
| `episode_end` | `id`, `exitCode` | Episode execution returned. The phase is now `finishing`; physical state requires independent verification. |
| `run_end` | `id`, `exitCode`, `summary`, `upload`, `logDir`, `events`, `runError`, `error`, `tail` | The run process exited. The phase is already `idle`, so the next prompt can go. `events: false` means a run from an older harness, which sent no events. `tail` (the last terminal lines) comes when the run failed before its episode. |
| `events_dropped` | `id`, `count` | The run's event pipe overflowed. |
| `hold` | `on` | New runs were paused or resumed. |
| `upload_queued`, `upload_start` | `run` | Background upload of a finished run. |
| `upload_end` | `run`, `ok`, `error`, `bytes` | A failed upload leaves the run on disk for `abstract robot upload`. |
| `check_start`, `check_result`, `check_end` | `result`; `ready`, `exitCode` | A `check`. |
| `preview` | `on` | The between-runs camera preview started or stopped. |

## Example: a relay that runs each message

Python 3 with `requests`. It waits while the rig is busy or held, starts the run, then
reads events until that run's `run_end`:

```python
import json, os, time
import requests

BASE = "http://127.0.0.1:8780"
HEADERS = {"Authorization": f"Bearer {os.environ['ABSTRACT_RPC_TOKEN']}"}


def rpc(kind, **fields):
    response = requests.post(
        f"{BASE}/rpc", json={"type": kind, **fields}, headers=HEADERS, timeout=10
    )
    response.raise_for_status()
    return response.json()


def run(message, tags=("x-demo",)):
    """Start one run with `message` and return its run_end event."""
    text = " ".join(message.split())[:2000]  # one line
    while True:
        since = rpc("get_state")["data"]["eventSeq"]
        started = rpc("prompt", message=text, tags=list(tags))
        if started["success"]:
            run_id = started["data"]["run"]
            break
        if not started["error"].startswith(("held", "the rig is")):
            raise RuntimeError(started["error"])  # a refusal waiting won't fix
        time.sleep(5)  # busy or paused: try again
    with requests.get(
        f"{BASE}/events", params={"since": since}, headers=HEADERS, stream=True, timeout=(10, 60)
    ) as stream:
        for line in stream.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                event = json.loads(line[6:])
                if event["type"] == "run_end" and event["id"] == run_id:
                    return event
```

A relay that only needs the outcome can poll instead: `get_state` until `phase` is
`idle` and `lastRun.id` is the run's id.

Without `--auto-start`, every run waits at its two gates. The relay (or someone on the
page) answers each with `extension_ui_response` `{"id": …, "confirmed": true}` when it
appears as `extension_ui_request`, and only while someone watches the arms.

## Safety

- One run at a time, from the robot TOML only. No client can pass `-E` or `-P`, so a
  client can't switch off a guardrail. Only the operator who starts serve chooses
  `--auto-start`.
- Without `--auto-start`, each run waits for two gates, the stand-clear before homing and
  the scene-ready before starting. Confirming one over RPC is pressing Enter at the rig.
- With `--auto-start`, a prompt homes the arms and starts the episode immediately, so the
  person at the rig is the only check.
  - Someone stays there with the e-stop for the whole demo.
  - Use `hold` before anyone reaches into the workspace.
  - `stop` and `abort` request cooperative cancellation and teardown. Use the hardware
    stop when physical stopping is required.
- `--allow-paid` lets prompts reach a paid model. The meter caps each run's spend (the
  TOML's budget), not the total across runs: rate-limit a public relay.
- The collision guardrail is off by default until the arm bases and table are measured.
  Nothing but the per-arm workspace bounds keeps the two arms apart when it is disabled.

Shutdown deadline expiry raises an error with physical state explicitly unverified.
Process completion and RPC success never independently prove parking or torque release.
See [SAFETY.md](SAFETY.md) for native-call and signal limitations.
