# Execution and safety contract

The harness is a supervisory robotics application, not a certified or hard real-time safety
controller. A software refusal means the harness has declined a command; it does not prove
that an actuator has stopped. Independent hardware protection and a present operator are
required for physical use.

## Ownership and defaults

CLI execution owns and closes its embodiment in `finally`. Python `eval()` closes a
registry-created embodiment; the caller must close an embodiment object that it supplies.
Low-level `rollout()` never owns the embodiment. Constructors must remain motion-free;
`reset()` may open devices, enable torque and home the robot.

Python and CLI evaluation share guardrail construction. Physical default control requires
declared action semantics and bounds. Absolute target chains require a measured initial
reference and reanchor before each command. YAM reads the latest driver state for that
reference, including after slow inference. Adapter guardrails contribute to the same chain.
Supplying an explicit approver transfers responsibility for the approval policy to the caller.

Joint-target differences are limited per control step, in the declared native units.
These are not measured velocity or acceleration limits. Tracking, actuator torque/current,
rate enforcement and emergency stopping also require driver/firmware controls appropriate
to the robot. Unsupported rotation representations may require a dedicated approver.

## Liveness and cancellation

Physical inference runs on a daemon worker while the owning thread checks optional adapter
health and polls operator input, at intervals of approximately 50 ms when the interpreter
can schedule it. The default inference deadline is 120 seconds; Python callers may set
`policy_timeout_s`. Timeout, health failure or operator stop halts the trial and dispatches
no late inference result. An abandoned policy must never be reused or access actuators
itself. Python cannot kill that thread; CPU/native code holding the GIL can still prevent
supervision. Process isolation is required for untrusted or uncooperative model code.

The CLI scopes SIGTERM and SIGHUP into the same unwinding path as Ctrl-C, restoring prior
handlers on exit. A second signal can interrupt cleanup. SIGKILL, process crashes, power
loss, blocked native calls and operating-system failure cannot promise Python cleanup.
Library imports and Python `eval()` do not install process signal handlers.

YAM initialization has a 60-second wait deadline. Late successfully constructed arms are
closed rather than adopted after cancellation. A permanently blocked native constructor
cannot be killed by Python; the device claim remains held in the live process while its
cleanup is outstanding. Use the hardware stop. Advisory device locks are per-user by
default and coordinate cooperating processes only.

## YAM requirements

Native bring-up claims the actual configured CAN/camera identities and checks positive
loss-of-comms TIMEOUT registers on all seven motors of each configured arm before enabling
either arm. Read failure refuses bring-up. An injected driver is caller-owned and must
provide equivalent protections itself. Replacing or disabling firmware safeguards is an
operator task outside this library's authority.

Homing and parking ramps respect configured per-step target differences. When collision
checking is enabled, native lifecycle trajectories are checked before dispatch. Policy
collision checking supports absolute joint control; EEF/delta paths need their own
appropriate validation. Collision geometry must match measured arm bases and the actual
workspace. Shipped configurations keep it disabled until those measurements exist.
A collision refusal can prevent parking; do not respond by blindly bypassing the check.

Thermal checking runs during native steps and inference waits. A thermal-trip transition
records `action_executed=false`; dataset export omits that predicted action. Thermal
parking itself is motion and may fail. Closing attempts park and driver shutdown; an
exception or timeout never establishes a verified parked or torque-free state.

RPC `stop` and `abort` are cooperative operator controls. `episode_end`, `run_end`, process
exit and a successful HTTP response are software events, not independent physical state
measurements. Keep a hardware stop reachable, and measure stop behavior on each rig.

## Data and timing

Numeric observation/action buffers are snapshotted for retained trial records. Custom
opaque metadata and lazy observation extras remain owned by their producer. Supply
monotonic timestamps from a consistent clock domain, document camera publication latency,
and do not reuse an observation object for a new gateway history sample.

Rerun completion includes the final captured observations and requires successful flush
with no detected camera/step drops or emitter failure. Its receipt also covers the raw
frames by path and hash; deletion rechecks that coverage. Completion receipts contain file hashes; they establish observed software
finalization, not sensor completeness under every external failure. Local raw frames stay
when export cannot be verified. Watch disk usage and repair retained runs deliberately.

LeRobot has nominal-rate timestamps. Actual camera publication, measured state and command
times belong to forensic action logs; a nominal dataset is not a real-time motion trace.
The agent spend meter estimates provider cost and reserves conservatively, but it is not
a provider-side billing cap. Keep provider account limits appropriate to the experiment.
