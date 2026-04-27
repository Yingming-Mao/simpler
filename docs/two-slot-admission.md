# Two-Slot Admission Test Guide

This document describes the current two-slot admission controls in the
`tensormap_and_ringbuffer` runtime and gives a reproducible hardware test
recipe for reviewers.

## Scope

The implementation is under:

- `src/a2a3/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp`
- `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/`

The policy only affects the scheduler pending phase. Idle-core dispatch still
runs first. Pending dispatch means the scheduler may preload a next task into a
core that is already running a task, so the next task can start without waiting
for a full scheduler round after the running task finishes.

## Default Behavior

By default, pending dispatch is enabled for both AIC and AIV:

```bash
PTO2_TWOSLOT_AIC_PENDING_ENABLE=1
PTO2_TWOSLOT_AIV_PENDING_ENABLE=1
```

The adaptive gates are disabled by default:

```bash
PTO2_TWOSLOT_ENABLE_STEADY_GATE=0
PTO2_TWOSLOT_ENABLE_RECENT_NEGATIVE=0
PTO2_TWOSLOT_ENABLE_KERNEL_GATE=0
PTO2_TWOSLOT_ENABLE_DIAG=0
```

With both `PTO2_TWOSLOT_ENABLE_STEADY_GATE=0` and
`PTO2_TWOSLOT_ENABLE_RECENT_NEGATIVE=0`, the scheduler keeps the original
aggressive pending behavior, except for explicit per-type pending disable.

## Mechanisms

### Pending Enable

These flags are hard enable switches for pending dispatch by core type:

```bash
PTO2_TWOSLOT_AIC_PENDING_ENABLE=1
PTO2_TWOSLOT_AIV_PENDING_ENABLE=1
```

Set either value to `0` to disable pending dispatch for that type.

### Probe Gate

When either steady gate or recent-negative gate is enabled, pending admission
requires enough visible work and ready-queue depth:

```bash
PTO2_TWOSLOT_AIC_PROBE_READY_MARGIN=2
PTO2_TWOSLOT_AIV_PROBE_READY_MARGIN=2
PTO2_TWOSLOT_AIC_PROBE_MIN_VISIBLE_TASKS=0
PTO2_TWOSLOT_AIV_PROBE_MIN_VISIBLE_TASKS=8
```

`ready_margin` is compared with the relevant ready queue depth. MIX tasks count
as available work for both AIC and AIV depth checks.

### Steady Gate

Enable with:

```bash
PTO2_TWOSLOT_ENABLE_STEADY_GATE=1
```

When enabled, pending admission also requires the steady thresholds:

```bash
PTO2_TWOSLOT_AIC_STEADY_READY_MARGIN=3
PTO2_TWOSLOT_AIV_STEADY_READY_MARGIN=4
PTO2_TWOSLOT_AIC_STEADY_MIN_VISIBLE_TASKS=0
PTO2_TWOSLOT_AIV_STEADY_MIN_VISIBLE_TASKS=8
```

This is intended to avoid paying pending-dispatch cost near the tail of a graph
where there is not enough remaining work to hide the overhead.

### Recent-Negative Cooldown

Enable with:

```bash
PTO2_TWOSLOT_ENABLE_RECENT_NEGATIVE=1
```

Recent-negative accounting only takes effect when the steady gate is also
enabled. This is deliberate: misses are only counted in the steady fallback
region, not near the graph tail.

Relevant knobs:

```bash
PTO2_TWOSLOT_AIC_RECENT_MISS_LIMIT=3
PTO2_TWOSLOT_AIV_RECENT_MISS_LIMIT=2
PTO2_TWOSLOT_AIC_RECENT_STOLEN_PENALTY=1
PTO2_TWOSLOT_AIV_RECENT_STOLEN_PENALTY=2
```

A miss means a core became idle without promoting a pending task while enough
ready work still existed. After `RECENT_MISS_LIMIT` misses, the policy blocks
pending admission for `RECENT_STOLEN_PENALTY` scheduler ticks for that type.

Important: setting only `*_RECENT_MISS_LIMIT` or
`*_RECENT_STOLEN_PENALTY` does not change behavior unless both gates below are
also enabled:

```bash
PTO2_TWOSLOT_ENABLE_STEADY_GATE=1
PTO2_TWOSLOT_ENABLE_RECENT_NEGATIVE=1
```

### Kernel Gate

Enable with:

```bash
PTO2_TWOSLOT_ENABLE_KERNEL_GATE=1
```

The kernel gate probes pending admission per kernel id, then admits more often
after successful pending completions:

```bash
PTO2_TWOSLOT_AIC_KERNEL_PROBE_INTERVAL=8
PTO2_TWOSLOT_AIV_KERNEL_PROBE_INTERVAL=16
PTO2_TWOSLOT_AIC_KERNEL_ADMIT_SCORE=3
PTO2_TWOSLOT_AIV_KERNEL_ADMIT_SCORE=3
PTO2_TWOSLOT_AIC_KERNEL_ADMIT_STRIDE=2
PTO2_TWOSLOT_AIV_KERNEL_ADMIT_STRIDE=4
```

MIX pending is disabled while kernel gate is enabled, because MIX currently has
one admission decision but spans both AIC and AIV resources.

### Diagnostics

Diagnostics are disabled by default:

```bash
PTO2_TWOSLOT_ENABLE_DIAG=0
```

Set it to `1` only for investigation. It enables extra counters and device log
summary lines. Do not use diagnostic runs for final performance comparisons.

## Recommended Review Test

Use the same benchmark command for every arm, and only change the environment
variables under test:

```bash
tools/benchmark_rounds.sh -p a2a3 -d 0 -n 10 \
  -r tensormap_and_ringbuffer --rounds 10
```

For acceptance-quality numbers, prefer A/B/A with at least `n=30` per arm. A
single n10 run is useful for smoke testing but is noisy on this suite.

### Baseline: Current Default

```bash
tools/benchmark_rounds.sh -p a2a3 -d 0 -n 30 \
  -r tensormap_and_ringbuffer --rounds 10
```

### Gate Baseline: Steady + Recent, No Cooldown

This checks the cost of enabling the gate logic while keeping cooldown disabled:

```bash
PTO2_TWOSLOT_ENABLE_STEADY_GATE=1 \
PTO2_TWOSLOT_ENABLE_RECENT_NEGATIVE=1 \
PTO2_TWOSLOT_AIV_RECENT_MISS_LIMIT=3 \
PTO2_TWOSLOT_AIV_RECENT_STOLEN_PENALTY=0 \
tools/benchmark_rounds.sh -p a2a3 -d 0 -n 30 \
  -r tensormap_and_ringbuffer --rounds 10
```

### Candidate: AIV Cooldown 1

```bash
PTO2_TWOSLOT_ENABLE_STEADY_GATE=1 \
PTO2_TWOSLOT_ENABLE_RECENT_NEGATIVE=1 \
PTO2_TWOSLOT_AIV_RECENT_MISS_LIMIT=3 \
PTO2_TWOSLOT_AIV_RECENT_STOLEN_PENALTY=1 \
tools/benchmark_rounds.sh -p a2a3 -d 0 -n 30 \
  -r tensormap_and_ringbuffer --rounds 10
```

### Candidate: AIV Cooldown 2

```bash
PTO2_TWOSLOT_ENABLE_STEADY_GATE=1 \
PTO2_TWOSLOT_ENABLE_RECENT_NEGATIVE=1 \
PTO2_TWOSLOT_AIV_RECENT_MISS_LIMIT=3 \
PTO2_TWOSLOT_AIV_RECENT_STOLEN_PENALTY=2 \
tools/benchmark_rounds.sh -p a2a3 -d 0 -n 30 \
  -r tensormap_and_ringbuffer --rounds 10
```

## Current Smoke-Test Observation

On one n10 smoke run using device 0, with diagnostics disabled, comparing
against the gate baseline (`STOLEN_PENALTY=0`) showed:

| Candidate | Mean | Losses | Min |
| --- | ---: | ---: | ---: |
| `AIV_RECENT_STOLEN_PENALTY=1` | `+0.447%` | `4/7` | `-0.981%` |
| `AIV_RECENT_STOLEN_PENALTY=2` | `+0.802%` | `3/7` | `-0.767%` |

The n10 result is not a final acceptance signal. It is only evidence that the
knobs are active when both gates are enabled. Use A/B/A n30 or higher before
choosing a default.

## Reviewer Notes

- The current branch preserves default aggressive pending behavior unless gates
  are explicitly enabled.
- `*_RECENT_MISS_LIMIT` and `*_RECENT_STOLEN_PENALTY` are inert unless both
  steady gate and recent-negative gate are enabled.
- Keep `PTO2_TWOSLOT_ENABLE_DIAG=0` for performance comparisons.
- Compare trimmed timing summaries rather than raw averages when possible,
  because this suite can show large single-run outliers.
