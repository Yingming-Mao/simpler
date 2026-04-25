#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Benchmark wrapper: run examples on hardware,
# then parse device-log timing lines to report per-round latency.
#
# Usage:
#   ./tools/benchmark_rounds.sh [-p <platform>] [-d <device>] [-n <rounds>] [-r <runtime>]
#
# Edit the EXAMPLE_CASES maps below to control which examples and cases to run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# Examples to benchmark and their case lists, per runtime.
# Key   = directory name under tests/st/<platform>/<runtime>/
# Value = comma-separated case names to run (empty string = run DEFAULT_CASE)
# ---------------------------------------------------------------------------

# --- tensormap_and_ringbuffer ---
declare -A TMR_EXAMPLE_CASES=(
    [alternating_matmul_add]="Case1"
    [benchmark_bgemm]="Case0"
    [paged_attention_unroll]="Case1,Case2"
    [batch_paged_attention]="Case1"
    [spmd_paged_attention]="Case1,Case2"
)
TMR_EXAMPLE_ORDER=(
    alternating_matmul_add
    benchmark_bgemm
    paged_attention_unroll
    batch_paged_attention
    spmd_paged_attention
)

# --- aicpu_build_graph ---
declare -A ABG_EXAMPLE_CASES=(
    [paged_attention_unroll]="Case1,Case2"
)
ABG_EXAMPLE_ORDER=(
    paged_attention_unroll
)

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DEVICE_ID=0
ROUNDS=100
PLATFORM=a2a3
RUNTIME=tensormap_and_ringbuffer
VERBOSE=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE_ID="$2"
            shift 2
            ;;
        -n|--rounds)
            ROUNDS="$2"
            shift 2
            ;;
        -r|--runtime)
            RUNTIME="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        --help|-h)
            cat <<'USAGE'
benchmark_rounds.sh — run all examples and report per-round timing from device logs

Usage:
  ./tools/benchmark_rounds.sh [-p <platform>] [-d <device>] [-n <rounds>] [-r <runtime>] [-v]

Options:
  -p, --platform Platform to run on (default: a2a3)
  -d, --device   Device ID (default: 0)
  -n, --rounds   Override number of rounds for each example (default: 100)
  -r, --runtime  Runtime to benchmark: tensormap_and_ringbuffer (default), aicpu_build_graph
  -v, --verbose  Save detailed test_*.py output to a timestamped log file
  -h, --help     Show this help

All other options are passed through to the underlying `python test_*.py`
invocation (e.g. --case).

Edit the EXAMPLE_CASES map at the top of this script to control which
examples and cases to benchmark.

Output:
  Average elapsed time in microseconds for each example.
USAGE
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Verbose logging setup
# ---------------------------------------------------------------------------
VERBOSE_LOG=""
if [[ $VERBOSE -eq 1 ]]; then
    mkdir -p "$PROJECT_ROOT/outputs"
    VERBOSE_LOG="$PROJECT_ROOT/outputs/benchmark_$(date +%Y%m%d_%H%M%S).log"
    echo "Verbose log: $VERBOSE_LOG"
fi

vlog() {
    if [[ -n "$VERBOSE_LOG" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$VERBOSE_LOG"
    fi
}

# ---------------------------------------------------------------------------
# Derive arch from platform and set examples directories
# ---------------------------------------------------------------------------
# Search both examples/ (migrated tests) and tests/st/ (legacy tests)
ARCH="${PLATFORM%%sim}"  # strip "sim" suffix if present
EXAMPLES_DIRS=(
    "$PROJECT_ROOT/tests/st/${ARCH}/${RUNTIME}"
    "$PROJECT_ROOT/examples/${ARCH}/${RUNTIME}"
)

# Clock frequency (MHz) for converting cycle counts to microseconds
case "$PLATFORM" in
    a2a3) FREQ=50 ;;
    a5)   FREQ=1000 ;;
    *)    echo "ERROR: unsupported platform '$PLATFORM'. Use a2a3 or a5."; exit 1 ;;
esac

# Select example cases and order based on runtime
case "$RUNTIME" in
    tensormap_and_ringbuffer)
        declare -n EXAMPLE_CASES=TMR_EXAMPLE_CASES
        EXAMPLE_ORDER=("${TMR_EXAMPLE_ORDER[@]}")
        ;;
    aicpu_build_graph)
        declare -n EXAMPLE_CASES=ABG_EXAMPLE_CASES
        EXAMPLE_ORDER=("${ABG_EXAMPLE_ORDER[@]}")
        ;;
    *)
        echo "ERROR: unknown runtime '$RUNTIME'. Use tensormap_and_ringbuffer or aicpu_build_graph."
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Resolve device log directory (mirrors simpler_setup/device_log_resolver.py)
# ---------------------------------------------------------------------------
if [[ -n "${ASCEND_WORK_PATH:-}" ]]; then
    LOG_ROOT="$ASCEND_WORK_PATH/log/debug"
    if [[ ! -d "$LOG_ROOT" ]]; then
        LOG_ROOT="$HOME/ascend/log/debug"
    fi
else
    LOG_ROOT="$HOME/ascend/log/debug"
fi
DEVICE_LOG_DIR="$LOG_ROOT/device-${DEVICE_ID}"

# ---------------------------------------------------------------------------
# parse_timing <log_file>
#   Grep for orch_start / end lines, compute per-round elapsed, print summary.
# ---------------------------------------------------------------------------
parse_timing() {
    local log_file="$1"

    local timing
    timing=$(grep -E 'Thread [0-9]+: (sched_start|orch_start|orch_end|sched_end|orch_stage_end)' "$log_file" || true)

    if [[ -z "$timing" ]]; then
        echo "  (no benchmark timing data — was PTO2_PROFILING enabled?)"
        return 1
    fi

    # NOTE: Use python instead of awk because common awk implementations on
    # hosts (e.g. mawk) don't support gawk-only `match(..., ..., array)` APIs.
    python3 - "$FREQ" "$log_file" <<'PY'
import re
import sys

freq = float(sys.argv[1])
log_file = sys.argv[2]

with open(log_file, "r", errors="ignore") as f:
    timing = [
        line.rstrip("\n")
        for line in f
        if re.search(r"Thread [0-9]+: (sched_start|orch_start|orch_end|sched_end|orch_stage_end)", line)
    ]

re_tid = re.compile(r"Thread ([0-9]+):")
re_sched_start = re.compile(r"sched_start=([0-9]+)")
re_sched_end = re.compile(r"sched_end[^=]*=([0-9]+)")
re_orch_start = re.compile(r"orch_start=([0-9]+)")
re_orch_end = re.compile(r"orch_end=([0-9]+)")
re_orch_stage_end = re.compile(r"orch_stage_end=([0-9]+)")


def flush_round(round_idx, state, out):
    if state["min_start"] and state["max_end"] and state["max_end"] > state["min_start"]:
        out["elapsed"].append((state["max_end"] - state["min_start"]) / freq)
        if state["min_sched_start"] and state["max_sched_end"] and state["max_sched_end"] > state["min_sched_start"]:
            out["sched"].append((state["max_sched_end"] - state["min_sched_start"]) / freq)
        else:
            out["sched"].append(None)
        if state["min_orch_start"] and state["max_orch_end"] and state["max_orch_end"] > state["min_orch_start"]:
            out["orch"].append((state["max_orch_end"] - state["min_orch_start"]) / freq)
        else:
            out["orch"].append(None)
        return round_idx + 1
    return round_idx


def new_state():
    return {
        "min_start": 0,
        "max_end": 0,
        "min_sched_start": 0,
        "max_sched_end": 0,
        "min_orch_start": 0,
        "max_orch_end": 0,
        "sched_seen": set(),
        "orch_seen": set(),
        "has_sched": False,
        "has_orch_end": False,
    }


state = new_state()
out = {"elapsed": [], "sched": [], "orch": []}
round_idx = 0

for line in timing:
    m = re_tid.search(line)
    tid = int(m.group(1)) if m else None

    ms = re_sched_start.search(line)
    if ms:
        if tid is not None and tid in state["sched_seen"]:
            round_idx = flush_round(round_idx, state, out)
            state = new_state()
        if tid is not None:
            state["sched_seen"].add(tid)
        state["has_sched"] = True
        val = int(ms.group(1))
        state["min_sched_start"] = val if state["min_sched_start"] == 0 else min(state["min_sched_start"], val)
        state["min_start"] = val if state["min_start"] == 0 else min(state["min_start"], val)

    mo = re_orch_start.search(line)
    if mo:
        if tid is not None and tid in state["orch_seen"]:
            round_idx = flush_round(round_idx, state, out)
            state = new_state()
        if tid is not None:
            state["orch_seen"].add(tid)
        val = int(mo.group(1))
        state["min_orch_start"] = val if state["min_orch_start"] == 0 else min(state["min_orch_start"], val)
        state["min_start"] = val if state["min_start"] == 0 else min(state["min_start"], val)

    me = re_sched_end.search(line)
    if me:
        val = int(me.group(1))
        state["max_sched_end"] = max(state["max_sched_end"], val)
        state["max_end"] = max(state["max_end"], val)

    moe = re_orch_end.search(line)
    if moe:
        val = int(moe.group(1))
        state["has_orch_end"] = True
        state["max_orch_end"] = max(state["max_orch_end"], val)
        state["max_end"] = max(state["max_end"], val)

    mse = re_orch_stage_end.search(line)
    if mse:
        val = int(mse.group(1))
        state["max_end"] = max(state["max_end"], val)

round_idx = flush_round(round_idx, state, out)

count = len(out["elapsed"])
if count == 0:
    print("  (no rounds parsed)")
    raise SystemExit(1)

show_sched = any(v is not None for v in out["sched"])
show_orch = any(v is not None for v in out["orch"])

hdr = f"  {'Round':<8}  {'Elapsed (us)':>12}"
sep = f"  {'-----':<8}  {'------------':>12}"
if show_sched:
    hdr += f"  {'Sched (us)':>12}"
    sep += f"  {'----------':>12}"
if show_orch:
    hdr += f"  {'Orch (us)':>12}"
    sep += f"  {'---------':>12}"
print(hdr)
print(sep)

def fmt(v):
    return f"{v:12.1f}" if v is not None else f"{'-':>12}"

sum_e = 0.0
sum_s = 0.0
sum_o = 0.0
cnt_s = 0
cnt_o = 0

for i in range(count):
    e = out["elapsed"][i]
    s = out["sched"][i]
    o = out["orch"][i]
    line = f"  {i:<8d}  {e:12.1f}"
    sum_e += e
    if show_sched:
        line += "  " + fmt(s)
        if s is not None:
            sum_s += s
            cnt_s += 1
    if show_orch:
        line += "  " + fmt(o)
        if o is not None:
            sum_o += o
            cnt_o += 1
    print(line)

avg_e = sum_e / count
msg = f"\n  Avg: {avg_e:.1f} us"
if show_sched:
    msg += f"  |  Sched Avg: {(sum_s / max(cnt_s,1)):.1f} us"
if show_orch:
    msg += f"  |  Orch Avg: {(sum_o / max(cnt_o,1)):.1f} us"
msg += f"  ({count} rounds)"
print(msg)

TRIM = 10
if count > 2 * TRIM:
    sv = sorted(out["elapsed"])
    trimmed = sv[TRIM:count-TRIM]
    tavg = sum(trimmed) / len(trimmed)
    print(f"  Trimmed Avg: {tavg:.1f} us  (dropped {TRIM} low + {TRIM} high, {len(trimmed)} rounds used)")
    if show_sched:
        ss = [v for v in out['sched'] if v is not None]
        if len(ss) > 2 * TRIM:
            ss.sort()
            tss = ss[TRIM:len(ss)-TRIM]
            print(f"  Sched Trimmed Avg: {sum(tss)/len(tss):.1f} us  (dropped {TRIM} low + {TRIM} high)")
    if show_orch:
        so = [v for v in out['orch'] if v is not None]
        if len(so) > 2 * TRIM:
            so.sort()
            tso = so[TRIM:len(so)-TRIM]
            print(f"  Orch Trimmed Avg: {sum(tso)/len(tso):.1f} us  (dropped {TRIM} low + {TRIM} high)")
PY
}

# ---------------------------------------------------------------------------
# wait_for_new_log <pre_run_logs_file>
#   Wait up to 15s for a new .log file in DEVICE_LOG_DIR. Prints the path.
# ---------------------------------------------------------------------------
wait_for_new_log() {
    local pre_file="$1"
    local new_log=""
    local deadline=$((SECONDS + 15))

    while [[ $SECONDS -lt $deadline ]]; do
        if [[ -d "$DEVICE_LOG_DIR" ]]; then
            new_log=$(comm -13 "$pre_file" <(ls -1 "$DEVICE_LOG_DIR"/*.log 2>/dev/null | sort) 2>/dev/null | tail -1 || true)
            if [[ -n "$new_log" ]]; then
                echo "$new_log"
                return 0
            fi
        fi
        sleep 0.5
    done

    # Fallback: newest log
    if [[ -d "$DEVICE_LOG_DIR" ]]; then
        new_log=$(ls -t "$DEVICE_LOG_DIR"/*.log 2>/dev/null | head -1 || true)
        if [[ -n "$new_log" ]]; then
            echo "$new_log"
            return 0
        fi
    fi
    return 1
}

# ---------------------------------------------------------------------------
# run_bench <example> <example_dir> [case_name]
#   Run one benchmark invocation (via `python test_*.py`) and parse timing
#   from the resulting log. Skips the example if it has no test_*.py.
#   Sets global PASS / FAIL counters.
# ---------------------------------------------------------------------------
run_bench() {
    local example="$1" example_dir="$2" case_name="${3:-}"

    if [[ -n "$case_name" ]]; then
        echo "  ---- $case_name ----"
    fi

    # Snapshot existing logs
    local pre_log_file
    pre_log_file=$(mktemp)
    trap 'rm -f -- "$pre_log_file"' RETURN
    ls -1 "$DEVICE_LOG_DIR"/*.log 2>/dev/null | sort > "$pre_log_file" || true

    # Build run command using test_*.py
    local test_file
    test_file=$(find "$example_dir" -maxdepth 1 -name 'test_*.py' -print -quit 2>/dev/null || true)

    local run_cmd
    if [[ -n "$test_file" ]]; then
        run_cmd=(
            python3 "$test_file"
            --platform "$PLATFORM" --device "$DEVICE_ID"
            --rounds "$ROUNDS" --skip-golden
        )
    else
        echo "  SKIPPED: no test_*.py found in $example_dir"
        return
    fi
    if [[ -n "$case_name" ]]; then
        run_cmd+=(--case "$case_name")
        [[ -n "$test_file" ]] && run_cmd+=(--manual include)
    fi
    run_cmd+=("${EXTRA_ARGS[@]}")

    # Run example
    vlog "Running: ${run_cmd[*]}"
    local rc=0
    if [[ -n "$VERBOSE_LOG" ]]; then
        local run_output
        run_output=$("${run_cmd[@]}" 2>&1) || rc=$?
        if [[ -n "$run_output" ]]; then echo "$run_output" >> "$VERBOSE_LOG"; fi
    else
        "${run_cmd[@]}" > /dev/null 2>&1 || rc=$?
    fi
    if [[ $rc -ne 0 ]]; then
        echo "  FAILED: benchmark run returned non-zero"
        vlog "FAILED: exit code $rc"
        ((FAIL++)) || true
        return
    fi

    # Find new device log
    local new_log
    new_log=$(wait_for_new_log "$pre_log_file")

    if [[ -z "$new_log" ]]; then
        echo "  FAILED: no device log found in $DEVICE_LOG_DIR"
        ((FAIL++)) || true
        return
    fi

    echo "  Log: $new_log"
    local timing_output
    local parse_rc=0
    timing_output=$(parse_timing "$new_log") || parse_rc=$?
    echo "$timing_output"

    if [[ $parse_rc -ne 0 ]]; then
        ((FAIL++)) || true
        return
    fi
    ((PASS++)) || true

    # Extract averages for summary table
    local label="$example"
    [[ -n "$case_name" ]] && label="$example ($case_name)"

    local avg_line
    avg_line=$(echo "$timing_output" | grep "^  Avg:" || true)
    local avg_elapsed="-" avg_sched="-" avg_orch="-"
    if [[ -n "$avg_line" ]]; then
        avg_elapsed=$(echo "$avg_line" | awk '{print $2}')
        avg_sched=$(echo "$avg_line" | grep -o 'Sched Avg: [0-9.]*' | awk '{print $3}') || avg_sched="-"
        avg_orch=$(echo "$avg_line" | grep -o 'Orch Avg: [0-9.]*' | awk '{print $3}') || avg_orch="-"
    fi

    SUMMARY_NAMES+=("$label")
    SUMMARY_ELAPSED+=("$avg_elapsed")
    SUMMARY_SCHED+=("$avg_sched")
    SUMMARY_ORCH+=("$avg_orch")
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
PASS=0
FAIL=0

# Summary collection arrays
SUMMARY_NAMES=()
SUMMARY_ELAPSED=()
SUMMARY_SCHED=()
SUMMARY_ORCH=()

echo ""
echo "Runtime: $RUNTIME"

for example in "${EXAMPLE_ORDER[@]}"; do
    case_list="${EXAMPLE_CASES[$example]:-}"

    # Search for example: prefer test_*.py (new style), fall back to golden.py (legacy).
    # tests/st/ is searched before examples/ since benchmarks use production-scale cases.
    EXAMPLE_DIR=""
    for dir in "${EXAMPLES_DIRS[@]}"; do
        candidate="$dir/$example"
        if [[ -d "$candidate" ]] && ls "$candidate"/test_*.py 1>/dev/null 2>&1; then
            EXAMPLE_DIR="$candidate"
            break
        fi
    done
    if [[ -z "$EXAMPLE_DIR" ]]; then
        for dir in "${EXAMPLES_DIRS[@]}"; do
            candidate="$dir/$example"
            if [[ -f "$candidate/golden.py" && -d "$candidate/kernels" ]]; then
                EXAMPLE_DIR="$candidate"
                break
            fi
        done
    fi

    echo ""
    echo "================================================================"
    echo "  $example"
    echo "================================================================"

    if [[ -z "$EXAMPLE_DIR" ]]; then
        echo "  SKIP: not found in any search directory"
        ((FAIL++)) || true
        continue
    fi

    if [[ -z "${case_list:-}" ]]; then
        run_bench "$example" "$EXAMPLE_DIR"
    else
        IFS=',' read -ra cases <<< "$case_list"
        for c in "${cases[@]}"; do
            run_bench "$example" "$EXAMPLE_DIR" "$c"
        done
    fi
done

# ---------------------------------------------------------------------------
# Performance Summary Table
# ---------------------------------------------------------------------------
if [[ ${#SUMMARY_NAMES[@]} -gt 0 ]]; then
    # Check if any sched/orch data exists across all runs
    _has_sched=0
    _has_orch=0
    for _i in "${!SUMMARY_NAMES[@]}"; do
        [[ "${SUMMARY_SCHED[$_i]}" != "-" ]] && _has_sched=1
        [[ "${SUMMARY_ORCH[$_i]}" != "-" ]] && _has_orch=1
    done

    echo ""
    echo "================================================================"
    echo "  Performance Summary ($RUNTIME)"
    echo "================================================================"
    echo ""

    # Header
    _hdr=$(printf "  %-40s  %12s" "Example" "Elapsed (us)")
    _sep=$(printf "  %-40s  %12s" "----------------------------------------" "------------")
    if [[ $_has_sched -eq 1 ]]; then
        _hdr=$(printf "%s  %12s" "$_hdr" "Sched (us)")
        _sep=$(printf "%s  %12s" "$_sep" "------------")
    fi
    if [[ $_has_orch -eq 1 ]]; then
        _hdr=$(printf "%s  %12s" "$_hdr" "Orch (us)")
        _sep=$(printf "%s  %12s" "$_sep" "------------")
    fi
    echo "$_hdr"
    echo "$_sep"

    # Rows
    for _i in "${!SUMMARY_NAMES[@]}"; do
        _row=$(printf "  %-40s  %12s" "${SUMMARY_NAMES[$_i]}" "${SUMMARY_ELAPSED[$_i]}")
        if [[ $_has_sched -eq 1 ]]; then
            _row=$(printf "%s  %12s" "$_row" "${SUMMARY_SCHED[$_i]}")
        fi
        if [[ $_has_orch -eq 1 ]]; then
            _row=$(printf "%s  %12s" "$_row" "${SUMMARY_ORCH[$_i]}")
        fi
        echo "$_row"
    done
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL=$((PASS + FAIL))
echo ""
echo "================================================================"
echo "  Benchmark complete ($RUNTIME): $PASS passed, $FAIL failed ($TOTAL total)"
echo "================================================================"

if [[ -n "$VERBOSE_LOG" ]]; then
    echo "  Verbose log saved to: $VERBOSE_LOG"
fi

[[ $FAIL -eq 0 ]]
