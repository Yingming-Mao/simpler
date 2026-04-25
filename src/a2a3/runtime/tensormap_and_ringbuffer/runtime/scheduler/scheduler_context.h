/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
#ifndef SCHEDULER_CONTEXT_H
#define SCHEDULER_CONTEXT_H

#include "scheduler_types.h"

#include "scheduler/pto_scheduler.h"

#include "pto2_dispatch_payload.h"

// These macros are defined in runtime.h, but we cannot include it here
// (it pulls in Handshake which we only forward-declare).  Mirror the
// authoritative values so the class layout compiles standalone.
#ifndef RUNTIME_MAX_WORKER
#define RUNTIME_MAX_WORKER 72
#endif
#ifndef RUNTIME_MAX_FUNC_ID
#define RUNTIME_MAX_FUNC_ID 32
#endif

// Forward declarations — avoid pulling in full headers for pointer/reference params.
class Runtime;
struct Handshake;
struct PTO2Runtime;

/**
 * SchedulerContext: owns all scheduler-side state and methods.
 *
 * Held as a member of AicpuExecutor (sched_ctx_).  The single public entry
 * point is resolve_and_dispatch(), called once per scheduler thread.
 *
 * All dispatch/completion/drain/cold-path logic is implemented as private
 * member methods, split across three .cpp files by responsibility:
 *   - scheduler_completion.cpp  (completion polling, drain protocol)
 *   - scheduler_cold_path.cpp   (exit checks, stall diagnostics, profiling)
 *   - scheduler_dispatch.cpp    (task dispatch loop and helpers)
 */
class SchedulerContext {
public:
    // =========================================================================
    // Lifecycle
    // =========================================================================

    // Initialize scheduler state from the given runtime and thread layout.
    // - Discovers cores via handshake_all_cores()
    // - Assigns cores to scheduler threads
    // - Resets task counters, payloads, per-core GlobalContext
    // - Binds func_id_to_addr_ / initial sched_ (if rt is already known)
    // - Captures AICore-register base (consumed by handshake_all_cores())
    // Returns 0 on success, negative on failure (handshake / assignment error).
    int32_t
    init(Runtime *runtime, int32_t thread_num, int32_t sched_thread_num, bool orch_to_sched, uint64_t regs_base);

    // Reset all SchedulerContext-owned state to its post-construction defaults.
    // Called by AicpuExecutor::deinit() during per-run teardown.
    void deinit();

    // =========================================================================
    // Per-thread execution entry points (called by AicpuExecutor::run)
    // =========================================================================

    // Main scheduler thread entry: poll completion + dispatch ready tasks.
    int32_t resolve_and_dispatch(Runtime *runtime, int32_t thread_idx);

    // Shutdown AICore registers for this thread's assigned cores.
    // Also runs PMU finalize (PTO2_PROFILING) before deinit when enabled.
    // Orchestrator threads (core_trackers_[thread_idx].core_num() == 0) are a no-op.
    int32_t shutdown(int32_t thread_idx);

    // Run all post-orchestration scheduler bookkeeping:
    //  - publishes core assignments to the perf collector (PTO2_PROFILING)
    //  - latches submitted task count from PTO2 shared memory
    //  - folds inline_completed_tasks into completed_tasks_
    //  - flips orchestrator_done_ and triggers core transition
    //    (skipped on fatal error — emergency_shutdown runs instead)
    // Callers must invoke pto2_rt_orchestration_done(rt) before this — that
    // step belongs to the orchestrator lifecycle, not the scheduler.
    void on_orchestration_done(Runtime *runtime, PTO2Runtime *rt, int32_t thread_idx, int32_t total_tasks);

    // Bind the PTO2Runtime scheduler pointer. Required in device-orchestration
    // mode where rt is created by the orchestrator thread after init().
    void bind_runtime(PTO2Runtime *rt);

    // =========================================================================
    // State queries / external synchronization points
    // =========================================================================

    int32_t aic_count() const { return aic_count_; }
    int32_t aiv_count() const { return aiv_count_; }
    bool is_completed() const { return completed_.load(std::memory_order_acquire); }
    int32_t completed_tasks_count() const { return completed_tasks_.load(std::memory_order_acquire); }

    // Block until the first scheduler thread has finished one-time PTO2 init.
    // Called by the orchestrator thread in device-orch mode.
    void wait_pto2_init_complete() const;

private:
    // =========================================================================
    // State
    // =========================================================================

    // --- Scheduler binding & per-core runtime state ---
    alignas(64) PTO2SchedulerState *sched_{nullptr};

    // =========================================================================
    // Two-slot admission (pending-phase gating)
    // =========================================================================
    //
    // The scheduler supports dual-issue by dispatching to IDLE cores first, then
    // dispatching to RUNNING cores' pending slots (CoreTracker::DispatchPhase::PENDING).
    //
    // For some graphs, aggressive pending dispatch can be a net loss. This is an
    // opt-in admission policy controlled via env vars read once at init() time.
    enum : int32_t { TWOSLOT_TYPE_AIC = 0, TWOSLOT_TYPE_AIV = 1, TWOSLOT_TYPE_NUM = 2 };

    struct TwoSlotAdmissionPolicy {
        // Config (defaults match simpler-PTO knobs)
        bool steady_gate_enabled{false};     // PTO2_TWOSLOT_ENABLE_STEADY_GATE
        bool recent_negative_enabled{false}; // PTO2_TWOSLOT_ENABLE_RECENT_NEGATIVE
        int32_t probe_ready_margin[TWOSLOT_TYPE_NUM]{2, 2};    // *_PROBE_READY_MARGIN
        int32_t probe_min_visible[TWOSLOT_TYPE_NUM]{0, 8};     // *_PROBE_MIN_VISIBLE_TASKS
        int32_t steady_ready_margin[TWOSLOT_TYPE_NUM]{3, 4};   // *_STEADY_READY_MARGIN
        int32_t steady_min_visible[TWOSLOT_TYPE_NUM]{0, 8};    // *_STEADY_MIN_VISIBLE_TASKS
        int32_t recent_miss_limit[TWOSLOT_TYPE_NUM]{3, 2};     // *_RECENT_MISS_LIMIT
        int32_t recent_stolen_penalty[TWOSLOT_TYPE_NUM]{1, 2}; // *_RECENT_STOLEN_PENALTY (used as cooldown)

        // State (shared across scheduler threads)
        std::atomic<int32_t> cooldown[TWOSLOT_TYPE_NUM]{{0}, {0}};
        std::atomic<int32_t> recent_miss_score[TWOSLOT_TYPE_NUM]{{0}, {0}};
        std::atomic<uint64_t> pending_dispatch_by_shape[PTO2_NUM_RESOURCE_SHAPES]{{0}, {0}, {0}};
        std::atomic<uint64_t> pending_blocked_by_shape[PTO2_NUM_RESOURCE_SHAPES]{{0}, {0}, {0}};
        std::atomic<uint64_t> pending_promote_by_type[TWOSLOT_TYPE_NUM]{{0}, {0}};
        std::atomic<uint64_t> idle_without_pending_by_type[TWOSLOT_TYPE_NUM]{{0}, {0}};

        void reset_runtime_state() {
            for (int i = 0; i < TWOSLOT_TYPE_NUM; i++) {
                cooldown[i].store(0, std::memory_order_relaxed);
                recent_miss_score[i].store(0, std::memory_order_relaxed);
                pending_promote_by_type[i].store(0, std::memory_order_relaxed);
                idle_without_pending_by_type[i].store(0, std::memory_order_relaxed);
            }
            for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
                pending_dispatch_by_shape[i].store(0, std::memory_order_relaxed);
                pending_blocked_by_shape[i].store(0, std::memory_order_relaxed);
            }
        }

        bool allow_pending_type(int32_t type_idx, uint64_t ready_depth, int32_t visible_tasks) const {
            if (!steady_gate_enabled && !recent_negative_enabled) return true; // default behavior unchanged
            if (recent_negative_enabled && cooldown[type_idx].load(std::memory_order_relaxed) > 0) return false;

            if (visible_tasks < probe_min_visible[type_idx]) return false;
            if (ready_depth < static_cast<uint64_t>(probe_ready_margin[type_idx])) return false;

            if (steady_gate_enabled) {
                if (visible_tasks < steady_min_visible[type_idx]) return false;
                if (ready_depth < static_cast<uint64_t>(steady_ready_margin[type_idx])) return false;
            }
            return true;
        }

        bool allow_pending_shape(
            PTO2ResourceShape shape, uint64_t ready_aic, uint64_t ready_aiv, int32_t visible_tasks
        ) const {
            if (shape == PTO2ResourceShape::AIC) return allow_pending_type(TWOSLOT_TYPE_AIC, ready_aic, visible_tasks);
            if (shape == PTO2ResourceShape::AIV) return allow_pending_type(TWOSLOT_TYPE_AIV, ready_aiv, visible_tasks);
            // MIX: conservative: only allow when both core types would allow.
            return allow_pending_type(TWOSLOT_TYPE_AIC, ready_aic, visible_tasks) &&
                   allow_pending_type(TWOSLOT_TYPE_AIV, ready_aiv, visible_tasks);
        }

        void tick() {
            if (!recent_negative_enabled) return;
            for (int i = 0; i < TWOSLOT_TYPE_NUM; i++) {
                int32_t c = cooldown[i].load(std::memory_order_relaxed);
                if (c > 0) cooldown[i].store(c - 1, std::memory_order_relaxed);
            }
        }

        void on_pending_hit(int32_t type_idx) {
            if (!recent_negative_enabled) return;
            int32_t s = recent_miss_score[type_idx].load(std::memory_order_relaxed);
            if (s > 0) recent_miss_score[type_idx].store(s - 1, std::memory_order_relaxed);
        }

        void on_pending_dispatch(PTO2ResourceShape shape) {
            pending_dispatch_by_shape[static_cast<int32_t>(shape)].fetch_add(1, std::memory_order_relaxed);
        }

        void on_pending_blocked(PTO2ResourceShape shape, int32_t blocked_count) {
            if (blocked_count <= 0) return;
            pending_blocked_by_shape[static_cast<int32_t>(shape)].fetch_add(blocked_count, std::memory_order_relaxed);
        }

        void on_pending_promote(int32_t type_idx) {
            pending_promote_by_type[type_idx].fetch_add(1, std::memory_order_relaxed);
            on_pending_hit(type_idx);
        }

        void on_idle_without_pending(int32_t type_idx) {
            idle_without_pending_by_type[type_idx].fetch_add(1, std::memory_order_relaxed);
        }

        void on_pending_miss(int32_t type_idx, uint64_t ready_depth, int32_t visible_tasks) {
            if (!recent_negative_enabled) return;
            if (!steady_gate_enabled) return; // only count misses in "steady fallback" region
            if (visible_tasks < steady_min_visible[type_idx]) return;
            if (ready_depth < static_cast<uint64_t>(steady_ready_margin[type_idx])) return;

            int32_t s = recent_miss_score[type_idx].fetch_add(1, std::memory_order_relaxed) + 1;
            if (s >= recent_miss_limit[type_idx]) {
                recent_miss_score[type_idx].store(0, std::memory_order_relaxed);
                int32_t penalty = recent_stolen_penalty[type_idx];
                if (penalty < 0) penalty = 0;
                cooldown[type_idx].store(penalty, std::memory_order_relaxed);
            }
        }
    };

    TwoSlotAdmissionPolicy twoslot_policy_;

    // Per-core execution state, indexed by core_id (= worker_id)
    CoreExecState core_exec_states_[RUNTIME_MAX_WORKER];

    // Cluster-ordered core trackers, one per scheduler thread
    CoreTracker core_trackers_[MAX_AICPU_THREADS];

    // Per-core dispatch payload storage: dual-buffer for pipelining.
    // buf_idx = reg_task_id & 1; adjacent dispatches alternate automatically.
    PTO2DispatchPayload payload_per_core_[RUNTIME_MAX_WORKER][2];

    // sync_start drain coordination
    SyncStartDrainState drain_state_;

#if PTO2_PROFILING
    SchedL2PerfCounters sched_l2_perf_[MAX_AICPU_THREADS];
#endif

    // --- Task-execution tracking ---
    std::atomic<int32_t> completed_tasks_{0};
    int32_t total_tasks_{0};
    // Device orchestration: set by last orchestrator when graph is built; schedulers poll it.
    // volatile prevents the compiler from hoisting the load out of spin loops.
    volatile bool orchestrator_done_{false};
    std::atomic<bool> completed_{false};
    uint64_t *func_id_to_addr_{nullptr};

    // --- Core-transition coordination ---
    std::atomic<bool> transition_requested_{false};
    std::atomic<int32_t> wait_reassign_{0};
    std::atomic<bool> reassigned_{false};

    // --- Thread/core configuration ---
    int32_t active_sched_threads_{0};
    int32_t sched_thread_num_{0};
    bool orch_to_sched_{false};
    int32_t thread_num_{0};
    int32_t cores_total_num_{0};

    // Cluster-ordered worker_id lists, populated by handshake_all_cores().
    int32_t aic_worker_ids_[RUNTIME_MAX_WORKER]{};
    int32_t aiv_worker_ids_[RUNTIME_MAX_WORKER]{};
    int32_t aic_count_{0};
    int32_t aiv_count_{0};

    // Platform AICore-register base array (set by AicpuExecutor before init()).
    uint64_t regs_{0};

#if PTO2_PROFILING
    // PMU profiling: physical core IDs for PMU MMIO base resolution.
    // Separate storage because CoreExecState's 64-byte budget has no room for
    // physical_core_id when PTO2_PROFILING=1.
    uint32_t physical_core_ids_[RUNTIME_MAX_WORKER]{};
#endif

    // --- One-time init coordination ---
    std::atomic<bool> pto2_init_done_{false};
    std::atomic<bool> pto2_init_complete_{false};

    // =========================================================================
    // Core management (scheduler_cold_path.cpp)
    // =========================================================================

    // Handshake with all AICore workers; populates core_exec_states_, worker id lists.
    int32_t handshake_all_cores(Runtime *runtime);

    // Assign discovered cores (cluster = 1 AIC + 2 AIV) round-robin across scheduler threads.
    bool assign_cores_to_threads();

    // Re-distribute all cores across all threads after orchestration completes.
    void reassign_cores_for_all_threads();

    // Emergency shutdown: broadcast exit signal to every handshake'd core and
    // deinit their AICore register blocks. Idempotent.
    void emergency_shutdown(Runtime *runtime);

    // =========================================================================
    // Dispatch (scheduler_dispatch.cpp)
    // =========================================================================

    static const char *shape_name(PTO2ResourceShape shape);
    static const PTO2ResourceShape *get_dispatch_order(int32_t thread_idx);

    int pop_ready_tasks_batch(
        PTO2ResourceShape shape, int32_t thread_idx, PTO2LocalReadyBuffer &local_buf, PTO2TaskSlotState **out,
        int max_count
    );

    void build_payload(PTO2DispatchPayload &dispatch_payload, PTO2TaskSlotState &slot_state, PTO2SubtaskSlot subslot);

    void dispatch_subtask_to_core(
        Runtime *runtime, int32_t thread_idx, int32_t core_offset, PTO2TaskSlotState &slot_state,
        PTO2ResourceShape shape, PTO2SubtaskSlot subslot, bool to_pending
    );

    void dispatch_mix_block_to_cluster(
        Runtime *runtime, int32_t thread_idx, int32_t cluster_offset, PTO2TaskSlotState &slot_state, bool to_pending
    );

    void dispatch_block(
        Runtime *runtime, int32_t thread_idx, int32_t core_offset, PTO2TaskSlotState &slot_state,
        PTO2ResourceShape shape, bool to_pending
    );

    void dispatch_shape(
        Runtime *runtime, int32_t thread_idx, PTO2ResourceShape shape, CoreTracker::DispatchPhase phase,
        PTO2LocalReadyBuffer &local_buf, CoreTracker &tracker, bool &entered_drain, bool &made_progress,
        bool &try_pushed
    );

    // =========================================================================
    // Completion & drain (scheduler_completion.cpp)
    // =========================================================================

    static SlotTransition
    decide_slot_transition(int32_t reg_task_id, int32_t reg_state, int32_t running_id, int32_t pending_id);

    void complete_slot_task(
        PTO2TaskSlotState &slot_state, int32_t expected_reg_task_id, PTO2SubtaskSlot subslot, int32_t thread_idx,
        int32_t core_id, Handshake *hank, int32_t &completed_this_turn,
        PTO2TaskSlotState *deferred_release_slot_states[], int32_t &deferred_release_count,
        PTO2LocalReadyBuffer *local_bufs
#if PTO2_PROFILING
        ,
        uint64_t dispatch_ts
#endif
    );

    static void promote_pending_to_running(CoreExecState &core);
    static void clear_running_slot(CoreExecState &core);

    void check_running_cores_for_completion(
        int32_t thread_idx, Handshake *hank, int32_t &completed_this_turn, int32_t &cur_thread_completed,
        bool &made_progress, PTO2TaskSlotState *deferred_release_slot_states[], int32_t &deferred_release_count,
        PTO2LocalReadyBuffer *local_bufs
    );

    bool enter_drain_mode(PTO2TaskSlotState *slot_state, int32_t block_num);
    int32_t count_global_available(PTO2ResourceShape shape);
    void drain_worker_dispatch(Runtime *runtime, int32_t block_num);
    void handle_drain_mode(Runtime *runtime, int32_t thread_idx);

    // =========================================================================
    // Cold path: exit checks, stall diagnostics, profiling (scheduler_cold_path.cpp)
    // =========================================================================

    __attribute__((noinline, cold)) LoopAction
    handle_orchestrator_exit(int32_t thread_idx, PTO2SharedMemoryHeader *header, Runtime *runtime, int32_t &task_count);

    __attribute__((noinline, cold)) LoopAction handle_core_transition(bool &cores_released);

    __attribute__((noinline, cold)) LoopAction
    check_idle_fatal_error(int32_t thread_idx, PTO2SharedMemoryHeader *header, Runtime *runtime);

    __attribute__((noinline, cold)) void
    log_stall_diagnostics(int32_t thread_idx, int32_t task_count, int32_t idle_iterations, int32_t last_progress_count);

    __attribute__((noinline, cold)) int32_t handle_timeout_exit(
        int32_t thread_idx, int32_t idle_iterations
#if PTO2_PROFILING
        ,
        uint64_t sched_start_ts
#endif
    );

#if PTO2_PROFILING
    __attribute__((noinline, cold)) void log_l2_perf_summary(int32_t thread_idx, int32_t cur_thread_completed);
#endif

    // =========================================================================
    // Small inline helpers
    // =========================================================================

    uint64_t get_function_bin_addr(int func_id) const {
        if (!func_id_to_addr_ || func_id < 0 || func_id >= RUNTIME_MAX_FUNC_ID) return 0;
        return func_id_to_addr_[func_id];
    }
};

#endif  // SCHEDULER_CONTEXT_H
