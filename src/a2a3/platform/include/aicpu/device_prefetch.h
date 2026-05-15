/**
 * @file device_prefetch.h
 * @brief SDMA Prefetch Interface for AICPU
 *
 * Provides device-side SDMA prefetch to warm AICore's L2 cache before
 * kernel execution. AICPU and AICore have separate L2 caches, so this
 * uses the SDMA engine (via direct STARS SQ submission) to push data
 * into AICore's L2.
 *
 * The mechanism writes CMO PREFETCH SQEs directly to the STARS submission
 * queue and rings the hardware doorbell — same protocol as shmem but from
 * AICPU instead of AICore.
 *
 * Platform Support:
 * - a2a3: Real hardware using STARS SDMA SQ (channel info set by host)
 * - a2a3sim: No-op (no SDMA engine in simulation)
 */

#ifndef PLATFORM_DEVICE_PREFETCH_H_
#define PLATFORM_DEVICE_PREFETCH_H_

#include <cstddef>
#include <cstdint>

/**
 * Initialize SDMA prefetch subsystem.
 *
 * On a2a3: Reads STARS channel info (SQ base, doorbell register) from the
 * workspace pointer set by the host. If workspace is null, prefetch is
 * silently disabled. Host seeds the workspace with stream SQ/CQ IDs, and
 * AICPU completes SQ base/register/depth resolution via HAL.
 *
 * On a2a3sim: No-op.
 *
 * @param sdma_workspace  Pointer to STARS channel workspace in device GM.
 *                        Host writes initial channel IDs; AICPU fills the
 *                        remaining SQ metadata via HAL query. NULL disables
 *                        prefetch.
 * @param suppress_window Number of future eligible attempts to suppress on the
 *                        same channel after one successful issue.
 */
void aicpu_prefetch_init(void* sdma_workspace, uint32_t suppress_window, bool debug_enabled);

/**
 * Shut down SDMA prefetch subsystem.
 *
 * On a2a3: Clears state. Host is responsible for freeing workspace.
 * On a2a3sim: No-op.
 */
void aicpu_prefetch_deinit();

/**
 * Reserve a channel for a potential prefetch attempt.
 *
 * Performs cheap availability/suppression checks and consumes one suppress
 * slot if the channel is currently suppressed.
 *
 * @param thread_idx   Scheduler thread index (selects which channel to use)
 * @return true if the caller should proceed to issue a prefetch
 */
bool aicpu_prefetch_reserve_channel(int thread_idx);

/**
 * Issue prefetch SQE(s) after reserve_channel has already succeeded.
 *
 * Skips suppression handling and writes the SQE(s) directly. When the STARS
 * queue depth is 1, only the tensor SQE is written. When depth > 1 and
 * @p instr_addr is non-null with @p instr_size > 0, a second SQE may be written
 * for code (single lock, one doorbell). Implementations may skip the instruction
 * SQE when the same kernel has already been prefetched recently.
 *
 * @param tensor_addr  Device memory address for tensor prefetch
 * @param tensor_size  Tensor prefetch length in bytes
 * @param instr_addr   Device address for instruction prefetch, or NULL to skip
 * @param instr_size   Instruction prefetch length (e.g. 2048), ignored if @p instr_addr is NULL
 * @param instr_kernel_id  Kernel id associated with @p instr_addr, or -1 to skip dedup
 * @param thread_idx   Scheduler thread index (selects which channel to use)
 */
void aicpu_prefetch_issue_reserved(
    void* tensor_addr, size_t tensor_size, void* instr_addr, size_t instr_size, int32_t instr_kernel_id,
    int thread_idx
);

/**
 * Issue an asynchronous SDMA prefetch for a device memory region.
 *
 * Writes a CMO PREFETCH SQE to the STARS submission queue and rings
 * the hardware doorbell. Non-blocking: SDMA runs in background.
 *
 * Safe to call when prefetch is unavailable (no-op).
 *
 * @param addr         Device memory address to prefetch
 * @param size         Number of bytes to prefetch
 * @param thread_idx   Scheduler thread index (selects which channel to use)
 */
void aicpu_prefetch_tensor(void* addr, size_t size, int thread_idx);

/**
 * Check whether SDMA prefetch is available and enabled.
 *
 * @return true if prefetch calls will actually issue SDMA commands
 */
bool aicpu_prefetch_available();

/**
 * Return current SDMA channel count observed by AICPU prefetch subsystem.
 */
uint32_t aicpu_prefetch_channel_count();

#endif  // PLATFORM_DEVICE_PREFETCH_H_
