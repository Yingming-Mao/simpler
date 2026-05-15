/**
 * @file device_prefetch.cpp
 * @brief SDMA Prefetch Stub for Simulation (a2a3sim)
 *
 * No-op implementation: simulation has no SDMA engine or separate L2 caches.
 */

#include "aicpu/device_prefetch.h"

void aicpu_prefetch_init(void* sdma_workspace, uint32_t suppress_window, bool debug_enabled) {
    (void)sdma_workspace;
    (void)suppress_window;
    (void)debug_enabled;
}

void aicpu_prefetch_deinit() {
}

bool aicpu_prefetch_reserve_channel(int thread_idx) {
    (void)thread_idx;
    return false;
}

void aicpu_prefetch_issue_reserved(
    void* tensor_addr, size_t tensor_size, void* instr_addr, size_t instr_size, int32_t instr_kernel_id,
    int thread_idx
) {
    (void)tensor_addr;
    (void)tensor_size;
    (void)instr_addr;
    (void)instr_size;
    (void)instr_kernel_id;
    (void)thread_idx;
}

void aicpu_prefetch_tensor(void* addr, size_t size, int thread_idx) {
    (void)addr;
    (void)size;
    (void)thread_idx;
}

bool aicpu_prefetch_available() {
    return false;
}

uint32_t aicpu_prefetch_channel_count() {
    return 0;
}
