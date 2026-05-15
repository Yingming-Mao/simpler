/**
 * @file device_prefetch.cpp
 * @brief SDMA Prefetch for Real Hardware (a2a3) via STARS SQ
 *
 * Implements AICPU-initiated SDMA prefetch by writing CMO PREFETCH SQEs
 * directly to the STARS submission queue and ringing the hardware doorbell.
 *
 * The host sets up STARS channels in a device workspace, and AICPU selects
 * one fixed channel per target core. This matches shmem's "stable executor
 * identity -> stable channel" model, but keeps a conservative single
 * outstanding SQE per channel because completion handling is not yet modeled.
 */

#include "aicpu/device_prefetch.h"
#include "aicpu/device_log.h"
#include "aicpu/device_time.h"
#include "aicpu/platform_regs.h"
#include "common/platform_config.h"
#include "common/unified_log.h"

#include <driver/ascend_hal.h>
#include <cinttypes>
#include <cstring>
#include <dlfcn.h>

#define DEV_ALWAYS(fmt, ...) LOG_INFO_V9(fmt, ##__VA_ARGS__)

struct stars_channel_flag_info_t {
    uint32_t flag;
    uint32_t totalQueueNum;
    uint8_t reserved[56];
};

struct stars_channel_info_t {
    uint32_t sq_head;
    uint32_t sq_tail;
    uint64_t sq_base;
    uint64_t sq_reg_base;
    uint32_t sq_depth;
    uint32_t sq_id;
    uint32_t cq_id;
    uint32_t logic_cq_id;
    uint64_t cqe_addr;
    uint32_t report_cqe_num;
    uint32_t stream_id;
    uint32_t dev_id;
    uint8_t reserved[4];
};

struct stars_sdma_cmo_sqe_t {
    uint8_t type : 6;
    uint8_t l1_lock : 1;
    uint8_t l1_unlock : 1;

    uint8_t ie : 2;
    uint8_t pre_p : 2;
    uint8_t post_p : 2;
    uint8_t wr_cqe : 1;
    uint8_t reserved_hdr : 1;

    uint16_t block_dim;
    uint16_t rt_streamid;
    uint16_t task_id;

    uint32_t res3;

    uint16_t res4;
    uint8_t kernel_credit;
    uint8_t ptr_mode : 1;
    uint8_t res5 : 7;

    uint32_t opcode : 8;
    uint32_t ie2 : 1;
    uint32_t sssv : 1;
    uint32_t dssv : 1;
    uint32_t sns : 1;
    uint32_t dns : 1;
    uint32_t qos : 4;
    uint32_t sro : 1;
    uint32_t dro : 1;
    uint32_t partid : 8;
    uint32_t mpam : 1;
    uint32_t d2d_offset_flag : 1;
    uint32_t res6 : 3;

    uint16_t src_streamid;
    uint16_t src_sub_streamid;
    uint16_t dst_streamid;
    uint16_t dst_sub_streamid;

    uint32_t length;
    uint32_t src_addr_low;
    uint32_t src_addr_high;
    uint32_t dst_addr_low;
    uint32_t dst_addr_high;

    uint32_t src_offset_low;
    uint32_t dst_offset_low;
    uint16_t src_offset_high;
    uint16_t dst_offset_high;
    uint32_t res_last[1];
};

static_assert(sizeof(stars_sdma_cmo_sqe_t) == 64, "CMO SQE must be 64 bytes");
static_assert(sizeof(stars_channel_flag_info_t) == 64, "Flag info must be 64 bytes");

static constexpr uint8_t STARS_SQE_TYPE_SDMA = 11;
static constexpr uint8_t CMO_OPCODE_PREFETCH = 6;
static constexpr uint8_t DEFAULT_KERNEL_CREDIT = 254;
// Mirrors a2a3 Runtime::RUNTIME_MAX_FUNC_ID without introducing a platform->runtime dependency.
static constexpr int PREFETCH_MAX_KERNEL_ID = 1024;
static constexpr int PREFETCH_TENSOR_DEDUP_SLOTS = 2048;
static constexpr int PREFETCH_TENSOR_DEDUP_PROBE_LIMIT = 16;

static bool g_prefetch_enabled = false;
static volatile stars_channel_info_t* g_channel_info = nullptr;
static uint32_t g_channel_count = 0;
static uint32_t g_prefetch_suppress_window = 0;
static volatile uint32_t g_prefetch_submit_lock[PLATFORM_MAX_CORES] = {0};
static volatile uint32_t g_prefetch_skip_queue_full[PLATFORM_MAX_CORES] = {0};
static volatile uint32_t g_prefetch_channel_suppress_remaining[PLATFORM_MAX_CORES] = {0};
static volatile uint32_t g_prefetch_skip_suppressed[PLATFORM_MAX_CORES] = {0};
static volatile uint64_t g_prefetch_last_instr_addr[PLATFORM_MAX_CORES] = {0};
static volatile uint32_t g_prefetch_last_instr_size[PLATFORM_MAX_CORES] = {0};
static volatile uint32_t g_prefetch_skip_duplicate_instr[PLATFORM_MAX_CORES] = {0};
static volatile uint8_t g_prefetch_instr_kernel_seen[PREFETCH_MAX_KERNEL_ID] = {0};
static volatile uint32_t g_prefetch_skip_duplicate_instr_kernel = 0;
static volatile uint32_t g_prefetch_tensor_dedup_lock = 0;
static volatile uint64_t g_prefetch_tensor_seen_addr[PREFETCH_TENSOR_DEDUP_SLOTS] = {0};
static volatile uint64_t g_prefetch_tensor_seen_size[PREFETCH_TENSOR_DEDUP_SLOTS] = {0};
static volatile uint32_t g_prefetch_skip_duplicate_tensor = 0;
static volatile uint32_t g_prefetch_tensor_dedup_overflow = 0;
static volatile uint32_t g_prefetch_attempt_count = 0;
static volatile uint32_t g_prefetch_issue_count = 0;
static volatile uint64_t g_prefetch_attempt_bytes = 0;
static volatile uint64_t g_prefetch_issue_bytes = 0;
static volatile uint64_t g_prefetch_attempt_cycles = 0;
static volatile uint64_t g_prefetch_issue_cycles = 0;
static bool g_prefetch_debug_enabled = false;

using HalSqCqQueryFn = drvError_t (*)(uint32_t devId, halSqCqQueryInfo* info);
using HalResourceIdRestoreFn = drvError_t (*)(drvResIdKey* info);

static HalSqCqQueryFn g_hal_sq_cq_query = nullptr;
static bool g_hal_sq_cq_query_resolved = false;
static HalResourceIdRestoreFn g_hal_resource_id_restore = nullptr;
static bool g_hal_resource_id_restore_resolved = false;

static void resolve_hal_sq_cq_query()
{
    if (g_hal_sq_cq_query_resolved) {
        return;
    }
    g_hal_sq_cq_query = reinterpret_cast<HalSqCqQueryFn>(dlsym(RTLD_DEFAULT, "halSqCqQuery"));
    if (g_hal_sq_cq_query == nullptr) {
        LOG_ERROR("SDMA prefetch: failed to resolve halSqCqQuery: %s", dlerror());
    }
    g_hal_sq_cq_query_resolved = true;
}

static void resolve_hal_resource_id_restore()
{
    if (g_hal_resource_id_restore_resolved) {
        return;
    }
    g_hal_resource_id_restore = reinterpret_cast<HalResourceIdRestoreFn>(dlsym(RTLD_DEFAULT, "halResourceIdRestore"));
    if (g_hal_resource_id_restore == nullptr && g_prefetch_debug_enabled) {
        LOG_INFO_V9("SDMA prefetch: halResourceIdRestore not found: %s", dlerror());
    }
    g_hal_resource_id_restore_resolved = true;
}

static drvError_t restore_stream_resource(uint32_t dev_id, uint32_t stream_id)
{
    resolve_hal_resource_id_restore();
    if (g_hal_resource_id_restore == nullptr) {
        return static_cast<drvError_t>(-1);
    }

    drvResIdKey key = {};
    key.ruDevId = dev_id;
    key.tsId = 0;
    key.resType = DRV_STREAM_ID;
    key.resId = stream_id;
    key.flag = 0;
    return g_hal_resource_id_restore(&key);
}

static uint64_t query_value_to_u64(const uint32_t value[SQCQ_QUERY_INFO_LENGTH])
{
    return static_cast<uint64_t>(value[0]) | (static_cast<uint64_t>(value[1]) << 32);
}

static uint64_t query_value_to_u64_reversed(const uint32_t value[SQCQ_QUERY_INFO_LENGTH])
{
    return static_cast<uint64_t>(value[1]) | (static_cast<uint64_t>(value[0]) << 32);
}

static bool query_sqcq_u64(
    uint32_t dev_id, uint32_t ts_id, drvSqCqType_t type, uint32_t sq_id, uint32_t cq_id, drvSqCqPropType_t prop,
    uint64_t* value
)
{
    halSqCqQueryInfo query = {};
    query.type = type;
    query.tsId = ts_id;
    query.sqId = sq_id;
    query.cqId = cq_id;
    query.prop = prop;
    drvError_t rc = g_hal_sq_cq_query(dev_id, &query);
    if (rc != 0) {
        return false;
    }
    if (prop == DRV_SQCQ_PROP_SQ_REG_BASE) {
        *value = query_value_to_u64_reversed(query.value);
    } else {
        *value = query_value_to_u64(query.value);
    }
    return true;
}

static bool populate_channel_with_hal(volatile stars_channel_info_t* ch, uint32_t channel_idx, uint32_t channel_count)
{
    resolve_hal_sq_cq_query();
    if (g_hal_sq_cq_query == nullptr) {
        return false;
    }

    static constexpr uint32_t TS_ID = 0;
    uint64_t sq_base = 0;
    uint64_t sq_reg_base = 0;
    uint64_t sq_depth = 0;
    bool ok =
        query_sqcq_u64(ch->dev_id, TS_ID, DRV_NORMAL_TYPE, ch->sq_id, ch->cq_id, DRV_SQCQ_PROP_SQ_BASE, &sq_base) &&
        query_sqcq_u64(
            ch->dev_id, TS_ID, DRV_NORMAL_TYPE, ch->sq_id, ch->cq_id, DRV_SQCQ_PROP_SQ_REG_BASE, &sq_reg_base
        ) &&
        query_sqcq_u64(ch->dev_id, TS_ID, DRV_NORMAL_TYPE, ch->sq_id, ch->cq_id, DRV_SQCQ_PROP_SQ_DEPTH, &sq_depth);
    if (ok && sq_base != 0 && sq_reg_base != 0 && sq_depth != 0) {
        ch->sq_base = sq_base;
        ch->sq_reg_base = sq_reg_base;
        ch->sq_depth = static_cast<uint32_t>(sq_depth);
        if (channel_idx < 4 || channel_idx + 1 == channel_count) {
            DEV_ALWAYS(
                "SDMA prefetch: HAL channel[%u] sid=%u sq=%u cq=%u dev=%u ts=%u type=%d base=0x%llx reg=0x%llx depth=%u",
                channel_idx, ch->stream_id, ch->sq_id, ch->cq_id, ch->dev_id, TS_ID, static_cast<int>(DRV_NORMAL_TYPE),
                (unsigned long long)ch->sq_base, (unsigned long long)ch->sq_reg_base, ch->sq_depth
            );
        }
        return true;
    }

    // Final diagnostic: current device + SHM query fails too. We do not use it as
    // the steady-state path, but logging it helps confirm whether a queue semantic
    // mismatch is the reason for failure.
    sq_base = 0;
    sq_reg_base = 0;
    sq_depth = 0;
    ok = query_sqcq_u64(ch->dev_id, TS_ID, DRV_SHM_TYPE, ch->sq_id, ch->cq_id, DRV_SQCQ_PROP_SQ_BASE, &sq_base) &&
         query_sqcq_u64(ch->dev_id, TS_ID, DRV_SHM_TYPE, ch->sq_id, ch->cq_id, DRV_SQCQ_PROP_SQ_REG_BASE, &sq_reg_base) &&
         query_sqcq_u64(ch->dev_id, TS_ID, DRV_SHM_TYPE, ch->sq_id, ch->cq_id, DRV_SQCQ_PROP_SQ_DEPTH, &sq_depth);
    if (ok && sq_base != 0 && sq_reg_base != 0 && sq_depth != 0) {
        DEV_ALWAYS(
            "SDMA prefetch: HAL found only SHM queue for channel[%u] sid=%u sq=%u cq=%u dev=%u ts=%u; NORMAL path unresolved",
            channel_idx, ch->stream_id, ch->sq_id, ch->cq_id, ch->dev_id, TS_ID
        );
    }

    DEV_ALWAYS(
        "SDMA prefetch: HAL query failed for channel[%u] sid=%u sq=%u cq=%u dev=%u", channel_idx, ch->stream_id,
        ch->sq_id, ch->cq_id, ch->dev_id
    );
    return false;
}

static inline void prefetch_lock(int channel_idx)
{
    while (__atomic_test_and_set(&g_prefetch_submit_lock[channel_idx], __ATOMIC_ACQUIRE)) {
    }
}

static inline void prefetch_unlock(int channel_idx)
{
    __atomic_clear(&g_prefetch_submit_lock[channel_idx], __ATOMIC_RELEASE);
}

static inline void tensor_dedup_lock()
{
    while (__atomic_exchange_n(&g_prefetch_tensor_dedup_lock, 1, __ATOMIC_ACQUIRE) != 0) {
    }
}

static inline void tensor_dedup_unlock()
{
    __atomic_store_n(&g_prefetch_tensor_dedup_lock, 0, __ATOMIC_RELEASE);
}

static inline uint32_t tensor_dedup_hash(uint64_t addr, uint64_t size)
{
    uint64_t mixed = (addr >> 12) ^ (addr >> 32) ^ (size >> 6) ^ (size << 7);
    return static_cast<uint32_t>(mixed) & (PREFETCH_TENSOR_DEDUP_SLOTS - 1);
}

static bool reserve_tensor_prefetch(void* tensor_addr, size_t tensor_size)
{
    uint64_t addr = reinterpret_cast<uint64_t>(tensor_addr);
    uint64_t size = static_cast<uint64_t>(tensor_size);
    uint32_t start = tensor_dedup_hash(addr, size);

    tensor_dedup_lock();
    for (int i = 0; i < PREFETCH_TENSOR_DEDUP_PROBE_LIMIT; ++i) {
        uint32_t slot = (start + static_cast<uint32_t>(i)) & (PREFETCH_TENSOR_DEDUP_SLOTS - 1);
        uint64_t seen_addr = __atomic_load_n(&g_prefetch_tensor_seen_addr[slot], __ATOMIC_RELAXED);
        uint64_t seen_size = __atomic_load_n(&g_prefetch_tensor_seen_size[slot], __ATOMIC_RELAXED);
        if (seen_addr == addr && seen_size == size) {
            uint32_t skipped = g_prefetch_debug_enabled
                ? __atomic_add_fetch(&g_prefetch_skip_duplicate_tensor, 1, __ATOMIC_RELAXED)
                : 0;
            tensor_dedup_unlock();
            if (g_prefetch_debug_enabled && (skipped <= 8 || (skipped % 64) == 0)) {
                DEV_ALWAYS(
                    "SDMA prefetch: skip duplicate tensor (addr=0x%llx size=%zu dup_tensor_count=%u)",
                    (unsigned long long)addr, tensor_size, skipped
                );
            }
            return false;
        }
        if (seen_addr == 0) {
            __atomic_store_n(&g_prefetch_tensor_seen_size[slot], size, __ATOMIC_RELAXED);
            __atomic_store_n(&g_prefetch_tensor_seen_addr[slot], addr, __ATOMIC_RELEASE);
            tensor_dedup_unlock();
            return true;
        }
    }

    if (g_prefetch_debug_enabled) {
        __atomic_add_fetch(&g_prefetch_tensor_dedup_overflow, 1, __ATOMIC_RELAXED);
    }
    tensor_dedup_unlock();
    return true;
}

static inline uint32_t consume_prefetch_suppress_window(int channel_idx)
{
    while (true) {
        uint32_t remaining = __atomic_load_n(&g_prefetch_channel_suppress_remaining[channel_idx], __ATOMIC_RELAXED);
        if (remaining == 0) {
            return 0;
        }
        uint32_t desired = remaining - 1;
        if (__atomic_compare_exchange_n(
                &g_prefetch_channel_suppress_remaining[channel_idx], &remaining, desired, false,
                __ATOMIC_ACQ_REL, __ATOMIC_RELAXED
            )) {
            return remaining;
        }
    }
}

static bool prepare_prefetch_channel(int* channel_idx, bool count_attempt, bool consume_suppress)
{
    if (!g_prefetch_enabled || g_channel_info == nullptr || g_channel_count == 0 || channel_idx == nullptr ||
        *channel_idx < 0) {
        return false;
    }

    *channel_idx %= static_cast<int>(g_channel_count);

    if (count_attempt && g_prefetch_debug_enabled) {
        __atomic_add_fetch(&g_prefetch_attempt_count, 1, __ATOMIC_RELAXED);
    }

    if (consume_suppress) {
        uint32_t suppress_remaining = consume_prefetch_suppress_window(*channel_idx);
        if (suppress_remaining != 0) {
            uint32_t skipped =
                g_prefetch_debug_enabled ? __atomic_add_fetch(&g_prefetch_skip_suppressed[*channel_idx], 1, __ATOMIC_RELAXED) : 0;
            if (g_prefetch_debug_enabled && (skipped <= 4 || (skipped % 64) == 0)) {
                DEV_ALWAYS(
                    "SDMA prefetch: suppressed, skip issue (channel=%d skipped=%u remaining_after=%u)",
                    *channel_idx, skipped, suppress_remaining - 1
                );
            }
            return false;
        }
    }

    return true;
}

static void fill_cmo_prefetch_sqe(volatile stars_sdma_cmo_sqe_t* sqe,
                                  uint64_t src_addr, uint32_t length,
                                  uint16_t stream_id, uint16_t task_id)
{
    memset(const_cast<stars_sdma_cmo_sqe_t*>(sqe), 0, sizeof(stars_sdma_cmo_sqe_t));

    sqe->type = STARS_SQE_TYPE_SDMA;
    sqe->rt_streamid = stream_id;
    sqe->task_id = task_id;
    sqe->kernel_credit = DEFAULT_KERNEL_CREDIT;

    sqe->opcode = CMO_OPCODE_PREFETCH;
    sqe->length = length;

    sqe->src_addr_low = static_cast<uint32_t>(src_addr & 0xFFFFFFFFu);
    sqe->src_addr_high = static_cast<uint32_t>((src_addr >> 32) & 0xFFFFFFFFu);

    sqe->qos = 6;
    sqe->partid = 63;
    sqe->sssv = 1;
    sqe->dssv = 1;
    sqe->sns = 1;
    sqe->dns = 1;
}

void aicpu_prefetch_init(void* sdma_workspace, uint32_t suppress_window, bool debug_enabled)
{
    g_prefetch_enabled = false;
    g_channel_info = nullptr;
    g_channel_count = 0;
    g_prefetch_suppress_window = suppress_window;
    g_prefetch_debug_enabled = debug_enabled;
    g_prefetch_attempt_count = 0;
    g_prefetch_issue_count = 0;
    g_prefetch_attempt_bytes = 0;
    g_prefetch_issue_bytes = 0;
    g_prefetch_attempt_cycles = 0;
    g_prefetch_issue_cycles = 0;
    for (int i = 0; i < PLATFORM_MAX_CORES; ++i) {
        g_prefetch_channel_suppress_remaining[i] = 0;
        g_prefetch_skip_suppressed[i] = 0;
        g_prefetch_last_instr_addr[i] = 0;
        g_prefetch_last_instr_size[i] = 0;
        g_prefetch_skip_duplicate_instr[i] = 0;
    }
    for (int i = 0; i < PREFETCH_MAX_KERNEL_ID; ++i) {
        g_prefetch_instr_kernel_seen[i] = 0;
    }
    g_prefetch_skip_duplicate_instr_kernel = 0;
    g_prefetch_tensor_dedup_lock = 0;
    for (int i = 0; i < PREFETCH_TENSOR_DEDUP_SLOTS; ++i) {
        g_prefetch_tensor_seen_addr[i] = 0;
        g_prefetch_tensor_seen_size[i] = 0;
    }
    g_prefetch_skip_duplicate_tensor = 0;
    g_prefetch_tensor_dedup_overflow = 0;

    if (sdma_workspace == nullptr) {
        DEV_ALWAYS("SDMA prefetch: disabled (no workspace provided)");
        return;
    }

    uint8_t* base = static_cast<uint8_t*>(sdma_workspace);
    auto* flag_info = reinterpret_cast<volatile stars_channel_flag_info_t*>(base);
    g_channel_info = reinterpret_cast<volatile stars_channel_info_t*>(base + sizeof(stars_channel_flag_info_t));
    if (g_prefetch_debug_enabled) {
        DEV_ALWAYS(
            "SDMA prefetch: workspace=%p flag_info=%p channel_info=%p flag=0x%x totalQueueNum=%u", sdma_workspace,
            const_cast<stars_channel_flag_info_t*>(flag_info), const_cast<stars_channel_info_t*>(g_channel_info),
            flag_info->flag, flag_info->totalQueueNum
        );
    }
    g_channel_count = flag_info->totalQueueNum;
    if (g_channel_count == 0 || g_channel_count > PLATFORM_MAX_CORES) {
        if (g_prefetch_debug_enabled) DEV_ALWAYS("SDMA prefetch: invalid channel count %u", g_channel_count);
        g_channel_info = nullptr;
        g_channel_count = 0;
        return;
    }

    for (uint32_t i = 0; i < g_channel_count; ++i) {
        volatile stars_channel_info_t* ch = g_channel_info + i;
        drvError_t restore_rc = restore_stream_resource(ch->dev_id, ch->stream_id);
        if (restore_rc != 0 && g_prefetch_debug_enabled) {
            DEV_ALWAYS(
                "SDMA prefetch: stream resource restore failed channel[%u] dev=%u stream=%u rc=%d",
                i, ch->dev_id, ch->stream_id, static_cast<int>(restore_rc)
            );
        }
        if (!populate_channel_with_hal(ch, i, g_channel_count)) {
            g_channel_info = nullptr;
            g_channel_count = 0;
            return;
        }
    }

    if (g_channel_info->sq_base == 0 || g_channel_info->sq_reg_base == 0 || g_channel_info->sq_depth == 0) {
        if (g_prefetch_debug_enabled) {
            DEV_ALWAYS(
                "SDMA prefetch: disabled (channel info not initialized after HAL query: sq_base=%llu sq_reg=%llu depth=%u)",
                (unsigned long long)g_channel_info->sq_base, (unsigned long long)g_channel_info->sq_reg_base,
                g_channel_info->sq_depth
            );
        }
        g_channel_info = nullptr;
        return;
    }

    g_prefetch_enabled = true;
    if (g_prefetch_debug_enabled) {
        for (uint32_t i = 0; i < g_channel_count && i < 4; ++i) {
            volatile stars_channel_info_t* ch = g_channel_info + i;
            DEV_ALWAYS(
                "SDMA prefetch: channel[%u] sq_head=%u sq_tail=%u sq_base=0x%llx sq_reg=0x%llx depth=%u sq_id=%u cq_id=%u stream=%u",
                i, ch->sq_head, ch->sq_tail, (unsigned long long)ch->sq_base, (unsigned long long)ch->sq_reg_base,
                ch->sq_depth, ch->sq_id, ch->cq_id, ch->stream_id
            );
        }
        DEV_ALWAYS("SDMA prefetch: enabled (channels=%u sq_base=0x%llx sq_reg=0x%llx depth=%u stream=%u)",
                   g_channel_count,
                   (unsigned long long)g_channel_info->sq_base,
                   (unsigned long long)g_channel_info->sq_reg_base,
                   g_channel_info->sq_depth,
                   g_channel_info->stream_id);
    }
}

void aicpu_prefetch_deinit()
{
    uint32_t attempt_count = __atomic_load_n(&g_prefetch_attempt_count, __ATOMIC_ACQUIRE);
    uint32_t issue_count = __atomic_load_n(&g_prefetch_issue_count, __ATOMIC_ACQUIRE);
    uint64_t attempt_bytes = __atomic_load_n(&g_prefetch_attempt_bytes, __ATOMIC_ACQUIRE);
    uint64_t issue_bytes = __atomic_load_n(&g_prefetch_issue_bytes, __ATOMIC_ACQUIRE);
    uint64_t attempt_cycles = __atomic_load_n(&g_prefetch_attempt_cycles, __ATOMIC_ACQUIRE);
    uint64_t issue_cycles = __atomic_load_n(&g_prefetch_issue_cycles, __ATOMIC_ACQUIRE);
    uint64_t suppressed_count = 0;
    uint64_t queue_full_count = 0;
    uint64_t duplicate_instr_count = 0;
    uint64_t duplicate_instr_kernel_count = __atomic_load_n(&g_prefetch_skip_duplicate_instr_kernel, __ATOMIC_ACQUIRE);
    uint64_t duplicate_tensor_count = __atomic_load_n(&g_prefetch_skip_duplicate_tensor, __ATOMIC_ACQUIRE);
    uint64_t tensor_dedup_overflow_count = __atomic_load_n(&g_prefetch_tensor_dedup_overflow, __ATOMIC_ACQUIRE);
    for (int i = 0; i < PLATFORM_MAX_CORES; ++i) {
        suppressed_count += __atomic_load_n(&g_prefetch_skip_suppressed[i], __ATOMIC_ACQUIRE);
        queue_full_count += __atomic_load_n(&g_prefetch_skip_queue_full[i], __ATOMIC_ACQUIRE);
        duplicate_instr_count += __atomic_load_n(&g_prefetch_skip_duplicate_instr[i], __ATOMIC_ACQUIRE);
    }
    if (g_prefetch_debug_enabled) {
        DEV_ALWAYS(
            "SDMA prefetch issue summary: enabled=%d attempts=%u issues=%u bytes=%" PRIu64
            " issue_bytes=%" PRIu64 " suppressed=%" PRIu64 " queue_full=%" PRIu64
            " dup_tensor=%" PRIu64 " dedup_overflow=%" PRIu64
            " dup_instr=%" PRIu64 " dup_instr_kernel=%" PRIu64 " total=%.3fus avg=%.3fus"
            " issue_total=%.3fus issue_avg=%.3fus",
            g_prefetch_enabled ? 1 : 0, attempt_count, issue_count, attempt_bytes, issue_bytes, suppressed_count,
            queue_full_count, duplicate_tensor_count, tensor_dedup_overflow_count, duplicate_instr_count,
            duplicate_instr_kernel_count, cycles_to_us(attempt_cycles),
            attempt_count > 0 ? cycles_to_us(attempt_cycles) / attempt_count : 0.0,
            cycles_to_us(issue_cycles), issue_count > 0 ? cycles_to_us(issue_cycles) / issue_count : 0.0
        );
    }

    g_prefetch_enabled = false;
    g_channel_info = nullptr;
    g_channel_count = 0;
    for (int i = 0; i < PLATFORM_MAX_CORES; ++i) {
        __atomic_store_n(&g_prefetch_submit_lock[i], 0, __ATOMIC_RELEASE);
        __atomic_store_n(&g_prefetch_skip_queue_full[i], 0, __ATOMIC_RELEASE);
        __atomic_store_n(&g_prefetch_channel_suppress_remaining[i], 0, __ATOMIC_RELEASE);
        __atomic_store_n(&g_prefetch_skip_suppressed[i], 0, __ATOMIC_RELEASE);
        __atomic_store_n(&g_prefetch_last_instr_addr[i], 0, __ATOMIC_RELEASE);
        __atomic_store_n(&g_prefetch_last_instr_size[i], 0, __ATOMIC_RELEASE);
        __atomic_store_n(&g_prefetch_skip_duplicate_instr[i], 0, __ATOMIC_RELEASE);
    }
    for (int i = 0; i < PREFETCH_MAX_KERNEL_ID; ++i) {
        __atomic_store_n(&g_prefetch_instr_kernel_seen[i], 0, __ATOMIC_RELEASE);
    }
    __atomic_store_n(&g_prefetch_skip_duplicate_instr_kernel, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&g_prefetch_tensor_dedup_lock, 0, __ATOMIC_RELEASE);
    for (int i = 0; i < PREFETCH_TENSOR_DEDUP_SLOTS; ++i) {
        __atomic_store_n(&g_prefetch_tensor_seen_addr[i], 0, __ATOMIC_RELEASE);
        __atomic_store_n(&g_prefetch_tensor_seen_size[i], 0, __ATOMIC_RELEASE);
    }
    __atomic_store_n(&g_prefetch_skip_duplicate_tensor, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&g_prefetch_tensor_dedup_overflow, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&g_prefetch_attempt_count, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&g_prefetch_issue_count, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&g_prefetch_attempt_bytes, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&g_prefetch_issue_bytes, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&g_prefetch_attempt_cycles, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&g_prefetch_issue_cycles, 0, __ATOMIC_RELEASE);
}

bool aicpu_prefetch_reserve_channel(int channel_idx)
{
    return prepare_prefetch_channel(&channel_idx, true, true);
}

void aicpu_prefetch_issue_reserved(
    void* tensor_addr, size_t tensor_size, void* instr_addr, size_t instr_size, int32_t instr_kernel_id, int channel_idx
)
{
    uint64_t start_cycle = g_prefetch_debug_enabled ? get_sys_cnt_aicpu() : 0;
    bool issued = false;

    do {
        if (!g_prefetch_enabled) {
            break;
        }

        bool issue_tensor = (tensor_addr != nullptr && tensor_size > 0);
        bool issue_instr = (instr_addr != nullptr && instr_size > 0);
        if (!issue_tensor && !issue_instr) {
            break;
        }

        if (!prepare_prefetch_channel(&channel_idx, false, false)) {
            break;
        }

        if (issue_tensor && !reserve_tensor_prefetch(tensor_addr, tensor_size)) {
            issue_tensor = false;
            if (!issue_instr) {
                break;
            }
        }

        volatile stars_channel_info_t* ch = g_channel_info + channel_idx;
        prefetch_lock(channel_idx);

        uint32_t sq_head = __atomic_load_n(&ch->sq_head, __ATOMIC_ACQUIRE);
        uint32_t sq_tail = __atomic_load_n(&ch->sq_tail, __ATOMIC_ACQUIRE);
        uint32_t sq_depth = ch->sq_depth;
        if (sq_depth == 0) {
            DEV_ALWAYS("SDMA prefetch: channel %d has zero depth", channel_idx);
            prefetch_unlock(channel_idx);
            break;
        }

        uint32_t new_tail = (sq_tail + 1) % sq_depth;
        if (new_tail == sq_head) {
            uint32_t skipped =
                g_prefetch_debug_enabled ? __atomic_add_fetch(&g_prefetch_skip_queue_full[channel_idx], 1, __ATOMIC_RELAXED) : 0;
            if (g_prefetch_debug_enabled && (skipped <= 4 || (skipped % 64) == 0)) {
                DEV_ALWAYS("SDMA prefetch: queue full, skip issue (channel=%d head=%u tail=%u depth=%u skipped=%u)",
                           channel_idx, sq_head, sq_tail, sq_depth, skipped);
            }
            prefetch_unlock(channel_idx);
            break;
        }

        bool duplicate_instr = false;
        bool duplicate_instr_kernel = false;
        if (issue_instr) {
            if (instr_kernel_id >= 0 && instr_kernel_id < PREFETCH_MAX_KERNEL_ID) {
                uint8_t expected = 0;
                duplicate_instr_kernel = !__atomic_compare_exchange_n(
                    &g_prefetch_instr_kernel_seen[instr_kernel_id], &expected, 1, false, __ATOMIC_ACQ_REL,
                    __ATOMIC_RELAXED
                );
                if (duplicate_instr_kernel && g_prefetch_debug_enabled) {
                    __atomic_add_fetch(&g_prefetch_skip_duplicate_instr_kernel, 1, __ATOMIC_RELAXED);
                }
            }
            duplicate_instr = (
                __atomic_load_n(&g_prefetch_last_instr_addr[channel_idx], __ATOMIC_RELAXED) ==
                    reinterpret_cast<uint64_t>(instr_addr) &&
                __atomic_load_n(&g_prefetch_last_instr_size[channel_idx], __ATOMIC_RELAXED) ==
                    static_cast<uint32_t>(instr_size)
            );
            if (duplicate_instr && g_prefetch_debug_enabled) {
                __atomic_add_fetch(&g_prefetch_skip_duplicate_instr[channel_idx], 1, __ATOMIC_RELAXED);
            }
        }
        issue_instr = issue_instr && !duplicate_instr && !duplicate_instr_kernel;
        if (issue_tensor && issue_instr) {
            uint32_t second_tail = (new_tail + 1u) % sq_depth;
            if (second_tail == sq_head) {
                issue_instr = false;
            }
        }
        if (!issue_tensor && !issue_instr) {
            prefetch_unlock(channel_idx);
            break;
        }

        uint32_t final_tail = new_tail;
        if (issue_tensor && issue_instr) {
            final_tail = (new_tail + 1u) % sq_depth;
        }

        if (g_prefetch_debug_enabled) {
            if (issue_tensor) {
                __atomic_add_fetch(&g_prefetch_attempt_bytes, static_cast<uint64_t>(tensor_size), __ATOMIC_RELAXED);
            }
            if (issue_instr) {
                __atomic_add_fetch(&g_prefetch_attempt_bytes, static_cast<uint64_t>(instr_size), __ATOMIC_RELAXED);
            }
        }

        volatile stars_sdma_cmo_sqe_t* sq_base =
            reinterpret_cast<volatile stars_sdma_cmo_sqe_t*>(ch->sq_base);

        uint32_t write_tail = sq_tail;
        uint32_t issue_idx = 0;
        if (issue_tensor) {
            volatile stars_sdma_cmo_sqe_t* tensor_sqe = sq_base + write_tail;
            uint16_t task_id = static_cast<uint16_t>((write_tail + sq_depth - sq_head) % sq_depth);
            fill_cmo_prefetch_sqe(
                tensor_sqe, reinterpret_cast<uint64_t>(tensor_addr), static_cast<uint32_t>(tensor_size),
                static_cast<uint16_t>(ch->stream_id), task_id
            );
            cache_flush_range(const_cast<stars_sdma_cmo_sqe_t*>(tensor_sqe), sizeof(stars_sdma_cmo_sqe_t));
            issue_idx =
                g_prefetch_debug_enabled ? __atomic_add_fetch(&g_prefetch_issue_count, 1, __ATOMIC_RELAXED) : 0;
            if (g_prefetch_debug_enabled) {
                __atomic_add_fetch(&g_prefetch_issue_bytes, static_cast<uint64_t>(tensor_size), __ATOMIC_RELAXED);
            }
            if (g_prefetch_debug_enabled && issue_idx <= 8) {
                DEV_ALWAYS(
                    "SDMA prefetch: issue[%u] channel=%d tensor addr=0x%llx size=%zu sq_head=%u sq_tail=%u new_tail=%u depth=%u",
                    issue_idx, channel_idx, (unsigned long long)reinterpret_cast<uint64_t>(tensor_addr), tensor_size,
                    sq_head, sq_tail, new_tail, sq_depth
                );
            }
            write_tail = (write_tail + 1u) % sq_depth;
        }

        if (issue_instr) {
            volatile stars_sdma_cmo_sqe_t* instr_sqe = sq_base + write_tail;
            uint16_t task_id = static_cast<uint16_t>((write_tail + sq_depth - sq_head) % sq_depth);
            fill_cmo_prefetch_sqe(
                instr_sqe, reinterpret_cast<uint64_t>(instr_addr), static_cast<uint32_t>(instr_size),
                static_cast<uint16_t>(ch->stream_id), task_id
            );
            cache_flush_range(const_cast<stars_sdma_cmo_sqe_t*>(instr_sqe), sizeof(stars_sdma_cmo_sqe_t));
            if (g_prefetch_debug_enabled) {
                issue_idx = __atomic_add_fetch(&g_prefetch_issue_count, 1, __ATOMIC_RELAXED);
                __atomic_add_fetch(&g_prefetch_issue_bytes, static_cast<uint64_t>(instr_size), __ATOMIC_RELAXED);
            }
            if (g_prefetch_debug_enabled && issue_idx <= 8) {
                DEV_ALWAYS(
                    "SDMA prefetch: issue[%u] channel=%d instr addr=0x%llx size=%zu head=%u tail=%u final_tail=%u depth=%u",
                    issue_idx, channel_idx, (unsigned long long)reinterpret_cast<uint64_t>(instr_addr), instr_size,
                    sq_head, sq_tail, final_tail, sq_depth
                );
            }
            __atomic_store_n(
                &g_prefetch_last_instr_addr[channel_idx], reinterpret_cast<uint64_t>(instr_addr), __ATOMIC_RELEASE
            );
            __atomic_store_n(
                &g_prefetch_last_instr_size[channel_idx], static_cast<uint32_t>(instr_size), __ATOMIC_RELEASE
            );
        } else if (duplicate_instr && g_prefetch_debug_enabled) {
            uint32_t dup_count = __atomic_load_n(&g_prefetch_skip_duplicate_instr[channel_idx], __ATOMIC_RELAXED);
            if (dup_count <= 4 || (dup_count % 64) == 0) {
                DEV_ALWAYS(
                    "SDMA prefetch: skip duplicate instr sqe (channel=%d addr=0x%llx size=%zu dup_count=%u)",
                    channel_idx, (unsigned long long)reinterpret_cast<uint64_t>(instr_addr), instr_size, dup_count
                );
            }
        } else if (duplicate_instr_kernel && g_prefetch_debug_enabled) {
            uint32_t dup_count =
                __atomic_load_n(&g_prefetch_skip_duplicate_instr_kernel, __ATOMIC_RELAXED);
            if (dup_count <= 4 || (dup_count % 64) == 0) {
                DEV_ALWAYS(
                    "SDMA prefetch: skip duplicate instr kernel (kid=%d addr=0x%llx size=%zu dup_kernel_count=%u)",
                    instr_kernel_id, (unsigned long long)reinterpret_cast<uint64_t>(instr_addr), instr_size, dup_count
                );
            }
        }

        __atomic_thread_fence(__ATOMIC_RELEASE);
        __atomic_store_n(&ch->sq_tail, final_tail, __ATOMIC_RELEASE);
        __atomic_store_n(&g_prefetch_channel_suppress_remaining[channel_idx], g_prefetch_suppress_window, __ATOMIC_RELEASE);

        volatile uint32_t* doorbell = reinterpret_cast<volatile uint32_t*>(ch->sq_reg_base + 8);
        *doorbell = final_tail;

        prefetch_unlock(channel_idx);
        issued = true;
    } while (false);

    if (g_prefetch_debug_enabled) {
        uint64_t elapsed = get_sys_cnt_aicpu() - start_cycle;
        __atomic_add_fetch(&g_prefetch_attempt_cycles, elapsed, __ATOMIC_RELAXED);
        if (issued) {
            __atomic_add_fetch(&g_prefetch_issue_cycles, elapsed, __ATOMIC_RELAXED);
        }
    }
}

void aicpu_prefetch_tensor(void* addr, size_t size, int channel_idx)
{
    if (!aicpu_prefetch_reserve_channel(channel_idx)) {
        return;
    }
    aicpu_prefetch_issue_reserved(addr, size, nullptr, 0, -1, channel_idx);
}

bool aicpu_prefetch_available()
{
    return g_prefetch_enabled;
}

uint32_t aicpu_prefetch_channel_count()
{
    return g_channel_count;
}
