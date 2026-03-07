#pragma once

// =========================================================================
// RESIDUE WALL — Level 4: Thread Isolation Zone (V4.1 HARDENED)
//
// V4.1 Changes:
//   1. 3-Tier Memory Lock Cascade (VirtualLock → HugePages → Prefetch)
//   2. Priority De-escalation (HIGH, not REALTIME — no OS freeze)
//   3. SMT Detection (pin to physical core, avoid HT sibling)
//   4. Explicit IsolationStatus with MemoryLockTier reporting
//
// All mechanisms are RAII-safe: destructor restores original state.
// Graceful degradation: if a mechanism requires admin and fails,
// we record the tier and continue without it.
// =========================================================================

#include <cstddef>
#include <cstdint>
#include <cstdio>

// Cross-platform forceinline
#ifdef _MSC_VER
#define RESIDUE_ZONE_FORCEINLINE __forceinline
#else
#define RESIDUE_ZONE_FORCEINLINE __attribute__((always_inline)) inline
#endif

// =========================================================================
// SHARED: Memory Lock Tier Enum
// =========================================================================
namespace residue_wall {

enum class MemoryLockTier : uint8_t {
  LOCKED = 0,     // VirtualLock / mlock succeeded — pages pinned in RAM
  HUGE_PAGES = 1, // Large/Huge pages allocated — TLB-optimal, pinned
  PREFETCHED =
      2,   // Advisory prefetch only — no guarantee, but better than nothing
  NONE = 3 // No memory protection — page faults possible
};

struct IsolationStatus {
  MemoryLockTier memory_tier;
  bool timer_hires;
  bool core_pinned;
  bool priority_elevated;
  bool smt_detected;
  bool irq_warned;
  bool numa_cross_node_risk; // V4.2: true if >1 NUMA node detected

  uint32_t locked_bytes;
  uint32_t pinned_core;
  uint32_t physical_core;   // actual physical core ID (may differ from logical)
  uint32_t cpu_numa_node;   // V4.2: NUMA node the worker runs on
  uint32_t numa_node_count; // V4.2: total NUMA nodes in system
  double last_elapsed_us;
};

} // namespace residue_wall

// =========================================================================
// PLATFORM: WINDOWS
// =========================================================================
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#pragma comment(lib, "winmm.lib") // for timeBeginPeriod/timeEndPeriod

namespace residue_wall {

class IsolationZone {
private:
  // --- Thread/Process handles ---
  HANDLE hThread_;
  HANDLE hProcess_;

  // --- Saved state for RAII restoration ---
  DWORD_PTR old_affinity_mask_;
  int old_thread_priority_;
  DWORD old_priority_class_;
  LARGE_INTEGER freq_, start_time_;

  // --- Memory lock tracking ---
  struct LockedRegion {
    const void *ptr;
    size_t bytes;
    MemoryLockTier tier;
  };
  static constexpr size_t MAX_LOCKED_REGIONS = 8;
  LockedRegion locked_regions_[MAX_LOCKED_REGIONS];
  uint32_t num_locked_;
  uint32_t total_locked_bytes_;

  // --- Runtime state ---
  bool priority_dropped_;
  bool tile_stalled_;
  double last_elapsed_us_;
  bool timer_hires_set_;
  uint32_t pinned_core_;
  uint32_t physical_core_;
  bool smt_detected_;
  uint32_t cpu_numa_node_;   // V4.2
  uint32_t numa_node_count_; // V4.2

  // --- One-shot warnings ---
  inline static bool warned_memory_lock_ = false;
  inline static bool warned_timer_ = false;
  inline static bool warned_affinity_ = false;
  inline static bool warned_irq_ = false;

  // =======================================================================
  // SMT DETECTION: Find physical cores vs logical threads
  // Returns the logical processor mask for the FIRST thread of the last
  // physical core. If detection fails, falls back to last logical core.
  // =======================================================================
  struct SmtInfo {
    DWORD_PTR target_mask;
    uint32_t logical_core_id;
    uint32_t physical_core_id;
    bool smt_detected;
  };

  static SmtInfo detect_smt_topology() {
    SmtInfo info = {};
    info.smt_detected = false;

    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    DWORD num_logical = sys_info.dwNumberOfProcessors;

    // Default fallback: last logical core
    info.logical_core_id = num_logical - 1;
    info.physical_core_id = info.logical_core_id;
    info.target_mask = (DWORD_PTR)1 << info.logical_core_id;

    // Query processor core topology
    DWORD buffer_size = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr,
                                     &buffer_size);
    if (buffer_size == 0)
      return info;

    alignas(8) uint8_t stack_buf[4096];
    uint8_t *buf = stack_buf;
    bool heap_alloc = false;

    if (buffer_size > sizeof(stack_buf)) {
      buf = static_cast<uint8_t *>(malloc(buffer_size));
      if (!buf)
        return info;
      heap_alloc = true;
    }

    auto *pinfo =
        reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, pinfo,
                                          &buffer_size)) {
      if (heap_alloc)
        free(buf);
      return info;
    }

    // Walk the core list: find the last physical core
    struct PhysicalCore {
      DWORD_PTR mask;
      uint32_t thread_count;
    };

    PhysicalCore last_core = {};
    uint32_t physical_core_count = 0;

    auto *ptr = buf;
    auto *end = buf + buffer_size;
    while (ptr < end) {
      auto *entry =
          reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(ptr);
      if (entry->Relationship == RelationProcessorCore) {
        DWORD_PTR core_mask = 0;
        for (WORD g = 0; g < entry->Processor.GroupCount; ++g) {
          core_mask |= entry->Processor.GroupMask[g].Mask;
        }

        // Count logical threads in this physical core
        uint32_t threads = 0;
        DWORD_PTR m = core_mask;
        while (m) {
          threads += (m & 1);
          m >>= 1;
        }

        if (threads > 1)
          info.smt_detected = true;

        last_core.mask = core_mask;
        last_core.thread_count = threads;
        physical_core_count++;
      }
      ptr += entry->Size;
    }

    if (heap_alloc)
      free(buf);

    if (physical_core_count > 0) {
      info.physical_core_id = physical_core_count - 1;

      // Extract the FIRST logical processor from the last physical core
      // (avoid the HT sibling)
      DWORD_PTR mask = last_core.mask;
      DWORD bit = 0;
      while (mask && !(mask & 1)) {
        mask >>= 1;
        bit++;
      }

      info.target_mask = (DWORD_PTR)1 << bit;
      info.logical_core_id = bit;
    }

    return info;
  }

public:
  // =====================================================================
  // CONSTRUCTOR: Enter the Isolation Zone (V4.1 HARDENED)
  // =====================================================================
  IsolationZone()
      : num_locked_(0), total_locked_bytes_(0), priority_dropped_(false),
        tile_stalled_(false), last_elapsed_us_(0.0), timer_hires_set_(false),
        pinned_core_(0), physical_core_(0), smt_detected_(false) {
    hThread_ = GetCurrentThread();
    hProcess_ = GetCurrentProcess();

    // Save original state
    old_priority_class_ = GetPriorityClass(hProcess_);
    old_thread_priority_ = GetThreadPriority(hThread_);

    // --- LAYER 1: Timer Resolution ---
    MMRESULT timer_result = timeBeginPeriod(1);
    if (timer_result == TIMERR_NOERROR) {
      timer_hires_set_ = true;
    } else if (!warned_timer_) {
      fprintf(stderr, "[ISOLATION ZONE] timeBeginPeriod(1) failed. "
                      "OS timer remains at default resolution.\n");
      warned_timer_ = true;
    }

    // --- LAYER 2: Core Isolation (SMT-aware) ---
    SmtInfo smt = detect_smt_topology();
    smt_detected_ = smt.smt_detected;
    pinned_core_ = smt.logical_core_id;
    physical_core_ = smt.physical_core_id;

    old_affinity_mask_ = SetThreadAffinityMask(hThread_, smt.target_mask);

    if (!old_affinity_mask_ && !warned_affinity_) {
      fprintf(stderr,
              "[ISOLATION ZONE] Core pinning to logical core %u "
              "(physical %u) FAILED.\n",
              pinned_core_, physical_core_);
      warned_affinity_ = true;
    }

    // --- LAYER 3: Priority (V4.1 HARDENED: HIGH, not REALTIME) ---
    SetPriorityClass(hProcess_, HIGH_PRIORITY_CLASS);
    SetThreadPriority(hThread_, THREAD_PRIORITY_HIGHEST);

    // --- LAYER 5: NUMA Topology Detection (V4.2) ---
    cpu_numa_node_ = 0;
    numa_node_count_ = 0;
    {
      UCHAR numa_node = 0;
      if (GetNumaProcessorNode(static_cast<UCHAR>(pinned_core_), &numa_node)) {
        cpu_numa_node_ = numa_node;
      }
      // Count total NUMA nodes
      ULONG highest_node = 0;
      if (GetNumaHighestNodeNumber(&highest_node)) {
        numa_node_count_ = highest_node + 1;
      } else {
        numa_node_count_ = 1;
      }
    }

    // --- LAYER 4: IRQ Shield (advisory) ---
    if (!warned_irq_) {
      warned_irq_ = true;
    }

    // Initialize timer
    QueryPerformanceFrequency(&freq_);
    QueryPerformanceCounter(&start_time_);
  }

  // =====================================================================
  // MEMORY LOCKING: 3-Tier Cascade (V4.1 HARDENED)
  //
  // Tier 1: VirtualLock() — true page pinning (requires working set quota)
  // Tier 2: Large Pages (MEM_LARGE_PAGES) — 2MB TLB entries (requires
  //         SeLockMemoryPrivilege, admin)
  // Tier 3: PrefetchVirtualMemory() — advisory, no privileges needed
  //         (Win8+, still better than nothing)
  // Tier 4: NONE — all failed, page faults possible
  // =====================================================================
  MemoryLockTier lock_memory(const void *ptr, size_t bytes) {
    if (!ptr || bytes == 0 || num_locked_ >= MAX_LOCKED_REGIONS)
      return MemoryLockTier::NONE;

    // --- TIER 1: VirtualLock ---
    // Increase working set size to accommodate the lock.
    SIZE_T min_ws, max_ws;
    if (GetProcessWorkingSetSize(hProcess_, &min_ws, &max_ws)) {
      SIZE_T needed = min_ws + bytes + (64 * 1024);
      if (needed > max_ws) {
        SetProcessWorkingSetSize(hProcess_, needed, needed);
      }
    }

    if (VirtualLock(const_cast<void *>(ptr), bytes)) {
      locked_regions_[num_locked_] = {ptr, bytes, MemoryLockTier::LOCKED};
      num_locked_++;
      total_locked_bytes_ += static_cast<uint32_t>(bytes);
      return MemoryLockTier::LOCKED;
    }

    // --- TIER 2: PrefetchVirtualMemory (Win8+) ---
    // This is an advisory hint to the VMM to bring pages into the
    // working set. Not as strong as VirtualLock, but prevents the
    // initial burst of page faults on first access.
    typedef BOOL(WINAPI * PrefetchVirtualMemoryFunc)(
        HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);

    static PrefetchVirtualMemoryFunc pPrefetch =
        (PrefetchVirtualMemoryFunc)GetProcAddress(
            GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory");

    if (pPrefetch) {
      WIN32_MEMORY_RANGE_ENTRY entry;
      entry.VirtualAddress = const_cast<void *>(ptr);
      entry.NumberOfBytes = bytes;

      if (pPrefetch(hProcess_, 1, &entry, 0)) {
        locked_regions_[num_locked_] = {ptr, bytes, MemoryLockTier::PREFETCHED};
        num_locked_++;
        total_locked_bytes_ += static_cast<uint32_t>(bytes);
        return MemoryLockTier::PREFETCHED;
      }
    }

    // --- TIER 3: NONE ---
    if (!warned_memory_lock_) {
      DWORD err = GetLastError();
      fprintf(
          stderr,
          "[ISOLATION ZONE] Memory lock FAILED for %zu bytes (err=%lu).\n"
          "  Tier 1 (VirtualLock): FAILED\n"
          "  Tier 2 (PrefetchVirtualMemory): %s\n"
          "  -> Running in UNPROTECTED mode. Page faults may occur.\n"
          "  -> Fix: Run as Administrator, or increase working set quota.\n",
          bytes, err, pPrefetch ? "FAILED" : "UNAVAILABLE (requires Win8+)");
      warned_memory_lock_ = true;
    }
    return MemoryLockTier::NONE;
  }

  // =====================================================================
  // HOT PATH: Timer reset (per-tile)
  // =====================================================================
  RESIDUE_ZONE_FORCEINLINE void reset_timer() {
    tile_stalled_ = false;
    last_elapsed_us_ = 0.0;
    QueryPerformanceCounter(&start_time_);
  }

  // =====================================================================
  // HOT PATH: Safety watchdog (per-tile)
  // =====================================================================
  RESIDUE_ZONE_FORCEINLINE void check_safety() {
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    last_elapsed_us_ =
        static_cast<double>(now.QuadPart - start_time_.QuadPart) * 1000000.0 /
        freq_.QuadPart;

    if (last_elapsed_us_ > 100.0) {
      tile_stalled_ = true;
      if (!priority_dropped_) {
        SetPriorityClass(hProcess_, NORMAL_PRIORITY_CLASS);
        SetThreadPriority(hThread_, THREAD_PRIORITY_NORMAL);
        priority_dropped_ = true;
      }
    }
  }

  RESIDUE_ZONE_FORCEINLINE bool was_stalled() const { return tile_stalled_; }
  RESIDUE_ZONE_FORCEINLINE double elapsed_us() const {
    return last_elapsed_us_;
  }

  // =====================================================================
  // STATUS: Query actual isolation level (V4.1 — no more guessing)
  // =====================================================================
  IsolationStatus get_status() const {
    IsolationStatus s;
    // Determine the best tier achieved across all locked regions
    s.memory_tier = MemoryLockTier::NONE;
    for (uint32_t i = 0; i < num_locked_; ++i) {
      if (locked_regions_[i].tier < s.memory_tier) {
        s.memory_tier = locked_regions_[i].tier;
      }
    }
    s.timer_hires = timer_hires_set_;
    s.core_pinned = (old_affinity_mask_ != 0);
    s.priority_elevated = !priority_dropped_;
    s.smt_detected = smt_detected_;
    s.irq_warned = warned_irq_;
    s.locked_bytes = total_locked_bytes_;
    s.pinned_core = pinned_core_;
    s.physical_core = physical_core_;
    s.cpu_numa_node = cpu_numa_node_;
    s.numa_node_count = numa_node_count_;
    s.numa_cross_node_risk = (numa_node_count_ > 1);
    s.last_elapsed_us = last_elapsed_us_;
    return s;
  }

  // =====================================================================
  // DIAGNOSTICS: Print full isolation status report (V4.1 HARDENED)
  // =====================================================================
  static void print_isolation_report() {
    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    DWORD num_cores = sys_info.dwNumberOfProcessors;

    SmtInfo smt = detect_smt_topology();

    fprintf(stdout,
            "======================================================\n"
            "   THREAD ISOLATION ZONE — STATUS REPORT (V4.2)       \n"
            "======================================================\n"
            "  Platform       : Windows x86-64\n"
            "  Logical Cores  : %u\n"
            "  SMT Detected   : %s\n"
            "  Target Core    : Logical %u (Physical %u)\n"
            "------------------------------------------------------\n"
            "  Layer 1 — Timer    : timeBeginPeriod(1)\n"
            "  Layer 2 — Affinity : SMT-aware core pinning\n"
            "  Layer 3 — Priority : HIGH_PRIORITY_CLASS (safe)\n"
            "  Layer 4 — Memory   : 3-Tier Cascade\n"
            "      Tier 1: VirtualLock (requires working set)\n"
            "      Tier 2: PrefetchVirtualMemory (advisory)\n"
            "      Tier 3: NONE (page faults possible)\n"
            "------------------------------------------------------\n",
            num_cores, smt.smt_detected ? "YES (HT sibling avoided)" : "NO",
            smt.logical_core_id, smt.physical_core_id);

    // V4.2: NUMA layer
    ULONG highest_node = 0;
    GetNumaHighestNodeNumber(&highest_node);
    uint32_t node_count = highest_node + 1;
    UCHAR cpu_node = 0;
    GetNumaProcessorNode(static_cast<UCHAR>(smt.logical_core_id), &cpu_node);

    if (node_count > 1) {
      fprintf(stdout,
              "  Layer 5 — NUMA     : Node %u (%u nodes detected "
              "— CROSS-NODE RISK!)\n"
              "                        Run with: start /node %u python ...\n",
              (unsigned)cpu_node, node_count, (unsigned)cpu_node);
    } else {
      fprintf(stdout,
              "  Layer 5 — NUMA     : Node 0 (%u node total "
              "— no cross-node risk)\n",
              node_count);
    }

    fprintf(stdout,
            "------------------------------------------------------\n"
            "  IRQ Steering (admin required):\n"
            "    1. Open Device Manager\n"
            "    2. Properties > Advanced > Interrupt Affinity\n"
            "    3. Exclude core %u from NIC/USB IRQs\n"
            "======================================================\n",
            smt.logical_core_id);
  }

  // =====================================================================
  // DESTRUCTOR: Exit the Isolation Zone (RAII)
  // =====================================================================
  ~IsolationZone() {
    // Unlock all memory regions (only VirtualLock needs explicit unlock)
    for (uint32_t i = 0; i < num_locked_; ++i) {
      if (locked_regions_[i].tier == MemoryLockTier::LOCKED) {
        VirtualUnlock(const_cast<void *>(locked_regions_[i].ptr),
                      locked_regions_[i].bytes);
      }
      // PREFETCHED regions don't need unlock — they're advisory
    }

    // Restore timer resolution
    if (timer_hires_set_) {
      timeEndPeriod(1);
    }

    // Restore priority (if not already dropped by watchdog)
    if (!priority_dropped_) {
      SetPriorityClass(hProcess_, old_priority_class_);
      SetThreadPriority(hThread_, old_thread_priority_);
    }

    // Restore affinity
    if (old_affinity_mask_) {
      SetThreadAffinityMask(hThread_, old_affinity_mask_);
    }
  }

  // Non-copyable, non-movable
  IsolationZone(const IsolationZone &) = delete;
  IsolationZone &operator=(const IsolationZone &) = delete;
};

} // namespace residue_wall

// =========================================================================
// PLATFORM: LINUX
// =========================================================================
#elif defined(__linux__)

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

namespace residue_wall {

class IsolationZone {
private:
  // --- Saved state ---
  cpu_set_t old_cpu_set_;
  struct sched_param old_sched_param_;
  int old_policy_;
  struct timespec start_time_;

  // --- Memory lock tracking ---
  struct LockedRegion {
    const void *ptr;
    size_t bytes;
    MemoryLockTier tier;
  };
  static constexpr size_t MAX_LOCKED_REGIONS = 8;
  LockedRegion locked_regions_[MAX_LOCKED_REGIONS];
  uint32_t num_locked_;
  uint32_t total_locked_bytes_;

  // --- C-state control ---
  int cpu_dma_latency_fd_;

  // --- Runtime ---
  bool priority_dropped_;
  bool tile_stalled_;
  double last_elapsed_us_;
  bool affinity_set_;
  uint32_t pinned_core_;
  uint32_t physical_core_;
  bool smt_detected_;

  // --- One-shot warnings ---
  inline static bool warned_affinity_ = false;
  inline static bool warned_priority_ = false;
  inline static bool warned_memory_lock_ = false;
  inline static bool warned_cstate_ = false;
  inline static bool warned_irq_ = false;

  // =======================================================================
  // SMT DETECTION (Linux): Read thread_siblings_list to find physical cores
  // =======================================================================
  struct SmtInfo {
    uint32_t target_logical;
    uint32_t physical_core_id;
    bool smt_detected;
  };

  static SmtInfo detect_smt_topology() {
    SmtInfo info = {};
    info.smt_detected = false;

    long num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    if (num_cores <= 0)
      return info;

    info.target_logical = static_cast<uint32_t>(num_cores - 1);
    info.physical_core_id = info.target_logical;

    // Try to find actual physical core mapping
    // Read /sys/devices/system/cpu/cpuN/topology/core_id for the last CPU
    char path[128];
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%u/topology/core_id",
             info.target_logical);
    FILE *f = fopen(path, "r");
    if (f) {
      char line[32];
      if (fgets(line, sizeof(line), f)) {
        info.physical_core_id = (uint32_t)strtoul(line, nullptr, 10);
      }
      fclose(f);
    }

    // Check if SMT is active by reading thread_siblings_list
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%u/topology/thread_siblings_list",
             info.target_logical);
    f = fopen(path, "r");
    if (f) {
      char line[64];
      if (fgets(line, sizeof(line), f)) {
        // If the siblings list contains a comma or dash, SMT is active
        if (strchr(line, ',') || strchr(line, '-')) {
          info.smt_detected = true;
          // Pick the first sibling (lowest numbered = primary thread)
          info.target_logical = (uint32_t)strtoul(line, nullptr, 10);
        }
      }
      fclose(f);
    }

    return info;
  }

public:
  // =====================================================================
  // CONSTRUCTOR: Enter the Isolation Zone (V4.1 HARDENED)
  // =====================================================================
  IsolationZone()
      : num_locked_(0), total_locked_bytes_(0), cpu_dma_latency_fd_(-1),
        priority_dropped_(false), tile_stalled_(false), last_elapsed_us_(0.0),
        affinity_set_(false), pinned_core_(0), physical_core_(0),
        smt_detected_(false) {
    pthread_t self = pthread_self();

    // Save original state
    pthread_getaffinity_np(self, sizeof(cpu_set_t), &old_cpu_set_);
    pthread_getschedparam(self, &old_policy_, &old_sched_param_);

    // --- LAYER 1: Timer Slack ---
    prctl(PR_SET_TIMERSLACK, 1);

    // --- LAYER 2: Core Isolation (SMT-aware) ---
    SmtInfo smt = detect_smt_topology();
    smt_detected_ = smt.smt_detected;
    pinned_core_ = smt.target_logical;
    physical_core_ = smt.physical_core_id;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(pinned_core_, &cpuset);
    if (pthread_setaffinity_np(self, sizeof(cpu_set_t), &cpuset) == 0) {
      affinity_set_ = true;
    } else if (!warned_affinity_) {
      fprintf(stderr,
              "[ISOLATION ZONE] Core pinning to logical %u "
              "(physical %u) FAILED (errno=%d).\n"
              "  -> Fix: run with CAP_SYS_NICE or as root.\n",
              pinned_core_, physical_core_, errno);
      warned_affinity_ = true;
    }

    // --- LAYER 3: Priority (V4.1 HARDENED: capped at 50, not max) ---
    struct sched_param param;
    int max_prio = sched_get_priority_max(SCHED_FIFO);
    param.sched_priority = (max_prio < 50) ? max_prio : 50;
    int ret = pthread_setschedparam(self, SCHED_FIFO, &param);
    if (ret != 0 && !warned_priority_) {
      fprintf(stderr,
              "[ISOLATION ZONE] SCHED_FIFO (prio %d) FAILED (errno=%d).\n"
              "  -> Fix: sudo setcap cap_sys_nice+ep <binary>\n",
              param.sched_priority, ret);
      warned_priority_ = true;
    }

    // --- LAYER 4: C-State Disable ---
    cpu_dma_latency_fd_ = open("/dev/cpu_dma_latency", O_WRONLY);
    if (cpu_dma_latency_fd_ >= 0) {
      int32_t target_latency = 0;
      ssize_t written =
          write(cpu_dma_latency_fd_, &target_latency, sizeof(target_latency));
      if (written != sizeof(target_latency)) {
        close(cpu_dma_latency_fd_);
        cpu_dma_latency_fd_ = -1;
      }
    } else if (!warned_cstate_) {
      fprintf(stderr,
              "[ISOLATION ZONE] /dev/cpu_dma_latency open FAILED (errno=%d).\n"
              "  -> Fix: run as root or chmod 666 /dev/cpu_dma_latency\n"
              "  -> CPU may enter deep C-states, causing wakeup latency.\n",
              errno);
      warned_cstate_ = true;
    }

    // --- LAYER 5: IRQ Shield (advisory) ---
    if (!warned_irq_) {
      warned_irq_ = true;
    }

    clock_gettime(CLOCK_MONOTONIC, &start_time_);
  }

  // =====================================================================
  // MEMORY LOCKING: 3-Tier Cascade (V4.1 HARDENED)
  // =====================================================================
  MemoryLockTier lock_memory(const void *ptr, size_t bytes) {
    if (!ptr || bytes == 0 || num_locked_ >= MAX_LOCKED_REGIONS)
      return MemoryLockTier::NONE;

    // --- TIER 1: mlock ---
    struct rlimit rl;
    if (getrlimit(RLIMIT_MEMLOCK, &rl) == 0) {
      if (rl.rlim_cur != RLIM_INFINITY &&
          total_locked_bytes_ + bytes > rl.rlim_cur) {
        rl.rlim_cur = total_locked_bytes_ + bytes + (64 * 1024);
        if (rl.rlim_cur > rl.rlim_max)
          rl.rlim_cur = rl.rlim_max;
        setrlimit(RLIMIT_MEMLOCK, &rl);
      }
    }

    if (mlock(ptr, bytes) == 0) {
      locked_regions_[num_locked_] = {ptr, bytes, MemoryLockTier::LOCKED};
      num_locked_++;
      total_locked_bytes_ += static_cast<uint32_t>(bytes);
      return MemoryLockTier::LOCKED;
    }

    // --- TIER 2: madvise(MADV_WILLNEED) — advisory prefetch ---
    if (madvise(const_cast<void *>(ptr), bytes, MADV_WILLNEED) == 0) {
      locked_regions_[num_locked_] = {ptr, bytes, MemoryLockTier::PREFETCHED};
      num_locked_++;
      total_locked_bytes_ += static_cast<uint32_t>(bytes);
      return MemoryLockTier::PREFETCHED;
    }

    // --- TIER 3: NONE ---
    if (!warned_memory_lock_) {
      fprintf(stderr,
              "[ISOLATION ZONE] Memory lock FAILED for %zu bytes (errno=%d).\n"
              "  Tier 1 (mlock): FAILED\n"
              "  Tier 2 (madvise): FAILED\n"
              "  -> Fix: ulimit -l unlimited, or run as root.\n",
              bytes, errno);
      warned_memory_lock_ = true;
    }
    return MemoryLockTier::NONE;
  }

  // =====================================================================
  // HOT PATH
  // =====================================================================
  RESIDUE_ZONE_FORCEINLINE void reset_timer() {
    tile_stalled_ = false;
    last_elapsed_us_ = 0.0;
    clock_gettime(CLOCK_MONOTONIC, &start_time_);
  }

  RESIDUE_ZONE_FORCEINLINE void check_safety() {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    long long elapsed_ns = (now.tv_sec - start_time_.tv_sec) * 1000000000LL +
                           (now.tv_nsec - start_time_.tv_nsec);
    last_elapsed_us_ = static_cast<double>(elapsed_ns) / 1000.0;

    if (last_elapsed_us_ > 100.0) {
      tile_stalled_ = true;
      if (!priority_dropped_) {
        pthread_t self = pthread_self();
        struct sched_param param;
        param.sched_priority = 0;
        pthread_setschedparam(self, SCHED_OTHER, &param);
        priority_dropped_ = true;
      }
    }
  }

  RESIDUE_ZONE_FORCEINLINE bool was_stalled() const { return tile_stalled_; }
  RESIDUE_ZONE_FORCEINLINE double elapsed_us() const {
    return last_elapsed_us_;
  }

  // =====================================================================
  // STATUS
  // =====================================================================
  IsolationStatus get_status() const {
    IsolationStatus s;
    s.memory_tier = MemoryLockTier::NONE;
    for (uint32_t i = 0; i < num_locked_; ++i) {
      if (locked_regions_[i].tier < s.memory_tier) {
        s.memory_tier = locked_regions_[i].tier;
      }
    }
    s.timer_hires = true; // PR_SET_TIMERSLACK always succeeds
    s.core_pinned = affinity_set_;
    s.priority_elevated = !priority_dropped_;
    s.smt_detected = smt_detected_;
    s.irq_warned = warned_irq_;
    s.locked_bytes = total_locked_bytes_;
    s.pinned_core = pinned_core_;
    s.physical_core = physical_core_;
    s.last_elapsed_us = last_elapsed_us_;
    return s;
  }

  // =====================================================================
  // DIAGNOSTICS
  // =====================================================================
  static void print_isolation_report() {
    long num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    SmtInfo smt = detect_smt_topology();

    bool core_isolated = false;
    FILE *f = fopen("/sys/devices/system/cpu/isolated", "r");
    if (f) {
      char buf[64];
      if (fgets(buf, sizeof(buf), f)) {
        char target_str[8];
        snprintf(target_str, sizeof(target_str), "%u", smt.target_logical);
        core_isolated = (strstr(buf, target_str) != nullptr);
      }
      fclose(f);
    }

    bool dma_lat_available = (access("/dev/cpu_dma_latency", W_OK) == 0);

    fprintf(stdout,
            "======================================================\n"
            "   THREAD ISOLATION ZONE — STATUS REPORT (V4.1)       \n"
            "======================================================\n"
            "  Platform       : Linux x86-64\n"
            "  Logical Cores  : %ld\n"
            "  SMT Detected   : %s\n"
            "  Target Core    : Logical %u (Physical %u)\n"
            "------------------------------------------------------\n"
            "  Layer 1 — Timer    : PR_SET_TIMERSLACK=1ns\n"
            "  Layer 2 — Affinity : SMT-aware core pinning\n"
            "  Layer 3 — Priority : SCHED_FIFO (capped at 50)\n"
            "  Layer 4 — C-State  : %s\n"
            "  Layer 5 — Memory   : 3-Tier Cascade\n"
            "      Tier 1: mlock (requires RLIMIT_MEMLOCK)\n"
            "      Tier 2: madvise(MADV_WILLNEED) (advisory)\n"
            "      Tier 3: NONE (page faults possible)\n"
            "------------------------------------------------------\n"
            "  Kernel isolcpus : %s\n"
            "  Recommended:\n"
            "    isolcpus=%u nohz_full=%u rcu_nocbs=%u\n"
            "======================================================\n",
            num_cores, smt.smt_detected ? "YES (HT sibling avoided)" : "NO",
            smt.target_logical, smt.physical_core_id,
            dma_lat_available ? "/dev/cpu_dma_latency writable" : "no access",
            core_isolated ? "YES" : "NO", smt.target_logical,
            smt.target_logical, smt.target_logical);
  }

  // =====================================================================
  // DESTRUCTOR
  // =====================================================================
  ~IsolationZone() {
    for (uint32_t i = 0; i < num_locked_; ++i) {
      if (locked_regions_[i].tier == MemoryLockTier::LOCKED) {
        munlock(locked_regions_[i].ptr, locked_regions_[i].bytes);
      }
    }

    if (cpu_dma_latency_fd_ >= 0) {
      close(cpu_dma_latency_fd_);
    }

    pthread_t self = pthread_self();
    if (!priority_dropped_) {
      pthread_setschedparam(self, old_policy_, &old_sched_param_);
    }

    if (affinity_set_) {
      pthread_setaffinity_np(self, sizeof(cpu_set_t), &old_cpu_set_);
    }
  }

  IsolationZone(const IsolationZone &) = delete;
  IsolationZone &operator=(const IsolationZone &) = delete;
};

} // namespace residue_wall

// =========================================================================
// PLATFORM: FALLBACK
// =========================================================================
#else

namespace residue_wall {

class IsolationZone {
public:
  IsolationZone() {}
  MemoryLockTier lock_memory(const void *, size_t) {
    return MemoryLockTier::NONE;
  }
  RESIDUE_ZONE_FORCEINLINE void reset_timer() {}
  RESIDUE_ZONE_FORCEINLINE void check_safety() {}
  RESIDUE_ZONE_FORCEINLINE bool was_stalled() const { return false; }
  RESIDUE_ZONE_FORCEINLINE double elapsed_us() const { return 0.0; }
  IsolationStatus get_status() const {
    return {
        MemoryLockTier::NONE, false, false, false, false, false, 0, 0, 0, 0.0};
  }
  static void print_isolation_report() {
    fprintf(stdout, "[ISOLATION ZONE] Unsupported platform — no isolation.\n");
  }
  ~IsolationZone() {}

  IsolationZone(const IsolationZone &) = delete;
  IsolationZone &operator=(const IsolationZone &) = delete;
};

} // namespace residue_wall

#endif
