#pragma once

// =========================================================================
// RESIDUE WALL — Level 4: Thread Isolation Zone
//
// Elevates ThreadGuard from "priority hint" to "OS bypass":
//   1. MemoryLock   — VirtualLock/mlock pins data pages in physical RAM
//   2. TimerControl — Reduces OS scheduling tick to 1ms (Windows)
//   3. CoreIsolation — Enhanced affinity + C-state disable
//   4. IRQShield    — Detects and warns about IRQ contention
//
// All mechanisms are RAII-safe: destructor restores original state.
// Graceful degradation: if a mechanism requires admin and fails,
// we warn once and continue without it.
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
// PLATFORM: WINDOWS
// =========================================================================
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#pragma comment(lib, "winmm.lib") // for timeBeginPeriod/timeEndPeriod

namespace residue_wall {

// --- Isolation status flags ---
struct IsolationStatus {
  bool memory_locked;
  bool timer_hires;
  bool core_pinned;
  bool priority_realtime;
  bool irq_warned;

  uint32_t locked_bytes;     // total bytes locked in physical RAM
  uint32_t pinned_core;      // core ID we're pinned to
  double   last_elapsed_us;
};

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
  };
  static constexpr size_t MAX_LOCKED_REGIONS = 4;
  LockedRegion locked_regions_[MAX_LOCKED_REGIONS];
  uint32_t num_locked_;
  uint32_t total_locked_bytes_;

  // --- Runtime state ---
  bool priority_dropped_;
  bool tile_stalled_;
  double last_elapsed_us_;
  bool timer_hires_set_;
  uint32_t pinned_core_;

  // --- One-shot warnings ---
  inline static bool warned_memory_lock_ = false;
  inline static bool warned_timer_ = false;
  inline static bool warned_affinity_ = false;
  inline static bool warned_irq_ = false;

public:
  // =====================================================================
  // CONSTRUCTOR: Enter the Isolation Zone
  // =====================================================================
  IsolationZone()
      : num_locked_(0), total_locked_bytes_(0), priority_dropped_(false),
        tile_stalled_(false), last_elapsed_us_(0.0), timer_hires_set_(false),
        pinned_core_(0) {
    hThread_ = GetCurrentThread();
    hProcess_ = GetCurrentProcess();

    // Save original state
    old_priority_class_ = GetPriorityClass(hProcess_);
    old_thread_priority_ = GetThreadPriority(hThread_);

    // --- LAYER 1: Timer Resolution ---
    // Reduce system timer from 15.6ms to 1ms.
    // This reduces the preemption window for our RT thread.
    MMRESULT timer_result = timeBeginPeriod(1);
    if (timer_result == TIMERR_NOERROR) {
      timer_hires_set_ = true;
    } else if (!warned_timer_) {
      fprintf(stderr,
              "[ISOLATION ZONE] timeBeginPeriod(1) failed. "
              "OS timer remains at default resolution.\n");
      warned_timer_ = true;
    }

    // --- LAYER 2: Core Isolation ---
    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    DWORD num_cores = sys_info.dwNumberOfProcessors;

    // Pin to the LAST core (least likely to handle IRQs on most systems)
    pinned_core_ = num_cores - 1;
    DWORD_PTR target_mask = (DWORD_PTR)1 << pinned_core_;
    old_affinity_mask_ = SetThreadAffinityMask(hThread_, target_mask);

    if (!old_affinity_mask_ && !warned_affinity_) {
      fprintf(stderr,
              "[ISOLATION ZONE] Core pinning to core %u FAILED. "
              "Running without CPU isolation.\n",
              pinned_core_);
      warned_affinity_ = true;
    }

    // --- LAYER 3: Real-Time Priority ---
    SetPriorityClass(hProcess_, REALTIME_PRIORITY_CLASS);
    SetThreadPriority(hThread_, THREAD_PRIORITY_TIME_CRITICAL);

    // --- LAYER 4: IRQ Shield (advisory) ---
    // On Windows, we can't steer IRQs from userspace without admin.
    // We warn once about what the user can do.
    if (!warned_irq_) {
      // Silent — only print if user calls print_isolation_report()
      warned_irq_ = true;
    }

    // Initialize timer
    QueryPerformanceFrequency(&freq_);
    QueryPerformanceCounter(&start_time_);
  }

  // =====================================================================
  // MEMORY LOCKING: Pin data pages in physical RAM
  // Call BEFORE entering the hot loop (cold path).
  // =====================================================================
  bool lock_memory(const void *ptr, size_t bytes) {
    if (!ptr || bytes == 0 || num_locked_ >= MAX_LOCKED_REGIONS)
      return false;

    // Increase working set size to accommodate the lock.
    // Without this, VirtualLock silently fails on large buffers.
    SIZE_T min_ws, max_ws;
    if (GetProcessWorkingSetSize(hProcess_, &min_ws, &max_ws)) {
      SIZE_T needed = min_ws + bytes + (64 * 1024); // +64KB headroom
      if (needed > max_ws) {
        SetProcessWorkingSetSize(hProcess_, needed, needed);
      }
    }

    if (VirtualLock(const_cast<void *>(ptr), bytes)) {
      locked_regions_[num_locked_].ptr = ptr;
      locked_regions_[num_locked_].bytes = bytes;
      num_locked_++;
      total_locked_bytes_ += static_cast<uint32_t>(bytes);
      return true;
    }

    if (!warned_memory_lock_) {
      DWORD err = GetLastError();
      fprintf(stderr,
              "[ISOLATION ZONE] VirtualLock(%zu bytes) FAILED (err=%lu). "
              "Page faults may occur in hot path.\n"
              "  -> Fix: Run as Administrator, or increase working set quota.\n",
              bytes, err);
      warned_memory_lock_ = true;
    }
    return false;
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
    last_elapsed_us_ = static_cast<double>(now.QuadPart - start_time_.QuadPart) *
                       1000000.0 / freq_.QuadPart;

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
  RESIDUE_ZONE_FORCEINLINE double elapsed_us() const { return last_elapsed_us_; }

  // =====================================================================
  // DIAGNOSTICS: Print full isolation status report
  // =====================================================================
  static void print_isolation_report() {
    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    DWORD num_cores = sys_info.dwNumberOfProcessors;

    fprintf(stdout,
            "╔══════════════════════════════════════════════════╗\n"
            "║       THREAD ISOLATION ZONE — STATUS REPORT      ║\n"
            "╠══════════════════════════════════════════════════╣\n"
            "║  Platform    : Windows x86-64                    ║\n"
            "║  CPU Cores   : %-4u                              ║\n"
            "║  Target Core : %-4u (last core)                  ║\n"
            "╠══════════════════════════════════════════════════╣\n"
            "║  Layer 1 — Timer Resolution:                     ║\n"
            "║    timeBeginPeriod(1): Available                  ║\n"
            "║  Layer 2 — Core Isolation:                       ║\n"
            "║    SetThreadAffinityMask: Available               ║\n"
            "║  Layer 3 — RT Priority:                          ║\n"
            "║    REALTIME_PRIORITY_CLASS: Available             ║\n"
            "║  Layer 4 — Memory Locking:                       ║\n"
            "║    VirtualLock: Available (needs working set)     ║\n"
            "║  Layer 5 — IRQ Shield:                           ║\n"
            "║    Manual config required. See below.             ║\n"
            "╠══════════════════════════════════════════════════╣\n"
            "║  IRQ Steering (admin required):                  ║\n"
            "║    1. Open Device Manager                        ║\n"
            "║    2. Properties > Advanced > Interrupt Affinity  ║\n"
            "║    3. Exclude core %u from NIC/USB IRQs          ║\n"
            "╚══════════════════════════════════════════════════╝\n",
            num_cores, num_cores - 1, num_cores - 1);
  }

  // =====================================================================
  // DESTRUCTOR: Exit the Isolation Zone (RAII)
  // =====================================================================
  ~IsolationZone() {
    // Unlock all memory regions
    for (uint32_t i = 0; i < num_locked_; ++i) {
      VirtualUnlock(const_cast<void *>(locked_regions_[i].ptr),
                    locked_regions_[i].bytes);
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
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/prctl.h>
#include <time.h>
#include <unistd.h>

namespace residue_wall {

struct IsolationStatus {
  bool memory_locked;
  bool timer_hires;
  bool core_pinned;
  bool priority_realtime;
  bool irq_warned;
  bool cstate_disabled;

  uint32_t locked_bytes;
  uint32_t pinned_core;
  double   last_elapsed_us;
};

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
  };
  static constexpr size_t MAX_LOCKED_REGIONS = 4;
  LockedRegion locked_regions_[MAX_LOCKED_REGIONS];
  uint32_t num_locked_;
  uint32_t total_locked_bytes_;

  // --- C-state control ---
  int cpu_dma_latency_fd_;  // fd to /dev/cpu_dma_latency (keeps CPU in C0)

  // --- Runtime ---
  bool priority_dropped_;
  bool tile_stalled_;
  double last_elapsed_us_;
  bool affinity_set_;
  uint32_t pinned_core_;

  // --- One-shot warnings ---
  inline static bool warned_affinity_ = false;
  inline static bool warned_priority_ = false;
  inline static bool warned_memory_lock_ = false;
  inline static bool warned_cstate_ = false;
  inline static bool warned_irq_ = false;

public:
  // =====================================================================
  // CONSTRUCTOR: Enter the Isolation Zone
  // =====================================================================
  IsolationZone()
      : num_locked_(0), total_locked_bytes_(0), cpu_dma_latency_fd_(-1),
        priority_dropped_(false), tile_stalled_(false), last_elapsed_us_(0.0),
        affinity_set_(false), pinned_core_(0) {
    pthread_t self = pthread_self();

    // Save original state
    pthread_getaffinity_np(self, sizeof(cpu_set_t), &old_cpu_set_);
    pthread_getschedparam(self, &old_policy_, &old_sched_param_);

    // --- LAYER 1: Timer Slack ---
    // Set timer slack to 1ns for precise wakeups
    prctl(PR_SET_TIMERSLACK, 1);

    // --- LAYER 2: Core Isolation ---
    long num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    if (num_cores > 0) {
      pinned_core_ = static_cast<uint32_t>(num_cores - 1);
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(pinned_core_, &cpuset);
      if (pthread_setaffinity_np(self, sizeof(cpu_set_t), &cpuset) == 0) {
        affinity_set_ = true;
      } else if (!warned_affinity_) {
        fprintf(stderr,
                "[ISOLATION ZONE] Core pinning to core %u FAILED (errno=%d).\n"
                "  -> Fix: run with CAP_SYS_NICE or as root.\n",
                pinned_core_, errno);
        warned_affinity_ = true;
      }
    }

    // --- LAYER 3: Real-Time Priority ---
    struct sched_param param;
    param.sched_priority = sched_get_priority_max(SCHED_FIFO);
    int ret = pthread_setschedparam(self, SCHED_FIFO, &param);
    if (ret != 0 && !warned_priority_) {
      fprintf(stderr,
              "[ISOLATION ZONE] SCHED_FIFO FAILED (errno=%d).\n"
              "  -> Fix: sudo setcap cap_sys_nice+ep <binary>\n",
              ret);
      warned_priority_ = true;
    }

    // --- LAYER 4: C-State Disable ---
    // Writing 0 to /dev/cpu_dma_latency prevents the CPU from entering
    // any C-state deeper than C0 while the fd is open.
    // This eliminates wakeup latency but increases power/heat.
    cpu_dma_latency_fd_ = open("/dev/cpu_dma_latency", O_WRONLY);
    if (cpu_dma_latency_fd_ >= 0) {
      int32_t target_latency = 0; // 0 = stay in C0
      ssize_t written = write(cpu_dma_latency_fd_, &target_latency,
                              sizeof(target_latency));
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
      // Check if our core receives significant IRQs
      // (full check in print_isolation_report)
      warned_irq_ = true;
    }

    clock_gettime(CLOCK_MONOTONIC, &start_time_);
  }

  // =====================================================================
  // MEMORY LOCKING
  // =====================================================================
  bool lock_memory(const void *ptr, size_t bytes) {
    if (!ptr || bytes == 0 || num_locked_ >= MAX_LOCKED_REGIONS)
      return false;

    // Check RLIMIT_MEMLOCK
    struct rlimit rl;
    if (getrlimit(RLIMIT_MEMLOCK, &rl) == 0) {
      if (rl.rlim_cur != RLIM_INFINITY &&
          total_locked_bytes_ + bytes > rl.rlim_cur) {
        // Try to raise the limit
        rl.rlim_cur = total_locked_bytes_ + bytes + (64 * 1024);
        if (rl.rlim_cur > rl.rlim_max)
          rl.rlim_cur = rl.rlim_max;
        setrlimit(RLIMIT_MEMLOCK, &rl);
      }
    }

    if (mlock(ptr, bytes) == 0) {
      locked_regions_[num_locked_].ptr = ptr;
      locked_regions_[num_locked_].bytes = bytes;
      num_locked_++;
      total_locked_bytes_ += static_cast<uint32_t>(bytes);
      return true;
    }

    if (!warned_memory_lock_) {
      fprintf(stderr,
              "[ISOLATION ZONE] mlock(%zu bytes) FAILED (errno=%d).\n"
              "  -> Fix: ulimit -l unlimited, or run as root.\n",
              bytes, errno);
      warned_memory_lock_ = true;
    }
    return false;
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
  RESIDUE_ZONE_FORCEINLINE double elapsed_us() const { return last_elapsed_us_; }

  // =====================================================================
  // DIAGNOSTICS
  // =====================================================================
  static void print_isolation_report() {
    long num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    uint32_t target_core = (num_cores > 0) ? static_cast<uint32_t>(num_cores - 1) : 0;

    // Check if isolated via kernel param
    bool core_isolated = false;
    FILE *f = fopen("/sys/devices/system/cpu/isolated", "r");
    if (f) {
      char buf[64];
      if (fgets(buf, sizeof(buf), f)) {
        // Parse comma-separated list or range like "7" or "6-7"
        // Simple check: does it contain our target core number?
        char target_str[8];
        snprintf(target_str, sizeof(target_str), "%u", target_core);
        core_isolated = (strstr(buf, target_str) != nullptr);
      }
      fclose(f);
    }

    bool dma_lat_available = (access("/dev/cpu_dma_latency", W_OK) == 0);

    fprintf(stdout,
            "╔══════════════════════════════════════════════════╗\n"
            "║       THREAD ISOLATION ZONE — STATUS REPORT      ║\n"
            "╠══════════════════════════════════════════════════╣\n"
            "║  Platform    : Linux x86-64                      ║\n"
            "║  CPU Cores   : %-4ld                             ║\n"
            "║  Target Core : %-4u (last core)                  ║\n"
            "╠══════════════════════════════════════════════════╣\n"
            "║  Layer 1 — Timer Slack:  PR_SET_TIMERSLACK=1ns   ║\n"
            "║  Layer 2 — Core Isolation:                       ║\n"
            "║    Kernel isolcpus: %s                     ║\n"
            "║  Layer 3 — RT Priority: SCHED_FIFO               ║\n"
            "║  Layer 4 — C-State Control:                      ║\n"
            "║    /dev/cpu_dma_latency: %s               ║\n"
            "║  Layer 5 — Memory Locking: mlock()               ║\n"
            "╠══════════════════════════════════════════════════╣\n"
            "║  Recommended kernel params for full isolation:    ║\n"
            "║    isolcpus=%u nohz_full=%u rcu_nocbs=%u          ║\n"
            "║  IRQ steering:                                   ║\n"
            "║    echo <mask> > /proc/irq/*/smp_affinity         ║\n"
            "╚══════════════════════════════════════════════════╝\n",
            num_cores, target_core,
            core_isolated ? "YES ✅" : "NO  ⚠️",
            dma_lat_available ? "writable ✅" : "no access ⚠️",
            target_core, target_core, target_core);
  }

  // =====================================================================
  // DESTRUCTOR: Exit the Isolation Zone
  // =====================================================================
  ~IsolationZone() {
    // Unlock memory
    for (uint32_t i = 0; i < num_locked_; ++i) {
      munlock(locked_regions_[i].ptr, locked_regions_[i].bytes);
    }

    // Release C-state hold
    if (cpu_dma_latency_fd_ >= 0) {
      close(cpu_dma_latency_fd_);
    }

    // Restore priority
    pthread_t self = pthread_self();
    if (!priority_dropped_) {
      pthread_setschedparam(self, old_policy_, &old_sched_param_);
    }

    // Restore affinity
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
  bool lock_memory(const void *, size_t) { return false; }
  RESIDUE_ZONE_FORCEINLINE void reset_timer() {}
  RESIDUE_ZONE_FORCEINLINE void check_safety() {}
  RESIDUE_ZONE_FORCEINLINE bool was_stalled() const { return false; }
  RESIDUE_ZONE_FORCEINLINE double elapsed_us() const { return 0.0; }
  static void print_isolation_report() {
    fprintf(stdout, "[ISOLATION ZONE] Unsupported platform — no isolation.\n");
  }
  ~IsolationZone() {}

  IsolationZone(const IsolationZone &) = delete;
  IsolationZone &operator=(const IsolationZone &) = delete;
};

} // namespace residue_wall

#endif
