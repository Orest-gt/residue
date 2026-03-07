#pragma once

// =========================================================================
// RESIDUE WALL — Component 1: Cache Topology Detection
// Runtime detection of L1/L2/L3 geometry via OS APIs.
// Zero allocation. Results cached as static const.
// =========================================================================

#include <cstdint>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#endif

namespace residue_wall {

struct CacheTopology {
  uint32_t l1d_size;          // L1 data cache size in bytes
  uint32_t l2_size;           // L2 unified cache size in bytes
  uint32_t l3_size;           // L3 unified cache size in bytes
  uint32_t cache_line_size;   // Cache line size in bytes (typically 64)
  uint32_t l1d_associativity; // L1d set associativity
  uint32_t num_physical_cores;

  // Derived constants (computed once)
  uint32_t l1d_lines;     // l1d_size / cache_line_size
  uint32_t lines_per_4kb; // 4096 / cache_line_size
};

// Conservative fallback — works on any x86-64
inline constexpr CacheTopology kDefaultTopology = {
    32768,  // 32KB L1d
    262144, // 256KB L2
    8388608, // 8MB L3
    64,     // 64-byte cache lines
    8,      // 8-way associative
    4,      // 4 physical cores
    512,    // 32KB / 64
    64      // 4096 / 64
};

#ifdef _WIN32

inline CacheTopology detect_cache_topology() {
  CacheTopology topo = kDefaultTopology;

  DWORD buffer_size = 0;
  GetLogicalProcessorInformationEx(RelationCache, nullptr, &buffer_size);
  if (buffer_size == 0)
    return topo;

  // Stack buffer for small systems; heap only if enormous
  alignas(8) uint8_t stack_buf[4096];
  uint8_t *buf = stack_buf;
  bool heap_alloc = false;

  if (buffer_size > sizeof(stack_buf)) {
    buf = static_cast<uint8_t *>(malloc(buffer_size));
    if (!buf)
      return topo;
    heap_alloc = true;
  }

  auto *info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf);
  if (!GetLogicalProcessorInformationEx(RelationCache, info, &buffer_size)) {
    if (heap_alloc) free(buf);
    return topo;
  }

  auto *ptr = buf;
  auto *end = buf + buffer_size;

  while (ptr < end) {
    auto *entry = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(ptr);
    if (entry->Relationship == RelationCache) {
      const auto &cache = entry->Cache;
      topo.cache_line_size = cache.LineSize;

      switch (cache.Level) {
      case 1:
        if (cache.Type == CacheData || cache.Type == CacheUnified) {
          topo.l1d_size = cache.CacheSize;
          topo.l1d_associativity = cache.Associativity;
        }
        break;
      case 2:
        topo.l2_size = cache.CacheSize;
        break;
      case 3:
        topo.l3_size = cache.CacheSize;
        break;
      }
    }
    ptr += entry->Size;
  }

  if (heap_alloc) free(buf);

  // Detect core count
  SYSTEM_INFO sysInfo;
  GetSystemInfo(&sysInfo);
  topo.num_physical_cores = sysInfo.dwNumberOfProcessors;

  // Compute derived
  topo.l1d_lines = topo.l1d_size / topo.cache_line_size;
  topo.lines_per_4kb = 4096 / topo.cache_line_size;

  return topo;
}

#elif defined(__linux__)

namespace detail {

inline uint32_t read_sysfs_uint(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f)
    return 0;
  char line[64];
  if (!fgets(line, sizeof(line), f)) {
    fclose(f);
    return 0;
  }
  fclose(f);

  // Handle suffixed values like "32K", "256K", "8192K"
  uint32_t val = static_cast<uint32_t>(strtoul(line, nullptr, 10));
  size_t len = strlen(line);
  if (len > 1) {
    char suffix = line[len - 2]; // before newline
    if (suffix == 'K' || suffix == 'k')
      val *= 1024;
    else if (suffix == 'M' || suffix == 'm')
      val *= 1024 * 1024;
  }
  return val;
}

inline uint32_t read_sysfs_type(const char *path, char *out, size_t out_size) {
  FILE *f = fopen(path, "r");
  if (!f)
    return 0;
  if (!fgets(out, static_cast<int>(out_size), f)) {
    fclose(f);
    return 0;
  }
  fclose(f);
  // Strip trailing newline
  size_t len = strlen(out);
  if (len > 0 && out[len - 1] == '\n')
    out[len - 1] = '\0';
  return 1;
}

} // namespace detail

inline CacheTopology detect_cache_topology() {
  CacheTopology topo = kDefaultTopology;

  // Iterate over cache indices 0..3
  for (int idx = 0; idx < 4; ++idx) {
    char path[128];

    // Read level
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu0/cache/index%d/level", idx);
    uint32_t level = detail::read_sysfs_uint(path);
    if (level == 0)
      continue;

    // Read type
    char type_str[32] = {0};
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu0/cache/index%d/type", idx);
    detail::read_sysfs_type(path, type_str, sizeof(type_str));

    // Read size
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu0/cache/index%d/size", idx);
    uint32_t size = detail::read_sysfs_uint(path);

    // Read line size
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu0/cache/index%d/coherency_line_size",
             idx);
    uint32_t line_size = detail::read_sysfs_uint(path);
    if (line_size > 0)
      topo.cache_line_size = line_size;

    // Read associativity
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu0/cache/index%d/ways_of_associativity",
             idx);
    uint32_t assoc = detail::read_sysfs_uint(path);

    switch (level) {
    case 1:
      if (strcmp(type_str, "Data") == 0 || strcmp(type_str, "Unified") == 0) {
        topo.l1d_size = size;
        if (assoc > 0)
          topo.l1d_associativity = assoc;
      }
      break;
    case 2:
      topo.l2_size = size;
      break;
    case 3:
      topo.l3_size = size;
      break;
    }
  }

  // Core count
  FILE *f = fopen("/proc/cpuinfo", "r");
  if (f) {
    uint32_t count = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
      if (strncmp(line, "processor", 9) == 0)
        count++;
    }
    fclose(f);
    if (count > 0)
      topo.num_physical_cores = count;
  }

  // Derived
  topo.l1d_lines = topo.l1d_size / topo.cache_line_size;
  topo.lines_per_4kb = 4096 / topo.cache_line_size;

  return topo;
}

#else

inline CacheTopology detect_cache_topology() { return kDefaultTopology; }

#endif

// Singleton — detected once, read many
inline const CacheTopology &get_cache_topology() {
  static const CacheTopology topo = detect_cache_topology();
  return topo;
}

} // namespace residue_wall
