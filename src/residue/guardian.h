#pragma once

// =========================================================================
// RESIDUE GUARDIAN — Thread Safety & Isolation
//
// V3.1: Basic ThreadGuard (core pin + RT priority + watchdog)
// V4.0: Full IsolationZone (+ memory lock + timer control + C-state + IRQ)
//
// ThreadGuard is now an alias for IsolationZone.
// All existing code using ThreadGuard works unchanged.
// =========================================================================

#include "../residue_wall/isolation_zone.h"

// Cross-platform forceinline (kept for backward compat with core_v3.cpp macros)
#ifdef _MSC_VER
#define RESIDUE_FORCEINLINE __forceinline
#define RESIDUE_POPCNT(x) __popcnt(x)
#else
#define RESIDUE_FORCEINLINE __attribute__((always_inline)) inline
#define RESIDUE_POPCNT(x) __builtin_popcount(x)
#endif

// ThreadGuard is now IsolationZone
using ThreadGuard = residue_wall::IsolationZone;
