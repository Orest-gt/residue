#pragma once

// =========================================================================
// RESIDUE WALL — Component 2: Aligned Gateway
// AlignedAllocator<T, Alignment> + AlignedBuffer RAII wrapper.
// Zero-copy when input is already aligned. Non-temporal copy otherwise.
// =========================================================================

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <new>
#include <utility>

#ifdef _MSC_VER
#include <intrin.h>
#define RESIDUE_WALL_FORCEINLINE __forceinline
#else
#define RESIDUE_WALL_FORCEINLINE __attribute__((always_inline)) inline
#endif

namespace residue_wall {

// =========================================================================
// Aligned Allocator — STL-compatible, usable with std::vector if ever needed
// =========================================================================
template <typename T, size_t Alignment = 64> class AlignedAllocator {
  static_assert(Alignment >= alignof(T),
                "Alignment must be >= natural alignment of T");
  static_assert((Alignment & (Alignment - 1)) == 0,
                "Alignment must be a power of 2");

public:
  using value_type = T;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using propagate_on_container_move_assignment = std::true_type;
  using is_always_equal = std::true_type;

  constexpr AlignedAllocator() noexcept = default;
  template <typename U>
  constexpr AlignedAllocator(const AlignedAllocator<U, Alignment> &) noexcept {}

  [[nodiscard]] T *allocate(size_t n) {
    if (n == 0)
      return nullptr;
    void *ptr = aligned_alloc_impl(Alignment, n * sizeof(T));
    if (!ptr)
      throw std::bad_alloc();
    return static_cast<T *>(ptr);
  }

  void deallocate(T *p, size_t /*n*/) noexcept { aligned_free_impl(p); }

  template <typename U> struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };

  bool operator==(const AlignedAllocator &) const noexcept { return true; }
  bool operator!=(const AlignedAllocator &) const noexcept { return false; }

private:
  static void *aligned_alloc_impl(size_t alignment, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    // aligned_alloc requires size to be a multiple of alignment
    size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);
    return aligned_alloc(alignment, aligned_size);
#endif
  }

  static void aligned_free_impl(void *ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
  }
};

// =========================================================================
// AlignedBuffer — RAII wrapper, zero-copy or staging buffer
// =========================================================================
template <typename T, size_t Alignment = 64> class AlignedBuffer {
public:
  AlignedBuffer() noexcept : data_(nullptr), size_(0), owns_(false) {}

  ~AlignedBuffer() { release(); }

  // Move only — no copying
  AlignedBuffer(const AlignedBuffer &) = delete;
  AlignedBuffer &operator=(const AlignedBuffer &) = delete;

  AlignedBuffer(AlignedBuffer &&other) noexcept
      : data_(other.data_), size_(other.size_), owns_(other.owns_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.owns_ = false;
  }

  AlignedBuffer &operator=(AlignedBuffer &&other) noexcept {
    if (this != &other) {
      release();
      data_ = other.data_;
      size_ = other.size_;
      owns_ = other.owns_;
      other.data_ = nullptr;
      other.size_ = 0;
      other.owns_ = false;
    }
    return *this;
  }

  // Allocate fresh aligned buffer
  void allocate(size_t count) {
    release();
    AlignedAllocator<T, Alignment> alloc;
    data_ = alloc.allocate(count);
    size_ = count;
    owns_ = true;
  }

  // Ensure capacity is at least `count` elements. No-op if already sufficient.
  void ensure_capacity(size_t count) {
    if (owns_ && size_ >= count)
      return;
    allocate(count);
  }

  // Release owned memory
  void release() {
    if (owns_ && data_) {
      AlignedAllocator<T, Alignment> alloc;
      alloc.deallocate(data_, size_);
    }
    data_ = nullptr;
    size_ = 0;
    owns_ = false;
  }

  // Adopt an external pointer (zero-copy path)
  void adopt(T *ptr, size_t count) {
    release();
    data_ = ptr;
    size_ = count;
    owns_ = false; // we do NOT own this pointer
  }

  T *data() noexcept { return data_; }
  const T *data() const noexcept { return data_; }
  size_t size() const noexcept { return size_; }
  bool owns() const noexcept { return owns_; }

private:
  T *data_;
  size_t size_;
  bool owns_;
};

// =========================================================================
// Alignment Check Utilities
// =========================================================================

RESIDUE_WALL_FORCEINLINE bool is_aligned(const void *ptr,
                                         size_t alignment) noexcept {
  return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

// =========================================================================
// Non-temporal copy for misaligned → aligned staging
// Uses streaming stores to avoid polluting L1d with destination data.
// Source is read with unaligned loads (it's the misaligned input).
// =========================================================================
RESIDUE_WALL_FORCEINLINE void
nontemporal_copy_float(float *__restrict dst, const float *__restrict src,
                       size_t count) {
  size_t i = 0;

  // AVX2 non-temporal stores (32 bytes = 8 floats at a time)
  for (; i + 8 <= count; i += 8) {
    __m256 v = _mm256_loadu_ps(src + i);
    _mm256_stream_ps(dst + i, v);
  }

  // SSE non-temporal for remainder (16 bytes = 4 floats)
  for (; i + 4 <= count; i += 4) {
    __m128 v = _mm_loadu_ps(src + i);
    _mm_stream_ps(dst + i, v);
  }

  // Scalar tail
  for (; i < count; ++i) {
    dst[i] = src[i];
  }

  // Ensure all streaming stores are visible
  _mm_sfence();
}

} // namespace residue_wall
