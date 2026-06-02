/*
 * platform.h — Cross-platform OS abstraction for Flash-MoE Universal
 */
#pragma once
#define _GNU_SOURCE
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// ============================================================================
// Architecture Detection
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64)
  #define ARCH_X86_64  1
  #define ARCH_NAME    "x86_64"
#elif defined(__aarch64__) || defined(_M_ARM64)
  #define ARCH_ARM64   1
  #define ARCH_NAME    "arm64"
#elif defined(__arm__) || defined(_M_ARM)
  #define ARCH_ARM32   1
  #define ARCH_NAME    "armv7"
#elif defined(__riscv)
  #define ARCH_RISCV   1
  #define ARCH_NAME    "riscv"
#else
  #define ARCH_GENERIC 1
  #define ARCH_NAME    "generic"
#endif

// ============================================================================
// OS Detection
// ============================================================================

#if defined(__APPLE__)
  #define OS_MACOS 1
  #define OS_NAME  "macOS"
  #include <TargetConditionals.h>
  #if TARGET_CPU_ARM64
    #define PLATFORM_APPLE_SILICON 1
  #endif
#elif defined(__linux__)
  #define OS_LINUX 1
  #define OS_NAME  "Linux"
#elif defined(_WIN32) || defined(_WIN64)
  #define OS_WINDOWS 1
  #define OS_NAME    "Windows"
  #define WIN32_LEAN_AND_MEAN
  #define NOMINMAX
  #include <windows.h>
#else
  #define OS_UNKNOWN 1
  #define OS_NAME    "Unknown"
#endif

// ============================================================================
// SIMD / Vector Extensions Detection
// ============================================================================

#ifdef ARCH_X86_64
  #if defined(__AVX512F__)
    #define HAS_AVX512 1
  #endif
  #if defined(__AVX2__)
    #define HAS_AVX2 1
    #include <immintrin.h>
  #elif defined(__AVX__)
    #define HAS_AVX 1
    #include <immintrin.h>
  #elif defined(__SSE4_1__)
    #define HAS_SSE41 1
    #include <smmintrin.h>
  #endif
#endif

#ifdef ARCH_ARM64
  #define HAS_NEON 1
  #include <arm_neon.h>
#endif

// ============================================================================
// File I/O Abstraction
// ============================================================================

#ifdef OS_WINDOWS
  #include <io.h>
  #include <fcntl.h>
  typedef HANDLE file_handle_t;
  #define FILE_INVALID  INVALID_HANDLE_VALUE

  static inline ssize_t platform_pread(file_handle_t fd, void *buf, size_t count, off_t offset) {
      OVERLAPPED ov = {0};
      ov.Offset     = (DWORD)(offset & 0xFFFFFFFF);
      ov.OffsetHigh = (DWORD)(offset >> 32);
      DWORD nread = 0;
      if (!ReadFile(fd, buf, (DWORD)count, &nread, &ov)) return -1;
      return (ssize_t)nread;
  }

  static inline file_handle_t platform_open(const char *path) {
      return CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                         OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, NULL);
  }

  static inline void platform_close(file_handle_t fd) {
      CloseHandle(fd);
  }

  static inline int64_t platform_file_size(file_handle_t fd) {
      LARGE_INTEGER sz;
      if (!GetFileSizeEx(fd, &sz)) return -1;
      return sz.QuadPart;
  }

#else  /* POSIX */
  #include <unistd.h>
  #include <fcntl.h>
  #include <sys/stat.h>
  typedef int file_handle_t;
  #define FILE_INVALID (-1)

  static inline ssize_t platform_pread(file_handle_t fd, void *buf, size_t count, off_t offset) {
      return pread(fd, buf, count, offset);
  }

  static inline file_handle_t platform_open(const char *path) {
      return open(path, O_RDONLY);
  }

  static inline void platform_close(file_handle_t fd) {
      close(fd);
  }

  static inline int64_t platform_file_size(file_handle_t fd) {
      struct stat st;
      if (fstat(fd, &st) != 0) return -1;
      return st.st_size;
  }
#endif

// ============================================================================
// Memory Mapping
// ============================================================================

#ifdef OS_WINDOWS
  typedef struct {
      HANDLE hFile;
      HANDLE hMap;
      void  *ptr;
      size_t size;
  } mmap_t;

  static inline void* platform_mmap(const char *path, size_t *out_size) {
      mmap_t *m = (mmap_t*)malloc(sizeof(mmap_t));
      m->hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
      if (m->hFile == INVALID_HANDLE_VALUE) { free(m); return NULL; }
      LARGE_INTEGER sz;
      GetFileSizeEx(m->hFile, &sz);
      m->size = (size_t)sz.QuadPart;
      if (out_size) *out_size = m->size;
      m->hMap = CreateFileMappingA(m->hFile, NULL, PAGE_READONLY, 0, 0, NULL);
      if (!m->hMap) { CloseHandle(m->hFile); free(m); return NULL; }
      m->ptr = MapViewOfFile(m->hMap, FILE_MAP_READ, 0, 0, 0);
      return m->ptr ? m : NULL;
  }

  static inline void platform_munmap(void *handle) {
      mmap_t *m = (mmap_t*)handle;
      UnmapViewOfFile(m->ptr);
      CloseHandle(m->hMap);
      CloseHandle(m->hFile);
      free(m);
  }
#else
  #include <sys/mman.h>

  typedef struct {
      void  *ptr;
      size_t size;
  } mmap_t;

  static inline void* platform_mmap(const char *path, size_t *out_size) {
      int fd = open(path, O_RDONLY);
      if (fd < 0) return NULL;
      struct stat st;
      fstat(fd, &st);
      if (out_size) *out_size = (size_t)st.st_size;
      void *ptr = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
      close(fd);
      if (ptr == MAP_FAILED) return NULL;
      mmap_t *m = (mmap_t*)malloc(sizeof(mmap_t));
      m->ptr  = ptr;
      m->size = st.st_size;
      return m;
  }

  static inline void platform_munmap(void *handle) {
      mmap_t *m = (mmap_t*)handle;
      munmap(m->ptr, m->size);
      free(m);
  }
#endif

// ============================================================================
// Threading
// ============================================================================

#ifdef OS_WINDOWS
  #include <process.h>
  typedef HANDLE     thread_t;
  typedef CRITICAL_SECTION mutex_t;

  #define mutex_init(m)    InitializeCriticalSection(m)
  #define mutex_lock(m)    EnterCriticalSection(m)
  #define mutex_unlock(m)  LeaveCriticalSection(m)
  #define mutex_destroy(m) DeleteCriticalSection(m)
#else
  #include <pthread.h>
  typedef pthread_t      thread_t;
  typedef pthread_mutex_t mutex_t;

  #define mutex_init(m)    pthread_mutex_init(m, NULL)
  #define mutex_lock(m)    pthread_mutex_lock(m)
  #define mutex_unlock(m)  pthread_mutex_unlock(m)
  #define mutex_destroy(m) pthread_mutex_destroy(m)
#endif

// ============================================================================
// Timing
// ============================================================================

#ifdef OS_WINDOWS
  static inline double platform_now_ms(void) {
      LARGE_INTEGER freq, cnt;
      QueryPerformanceFrequency(&freq);
      QueryPerformanceCounter(&cnt);
      return (double)cnt.QuadPart / freq.QuadPart * 1000.0;
  }
#else
  #include <sys/time.h>
  static inline double platform_now_ms(void) {
      struct timeval tv;
      gettimeofday(&tv, NULL);
      return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
  }
#endif

// ============================================================================
// CPU core count
// ============================================================================

#ifdef OS_WINDOWS
  static inline int platform_cpu_count(void) {
      SYSTEM_INFO si;
      GetSystemInfo(&si);
      return (int)si.dwNumberOfProcessors;
  }
#elif defined(OS_MACOS)
  #include <sys/sysctl.h>
  static inline int platform_cpu_count(void) {
      int count = 1;
      size_t len = sizeof(count);
      sysctlbyname("hw.logicalcpu", &count, &len, NULL, 0);
      return count;
  }
#else
  static inline int platform_cpu_count(void) {
      int count = sysconf(_SC_NPROCESSORS_ONLN);
      return count > 0 ? count : 1;
  }
#endif

// ============================================================================
// Path separator
// ============================================================================

#ifdef OS_WINDOWS
  #define PATH_SEP "\\"
#else
  #define PATH_SEP "/"
#endif

// ============================================================================
// Aligned allocation
// ============================================================================

static inline void* platform_aligned_alloc(size_t alignment, size_t size) {
#ifdef OS_WINDOWS
    return _aligned_malloc(size, alignment);
#else
    void *ptr = NULL;
    posix_memalign(&ptr, alignment, size);
    return ptr;
#endif
}

static inline void platform_aligned_free(void *ptr) {
#ifdef OS_WINDOWS
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}
