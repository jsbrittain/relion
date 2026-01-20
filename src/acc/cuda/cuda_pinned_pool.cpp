// src/acc/cuda/cuda_pinned_pool.cpp
#include "src/acc/cuda/cuda_pinned_pool.h"

#include <mutex>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <list>

#ifdef _CUDA_ENABLED
#include <cuda_runtime.h>
#endif

// Pooled pinned allocator
// - Minimum bucket: 64 kB
// - Buckets are power-of-two
// - Acquire returns a whole bucket (no suballocation)
// - Release returns pointer to freelist
// - When total pinned bytes exceed max, trim freelists (free oldest items)

namespace {
    std::mutex pool_mutex;
    std::unordered_map<size_t, std::vector<void*>> freelists; // bucket size -> list of free blocks
    std::unordered_map<void*, size_t> ptr_sizes; // pointer -> bucket size
    std::list<void*> lru_order; // optional LRU ordering of allocated blocks (only freelist entries tracked)
    size_t current_pinned_bytes = 0;
    size_t max_pinned_bytes = SIZE_MAX;  // 64ull * 1024 * 1024; // default 64 MB
    bool initialized = false;

    static inline size_t round_to_bucket(size_t bytes) {
        const size_t min_bucket = 64 * 1024; // 64 KiB minimum bucket
        size_t b = std::max(min_bucket, bytes);
        // round up to next power-of-two
        b--;
        b |= b >> 1;
        b |= b >> 2;
        b |= b >> 4;
        b |= b >> 8;
        b |= b >> 16;
#if SIZE_MAX > UINT32_MAX
        b |= b >> 32;
#endif
        b++;
        return b;
    }

#ifdef _CUDA_ENABLED
    static bool try_cuda_malloc_host(void** pptr, size_t bytes) {
        cudaError_t e = cudaMallocHost(pptr, bytes);
        return (e == cudaSuccess);
    }
    static void cuda_free_host(void* p) {
        cudaFreeHost(p);
    }
#else
    static bool try_cuda_malloc_host(void** pptr, size_t bytes) {
        // Non-CUDA builds: fall back to aligned malloc
        void* p = nullptr;
        if (posix_memalign(&p, 64, bytes) != 0) return false;
        *pptr = p;
        return true;
    }
    static void cuda_free_host(void* p) {
        free(p);
    }
#endif

    // Helper to trim freelists when current_pinned_bytes > max_pinned_bytes.
    // Frees oldest freelist entries until under cap.
    static void trim_freelists_if_needed_unlocked() {
        if (current_pinned_bytes <= max_pinned_bytes) return;

        // iterate freelists buckets smallest-first to free some blocks.
        // Simpler policy: iterate buckets and free blocks until under cap.
        for (auto it = freelists.begin(); it != freelists.end() && current_pinned_bytes > max_pinned_bytes;) {
            auto &vec = it->second;
            while (!vec.empty() && current_pinned_bytes > max_pinned_bytes) {
                void* p = vec.back();
                vec.pop_back();
                auto sit = ptr_sizes.find(p);
                if (sit != ptr_sizes.end()) {
                    size_t b = sit->second;
                    // free actual host memory
                    cuda_free_host(p);
                    current_pinned_bytes -= b;
                    ptr_sizes.erase(sit);
                } else {
                    // unknown pointer (shouldn't happen), free defensively
                    cuda_free_host(p);
                }
            }
            // erase empty vector entries to keep map small
            if (vec.empty()) it = freelists.erase(it);
            else ++it;
        }
    }

}

extern "C" {

void cuda_pinned_pool_init(size_t max_total_bytes) {
    std::lock_guard<std::mutex> lk(pool_mutex);
    if (initialized) return;
    if (max_total_bytes > 0) max_pinned_bytes = max_total_bytes;
    freelists.clear();
    ptr_sizes.clear();
    lru_order.clear();
    current_pinned_bytes = 0;
    initialized = true;
}

void* cuda_pinned_pool_acquire(size_t bytes) {
    if (!initialized) cuda_pinned_pool_init(0);

    // std::cout << "cuda_pinned_pool_acquire: requesting " << bytes << " bytes." << std::endl;

    size_t b = round_to_bucket(bytes);
    std::lock_guard<std::mutex> lk(pool_mutex);

    // Try to pop from freelist for this bucket
    auto it = freelists.find(b);
    if (it != freelists.end() && !it->second.empty()) {
        void* p = it->second.back();
        it->second.pop_back();
        // book-keeping: this pointer remains known (ptr_sizes already contains it)
        // move in LRU: remove from list if present (we won't track exact LRU per freelist entry here)
        ptr_sizes[p] = b; // ensure entry exists
        return p;
    }

    // If allocating a new block would exceed cap, attempt to trim freelists first
    if (current_pinned_bytes + b > max_pinned_bytes) {
        // try to free some freelist blocks to make room
        trim_freelists_if_needed_unlocked();
        if (current_pinned_bytes + b > max_pinned_bytes) {
            // still can't allocate, fail
            // std::cerr << "cuda_pinned_pool_acquire: allocation of " << b << " bytes would exceed max pinned pool size of "
            //   << max_pinned_bytes << " bytes. Allocation failed." << std::endl;
            return nullptr;
        }
    }

    // allocate underlying pinned memory
    void* p = nullptr;
    if (!try_cuda_malloc_host(&p, b)) {
        // allocation failed
        return nullptr;
    }
    // success: record its bucket and account bytes
    ptr_sizes[p] = b;
    current_pinned_bytes += b;
    return p;
}

void cuda_pinned_pool_release(void* ptr) {
    if (ptr == nullptr) return;
    std::lock_guard<std::mutex> lk(pool_mutex);

    auto it = ptr_sizes.find(ptr);
    if (it == ptr_sizes.end()) {
        // Unknown pointer: caller returned something we didn't allocate via pool.
        // Best-effort: free it via cudaFreeHost (this is conservative).
        cuda_free_host(ptr);
        return;
    }
    size_t b = it->second;

    // return pointer to freelist (do NOT free underlying memory here)
    freelists[b].push_back(ptr);

    // Optionally trim freelists immediately if we are above cap:
    // (trim will free some freelist blocks back to driver to enforce max_pinned_bytes)
    if (current_pinned_bytes > max_pinned_bytes) {
        trim_freelists_if_needed_unlocked();
    }
}

void cuda_pinned_pool_destroy() {
    std::lock_guard<std::mutex> lk(pool_mutex);
    // free all freelist blocks
    for (auto &kv : freelists) {
        for (void* p : kv.second) {
            auto psit = ptr_sizes.find(p);
            if (psit != ptr_sizes.end()) {
                cuda_free_host(p);
                current_pinned_bytes -= psit->second;
                ptr_sizes.erase(psit);
            } else {
                cuda_free_host(p);
            }
        }
    }
    freelists.clear();

    // There may be outstanding checked-out pointers in ptr_sizes; we can't free them safely here.
    // Clear bookkeeping so repeated init/destroy works but warn in debug.
#ifndef NDEBUG
    if (!ptr_sizes.empty()) {
        std::cerr << "cuda_pinned_pool_destroy: outstanding checked-out pinned pointers remain (not freed)." << std::endl;
    }
#endif
    ptr_sizes.clear();
    current_pinned_bytes = 0;
    initialized = false;
}

size_t cuda_pinned_pool_get_size(void* ptr) {
    std::lock_guard<std::mutex> lk(pool_mutex);
    auto it = ptr_sizes.find(ptr);
    if (it == ptr_sizes.end()) return 0;
    return it->second;
}

} // extern "C"
