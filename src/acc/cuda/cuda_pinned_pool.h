#ifndef CUDA_PINNED_POOL_H_
#define CUDA_PINNED_POOL_H_

#include <cstddef>

#ifdef _CUDA_ENABLED
#include <cuda_runtime.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

void cuda_pinned_pool_init(size_t max_total_bytes);
void* cuda_pinned_pool_acquire(size_t bytes);
void cuda_pinned_pool_release(void* ptr);
void cuda_pinned_pool_destroy(void);
size_t cuda_pinned_pool_get_size(void* ptr);

#ifdef __cplusplus
}
#endif

#endif // CUDA_PINNED_POOL_H_
