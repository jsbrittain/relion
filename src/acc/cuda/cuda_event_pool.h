#ifndef CUDA_EVENT_POOL_H_
#define CUDA_EVENT_POOL_H_

#include <cuda_runtime.h>
#include <cstddef>
#include <ostream>

// Initialize the pool (once). Safe to call repeatedly.
void cuda_event_pool_init(size_t initial_per_device = 32, size_t max_per_device = 0);

// Get a CUDA event (may create if none available).
cudaEvent_t cuda_event_pool_acquire();

// Return an event to the pool.
void cuda_event_pool_release(cudaEvent_t ev);

// Destroy all pooled events.
void cuda_event_pool_shutdown();

// Print simple pool stats.
void cuda_event_pool_get_stats(std::ostream &os);

#endif
