#include "src/acc/cuda/cuda_event_pool.h"

#include <mutex>
#include <unordered_map>
#include <vector>
#include <cstdio>

struct EventPoolDevice {
    std::vector<cudaEvent_t> free_events;
};

static std::mutex pool_mutex;
static std::unordered_map<int, EventPoolDevice> device_pools;
static size_t g_initial = 32;
static size_t g_max = 0;
static bool g_initialized = false;

static inline void warn(const char* msg) { fprintf(stderr, "[event_pool] %s\n", msg); }

void cuda_event_pool_init(size_t initial_per_device, size_t max_per_device) {
    std::lock_guard<std::mutex> lk(pool_mutex);
    if (g_initialized) return;
    g_initial = initial_per_device;
    g_max = max_per_device;

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        warn("No CUDA devices found; pool disabled.");
        g_initialized = true;
        return;
    }

    int cur_dev = 0;
    cudaGetDevice(&cur_dev);

    for (int d = 0; d < ndev; ++d) {
        cudaSetDevice(d);
        EventPoolDevice pool;
        for (size_t i = 0; i < g_initial; ++i) {
            cudaEvent_t ev;
            if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) break;
            pool.free_events.push_back(ev);
        }
        device_pools.emplace(d, std::move(pool));
    }
    cudaSetDevice(cur_dev);
    g_initialized = true;
}

cudaEvent_t cuda_event_pool_acquire() {
    std::lock_guard<std::mutex> lk(pool_mutex);
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return nullptr;

    auto &pool = device_pools[dev];
    if (!pool.free_events.empty()) {
        auto ev = pool.free_events.back();
        pool.free_events.pop_back();
        return ev;
    }

    cudaEvent_t ev;
    if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) return nullptr;
    return ev;
}

void cuda_event_pool_release(cudaEvent_t ev) {
    if (!ev) return;
    std::lock_guard<std::mutex> lk(pool_mutex);
    int dev = 0;
    cudaGetDevice(&dev);
    auto &pool = device_pools[dev];
    if (g_max && pool.free_events.size() >= g_max) {
        cudaEventDestroy(ev);
    } else {
        pool.free_events.push_back(ev);
    }
}

void cuda_event_pool_shutdown() {
    std::lock_guard<std::mutex> lk(pool_mutex);
    for (auto &kv : device_pools) {
        for (auto ev : kv.second.free_events) cudaEventDestroy(ev);
        kv.second.free_events.clear();
    }
    device_pools.clear();
    g_initialized = false;
}

void cuda_event_pool_get_stats(std::ostream &os) {
    std::lock_guard<std::mutex> lk(pool_mutex);
    os << "[event_pool] devices: " << device_pools.size() << "\n";
    for (auto &kv : device_pools)
        os << "  dev " << kv.first << " free=" << kv.second.free_events.size() << "\n";
}
