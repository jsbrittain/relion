#ifndef ACC_BACKEND_H_
#define ACC_BACKEND_H_

#include "src/ml_optimiser.h"

// Forward declaration of the global threading function defined in ml_optimiser.cpp
void globalThreadExpectationSomeParticles(void *self, int thread_id);

class AccBackend {
public:
    // Allocate data bundles and fixed-size device objects; push to opt.accDataBundles
    virtual void createBundles(MlOptimiser &opt) = 0;
    // Allocate per-thread optimisers; push to opt.accOptimisers
    virtual void createOptimisers(MlOptimiser &opt) = 0;
    // Second-phase GPU memory finalisation (tunable alloc). No-op by default.
    virtual void finalizeBundles(MlOptimiser &opt) {}
    // Sync, extract, clear, and free all accelerator resources
    virtual void teardown(MlOptimiser &opt) = 0;

    // Dispatch particle range to per-thread workers.
    // Default implementation uses OpenMP + accOptimisers (or doThreadExpectationSomeParticles).
    // ALTCPU overrides to use TBB.
    virtual void runParticles(MlOptimiser &opt, long int first, long int last)
    {
        opt.exp_ipart_ThreadTaskDistributor->resize(last - first + 1, 1);
        opt.exp_ipart_ThreadTaskDistributor->reset();
        #pragma omp parallel for num_threads(opt.nr_threads)
        for (int tid = 0; tid < opt.nr_threads; tid++)
            globalThreadExpectationSomeParticles(&opt, tid);
    }

    virtual ~AccBackend() = default;
};

#endif /* ACC_BACKEND_H_ */
