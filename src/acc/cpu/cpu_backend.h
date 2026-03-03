#ifndef CPU_BACKEND_H_
#define CPU_BACKEND_H_

#include "src/acc/acc_backend.h"

#ifdef ALTCPU
#include <atomic>
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#define TBB_PREVIEW_GLOBAL_CONTROL 1
#include <tbb/global_control.h>
#include "src/acc/cpu/cpu_ml_optimiser.h"

// Backend for the TBB/ALTCPU path
class AltCpuBackend : public AccBackend {
public:
    void createBundles(MlOptimiser &opt) override
    {
        unsigned nr_classes = opt.mymodel.PPref.size();
        // Allocate array of complex arrays for each class
        if (posix_memalign((void **)&opt.mdlClassComplex, MEM_ALIGN,
                           nr_classes * sizeof(std::complex<XFLOAT> *)))
            CRITICAL(RAMERR);

        for (unsigned iclass = 0; iclass < nr_classes; iclass++)
        {
            int mdlX = opt.mymodel.PPref[iclass].data.xdim;
            int mdlY = opt.mymodel.PPref[iclass].data.ydim;
            int mdlZ = opt.mymodel.PPref[iclass].data.zdim;
            size_t mdlXYZ = (mdlZ == 0)
                ? (size_t)mdlX * (size_t)mdlY
                : (size_t)mdlX * (size_t)mdlY * (size_t)mdlZ;

            try { opt.mdlClassComplex[iclass] = new std::complex<XFLOAT>[mdlXYZ]; }
            catch (std::bad_alloc &) { CRITICAL(RAMERR); }

            std::complex<XFLOAT> *pData = opt.mdlClassComplex[iclass];
            for (size_t i = 0; i < mdlXYZ; i++)
            {
                pData[i] = std::complex<XFLOAT>(
                    (XFLOAT)opt.mymodel.PPref[iclass].data.data[i].real,
                    (XFLOAT)opt.mymodel.PPref[iclass].data.data[i].imag);
            }
        }

        MlDataBundle *b = new MlDataBundle();
        b->setup(&opt);
        opt.accDataBundles.push_back(b);
    }

    void createOptimisers(MlOptimiser &opt) override
    {
        // TBB creates MlOptimiserCpu lazily in thread-local storage; nothing to do here
    }

    void teardown(MlOptimiser &opt) override
    {
        for (auto *ab : opt.accDataBundles)
        {
            ab->clearProjData();
            ab->extractAndAccumulate(opt.wsum_model);
            delete ab;
        }
        opt.accDataBundles.clear();

        // Free the class complex arrays
        unsigned nr_classes = opt.mymodel.nr_classes;
        for (unsigned iclass = 0; iclass < nr_classes; iclass++)
            delete[] opt.mdlClassComplex[iclass];
        free(opt.mdlClassComplex);

        opt.tbbCpuOptimiser.clear();
    }

    void runParticles(MlOptimiser &opt, long int first, long int last) override
    {
        std::atomic<int> tCount(0);
        tbb::global_control gc(tbb::global_control::max_allowed_parallelism, opt.nr_threads);
        tbb::parallel_for(first, last + 1, [&](long int i) {
            MlOptimiser::CpuOptimiserType::reference ref = opt.tbbCpuOptimiser.local();
            MlOptimiserCpu *cpuOptimiser = (MlOptimiserCpu *)ref;
            if (cpuOptimiser == NULL)
            {
                cpuOptimiser = new MlOptimiserCpu(
                    &opt, (MlDataBundle*)opt.accDataBundles[0], "cpu_optimiser");
                cpuOptimiser->resetData();
                ref = cpuOptimiser;
                cpuOptimiser->thread_id = tCount.fetch_add(1);
            }
            cpuOptimiser->expectationOneParticle(i, cpuOptimiser->thread_id);
        });
    }
};

#endif /* ALTCPU */

// Backend for the plain CPU path (no accelerator)
class PlainCpuBackend : public AccBackend {
public:
    void createBundles(MlOptimiser &)    override {}
    void createOptimisers(MlOptimiser &) override {}
    void teardown(MlOptimiser &)         override {}
    // runParticles: inherit the default OpenMP implementation from AccBackend
};

#endif /* CPU_BACKEND_H_ */
