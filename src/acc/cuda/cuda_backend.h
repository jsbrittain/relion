#ifndef CUDA_BACKEND_H_
#define CUDA_BACKEND_H_

#if defined _CUDA_ENABLED || defined _HIP_ENABLED

#include "src/acc/acc_backend.h"
#ifdef _CUDA_ENABLED
#include "src/acc/cuda/cuda_ml_optimiser.h"
#elif _HIP_ENABLED
#include "src/acc/hip/hip_ml_optimiser.h"
#endif

class CudaBackend : public AccBackend {
    std::vector<size_t>   allocationSizes_;
    std::vector<unsigned> threadcountOnDevice_;

public:
    void createBundles(MlOptimiser &opt) override
    {
        for (int i = 0; i < (int)opt.gpuDevices.size(); i++)
        {
            MlDeviceBundle *b = new MlDeviceBundle(&opt);
            b->setDevice(opt.gpuDevices[i]);
            b->setupFixedSizedObjects();
            opt.accDataBundles.push_back(b);
        }

        threadcountOnDevice_.assign(opt.accDataBundles.size(), 0);
        for (int i = 0; i < (int)opt.gpuOptimiserDeviceMap.size(); i++)
            threadcountOnDevice_[opt.gpuOptimiserDeviceMap[i]]++;

        int devCount;
        HANDLE_ERROR(accGPUGetDeviceCount(&devCount));
        HANDLE_ERROR(accGPUDeviceSynchronize());

        for (int i = 0; i < (int)opt.accDataBundles.size(); i++)
        {
            MlDeviceBundle *b = static_cast<MlDeviceBundle*>(opt.accDataBundles[i]);
            if (b->device_id >= devCount || b->device_id < 0) {
                CRITICAL(ERR_GPUID);
            } else {
                HANDLE_ERROR(accGPUSetDevice(b->device_id));
            }

            size_t free, total, allocationSize;
            HANDLE_ERROR(accGPUMemGetInfo(&free, &total));

            // In MPI mode, divide memory by number of ranks sharing this device
            free = (size_t)((double)free / (double)opt.gpuDeviceShareAt(i));

            size_t required_free = opt.requested_free_gpu_memory +
                                   GPU_THREAD_MEMORY_OVERHEAD_MB * 1000 * 1000 *
                                   threadcountOnDevice_[i];

            if (free < required_free)
            {
                printf("WARNING: Ignoring required free GPU memory amount of %zu MB, "
                       "due to space insufficiency.\n", required_free / 1000000);
                allocationSize = (double)free * 0.7;
            }
            else
                allocationSize = free - required_free;

            if (allocationSize < 200000000)
                printf("WARNING: The available space on the GPU after initialization "
                       "(%zu MB) might be insufficient for the expectation step.\n",
                       allocationSize / 1000000);

#ifdef PRINT_GPU_MEM_INFO
            printf("INFO: Projector model size %dx%dx%d\n",
                   (int)opt.mymodel.PPref[0].data.xdim,
                   (int)opt.mymodel.PPref[0].data.ydim,
                   (int)opt.mymodel.PPref[0].data.zdim);
            printf("INFO: Free memory for Custom Allocator of device bundle %d is %d MB\n",
                   i, (int)((float)allocationSize / 1000000.0));
#endif
            allocationSizes_.push_back(allocationSize);
        }
    }

    void createOptimisers(MlOptimiser &opt) override
    {
        for (int i = 0; i < (int)opt.gpuOptimiserDeviceMap.size(); i++)
        {
            std::string name = opt.accThreadName(i);
            MlDeviceBundle *bundle = static_cast<MlDeviceBundle*>(
                opt.accDataBundles[opt.gpuOptimiserDeviceMap[i]]);
#ifdef _CUDA_ENABLED
            MlOptimiserCuda *b = new MlOptimiserCuda(&opt, bundle, name.c_str());
#elif _HIP_ENABLED
            MlOptimiserHip  *b = new MlOptimiserHip (&opt, bundle, name.c_str());
#endif
            b->resetData();
            opt.accOptimisers.push_back(b);
        }
    }

    void finalizeBundles(MlOptimiser &opt) override
    {
        for (int i = 0; i < (int)opt.accDataBundles.size(); i++)
            static_cast<MlDeviceBundle*>(opt.accDataBundles[i])
                ->setupTunableSizedObjects(allocationSizes_[i]);
    }

    void teardown(MlOptimiser &opt) override
    {
        for (auto *ab : opt.accDataBundles)
        {
            ab->syncAllBackprojects();
            ab->clearProjData();
            ab->extractAndAccumulate(opt.wsum_model);
        }

        for (auto *o : opt.accOptimisers)
            delete o;
        opt.accOptimisers.clear();

        for (auto *ab : opt.accDataBundles)
        {
            MlDeviceBundle *b = static_cast<MlDeviceBundle*>(ab);
            b->allocator->syncReadyEvents();
            b->allocator->freeReadyAllocs();

#if defined DEBUG_CUDA || defined DEBUG_HIP
            if (b->allocator->getNumberOfAllocs() != 0)
            {
                printf("DEBUG_ERROR: Non-zero allocation count encountered in "
                       "custom allocator between iterations.\n");
                b->allocator->printState();
                fflush(stdout);
                CRITICAL(ERR_CANZ);
            }
#endif
        }

        for (auto *ab : opt.accDataBundles)
            delete ab;
        opt.accDataBundles.clear();
    }
};

#endif /* _CUDA_ENABLED || _HIP_ENABLED */
#endif /* CUDA_BACKEND_H_ */
