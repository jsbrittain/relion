#ifndef SYCL_BACKEND_H_
#define SYCL_BACKEND_H_

#ifdef _SYCL_ENABLED

#include <cstdlib>
#include <cstring>
#include <tuple>
#include <algorithm>

#include "src/acc/acc_backend.h"
#include "src/acc/sycl/sycl_ml_optimiser.h"

class SyclBackend : public AccBackend {
public:
    void createBundles(MlOptimiser &opt) override
    {
        char* pEnvStream = std::getenv("relionSyclUseStream");
        std::string strStream = (pEnvStream == nullptr) ? "0" : pEnvStream;
        std::transform(strStream.begin(), strStream.end(), strStream.begin(),
                       [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        isStream_ = (strStream == "1" || strStream == "on");

        for (int i = 0; i < (int)opt.gpuDevices.size(); i++)
        {
            MlSyclDataBundle *b = new MlSyclDataBundle(opt.syclDeviceList[opt.gpuDevices[i]]);
            b->setup(&opt);
            opt.accDataBundles.push_back(b);
        }
    }

    void createOptimisers(MlOptimiser &opt) override
    {
        for (int i = 0; i < (int)opt.gpuOptimiserDeviceMap.size(); i++)
        {
            MlSyclDataBundle *bundle = static_cast<MlSyclDataBundle*>(
                opt.accDataBundles[opt.gpuOptimiserDeviceMap[i]]);
            MlOptimiserSYCL *b = new MlOptimiserSYCL(&opt, bundle, isStream_, "sycl_optimiser");
            b->resetData();
            b->threadID = i;
            opt.accOptimisers.push_back(b);
        }
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
            delete ab;
        opt.accDataBundles.clear();
    }

private:
    bool isStream_ = false;
};

#endif /* _SYCL_ENABLED */
#endif /* SYCL_BACKEND_H_ */
