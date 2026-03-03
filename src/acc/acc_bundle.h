#ifndef ACC_BUNDLE_H_
#define ACC_BUNDLE_H_

#include "src/acc/acc_projector.h"
#include "src/acc/acc_backprojector.h"
#include "src/acc/acc_projector_plan.h"
#include <vector>

class MlWsumModel;

class AccBundle {
public:
    std::vector<AccProjector>     projectors;
    std::vector<AccBackprojector> backprojectors;
    std::vector<AccProjectorPlan> coarseProjectionPlans;
    bool generateProjectionPlanOnTheFly = false;

    // Default no-op for CPU backend (synchronous)
    virtual void syncAllBackprojects() {}

    // Extract backprojector data into wsum_model and clear backprojectors
    virtual void extractAndAccumulate(MlWsumModel &wsum_model) = 0;

    // Clear projectors and coarse projection plans (call before extractAndAccumulate)
    void clearProjData() {
        for (auto &p : projectors)            p.clear();
        for (auto &p : coarseProjectionPlans) p.clear();
    }

    virtual ~AccBundle() = default;
};

#endif /* ACC_BUNDLE_H_ */
