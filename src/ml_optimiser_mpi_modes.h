/***************************************************************************
 *
 * Mode-specific subclasses of MlOptimiserMpi.
 *
 * MlOptimiserMpi is a general-purpose MPI optimiser that uses runtime boolean
 * flags (do_auto_refine, do_split_random_halves, mymodel.ref_dim, …) to
 * choose between mode-specific code paths.  This file provides concrete
 * subclasses that "lock in" a single mode and override the virtual hooks
 * established in Phases 1 & 2 with mode-specific implementations, removing
 * the need for per-mode flag checks at call-sites.
 *
 * Hierarchy:
 *   MlOptimiser
 *     └─ MlOptimiserMpi            (base: flag-dispatch fallback)
 *           ├─ AutoRefineOptimiserMpi    (do_auto_refine=true, split halves)
 *           └─ ClassificationOptimiserMpi (do_auto_refine=false, 2D or 3D)
 *
 * Factory:
 *   makeMlOptimiserMpi(argc, argv) peeks --auto_refine in argv and returns
 *   the appropriate concrete type.  The caller is responsible for calling
 *   read(), initialise(), iterate() in the usual order.
 *
 ***************************************************************************/

#ifndef ML_OPTIMISER_MPI_MODES_H_
#define ML_OPTIMISER_MPI_MODES_H_

#include "src/ml_optimiser_mpi.h"
#include <memory>

// ---------------------------------------------------------------------------
// AutoRefineOptimiserMpi
//
// Gold-standard auto-refinement: one 3D reference, data split into two
// random halfsets, FSC-based convergence.
//
// Overrides:
//   reconstructionRankForClass() — halfset-split rank formula (no flag check)
//   computeMemoryConfig()        — adds GPU headroom for the large final-
//                                  iteration box size (ori_size >> current_size)
// ---------------------------------------------------------------------------
class AutoRefineOptimiserMpi : public MlOptimiserMpi
{
public:
    int          reconstructionRankForClass(int ith_recons,
                                           int nr_followers) const override;
    MemoryConfig computeMemoryConfig()                        const override;
};

// ---------------------------------------------------------------------------
// ClassificationOptimiserMpi
//
// 2D or 3D classification: one or more references, no halfset split,
// classification-specific convergence.
//
// Overrides:
//   reconstructionRankForClass() — round-robin rank formula (no flag check)
//   computeMemoryConfig()        — caps nr_pool proportionally to nr_classes
//                                  for 3D runs (prevents per-class OOM);
//                                  2D runs are exempt (small reference vols)
// ---------------------------------------------------------------------------
class ClassificationOptimiserMpi : public MlOptimiserMpi
{
public:
    int          reconstructionRankForClass(int ith_recons,
                                            int nr_followers) const override;
    MemoryConfig computeMemoryConfig()                         const override;
};

// ---------------------------------------------------------------------------
// Factory
//
// Peeks argv for --auto_refine and returns the appropriate concrete subclass.
// ---------------------------------------------------------------------------
std::unique_ptr<MlOptimiserMpi> makeMlOptimiserMpi(int argc, char **argv);

#endif /* ML_OPTIMISER_MPI_MODES_H_ */
