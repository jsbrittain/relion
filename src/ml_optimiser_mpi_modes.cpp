/***************************************************************************
 *
 * Mode-specific subclass implementations for MlOptimiserMpi.
 *
 ***************************************************************************/

#include "src/ml_optimiser_mpi_modes.h"
#include "src/args.h"

// ---------------------------------------------------------------------------
// AutoRefineOptimiserMpi
// ---------------------------------------------------------------------------

int AutoRefineOptimiserMpi::reconstructionRankForClass(int ith_recons,
                                                       int nr_followers) const
{
    // Halfset-split: followers are split into two equal halves.
    // Reconstruction ranks are the odd-numbered followers within halfset 1.
    return 2 * (ith_recons % (nr_followers / 2)) + 1;
}

MemoryConfig AutoRefineOptimiserMpi::computeMemoryConfig() const
{
    MemoryConfig cfg = MlOptimiserMpi::computeMemoryConfig();

    // AutoRefine iterates at current_size but produces a final map at ori_size.
    // Reserve extra GPU memory for the larger final-iteration backprojector
    // (two halfsets × complex + weight arrays at ori_size padding).
    if (mymodel.ori_size > mymodel.current_size && mymodel.current_size > 0)
    {
        const double pf       = mymodel.padding_factor;
        const double ori_pad  = pf * mymodel.ori_size;
        const double curr_pad = pf * mymodel.current_size;

        // Number of Fourier-half elements: NX * NY * (NZ/2 + 1)
        const size_t ori_elems =
            static_cast<size_t>(ori_pad) *
            static_cast<size_t>(ori_pad) *
            static_cast<size_t>(ori_pad / 2 + 1);
        const size_t curr_elems =
            static_cast<size_t>(curr_pad) *
            static_cast<size_t>(curr_pad) *
            static_cast<size_t>(curr_pad / 2 + 1);

        if (ori_elems > curr_elems)
        {
            // Two halfsets, each needs complex data + real weight arrays.
            // complex = 2 * RFLOAT, weight = 1 * RFLOAT → factor 3 per halfset
            cfg.requested_free_gpu_bytes +=
                (ori_elems - curr_elems) * 2 * 3 * sizeof(RFLOAT);
        }
    }

    return cfg;
}

// ---------------------------------------------------------------------------
// ClassificationOptimiserMpi
// ---------------------------------------------------------------------------

int ClassificationOptimiserMpi::reconstructionRankForClass(int ith_recons,
                                                           int nr_followers) const
{
    // Round-robin across all followers (1-indexed).
    return ith_recons % nr_followers + 1;
}

MemoryConfig ClassificationOptimiserMpi::computeMemoryConfig() const
{
    MemoryConfig cfg = MlOptimiserMpi::computeMemoryConfig();

    // For 3D classification each pool step holds particles against all K
    // reference volumes simultaneously, so GPU pressure scales with K.
    // Cap the pool so that K × pool_size ≤ x_pool × nr_threads (the
    // thread-budget before any class-count scaling).
    //
    // 2D classification is exempt: 2D Fourier references are far smaller than
    // 3D ones and do not create the same pressure.
    //
    // Guards:
    //   nr_classes == 0 → model not yet initialised; leave uncapped.
    //   nr_classes == 1 → single reference, no multi-class pressure.
    //   nr_threads == 0 → thread count not yet read; leave uncapped.
    if (mymodel.ref_dim == 3 && mymodel.nr_classes > 1 && nr_threads > 0)
    {
        const int base_pool = x_pool * nr_threads;
        cfg.max_pool_size   = std::max(1, base_pool / mymodel.nr_classes);
    }

    return cfg;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<MlOptimiserMpi> makeMlOptimiserMpi(int argc, char **argv)
{
    if (checkParameter(argc, argv, "--auto_refine"))
        return std::make_unique<AutoRefineOptimiserMpi>();
    return std::make_unique<ClassificationOptimiserMpi>();
}
