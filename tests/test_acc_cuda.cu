/*
 * Unit tests for pure CPU-side logic in src/acc/cuda/.
 *
 * Covers:
 *   cuda_benchmark_utils.cu  — relion_timer::cuda_benchmark_find_id()
 *   cuda_helper_functions.cu — mapWeights()
 *   cuda_settings.h          — compile-time configuration constants
 *
 * These functions contain no GPU kernel launches and therefore run
 * correctly without a physical GPU being present.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <limits>
#include <sys/stat.h>

// XFLOAT: float (or double with ACC_DOUBLE_PRECISION), no CUDA headers needed
#include "src/acc/settings.h"

// cuda_benchmark_utils — relion_timer class definition
#include "src/acc/cuda/cuda_benchmark_utils.h"

// cuda_settings — compile-time block-size / overhead constants
#include "src/acc/cuda/cuda_settings.h"

// ---------------------------------------------------------------------------
// mapWeights forward declaration
//
// Defined in src/acc/acc_helper_functions_impl.h, compiled into relion_gpu_util
// via src/acc/cuda/cuda_helper_functions.cu, and linked through relion_lib.
// Signature uses only standard C++ types (XFLOAT = float/double, raw pointers).
// ---------------------------------------------------------------------------
void mapWeights(
        unsigned long   orientation_start,
        XFLOAT         *mapped_weights,
        unsigned long   orientation_num,
        unsigned long   idxArr_start,
        unsigned long   idxArr_end,
        unsigned long   translation_num,
        XFLOAT         *weights,
        long unsigned  *rot_idx,
        long unsigned  *trans_idx,
        unsigned long   current_oversampling);

// ---------------------------------------------------------------------------
// Global fixture: ensure "output/" directory exists so relion_timer can open
// its log files.  The directory is created once before any test runs.
// ---------------------------------------------------------------------------
class OutputDirEnvironment : public ::testing::Environment
{
public:
    void SetUp() override { mkdir("output", 0755); }
};

// ---------------------------------------------------------------------------
// relion_timer::cuda_benchmark_find_id
// ---------------------------------------------------------------------------

TEST(CudaBenchmarkUtilsTest, FindId_EmptyVector_ReturnsMinusOne)
{
    relion_timer timer("bm_empty");
    std::vector<std::string> v;
    EXPECT_EQ(timer.cuda_benchmark_find_id("anything", v), -1);
}

TEST(CudaBenchmarkUtilsTest, FindId_PresentAtIndex0)
{
    relion_timer timer("bm_idx0");
    std::vector<std::string> v = {"alpha", "beta", "gamma"};
    EXPECT_EQ(timer.cuda_benchmark_find_id("alpha", v), 0);
}

TEST(CudaBenchmarkUtilsTest, FindId_PresentAtIndex2)
{
    relion_timer timer("bm_idx2");
    std::vector<std::string> v = {"alpha", "beta", "gamma"};
    EXPECT_EQ(timer.cuda_benchmark_find_id("gamma", v), 2);
}

TEST(CudaBenchmarkUtilsTest, FindId_NotPresent_ReturnsMinusOne)
{
    relion_timer timer("bm_notfound");
    std::vector<std::string> v = {"alpha", "beta", "gamma"};
    EXPECT_EQ(timer.cuda_benchmark_find_id("delta", v), -1);
}

TEST(CudaBenchmarkUtilsTest, FindId_ReturnsFirstOccurrence)
{
    relion_timer timer("bm_first");
    std::vector<std::string> v = {"x", "y", "x"};
    EXPECT_EQ(timer.cuda_benchmark_find_id("x", v), 0);
}

TEST(CudaBenchmarkUtilsTest, FindId_SingleElementMatch)
{
    relion_timer timer("bm_single");
    std::vector<std::string> v = {"only"};
    EXPECT_EQ(timer.cuda_benchmark_find_id("only", v), 0);
    EXPECT_EQ(timer.cuda_benchmark_find_id("other", v), -1);
}

// ---------------------------------------------------------------------------
// cuda_settings.h — compile-time constants
// ---------------------------------------------------------------------------

TEST(CudaSettingsTest, GpuThreadMemoryOverheadMB_Is200)
{
    EXPECT_EQ(GPU_THREAD_MEMORY_OVERHEAD_MB, 200);
}

TEST(CudaSettingsTest, ComputeCapabilityMinimum)
{
    EXPECT_EQ(CUDA_CC_MAJOR, 3);
    EXPECT_EQ(CUDA_CC_MINOR, 5);
}

TEST(CudaSettingsTest, BlockSizePositive)
{
    EXPECT_GT(BLOCK_SIZE, 0);
}

TEST(CudaSettingsTest, Diff2CoarseBlockSizesPositive)
{
    EXPECT_GT(D2C_BLOCK_SIZE_2D,    0);
    EXPECT_GT(D2C_BLOCK_SIZE_REF3D, 0);
    EXPECT_GT(D2C_BLOCK_SIZE_DATA3D,0);
}

TEST(CudaSettingsTest, Diff2FineChunkSizesPositive)
{
    EXPECT_GT(D2F_CHUNK_2D,    0);
    EXPECT_GT(D2F_CHUNK_REF3D, 0);
    EXPECT_GT(D2F_CHUNK_DATA3D,0);
}

TEST(CudaSettingsTest, BackprojectionBlockSizesPositive)
{
    EXPECT_GT(BP_2D_BLOCK_SIZE,    0);
    EXPECT_GT(BP_REF3D_BLOCK_SIZE, 0);
    EXPECT_GT(BP_DATA3D_BLOCK_SIZE,0);
}

// ---------------------------------------------------------------------------
// mapWeights
//
// The function fills an (orientation_num × translation_num) dense array from
// a sparse (rot_idx, trans_idx, weights) representation:
//
//   for i in [idxArr_start, idxArr_end):
//       mapped[ (rot_idx[i] - orientation_start) * trans_num + trans_idx[i] ]
//           = weights[i]
//
// All positions not covered by a sparse entry remain at
// std::numeric_limits<XFLOAT>::lowest().
// ---------------------------------------------------------------------------

TEST(MapWeightsTest, EmptyRange_AllLowest)
{
    // idxArr_start == idxArr_end → nothing is mapped
    const unsigned long orient_num = 2;
    const unsigned long trans_num  = 3;
    std::vector<XFLOAT> mapped(orient_num * trans_num, 0.f);

    XFLOAT        weights[]   = {1.f, 2.f};
    long unsigned rot_idx[]   = {0, 0};
    long unsigned trans_idx[] = {0, 1};

    mapWeights(0, mapped.data(), orient_num, 5, 5, trans_num,
               weights, rot_idx, trans_idx, 1);

    for (auto v : mapped)
        EXPECT_EQ(v, std::numeric_limits<XFLOAT>::lowest());
}

TEST(MapWeightsTest, SingleWeight_PlacedAtCorrectIndex)
{
    const unsigned long orient_num = 1;
    const unsigned long trans_num  = 3;
    std::vector<XFLOAT> mapped(orient_num * trans_num);

    XFLOAT        weights[]   = {3.14f};
    long unsigned rot_idx[]   = {0};       // absolute rot 0
    long unsigned trans_idx[] = {2};       // translation 2

    mapWeights(0, mapped.data(), orient_num, 0, 1, trans_num,
               weights, rot_idx, trans_idx, 1);

    EXPECT_EQ(mapped[0 * trans_num + 2], 3.14f);
    EXPECT_EQ(mapped[0 * trans_num + 0], std::numeric_limits<XFLOAT>::lowest());
    EXPECT_EQ(mapped[0 * trans_num + 1], std::numeric_limits<XFLOAT>::lowest());
}

TEST(MapWeightsTest, MultipleWeights_SparseToDense)
{
    const unsigned long orient_num = 2;
    const unsigned long trans_num  = 2;
    std::vector<XFLOAT> mapped(orient_num * trans_num);

    XFLOAT        weights[]   = {1.f, 2.f, 3.f};
    long unsigned rot_idx[]   = {0, 0, 1};   // three entries: two in rot 0, one in rot 1
    long unsigned trans_idx[] = {0, 1, 0};

    mapWeights(0, mapped.data(), orient_num, 0, 3, trans_num,
               weights, rot_idx, trans_idx, 1);

    EXPECT_EQ(mapped[0 * trans_num + 0], 1.f);  // rot=0, trans=0
    EXPECT_EQ(mapped[0 * trans_num + 1], 2.f);  // rot=0, trans=1
    EXPECT_EQ(mapped[1 * trans_num + 0], 3.f);  // rot=1, trans=0
    EXPECT_EQ(mapped[1 * trans_num + 1], std::numeric_limits<XFLOAT>::lowest());
}

TEST(MapWeightsTest, OrientationStartOffset_CorrectlyShiftsIndex)
{
    // orientation_start=4: absolute rot_idx=5 maps to relative index 5-4=1
    const unsigned long orient_num = 2;
    const unsigned long trans_num  = 1;
    std::vector<XFLOAT> mapped(orient_num * trans_num);

    XFLOAT        weights[]   = {7.f};
    long unsigned rot_idx[]   = {5};  // absolute
    long unsigned trans_idx[] = {0};

    mapWeights(4, mapped.data(), orient_num, 0, 1, trans_num,
               weights, rot_idx, trans_idx, 1);

    EXPECT_EQ(mapped[0 * trans_num + 0], std::numeric_limits<XFLOAT>::lowest());
    EXPECT_EQ(mapped[1 * trans_num + 0], 7.f);
}

TEST(MapWeightsTest, PartialRange_OnlyRangeWeightsMapped)
{
    // idxArr_start=1, idxArr_end=2: only the middle element of the arrays is used
    const unsigned long orient_num = 1;
    const unsigned long trans_num  = 3;
    std::vector<XFLOAT> mapped(orient_num * trans_num);

    XFLOAT        weights[]   = {10.f, 20.f, 30.f};
    long unsigned rot_idx[]   = {0, 0, 0};
    long unsigned trans_idx[] = {0, 1, 2};

    mapWeights(0, mapped.data(), orient_num, 1, 2, trans_num,
               weights, rot_idx, trans_idx, 1);

    EXPECT_EQ(mapped[0], std::numeric_limits<XFLOAT>::lowest()); // trans 0 skipped
    EXPECT_EQ(mapped[1], 20.f);                                   // trans 1 included
    EXPECT_EQ(mapped[2], std::numeric_limits<XFLOAT>::lowest()); // trans 2 skipped
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    ::testing::AddGlobalTestEnvironment(new OutputDirEnvironment());
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
