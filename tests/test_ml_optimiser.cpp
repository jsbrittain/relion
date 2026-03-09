/*
 * Unit tests for src/ml_optimiser.h / src/ml_optimiser.cpp (MlOptimiser)
 *
 * Covers: refinementMode() — all three enum values,
 *         computeMemoryConfig() — returns requested_free_gpu_memory,
 *         accThreadName() — generates correct thread-name string
 */

#include <gtest/gtest.h>
#include "src/ml_optimiser.h"

// ---------------------------------------------------------------------------
// refinementMode()
// ---------------------------------------------------------------------------

TEST(MlOptimiserTest, RefinementMode_Class2D)
{
    MlOptimiser opt;
    opt.do_auto_refine     = false;
    opt.mymodel.ref_dim    = 2;
    EXPECT_EQ(opt.refinementMode(), RefinementMode::Class2D);
}

TEST(MlOptimiserTest, RefinementMode_Class3D)
{
    MlOptimiser opt;
    opt.do_auto_refine     = false;
    opt.mymodel.ref_dim    = 3;
    EXPECT_EQ(opt.refinementMode(), RefinementMode::Class3D);
}

TEST(MlOptimiserTest, RefinementMode_AutoRefine_TakesPrecedence)
{
    MlOptimiser opt;
    opt.do_auto_refine  = true;
    opt.mymodel.ref_dim = 2; // even with ref_dim==2, auto_refine wins
    EXPECT_EQ(opt.refinementMode(), RefinementMode::AutoRefine);
}

TEST(MlOptimiserTest, RefinementMode_AutoRefine_3D)
{
    MlOptimiser opt;
    opt.do_auto_refine  = true;
    opt.mymodel.ref_dim = 3;
    EXPECT_EQ(opt.refinementMode(), RefinementMode::AutoRefine);
}

// ---------------------------------------------------------------------------
// computeMemoryConfig()
// ---------------------------------------------------------------------------

TEST(MlOptimiserTest, ComputeMemoryConfig_Default_FreeBytesZero)
{
    MlOptimiser opt;
    // Default constructor sets requested_free_gpu_memory = 0
    MemoryConfig cfg = opt.computeMemoryConfig();
    EXPECT_EQ(cfg.requested_free_gpu_bytes, (size_t)0);
    EXPECT_EQ(cfg.max_pool_size, 0);
}

TEST(MlOptimiserTest, ComputeMemoryConfig_CustomValue)
{
    MlOptimiser opt;
    opt.requested_free_gpu_memory = 512 * 1024 * 1024ULL; // 512 MB
    MemoryConfig cfg = opt.computeMemoryConfig();
    EXPECT_EQ(cfg.requested_free_gpu_bytes, 512ULL * 1024 * 1024);
    EXPECT_EQ(cfg.max_pool_size, 0);
}

// ---------------------------------------------------------------------------
// accThreadName()
// ---------------------------------------------------------------------------

TEST(MlOptimiserTest, AccThreadName_Thread0)
{
    MlOptimiser opt;
    std::string name = opt.accThreadName(0);
    EXPECT_FALSE(name.empty());
    EXPECT_NE(name.find("0"), std::string::npos);
}

TEST(MlOptimiserTest, AccThreadName_Thread3)
{
    MlOptimiser opt;
    std::string name = opt.accThreadName(3);
    EXPECT_NE(name.find("3"), std::string::npos);
}

TEST(MlOptimiserTest, AccThreadName_DifferentThreadsDistinct)
{
    MlOptimiser opt;
    EXPECT_NE(opt.accThreadName(0), opt.accThreadName(1));
}

// ---------------------------------------------------------------------------
// gpuDeviceShareAt()
// ---------------------------------------------------------------------------

TEST(MlOptimiserTest, GpuDeviceShareAt_ReturnsOne)
{
    MlOptimiser opt;
    // Base-class implementation always returns 1 (overridden in MpiOptimiser)
    EXPECT_EQ(opt.gpuDeviceShareAt(0), 1);
}

TEST(MlOptimiserTest, GpuDeviceShareAt_AnyIndexReturnsOne)
{
    MlOptimiser opt;
    EXPECT_EQ(opt.gpuDeviceShareAt(5),  1);
    EXPECT_EQ(opt.gpuDeviceShareAt(-1), 1);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
