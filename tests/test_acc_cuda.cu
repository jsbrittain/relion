/*
 * Unit tests for src/acc/cuda/.
 *
 * Covers:
 *   cuda_benchmark_utils.cu  — relion_timer::cuda_benchmark_find_id()
 *   cuda_helper_functions.cu — mapWeights()
 *   cuda_settings.h          — compile-time configuration constants
 *   cuda_kernels/helper.cuh  — GPU kernel launch verification (GPU required)
 *
 * CPU-side tests run without a GPU.
 * GPU kernel tests skip gracefully when no CUDA device is available.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <limits>
#include <sys/stat.h>
#include <cmath>

// XFLOAT: float (or double with ACC_DOUBLE_PRECISION), no CUDA headers needed
#include "src/acc/settings.h"

// GPU kernel definitions (inline templates in helper.cuh)
#include "src/acc/cuda/cuda_kernels/helper.cuh"

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
// GPU kernel launch verification
//
// All tests in this fixture skip automatically when no CUDA device is present.
// They verify that kernels launch without error and produce correct output
// for the simplest possible inputs.
// ---------------------------------------------------------------------------

class GpuKernelTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        int n = 0;
        cudaGetDeviceCount(&n);
        if (n == 0)
            GTEST_SKIP() << "No CUDA device available";
        cudaSetDevice(0);
    }
};

// Helper: allocate device memory, fill from host, return device pointer.
// Caller must cudaFree the returned pointer.
static XFLOAT *makeDeviceArray(const std::vector<XFLOAT> &h)
{
    XFLOAT *d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(XFLOAT));
    cudaMemcpy(d, h.data(), h.size() * sizeof(XFLOAT), cudaMemcpyHostToDevice);
    return d;
}

// ---------------------------------------------------------------------------
// cuda_kernel_exponentiate<XFLOAT>
// kernel: g_array[i] = exp(g_array[i] + add)
// ---------------------------------------------------------------------------

TEST_F(GpuKernelTest, Exponentiate_ZeroAdd_YieldsOne)
{
    const int N = 8;
    std::vector<XFLOAT> h(N, 0.f);
    XFLOAT *d = makeDeviceArray(h);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cuda_kernel_exponentiate<XFLOAT><<<blocks, BLOCK_SIZE>>>(d, (XFLOAT)0, (size_t)N);
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    cudaMemcpy(h.data(), d, N * sizeof(XFLOAT), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i)
        EXPECT_NEAR(h[i], 1.f, 1e-5f);
    cudaFree(d);
}

TEST_F(GpuKernelTest, Exponentiate_AddShiftsExponent)
{
    // g_array = [0, 0, 0], add = 1.0 → exp(1) for each element
    const int N = 4;
    std::vector<XFLOAT> h(N, 0.f);
    XFLOAT *d = makeDeviceArray(h);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cuda_kernel_exponentiate<XFLOAT><<<blocks, BLOCK_SIZE>>>(d, (XFLOAT)1, (size_t)N);
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    cudaMemcpy(h.data(), d, N * sizeof(XFLOAT), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i)
        EXPECT_NEAR(h[i], std::exp(1.f), 1e-4f);
    cudaFree(d);
}

TEST_F(GpuKernelTest, Exponentiate_LargeNegative_YieldsZero)
{
    // Values far below -88 (float) / -700 (double) saturate to 0.
    const int N = 4;
    std::vector<XFLOAT> h(N, -200.f);
    XFLOAT *d = makeDeviceArray(h);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cuda_kernel_exponentiate<XFLOAT><<<blocks, BLOCK_SIZE>>>(d, (XFLOAT)0, (size_t)N);
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    cudaMemcpy(h.data(), d, N * sizeof(XFLOAT), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i)
        EXPECT_EQ(h[i], (XFLOAT)0);
    cudaFree(d);
}

// ---------------------------------------------------------------------------
// CudaKernels::cuda_kernel_multi<XFLOAT> — in-place A[i] = A[i]*S
// ---------------------------------------------------------------------------

TEST_F(GpuKernelTest, Multi_InPlace_ScalesElements)
{
    const int N = 6;
    std::vector<XFLOAT> h = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    XFLOAT *d = makeDeviceArray(h);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CudaKernels::cuda_kernel_multi<XFLOAT><<<blocks, BLOCK_SIZE>>>(d, (XFLOAT)3, N);
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    cudaMemcpy(h.data(), d, N * sizeof(XFLOAT), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i)
        EXPECT_NEAR(h[i], (i + 1) * 3.f, 1e-5f);
    cudaFree(d);
}

TEST_F(GpuKernelTest, Multi_InPlace_ByZero_YieldsZeros)
{
    const int N = 4;
    std::vector<XFLOAT> h = {1.f, 2.f, 3.f, 4.f};
    XFLOAT *d = makeDeviceArray(h);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CudaKernels::cuda_kernel_multi<XFLOAT><<<blocks, BLOCK_SIZE>>>(d, (XFLOAT)0, N);
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    cudaMemcpy(h.data(), d, N * sizeof(XFLOAT), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i)
        EXPECT_EQ(h[i], (XFLOAT)0);
    cudaFree(d);
}

// ---------------------------------------------------------------------------
// CudaKernels::cuda_kernel_add<XFLOAT> — in-place A[i] += S
// ---------------------------------------------------------------------------

TEST_F(GpuKernelTest, Add_InPlace_AddsScalar)
{
    const int N = 5;
    std::vector<XFLOAT> h = {0.f, 1.f, 2.f, 3.f, 4.f};
    XFLOAT *d = makeDeviceArray(h);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CudaKernels::cuda_kernel_add<XFLOAT><<<blocks, BLOCK_SIZE>>>(d, (XFLOAT)10, N);
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    cudaMemcpy(h.data(), d, N * sizeof(XFLOAT), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i)
        EXPECT_NEAR(h[i], i + 10.f, 1e-5f);
    cudaFree(d);
}

TEST_F(GpuKernelTest, Add_InPlace_AddZero_Unchanged)
{
    const int N = 4;
    std::vector<XFLOAT> h = {5.f, 6.f, 7.f, 8.f};
    XFLOAT *d = makeDeviceArray(h);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CudaKernels::cuda_kernel_add<XFLOAT><<<blocks, BLOCK_SIZE>>>(d, (XFLOAT)0, N);
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    cudaMemcpy(h.data(), d, N * sizeof(XFLOAT), cudaMemcpyDeviceToHost);
    EXPECT_NEAR(h[0], 5.f, 1e-5f);
    EXPECT_NEAR(h[1], 6.f, 1e-5f);
    EXPECT_NEAR(h[2], 7.f, 1e-5f);
    EXPECT_NEAR(h[3], 8.f, 1e-5f);
    cudaFree(d);
}

// ---------------------------------------------------------------------------
// cuda_kernel_multi<XFLOAT> — out-of-place OUT[i] = A[i]*S
// ---------------------------------------------------------------------------

TEST_F(GpuKernelTest, Multi_OutOfPlace_ScalesIntoOutput)
{
    const int N = 4;
    std::vector<XFLOAT> hA = {1.f, 2.f, 3.f, 4.f};
    std::vector<XFLOAT> hOut(N, 0.f);
    XFLOAT *dA   = makeDeviceArray(hA);
    XFLOAT *dOut = makeDeviceArray(hOut);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cuda_kernel_multi<XFLOAT><<<blocks, BLOCK_SIZE>>>(dA, dOut, (XFLOAT)2, N);
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    cudaMemcpy(hOut.data(), dOut, N * sizeof(XFLOAT), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i)
        EXPECT_NEAR(hOut[i], (i + 1) * 2.f, 1e-5f);
    cudaFree(dA);
    cudaFree(dOut);
}

// ---------------------------------------------------------------------------
// cuda_kernel_make_eulers_2D<false> — rotation matrix from angles
// For alpha=0: ca=1, sa=0 → matrix [1,0,0, -0,1,0, 0,0,1] (entries 0,1,2…8)
// ---------------------------------------------------------------------------

TEST_F(GpuKernelTest, MakeEulers2D_ZeroAngle_IsIdentityLike)
{
    const unsigned nOrient = 1;
    std::vector<XFLOAT> hAlpha(nOrient, 0.f);   // alpha = 0 degrees
    std::vector<XFLOAT> hEulers(9 * nOrient, 0.f);

    XFLOAT *dAlpha  = makeDeviceArray(hAlpha);
    XFLOAT *dEulers = makeDeviceArray(hEulers);

    int blocks = (nOrient + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cuda_kernel_make_eulers_2D<false><<<blocks, BLOCK_SIZE>>>(dAlpha, dEulers, nOrient);
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    cudaMemcpy(hEulers.data(), dEulers, 9 * nOrient * sizeof(XFLOAT), cudaMemcpyDeviceToHost);

    // For alpha=0: ca=1, sa=0
    EXPECT_NEAR(hEulers[0],  1.f, 1e-5f);  // cos(0)
    EXPECT_NEAR(hEulers[1],  0.f, 1e-5f);  // sin(0)
    EXPECT_NEAR(hEulers[3], -0.f, 1e-5f);  // -sin(0)
    EXPECT_NEAR(hEulers[4],  1.f, 1e-5f);  // cos(0)
    EXPECT_NEAR(hEulers[8],  1.f, 1e-5f);  // z-diagonal

    cudaFree(dAlpha);
    cudaFree(dEulers);
}

TEST_F(GpuKernelTest, MakeEulers2D_90Degrees_SineAndCosine)
{
    const unsigned nOrient = 1;
    std::vector<XFLOAT> hAlpha(nOrient, 90.f);  // alpha = 90 degrees
    std::vector<XFLOAT> hEulers(9 * nOrient, 0.f);

    XFLOAT *dAlpha  = makeDeviceArray(hAlpha);
    XFLOAT *dEulers = makeDeviceArray(hEulers);

    int blocks = (nOrient + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cuda_kernel_make_eulers_2D<false><<<blocks, BLOCK_SIZE>>>(dAlpha, dEulers, nOrient);
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    cudaMemcpy(hEulers.data(), dEulers, 9 * nOrient * sizeof(XFLOAT), cudaMemcpyDeviceToHost);

    // For alpha=90: ca=cos(90°)=0, sa=sin(90°)=1
    EXPECT_NEAR(hEulers[0],  0.f, 1e-5f);  // ca
    EXPECT_NEAR(hEulers[1],  1.f, 1e-5f);  // sa
    EXPECT_NEAR(hEulers[3], -1.f, 1e-5f);  // -sa
    EXPECT_NEAR(hEulers[4],  0.f, 1e-5f);  // ca

    cudaFree(dAlpha);
    cudaFree(dEulers);
}

// ---------------------------------------------------------------------------
// CudaKernels::cuda_kernel_translate2D<XFLOAT>
// Copies input pixels to output shifted by (dx, dy).
// ---------------------------------------------------------------------------

TEST_F(GpuKernelTest, Translate2D_ZeroShift_CopiesInput)
{
    // 2x2 image, shift (0,0) → output == input
    const int xdim = 2, ydim = 2;
    const int N = xdim * ydim;
    std::vector<XFLOAT> hIn  = {1.f, 2.f, 3.f, 4.f};
    std::vector<XFLOAT> hOut(N, 0.f);
    XFLOAT *dIn  = makeDeviceArray(hIn);
    XFLOAT *dOut = makeDeviceArray(hOut);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CudaKernels::cuda_kernel_translate2D<XFLOAT><<<blocks, BLOCK_SIZE>>>(
        dIn, dOut, N, xdim, ydim, 0, 0);
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    cudaMemcpy(hOut.data(), dOut, N * sizeof(XFLOAT), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i)
        EXPECT_NEAR(hOut[i], hIn[i], 1e-5f);
    cudaFree(dIn);
    cudaFree(dOut);
}

TEST_F(GpuKernelTest, Translate2D_ShiftRight_MovesPixels)
{
    // 4x1 image [1,2,3,4], shift (1,0) → pixel at x maps to x+1
    // output[1]=1, output[2]=2, output[3]=3; output[0] untouched (0)
    const int xdim = 4, ydim = 1;
    const int N = xdim * ydim;
    std::vector<XFLOAT> hIn  = {1.f, 2.f, 3.f, 4.f};
    std::vector<XFLOAT> hOut(N, 0.f);
    XFLOAT *dIn  = makeDeviceArray(hIn);
    XFLOAT *dOut = makeDeviceArray(hOut);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CudaKernels::cuda_kernel_translate2D<XFLOAT><<<blocks, BLOCK_SIZE>>>(
        dIn, dOut, N, xdim, ydim, 1, 0);
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    cudaMemcpy(hOut.data(), dOut, N * sizeof(XFLOAT), cudaMemcpyDeviceToHost);
    EXPECT_NEAR(hOut[0], 0.f, 1e-5f);  // no source maps here (dx=1 wraps out of bounds at x=3)
    EXPECT_NEAR(hOut[1], 1.f, 1e-5f);  // pixel 0 shifted to index 1
    EXPECT_NEAR(hOut[2], 2.f, 1e-5f);
    EXPECT_NEAR(hOut[3], 3.f, 1e-5f);
    cudaFree(dIn);
    cudaFree(dOut);
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
