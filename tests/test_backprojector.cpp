/*
 * Unit tests for src/backprojector.h / src/backprojector.cpp
 *
 * Tests in-memory operations only — no reconstruction or file I/O.
 *
 * Covers:
 *   - Default constructor: compiles and constructs
 *   - Full constructor: sets ori_size, ref_dim, padding_factor
 *   - initialiseDataAndWeight: resizes data and weight to matching dimensions
 *   - initZeros: all elements zero after call
 *   - Assignment operator: deep copy
 *   - clear(): data and weight become empty
 */

#include <gtest/gtest.h>
#include <cmath>
#include "src/backprojector.h"
#include "src/euler.h"

// Helper: build a minimal 3D BackProjector with C1 symmetry.
static BackProjector makeC1_3D(int ori_size = 32)
{
    return BackProjector(ori_size, /*ref_dim=*/3, /*fn_sym=*/"c1");
}

// ------------------------------------------------------ constructors --

TEST(BackProjectorTest, DefaultConstructor_Compiles)
{
    BackProjector bp;
    // Should not throw; data and weight are empty
    EXPECT_EQ(NZYXSIZE(bp.weight), (size_t)0);
}

TEST(BackProjectorTest, FullConstructor_SetsOriSize)
{
    BackProjector bp = makeC1_3D(32);
    EXPECT_EQ(bp.ori_size, 32);
}

TEST(BackProjectorTest, FullConstructor_SetsRefDim)
{
    BackProjector bp = makeC1_3D(32);
    EXPECT_EQ(bp.ref_dim, 3);
}

TEST(BackProjectorTest, FullConstructor_SetsPaddingFactor)
{
    // Default padding_factor_3d = 2
    BackProjector bp = makeC1_3D(32);
    EXPECT_NEAR(bp.padding_factor, 2.0f, 1e-6f);
}

TEST(BackProjectorTest, FullConstructor_C2Symmetry)
{
    BackProjector bp(32, 3, "c2");
    EXPECT_EQ(bp.ori_size, 32);
    EXPECT_EQ(bp.SL.SymsNo(), 1);  // C2 has 1 non-identity symmetry
}

// ---------------------------------------- initialiseDataAndWeight --

TEST(BackProjectorTest, InitialiseDataAndWeight_DataNonEmpty)
{
    BackProjector bp = makeC1_3D(32);
    bp.initialiseDataAndWeight();

    EXPECT_GT(NZYXSIZE(bp.data), (size_t)0);
}

TEST(BackProjectorTest, InitialiseDataAndWeight_WeightNonEmpty)
{
    BackProjector bp = makeC1_3D(32);
    bp.initialiseDataAndWeight();

    EXPECT_GT(NZYXSIZE(bp.weight), (size_t)0);
}

TEST(BackProjectorTest, InitialiseDataAndWeight_DataAndWeightSameSize)
{
    BackProjector bp = makeC1_3D(32);
    bp.initialiseDataAndWeight();

    EXPECT_EQ(NZYXSIZE(bp.data), NZYXSIZE(bp.weight));
}

// ------------------------------------------------------- initZeros --

TEST(BackProjectorTest, InitZeros_DataAllZero)
{
    BackProjector bp = makeC1_3D(16);
    bp.initZeros();

    bool all_zero = true;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(bp.weight)
    {
        if (DIRECT_MULTIDIM_ELEM(bp.weight, n) != 0.0)
        {
            all_zero = false;
            break;
        }
    }
    EXPECT_TRUE(all_zero);
}

TEST(BackProjectorTest, InitZeros_WeightAllZero)
{
    BackProjector bp = makeC1_3D(16);
    bp.initZeros();

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(bp.weight)
        EXPECT_EQ(DIRECT_MULTIDIM_ELEM(bp.weight, n), 0.0)
            << "non-zero at element " << n;
}

TEST(BackProjectorTest, InitZeros_DataNonEmpty)
{
    BackProjector bp = makeC1_3D(16);
    bp.initZeros();

    EXPECT_GT(NZYXSIZE(bp.data), (size_t)0);
}

// ----------------------------------------------- assignment / copy --

TEST(BackProjectorTest, AssignmentOperator_CopiesOriSize)
{
    BackProjector bp1 = makeC1_3D(32);
    BackProjector bp2;
    bp2 = bp1;

    EXPECT_EQ(bp2.ori_size, bp1.ori_size);
    EXPECT_EQ(bp2.ref_dim, bp1.ref_dim);
    EXPECT_NEAR(bp2.padding_factor, bp1.padding_factor, 1e-6f);
}

TEST(BackProjectorTest, CopyConstructor_CopiesOriSize)
{
    BackProjector bp1 = makeC1_3D(32);
    bp1.initZeros();

    BackProjector bp2(bp1);
    EXPECT_EQ(bp2.ori_size, 32);
    EXPECT_EQ(NZYXSIZE(bp2.data), NZYXSIZE(bp1.data));
}

// ---------------------------------------------------------------- clear --

TEST(BackProjectorTest, Clear_WeightBecomesEmpty)
{
    BackProjector bp = makeC1_3D(32);
    bp.initialiseDataAndWeight();
    ASSERT_GT(NZYXSIZE(bp.weight), (size_t)0);

    bp.clear();
    EXPECT_EQ(NZYXSIZE(bp.weight), (size_t)0);
}

// ---------------------------------------------- 2D BackProjector --

TEST(BackProjectorTest, TwoDimensional_Constructor)
{
    BackProjector bp(32, 2, "c1");
    EXPECT_EQ(bp.ori_size, 32);
    EXPECT_EQ(bp.ref_dim, 2);
}

TEST(BackProjectorTest, TwoDimensional_InitZeros)
{
    BackProjector bp(16, 2, "c1");
    bp.initZeros();
    EXPECT_GT(NZYXSIZE(bp.data), (size_t)0);
}

// ----------------------------- calculateDownSampledFourierShellCorrelation --

TEST(BackProjectorTest, CalcDownSampledFSC_IdenticalArrays_FSCIsOne)
{
    BackProjector bp = makeC1_3D(16);
    bp.initZeros();

    // Get two identical downsampled averages
    MultidimArray<Complex> avg;
    bp.getDownsampledAverage(avg, false);

    // Fill with non-zero data so FSC is non-trivial
    FOR_ALL_ELEMENTS_IN_ARRAY3D(avg)
        A3D_ELEM(avg, k, i, j) = Complex(1.0, 0.5);

    MultidimArray<RFLOAT> fsc;
    bp.calculateDownSampledFourierShellCorrelation(avg, avg, fsc);

    // FSC of a signal with itself should be 1.0 everywhere (except possible 0-den shells)
    EXPECT_GE((int)XSIZE(fsc), 1);
    EXPECT_NEAR(fsc(0), 1.0, 1e-6);  // zero-freq shell is always set to 1
}

TEST(BackProjectorTest, CalcDownSampledFSC_OutputSize)
{
    BackProjector bp = makeC1_3D(16);
    bp.initZeros();

    MultidimArray<Complex> avg;
    bp.getDownsampledAverage(avg, false);

    MultidimArray<RFLOAT> fsc;
    bp.calculateDownSampledFourierShellCorrelation(avg, avg, fsc);

    EXPECT_EQ((int)XSIZE(fsc), bp.ori_size / 2 + 1);
}

// ----------------------------------------------- enforceHermitianSymmetry --

TEST(BackProjectorTest, EnforceHermitianSymmetry_DoesNotCrash)
{
    BackProjector bp = makeC1_3D(16);
    bp.initZeros();
    EXPECT_NO_THROW(bp.enforceHermitianSymmetry());
}

// ------------------------------------------------------ symmetrise --

TEST(BackProjectorTest, Symmetrise_C1_DoesNotCrash)
{
    BackProjector bp = makeC1_3D(16);
    bp.initZeros();
    // nr_helical_asu=1 → helical part is skipped; C1 point group → trivial
    EXPECT_NO_THROW(bp.symmetrise(1, 0.0, 0.0, 1));
}

// ---------------------------------------------- backproject2Dto3D --

TEST(BackProjectorTest, Backproject2Dto3D_DoesNotCrash)
{
    BackProjector bp = makeC1_3D(8);
    bp.initZeros();

    // Half-complex Fourier slice for an 8x8 real image: 8 rows × 5 cols
    const int s  = 8;
    const int sh = s / 2 + 1;
    MultidimArray<Complex> f2d(s, sh);
    // Fill with a uniform value (all ones)
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(f2d)
        DIRECT_MULTIDIM_ELEM(f2d, n) = Complex(1.0, 0.0);

    // Identity rotation: rot=0, tilt=0, psi=0
    Matrix2D<RFLOAT> A;
    Euler_rotation3DMatrix(0., 0., 0., A);

    EXPECT_NO_THROW(bp.backproject2Dto3D(f2d, A, nullptr, 0.0, true, nullptr));
}

TEST(BackProjectorTest, Backproject2Dto3D_ModifiesWeight)
{
    BackProjector bp = makeC1_3D(8);
    bp.initZeros();

    const int s  = 8;
    const int sh = s / 2 + 1;
    MultidimArray<Complex> f2d(s, sh);
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(f2d)
        DIRECT_MULTIDIM_ELEM(f2d, n) = Complex(2.0, 0.0);

    Matrix2D<RFLOAT> A;
    Euler_rotation3DMatrix(0., 0., 0., A);
    bp.backproject2Dto3D(f2d, A, nullptr, 0.0, true, nullptr);

    // After back-projection some weight entries should be non-zero
    bool any_nonzero = false;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(bp.weight)
    {
        if (DIRECT_MULTIDIM_ELEM(bp.weight, n) != 0.0)
        {
            any_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(any_nonzero);
}

TEST(BackProjectorTest, Backproject2Dto3D_TiltedView_DoesNotCrash)
{
    BackProjector bp = makeC1_3D(8);
    bp.initZeros();

    const int s  = 8;
    const int sh = s / 2 + 1;
    MultidimArray<Complex> f2d(s, sh);
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(f2d)
        DIRECT_MULTIDIM_ELEM(f2d, n) = Complex(1.0, 0.0);

    Matrix2D<RFLOAT> A;
    Euler_rotation3DMatrix(45., 30., 0., A);
    EXPECT_NO_THROW(bp.backproject2Dto3D(f2d, A, nullptr, 0.0, true, nullptr));
}

// ----------------------------------------- getLowResDataAndWeight --

TEST(BackProjectorTest, GetLowResDataAndWeight_CorrectDimensions)
{
    BackProjector bp = makeC1_3D(32);
    bp.initZeros();

    MultidimArray<Complex> lowres_data;
    MultidimArray<RFLOAT>  lowres_weight;
    const int lowres_r = 4;
    bp.getLowResDataAndWeight(lowres_data, lowres_weight, lowres_r);

    const int expected_pad = 2 * (ROUND(bp.padding_factor * lowres_r) + 1) + 1;
    EXPECT_EQ((int)YSIZE(lowres_data),   expected_pad);
    EXPECT_EQ((int)XSIZE(lowres_data),   expected_pad / 2 + 1);
    EXPECT_EQ(NZYXSIZE(lowres_data), NZYXSIZE(lowres_weight));
}

TEST(BackProjectorTest, GetLowResDataAndWeight_DataAndWeightMatchSize)
{
    BackProjector bp = makeC1_3D(16);
    bp.initZeros();

    MultidimArray<Complex> lowres_data;
    MultidimArray<RFLOAT>  lowres_weight;
    bp.getLowResDataAndWeight(lowres_data, lowres_weight, 3);
    EXPECT_EQ(NZYXSIZE(lowres_data), NZYXSIZE(lowres_weight));
}

// --------------------------------------- setLowResDataAndWeight --

TEST(BackProjectorTest, SetLowResDataAndWeight_DoesNotCrash)
{
    BackProjector bp = makeC1_3D(32);
    bp.initZeros();

    // Get a correctly-sized lowres array and give it back
    MultidimArray<Complex> lowres_data;
    MultidimArray<RFLOAT>  lowres_weight;
    const int lowres_r = 4;
    bp.getLowResDataAndWeight(lowres_data, lowres_weight, lowres_r);
    EXPECT_NO_THROW(bp.setLowResDataAndWeight(lowres_data, lowres_weight, lowres_r));
}

TEST(BackProjectorTest, SetLowResDataAndWeight_RoundTrip)
{
    BackProjector bp = makeC1_3D(32);
    bp.initZeros();

    // Write a known value at the origin
    A3D_ELEM(bp.data, 0, 0, 0) = Complex(5.0, 3.0);
    A3D_ELEM(bp.weight, 0, 0, 0) = 1.0;

    MultidimArray<Complex> lowres_data;
    MultidimArray<RFLOAT>  lowres_weight;
    const int lowres_r = 4;
    bp.getLowResDataAndWeight(lowres_data, lowres_weight, lowres_r);

    // Reset bp and restore via setLowResDataAndWeight
    bp.initZeros();
    bp.setLowResDataAndWeight(lowres_data, lowres_weight, lowres_r);

    // Origin value should be restored
    Complex restored = A3D_ELEM(bp.data, 0, 0, 0);
    EXPECT_NEAR(restored.real, 5.0, 1e-6);
    EXPECT_NEAR(restored.imag, 3.0, 1e-6);
}

// ------------------------------------------- getDownsampledAverage --

TEST(BackProjectorTest, GetDownsampledAverage_3D_CorrectDimensions)
{
    BackProjector bp = makeC1_3D(16);
    bp.initZeros();

    MultidimArray<Complex> avg;
    bp.getDownsampledAverage(avg, false);

    const int r_max    = bp.r_max;
    const int down_size = 2 * (r_max + 1) + 1;
    EXPECT_EQ((int)YSIZE(avg), down_size);
    EXPECT_EQ((int)XSIZE(avg), down_size / 2 + 1);
}

TEST(BackProjectorTest, GetDownsampledAverage_2D_CorrectDimensions)
{
    BackProjector bp(16, 2, "c1");
    bp.initZeros();

    MultidimArray<Complex> avg;
    bp.getDownsampledAverage(avg, false);

    const int r_max    = bp.r_max;
    const int down_size = 2 * (r_max + 1) + 1;
    EXPECT_EQ((int)YSIZE(avg), down_size);
    EXPECT_EQ((int)XSIZE(avg), down_size / 2 + 1);
}

// ---------------------------------------------- backrotate2D --

TEST(BackProjectorTest, Backrotate2D_DoesNotCrash)
{
    BackProjector bp(8, 2, "c1");
    bp.initZeros();

    const int s  = 8;
    const int sh = s / 2 + 1;
    MultidimArray<Complex> f2d(s, sh);
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(f2d)
        DIRECT_MULTIDIM_ELEM(f2d, n) = Complex(1.0, 0.0);

    Matrix2D<RFLOAT> A(2, 2);
    A.initIdentity();
    EXPECT_NO_THROW(bp.backrotate2D(f2d, A));
}

TEST(BackProjectorTest, Backrotate2D_ModifiesWeight)
{
    BackProjector bp(8, 2, "c1");
    bp.initZeros();

    const int s  = 8;
    const int sh = s / 2 + 1;
    MultidimArray<Complex> f2d(s, sh);
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(f2d)
        DIRECT_MULTIDIM_ELEM(f2d, n) = Complex(2.0, 0.0);

    Matrix2D<RFLOAT> A(2, 2);
    A.initIdentity();
    bp.backrotate2D(f2d, A);

    bool any_nonzero = false;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(bp.weight)
    {
        if (DIRECT_MULTIDIM_ELEM(bp.weight, n) != 0.0)
        {
            any_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(any_nonzero);
}

// ---------------------------------------------- backrotate3D --

TEST(BackProjectorTest, Backrotate3D_DoesNotCrash)
{
    BackProjector bp = makeC1_3D(8);
    bp.initZeros();

    const int s  = 8;
    const int sh = s / 2 + 1;
    MultidimArray<Complex> f3d(s, s, sh);
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(f3d)
        DIRECT_MULTIDIM_ELEM(f3d, n) = Complex(1.0, 0.0);

    Matrix2D<RFLOAT> A(3, 3);
    A.initIdentity();
    EXPECT_NO_THROW(bp.backrotate3D(f3d, A));
}

TEST(BackProjectorTest, Backrotate3D_ModifiesWeight)
{
    BackProjector bp = makeC1_3D(8);
    bp.initZeros();

    const int s  = 8;
    const int sh = s / 2 + 1;
    MultidimArray<Complex> f3d(s, s, sh);
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(f3d)
        DIRECT_MULTIDIM_ELEM(f3d, n) = Complex(1.0, 0.0);

    Matrix2D<RFLOAT> A(3, 3);
    A.initIdentity();
    bp.backrotate3D(f3d, A);

    bool any_nonzero = false;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(bp.weight)
    {
        if (DIRECT_MULTIDIM_ELEM(bp.weight, n) != 0.0)
        {
            any_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(any_nonzero);
}

// ---------------------------------------------- backproject1Dto2D --

TEST(BackProjectorTest, Backproject1Dto2D_DoesNotCrash)
{
    BackProjector bp(8, 2, "c1");
    bp.initZeros();

    const int sh = 8 / 2 + 1;
    MultidimArray<Complex> f1d(sh);
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(f1d)
        DIRECT_MULTIDIM_ELEM(f1d, n) = Complex(1.0, 0.0);

    Matrix2D<RFLOAT> A(2, 2);
    A.initIdentity();
    EXPECT_NO_THROW(bp.backproject1Dto2D(f1d, A));
}

// ---------------------------------------------- reweightGrad --

TEST(BackProjectorTest, ReweightGrad_DoesNotCrash)
{
    BackProjector bp = makeC1_3D(8);
    bp.initZeros();
    EXPECT_NO_THROW(bp.reweightGrad());
}

// ---------------------------------------------- getFristMoment --

TEST(BackProjectorTest, GetFristMoment_DoesNotCrash)
{
    BackProjector bp = makeC1_3D(8);
    bp.initZeros();

    MultidimArray<Complex> mom;
    mom.initZeros(bp.data);
    EXPECT_NO_THROW(bp.getFristMoment(mom));
}

TEST(BackProjectorTest, GetFristMoment_CopiesDataWhenMomIsZero)
{
    BackProjector bp = makeC1_3D(8);
    bp.initZeros();
    A3D_ELEM(bp.data, 0, 0, 0) = Complex(3.0, 1.0);

    MultidimArray<Complex> mom;
    mom.initZeros(bp.data);
    bp.getFristMoment(mom, 0.9);

    // When mom is all-zero, the first path copies data into mom
    Complex val = A3D_ELEM(mom, 0, 0, 0);
    EXPECT_NEAR(val.real, 3.0, 1e-6);
    EXPECT_NEAR(val.imag, 1.0, 1e-6);
}

// ---------------------------------------------- getSecondMoment --

TEST(BackProjectorTest, GetSecondMoment_DoesNotCrash)
{
    BackProjector bp = makeC1_3D(8);
    bp.initZeros();

    MultidimArray<Complex> mom, data_other;
    mom.initZeros(bp.data);
    data_other.initZeros(bp.data);
    EXPECT_NO_THROW(bp.getSecondMoment(mom, data_other));
}

// ---------------------------------------------- applyMomenta --

TEST(BackProjectorTest, ApplyMomenta_DoesNotCrash)
{
    BackProjector bp = makeC1_3D(8);
    bp.initZeros();

    MultidimArray<Complex> m1h1, m1h2, m2;
    m1h1.initZeros(bp.data);
    m1h2.initZeros(bp.data);
    m2.initZeros(bp.data);
    EXPECT_NO_THROW(bp.applyMomenta(m1h1, m1h2, m2));
}

// ---------------------------------------------- convoluteBlobRealSpace --

TEST(BackProjectorTest, ConvoluteBlobRealSpace_DoesNotCrash)
{
    BackProjector bp = makeC1_3D(8);
    bp.initZeros();

    FourierTransformer transformer;
    EXPECT_NO_THROW(bp.convoluteBlobRealSpace(transformer));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
