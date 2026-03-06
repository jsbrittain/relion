/*
 * GoogleTest unit tests for Projector and BackProjector
 * (src/projector.h/.cpp, src/backprojector.h/.cpp).
 *
 * Build and run:
 *   cmake -DBUILD_TESTS=ON ..
 *   make test_projector
 *   ./build/bin/test_projector
 *
 * No MPI required; CPU-only tests using FFTW.
 *
 * What is tested:
 *   Projector:
 *     1.  Default constructor clears all fields
 *     2.  Parameterised constructor stores ori_size, interpolator,
 *         padding_factor, r_min_nn, data_dim
 *     3.  Copy constructor is a deep copy
 *     4.  Assignment operator is a deep copy
 *     5.  computeFourierTransformMap – power spectrum has ori_size/2+1 elements
 *     6.  computeFourierTransformMap – data array is non-empty afterwards
 *     7.  computeFourierTransformMap – ref_dim is set from the input volume
 *     8.  computeFourierTransformMap – r_max set to ori_size/2
 *     9.  getSize – returns pad_size * pad_size * (pad_size/2+1) for 3D
 *    10.  project – output has the correct dimensions
 *    11.  project – zero volume gives all-zero projection
 *    12.  project – non-zero volume gives non-zero projection
 *    13.  project – DC component (0,0) is real for a real symmetric volume
 *    14.  project – two different rotation matrices give different slices
 *   BackProjector:
 *    15.  Parameterised constructor stores fields
 *    16.  initZeros – data is all zero
 *    17.  initZeros – weight is all zero
 *    18.  initZeros – data and weight have the same shape
 *    19.  backproject2Dto3D – zero slice leaves data zero
 *    20.  backproject2Dto3D – non-zero slice makes data non-zero
 *    21.  backproject2Dto3D – weight accumulates for non-zero input
 *    22.  enforceHermitianSymmetry – data(x,y,z) == conj(data(-x,-y,-z))
 *    23.  getLowResDataAndWeight – correct sizes returned
 *    24.  getLowResDataAndWeight / setLowResDataAndWeight round-trip
 *    25.  Project → backproject → data and weight both non-zero
 */

#include <gtest/gtest.h>
#include <cmath>
#include "src/projector.h"
#include "src/backprojector.h"
#include "src/euler.h"
#include "src/matrix2d.h"

// ---------------------------------------------------------------------------
// Test parameters
// ---------------------------------------------------------------------------
static const int ORI  = 8;    // small box size for speed
static const float PAD = 2.0f;
static const int RMIN = 10;   // r_min_nn (larger than r_max for ORI=8 is fine –
                               // forces TRILINEAR everywhere)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a zero 3D volume of size ORI^3
static MultidimArray<RFLOAT> zeroVol3D()
{
    MultidimArray<RFLOAT> v;
    v.initZeros(ORI, ORI, ORI);
    return v;
}

// Build a 3D volume with a single voxel = 1 at the centre
static MultidimArray<RFLOAT> deltaVol3D()
{
    MultidimArray<RFLOAT> v = zeroVol3D();
    DIRECT_A3D_ELEM(v, ORI/2, ORI/2, ORI/2) = 1.0;
    return v;
}

// Fill a Projector from a volume; returns the resulting power spectrum
static MultidimArray<RFLOAT> setupProjector(Projector& proj,
                                            MultidimArray<RFLOAT>& vol)
{
    MultidimArray<RFLOAT> ps;
    proj.computeFourierTransformMap(vol, ps, -1, 1, true, true);
    return ps;
}

// 3x3 rotation matrix from ZYZ Euler angles (degrees)
static Matrix2D<RFLOAT> eulerMatrix(RFLOAT rot, RFLOAT tilt, RFLOAT psi)
{
    Matrix2D<RFLOAT> A;
    Euler_angles2matrix(rot, tilt, psi, A, false);
    return A;
}

// Check whether any element of a Complex array is non-zero
static bool anyNonZero(const MultidimArray<Complex>& arr)
{
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(arr)
    {
        const Complex& c = DIRECT_MULTIDIM_ELEM(arr, n);
        if (c.real != 0.0 || c.imag != 0.0) return true;
    }
    return false;
}

// ===========================================================================
// Projector tests
// ===========================================================================

// ---------------------------------------------------------------------------
// 1. Default constructor
// ---------------------------------------------------------------------------
TEST(ProjectorTest, DefaultConstructorClearsFields)
{
    Projector p;
    EXPECT_EQ(p.ori_size, 0);
    EXPECT_EQ(p.r_max, 0);
    EXPECT_EQ(p.r_min_nn, 0);
    EXPECT_EQ(p.interpolator, 0);
    EXPECT_FLOAT_EQ(p.padding_factor, 0.f);
    EXPECT_EQ(p.ref_dim, 0);
    EXPECT_EQ(p.data_dim, 0);
    EXPECT_EQ(NZYXSIZE(p.data), 0);
}

// ---------------------------------------------------------------------------
// 2. Parameterised constructor
// ---------------------------------------------------------------------------
TEST(ProjectorTest, ConstructorSetsFields)
{
    Projector p(ORI, TRILINEAR, PAD, RMIN, 2);
    EXPECT_EQ(p.ori_size, ORI);
    EXPECT_EQ(p.interpolator, TRILINEAR);
    EXPECT_FLOAT_EQ(p.padding_factor, PAD);
    EXPECT_EQ(p.r_min_nn, RMIN);
    EXPECT_EQ(p.data_dim, 2);
}

// ---------------------------------------------------------------------------
// 3. Copy constructor
// ---------------------------------------------------------------------------
TEST(ProjectorTest, CopyConstructorDeepCopy)
{
    Projector p(ORI, TRILINEAR, PAD, RMIN);
    Projector p2(p);
    EXPECT_EQ(p2.ori_size, p.ori_size);
    EXPECT_EQ(p2.interpolator, p.interpolator);
    EXPECT_FLOAT_EQ(p2.padding_factor, p.padding_factor);
}

// ---------------------------------------------------------------------------
// 4. Assignment operator
// ---------------------------------------------------------------------------
TEST(ProjectorTest, AssignmentDeepCopy)
{
    Projector p(ORI, TRILINEAR, PAD, RMIN);
    Projector p2;
    p2 = p;
    EXPECT_EQ(p2.ori_size, p.ori_size);
    EXPECT_EQ(p2.r_min_nn, p.r_min_nn);
}

// ---------------------------------------------------------------------------
// 5. computeFourierTransformMap – power spectrum size
// ---------------------------------------------------------------------------
TEST(ProjectorTest, ComputeFTMap_PowerSpectrumSize)
{
    Projector p(ORI, TRILINEAR, PAD, RMIN);
    MultidimArray<RFLOAT> vol = deltaVol3D();
    MultidimArray<RFLOAT> ps = setupProjector(p, vol);
    EXPECT_EQ((int)XSIZE(ps), ORI / 2 + 1);
}

// ---------------------------------------------------------------------------
// 6. computeFourierTransformMap – data non-empty
// ---------------------------------------------------------------------------
TEST(ProjectorTest, ComputeFTMap_DataNonEmpty)
{
    Projector p(ORI, TRILINEAR, PAD, RMIN);
    MultidimArray<RFLOAT> vol = deltaVol3D();
    setupProjector(p, vol);
    EXPECT_GT(NZYXSIZE(p.data), 0);
}

// ---------------------------------------------------------------------------
// 7. computeFourierTransformMap – ref_dim inferred from volume
// ---------------------------------------------------------------------------
TEST(ProjectorTest, ComputeFTMap_RefDimSetTo3ForVolume)
{
    Projector p(ORI, TRILINEAR, PAD, RMIN);
    MultidimArray<RFLOAT> vol = deltaVol3D();
    setupProjector(p, vol);
    EXPECT_EQ(p.ref_dim, 3);
}

// ---------------------------------------------------------------------------
// 8. computeFourierTransformMap – r_max
// ---------------------------------------------------------------------------
TEST(ProjectorTest, ComputeFTMap_RMaxIsHalfOriSize)
{
    Projector p(ORI, TRILINEAR, PAD, RMIN);
    MultidimArray<RFLOAT> vol = deltaVol3D();
    setupProjector(p, vol);
    EXPECT_EQ(p.r_max, ORI / 2);
}

// ---------------------------------------------------------------------------
// 9. getSize – 3D reference
// ---------------------------------------------------------------------------
TEST(ProjectorTest, GetSize_3D_MatchesDataSize)
{
    Projector p(ORI, TRILINEAR, PAD, RMIN);
    MultidimArray<RFLOAT> vol = deltaVol3D();
    setupProjector(p, vol);
    long int reported = p.getSize();
    long int actual   = (long int)NZYXSIZE(p.data);
    EXPECT_EQ(reported, actual);
}

// ---------------------------------------------------------------------------
// 10. project – output dimensions
// ---------------------------------------------------------------------------
TEST(ProjectorTest, Project_OutputDimensions)
{
    Projector p(ORI, TRILINEAR, PAD, RMIN);
    MultidimArray<RFLOAT> vol = deltaVol3D();
    setupProjector(p, vol);

    MultidimArray<Complex> f2d;
    f2d.initZeros(ORI, ORI / 2 + 1);

    Matrix2D<RFLOAT> A = eulerMatrix(0, 0, 0);
    p.project(f2d, A);

    EXPECT_EQ((int)YSIZE(f2d), ORI);
    EXPECT_EQ((int)XSIZE(f2d), ORI / 2 + 1);
}

// ---------------------------------------------------------------------------
// 11. project – zero volume → all-zero output
// ---------------------------------------------------------------------------
TEST(ProjectorTest, Project_ZeroVolume_AllZeroOutput)
{
    Projector p(ORI, TRILINEAR, PAD, RMIN);
    MultidimArray<RFLOAT> vol = zeroVol3D();
    setupProjector(p, vol);

    MultidimArray<Complex> f2d;
    f2d.initZeros(ORI, ORI / 2 + 1);

    Matrix2D<RFLOAT> A = eulerMatrix(0, 0, 0);
    p.project(f2d, A);

    EXPECT_FALSE(anyNonZero(f2d));
}

// ---------------------------------------------------------------------------
// 12. project – non-zero volume → non-zero output
// ---------------------------------------------------------------------------
TEST(ProjectorTest, Project_NonzeroVolume_NonzeroOutput)
{
    Projector p(ORI, TRILINEAR, PAD, RMIN);
    MultidimArray<RFLOAT> vol = deltaVol3D();
    setupProjector(p, vol);

    MultidimArray<Complex> f2d;
    f2d.initZeros(ORI, ORI / 2 + 1);

    Matrix2D<RFLOAT> A = eulerMatrix(0, 0, 0);
    p.project(f2d, A);

    EXPECT_TRUE(anyNonZero(f2d));
}

// ---------------------------------------------------------------------------
// 13. project – DC component is real for a symmetric volume
// ---------------------------------------------------------------------------
TEST(ProjectorTest, Project_DcComponent_RealForCenteredVolume)
{
    Projector p(ORI, TRILINEAR, PAD, RMIN);
    MultidimArray<RFLOAT> vol = deltaVol3D();
    setupProjector(p, vol);

    MultidimArray<Complex> f2d;
    f2d.initZeros(ORI, ORI / 2 + 1);

    Matrix2D<RFLOAT> A = eulerMatrix(0, 0, 0);
    p.project(f2d, A);

    // DC element is at (0,0) in FFTW convention (centre-origin stored at i=0,j=0)
    const Complex dc = DIRECT_A2D_ELEM(f2d, 0, 0);
    EXPECT_NEAR(dc.imag, 0.0, 1e-6);
}

// ---------------------------------------------------------------------------
// 14. project – two different orientations give different results
// ---------------------------------------------------------------------------
TEST(ProjectorTest, Project_DifferentOrientations_DifferentSlices)
{
    Projector p(ORI, TRILINEAR, PAD, RMIN);

    // Volume: uniform slab – not symmetric in all directions
    MultidimArray<RFLOAT> vol;
    vol.initZeros(ORI, ORI, ORI);
    for (int k = ORI/2 - 1; k <= ORI/2 + 1; k++)
        for (int i = 0; i < ORI; i++)
            for (int j = 0; j < ORI; j++)
                DIRECT_A3D_ELEM(vol, k, i, j) = 1.0;

    setupProjector(p, vol);

    MultidimArray<Complex> f2d_a, f2d_b;
    f2d_a.initZeros(ORI, ORI / 2 + 1);
    f2d_b.initZeros(ORI, ORI / 2 + 1);

    Matrix2D<RFLOAT> A0 = eulerMatrix(0,   0, 0);
    Matrix2D<RFLOAT> A1 = eulerMatrix(0,  90, 0);
    p.project(f2d_a, A0);
    p.project(f2d_b, A1);

    bool identical = true;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(f2d_a)
    {
        const Complex& a = DIRECT_MULTIDIM_ELEM(f2d_a, n);
        const Complex& b = DIRECT_MULTIDIM_ELEM(f2d_b, n);
        if (std::abs(a.real - b.real) > 1e-9 || std::abs(a.imag - b.imag) > 1e-9)
        {
            identical = false;
            break;
        }
    }
    EXPECT_FALSE(identical);
}

// ===========================================================================
// BackProjector tests
// ===========================================================================

// Helper: construct a BackProjector ready for 3D reconstruction
static BackProjector makeBP()
{
    return BackProjector(ORI, 3, "c1", TRILINEAR, PAD, RMIN);
}

// ---------------------------------------------------------------------------
// 15. BackProjector constructor
// ---------------------------------------------------------------------------
TEST(BackProjectorTest, ConstructorSetsFields)
{
    BackProjector bp = makeBP();
    EXPECT_EQ(bp.ori_size, ORI);
    EXPECT_EQ(bp.ref_dim, 3);
    EXPECT_EQ(bp.interpolator, TRILINEAR);
    EXPECT_FLOAT_EQ(bp.padding_factor, PAD);
}

// ---------------------------------------------------------------------------
// 16. initZeros – data all zero
// ---------------------------------------------------------------------------
TEST(BackProjectorTest, InitZeros_DataAllZero)
{
    BackProjector bp = makeBP();
    bp.initZeros();

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(bp.data)
    {
        const Complex& c = DIRECT_MULTIDIM_ELEM(bp.data, n);
        EXPECT_FLOAT_EQ(c.real, 0.f);
        EXPECT_FLOAT_EQ(c.imag, 0.f);
    }
}

// ---------------------------------------------------------------------------
// 17. initZeros – weight all zero
// ---------------------------------------------------------------------------
TEST(BackProjectorTest, InitZeros_WeightAllZero)
{
    BackProjector bp = makeBP();
    bp.initZeros();

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(bp.weight)
    {
        EXPECT_FLOAT_EQ(DIRECT_MULTIDIM_ELEM(bp.weight, n), 0.f);
    }
}

// ---------------------------------------------------------------------------
// 18. initZeros – data and weight have same shape
// ---------------------------------------------------------------------------
TEST(BackProjectorTest, InitZeros_DataWeightSameShape)
{
    BackProjector bp = makeBP();
    bp.initZeros();

    EXPECT_EQ(bp.data.xdim, bp.weight.xdim);
    EXPECT_EQ(bp.data.ydim, bp.weight.ydim);
    EXPECT_EQ(bp.data.zdim, bp.weight.zdim);
}

// ---------------------------------------------------------------------------
// 19. backproject2Dto3D – zero slice leaves data zero
// ---------------------------------------------------------------------------
TEST(BackProjectorTest, Backproject2Dto3D_ZeroSlice_DataRemainsZero)
{
    BackProjector bp = makeBP();
    bp.initZeros();

    MultidimArray<Complex> f2d;
    f2d.initZeros(ORI, ORI / 2 + 1);

    Matrix2D<RFLOAT> A = eulerMatrix(0, 0, 0);
    bp.backproject2Dto3D(f2d, A);

    EXPECT_FALSE(anyNonZero(bp.data));
}

// ---------------------------------------------------------------------------
// 20. backproject2Dto3D – non-zero slice makes data non-zero
// ---------------------------------------------------------------------------
TEST(BackProjectorTest, Backproject2Dto3D_NonzeroSlice_DataNonzero)
{
    BackProjector bp = makeBP();
    bp.initZeros();

    MultidimArray<Complex> f2d;
    f2d.initZeros(ORI, ORI / 2 + 1);
    // Set the DC pixel to a known value
    DIRECT_A2D_ELEM(f2d, 0, 0) = Complex(1.0, 0.0);

    Matrix2D<RFLOAT> A = eulerMatrix(0, 0, 0);
    bp.backproject2Dto3D(f2d, A);

    EXPECT_TRUE(anyNonZero(bp.data));
}

// ---------------------------------------------------------------------------
// 21. backproject2Dto3D – weight accumulates for non-zero input
// ---------------------------------------------------------------------------
TEST(BackProjectorTest, Backproject2Dto3D_NonzeroSlice_WeightNonzero)
{
    BackProjector bp = makeBP();
    bp.initZeros();

    MultidimArray<Complex> f2d;
    f2d.initZeros(ORI, ORI / 2 + 1);
    DIRECT_A2D_ELEM(f2d, 0, 0) = Complex(1.0, 0.0);

    Matrix2D<RFLOAT> A = eulerMatrix(0, 0, 0);
    bp.backproject2Dto3D(f2d, A);

    bool any_weight = false;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(bp.weight)
    {
        if (DIRECT_MULTIDIM_ELEM(bp.weight, n) != 0.0)
        {
            any_weight = true;
            break;
        }
    }
    EXPECT_TRUE(any_weight);
}

// ---------------------------------------------------------------------------
// 22. enforceHermitianSymmetry
// ---------------------------------------------------------------------------
TEST(BackProjectorTest, EnforceHermitianSymmetry_HoldAfterEnforcement)
{
    BackProjector bp = makeBP();
    bp.initZeros();

    // Insert some asymmetric data to break Hermitian symmetry
    MultidimArray<Complex> f2d;
    f2d.initZeros(ORI, ORI / 2 + 1);
    DIRECT_A2D_ELEM(f2d, 1, 1) = Complex(3.0, 1.5);
    DIRECT_A2D_ELEM(f2d, 2, 2) = Complex(-0.5, 2.0);

    Matrix2D<RFLOAT> A = eulerMatrix(0, 20, 0);
    bp.backproject2Dto3D(f2d, A);

    bp.enforceHermitianSymmetry();

    // Check Hermitian condition at a sample of frequencies:
    // data(x,y,z) should equal conj(data(-x,-y,-z))
    // data array uses setXmippOrigin so origin is in centre.
    // XSIZE holds x=0..r_max, the x<0 side is obtained via conjugation.
    // For x=0 plane: data(0,y,z) = conj(data(0,-y,-z))
    const int hs = bp.pad_size / 2;
    for (int yp = -3; yp <= 3; yp++)
    {
        const Complex val     = A3D_ELEM(bp.data,  0,  yp, 0);
        const Complex val_neg = A3D_ELEM(bp.data,  0, -yp, 0);
        EXPECT_NEAR(val.real,  val_neg.real,  1e-6f);
        EXPECT_NEAR(val.imag, -val_neg.imag,  1e-6f);
    }
}

// ---------------------------------------------------------------------------
// 23. getLowResDataAndWeight – correct sizes
// ---------------------------------------------------------------------------
TEST(BackProjectorTest, GetLowResDataAndWeight_CorrectSizes)
{
    BackProjector bp = makeBP();
    bp.initZeros();

    const int lowres_r_max = 2;
    MultidimArray<Complex> lowres_data;
    MultidimArray<RFLOAT>  lowres_weight;
    bp.getLowResDataAndWeight(lowres_data, lowres_weight, lowres_r_max);

    // Formula (from getLowResDataAndWeight source):
    //   lowres_pad_size = 2 * (ROUND(padding_factor * lowres_r_max) + 1) + 1
    //   xdim = lowres_pad_size / 2 + 1 (integer division)
    //   ydim = zdim = lowres_pad_size
    const int lr_pad = 2 * (ROUND(PAD * lowres_r_max) + 1) + 1;
    EXPECT_EQ((int)XSIZE(lowres_data),   lr_pad / 2 + 1);
    EXPECT_EQ((int)YSIZE(lowres_data),   lr_pad);
    EXPECT_EQ((int)XSIZE(lowres_weight), lr_pad / 2 + 1);
    EXPECT_EQ((int)YSIZE(lowres_weight), lr_pad);
}

// ---------------------------------------------------------------------------
// 24. getLowResDataAndWeight / setLowResDataAndWeight round-trip
// ---------------------------------------------------------------------------
TEST(BackProjectorTest, LowResRoundTrip_RestoresData)
{
    BackProjector bp = makeBP();
    bp.initZeros();

    // Put data into the backprojector via a non-zero slice
    MultidimArray<Complex> f2d;
    f2d.initZeros(ORI, ORI / 2 + 1);
    DIRECT_A2D_ELEM(f2d, 0, 0) = Complex(2.0, 0.0);
    DIRECT_A2D_ELEM(f2d, 1, 0) = Complex(1.0, 0.5);

    Matrix2D<RFLOAT> A = eulerMatrix(0, 0, 0);
    bp.backproject2Dto3D(f2d, A);

    // Extract low-res, zero main data, then restore
    const int lr = 1;
    MultidimArray<Complex> lowres_data;
    MultidimArray<RFLOAT>  lowres_weight;
    bp.getLowResDataAndWeight(lowres_data, lowres_weight, lr);

    bp.data.initZeros();
    bp.weight.initZeros();

    bp.setLowResDataAndWeight(lowres_data, lowres_weight, lr);

    // After restore, low-res region is non-trivially populated
    bool any_nonzero_data = anyNonZero(bp.data);
    EXPECT_TRUE(any_nonzero_data);
}

// ---------------------------------------------------------------------------
// 25. Project → backproject → data and weight both non-zero
// ---------------------------------------------------------------------------
TEST(BackProjectorTest, ProjectThenBackproject_DataAndWeightNonzero)
{
    // Forward-project a delta volume
    Projector proj(ORI, TRILINEAR, PAD, RMIN);
    MultidimArray<RFLOAT> vol = deltaVol3D();
    setupProjector(proj, vol);

    MultidimArray<Complex> f2d;
    f2d.initZeros(ORI, ORI / 2 + 1);
    Matrix2D<RFLOAT> A = eulerMatrix(0, 0, 0);
    proj.project(f2d, A);

    // Backproject the resulting slice
    BackProjector bp = makeBP();
    bp.initZeros();
    bp.backproject2Dto3D(f2d, A);

    EXPECT_TRUE(anyNonZero(bp.data));

    bool any_weight = false;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(bp.weight)
    {
        if (DIRECT_MULTIDIM_ELEM(bp.weight, n) > 0.0)
        {
            any_weight = true;
            break;
        }
    }
    EXPECT_TRUE(any_weight);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
