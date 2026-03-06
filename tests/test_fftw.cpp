/*
 * Unit tests for FourierTransformer and FFTW free functions in src/fftw.h/.cpp.
 */

#include <gtest/gtest.h>
#include "src/fftw.h"
#include "src/multidim_array.h"
#include <cmath>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void fillRamp1D(MultidimArray<RFLOAT>& v, int n)
{
    v.resize(n);
    for (int i = 0; i < n; i++)
        DIRECT_A1D_ELEM(v, i) = static_cast<RFLOAT>(i + 1);
}

static void fillRamp2D(MultidimArray<RFLOAT>& v, int ny, int nx)
{
    v.resize(ny, nx);
    for (int i = 0; i < ny; i++)
        for (int j = 0; j < nx; j++)
            DIRECT_A2D_ELEM(v, i, j) = static_cast<RFLOAT>(i * nx + j + 1);
}

static void fillRamp3D(MultidimArray<RFLOAT>& v, int nz, int ny, int nx)
{
    v.resize(nz, ny, nx);
    for (int k = 0; k < nz; k++)
        for (int i = 0; i < ny; i++)
            for (int j = 0; j < nx; j++)
                DIRECT_A3D_ELEM(v, k, i, j) =
                    static_cast<RFLOAT>(k * ny * nx + i * nx + j + 1);
}

static double maxDiff(const MultidimArray<RFLOAT>& a, const MultidimArray<RFLOAT>& b)
{
    double md = 0.0;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(a)
    {
        double d = std::abs(static_cast<double>(DIRECT_MULTIDIM_ELEM(a, n)) -
                            static_cast<double>(DIRECT_MULTIDIM_ELEM(b, n)));
        if (d > md) md = d;
    }
    return md;
}

// ---------------------------------------------------------------------------
// FourierTransformer tests
// ---------------------------------------------------------------------------

class FourierTransformerTest : public ::testing::Test {};

TEST_F(FourierTransformerTest, OutputDimensions_1D)
{
    MultidimArray<RFLOAT> v;
    v.resize(8);
    MultidimArray<Complex> F;
    FourierTransformer ft;
    ft.FourierTransform(v, F, true);
    // Expected Fourier x-size = N/2 + 1 = 5
    EXPECT_EQ(XSIZE(F), 5);
    EXPECT_EQ(YSIZE(F), 1);
    EXPECT_EQ(ZSIZE(F), 1);
}

TEST_F(FourierTransformerTest, OutputDimensions_2D)
{
    MultidimArray<RFLOAT> v;
    v.resize(8, 8);
    MultidimArray<Complex> F;
    FourierTransformer ft;
    ft.FourierTransform(v, F, true);
    // FFTW half-transform: xdim = N/2+1 = 5, ydim = N = 8
    EXPECT_EQ(XSIZE(F), 5);
    EXPECT_EQ(YSIZE(F), 8);
    EXPECT_EQ(ZSIZE(F), 1);
}

TEST_F(FourierTransformerTest, OutputDimensions_3D)
{
    MultidimArray<RFLOAT> v;
    v.resize(4, 4, 4);
    MultidimArray<Complex> F;
    FourierTransformer ft;
    ft.FourierTransform(v, F, true);
    EXPECT_EQ(XSIZE(F), 3);  // 4/2+1
    EXPECT_EQ(YSIZE(F), 4);
    EXPECT_EQ(ZSIZE(F), 4);
}

TEST_F(FourierTransformerTest, RoundTrip_1D)
{
    MultidimArray<RFLOAT> orig, work;
    fillRamp1D(orig, 8);
    work = orig;

    MultidimArray<Complex> F;
    FourierTransformer ft;
    ft.FourierTransform(work, F, false);  // alias, work == fReal
    ft.inverseFourierTransform();         // write back to work

    EXPECT_LT(maxDiff(work, orig), 1e-10);
}

TEST_F(FourierTransformerTest, RoundTrip_2D)
{
    MultidimArray<RFLOAT> orig, work;
    fillRamp2D(orig, 8, 8);
    work = orig;

    MultidimArray<Complex> F;
    FourierTransformer ft;
    ft.FourierTransform(work, F, false);
    ft.inverseFourierTransform();

    EXPECT_LT(maxDiff(work, orig), 1e-9);
}

TEST_F(FourierTransformerTest, RoundTrip_3D)
{
    MultidimArray<RFLOAT> orig, work;
    fillRamp3D(orig, 4, 4, 4);
    work = orig;

    MultidimArray<Complex> F;
    FourierTransformer ft;
    ft.FourierTransform(work, F, false);
    ft.inverseFourierTransform();

    EXPECT_LT(maxDiff(work, orig), 1e-9);
}

TEST_F(FourierTransformerTest, ZeroInput_ZeroOutput)
{
    MultidimArray<RFLOAT> v;
    v.initZeros(8, 8);
    MultidimArray<Complex> F;
    FourierTransformer ft;
    ft.FourierTransform(v, F, true);

    double maxAbs = 0.0;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(F)
    {
        double a = std::abs(DIRECT_MULTIDIM_ELEM(F, n).real) +
                   std::abs(DIRECT_MULTIDIM_ELEM(F, n).imag);
        if (a > maxAbs) maxAbs = a;
    }
    EXPECT_NEAR(maxAbs, 0.0, 1e-15);
}

TEST_F(FourierTransformerTest, ConstantInput_DCOnly)
{
    // A constant real-space image has energy only at DC (k=i=j=0)
    const int N = 8;
    const RFLOAT val = 3.0;
    MultidimArray<RFLOAT> v;
    v.resize(N, N);
    v.initConstant(val);

    MultidimArray<Complex> F;
    FourierTransformer ft;
    ft.FourierTransform(v, F, true);

    // DC element = mean of the image = val (because forward transform is normalized)
    Complex dc = DIRECT_A2D_ELEM(F, 0, 0);
    EXPECT_NEAR(dc.real, val, 1e-10);
    EXPECT_NEAR(dc.imag, 0.0, 1e-10);

    // All other coefficients should be near zero
    double maxNonDC = 0.0;
    for (long int i = 0; i < YSIZE(F); i++)
        for (long int j = 0; j < XSIZE(F); j++)
        {
            if (i == 0 && j == 0) continue;
            Complex c = DIRECT_A2D_ELEM(F, i, j);
            double a = std::abs(c.real) + std::abs(c.imag);
            if (a > maxNonDC) maxNonDC = a;
        }
    EXPECT_LT(maxNonDC, 1e-10);
}

TEST_F(FourierTransformerTest, GetCopy_IsIndependent)
{
    MultidimArray<RFLOAT> v;
    v.resize(8);
    v.initConstant(1.0);

    MultidimArray<Complex> F;
    FourierTransformer ft;
    ft.FourierTransform(v, F, true);  // getCopy=true

    // Modify internal Fourier data (by doing another transform)
    v.initZeros(8);
    MultidimArray<Complex> F2;
    ft.FourierTransform(v, F2, true);

    // F should be unchanged (it is a copy)
    EXPECT_NEAR(DIRECT_A1D_ELEM(F, 0).real, 1.0, 1e-10);
}

TEST_F(FourierTransformerTest, TwoStep_InverseFourierTransform)
{
    // Test the two-argument inverseFourierTransform(V, v)
    MultidimArray<RFLOAT> orig, result;
    fillRamp2D(orig, 6, 6);
    result.initZeros(6, 6);

    MultidimArray<Complex> F;
    FourierTransformer ft;
    ft.FourierTransform(orig, F, true);

    ft.inverseFourierTransform(F, result);

    EXPECT_LT(maxDiff(result, orig), 1e-9);
}

// ---------------------------------------------------------------------------
// CenterFFT tests
// ---------------------------------------------------------------------------

class CenterFFTTest : public ::testing::Test {};

TEST_F(CenterFFTTest, ForwardThenBackward_Identity_1D)
{
    MultidimArray<RFLOAT> v, orig;
    fillRamp1D(v, 8);
    orig = v;

    CenterFFT(v, true);   // forward
    CenterFFT(v, false);  // backward

    EXPECT_LT(maxDiff(v, orig), 1e-12);
}

TEST_F(CenterFFTTest, ForwardThenBackward_Identity_2D)
{
    MultidimArray<RFLOAT> v, orig;
    fillRamp2D(v, 8, 8);
    orig = v;

    CenterFFT(v, true);
    CenterFFT(v, false);

    EXPECT_LT(maxDiff(v, orig), 1e-12);
}

TEST_F(CenterFFTTest, ForwardThenBackward_Identity_3D)
{
    MultidimArray<RFLOAT> v, orig;
    fillRamp3D(v, 4, 4, 4);
    orig = v;

    CenterFFT(v, true);
    CenterFFT(v, false);

    EXPECT_LT(maxDiff(v, orig), 1e-12);
}

TEST_F(CenterFFTTest, Forward_ShiftsOrigin_1D)
{
    // For a 1D array of size N, forward CenterFFT moves element at N/2 to index 0
    MultidimArray<RFLOAT> v;
    v.resize(8);
    v.initZeros();
    DIRECT_A1D_ELEM(v, 4) = 1.0;

    CenterFFT(v, true);

    EXPECT_NEAR(DIRECT_A1D_ELEM(v, 0), 1.0, 1e-12);
}

TEST_F(CenterFFTTest, Forward_2D_OriginAtCenter)
{
    // 8x8 array with 1 at (4,4) — after forward shift, 1 should be at (0,0)
    MultidimArray<RFLOAT> v;
    v.initZeros(8, 8);
    DIRECT_A2D_ELEM(v, 4, 4) = 1.0;

    CenterFFT(v, true);

    EXPECT_NEAR(DIRECT_A2D_ELEM(v, 0, 0), 1.0, 1e-12);
}

// ---------------------------------------------------------------------------
// windowFourierTransform tests
// ---------------------------------------------------------------------------

class WindowFourierTransformTest : public ::testing::Test {};

TEST_F(WindowFourierTransformTest, SameSize_NoChange_2D)
{
    MultidimArray<RFLOAT> v;
    fillRamp2D(v, 8, 8);
    MultidimArray<Complex> Fin, Fout;
    FourierTransformer ft;
    ft.FourierTransform(v, Fin, true);

    windowFourierTransform(Fin, Fout, 8);

    // Same shape
    EXPECT_EQ(XSIZE(Fout), XSIZE(Fin));
    EXPECT_EQ(YSIZE(Fout), YSIZE(Fin));
}

TEST_F(WindowFourierTransformTest, Crop_2D_ReducesDimension)
{
    MultidimArray<RFLOAT> v;
    fillRamp2D(v, 8, 8);
    MultidimArray<Complex> Fin, Fout;
    FourierTransformer ft;
    ft.FourierTransform(v, Fin, true);

    windowFourierTransform(Fin, Fout, 4);

    EXPECT_EQ(XSIZE(Fout), 3);  // 4/2+1
    EXPECT_EQ(YSIZE(Fout), 4);
}

TEST_F(WindowFourierTransformTest, Pad_2D_IncreaseDimension)
{
    MultidimArray<RFLOAT> v;
    fillRamp2D(v, 4, 4);
    MultidimArray<Complex> Fin, Fout;
    FourierTransformer ft;
    ft.FourierTransform(v, Fin, true);

    windowFourierTransform(Fin, Fout, 8);

    EXPECT_EQ(XSIZE(Fout), 5);  // 8/2+1
    EXPECT_EQ(YSIZE(Fout), 8);
}

TEST_F(WindowFourierTransformTest, Crop_PreservesLowFreq_1D)
{
    // A low-frequency signal should survive cropping
    const int N = 8;
    MultidimArray<RFLOAT> v;
    v.resize(N);
    for (int i = 0; i < N; i++)
        DIRECT_A1D_ELEM(v, i) = std::cos(2.0 * M_PI * i / N);

    MultidimArray<Complex> Fin, Fout;
    FourierTransformer ft;
    ft.FourierTransform(v, Fin, true);

    windowFourierTransform(Fin, Fout, 4);

    // First element (DC) should still be near zero for a pure cosine at freq 1
    // Element at freq 1 should be non-trivial
    double amp1 = std::sqrt(DIRECT_A1D_ELEM(Fout, 1).real * DIRECT_A1D_ELEM(Fout, 1).real +
                            DIRECT_A1D_ELEM(Fout, 1).imag * DIRECT_A1D_ELEM(Fout, 1).imag);
    EXPECT_GT(amp1, 0.0);
}

// ---------------------------------------------------------------------------
// getSpectrum tests
// ---------------------------------------------------------------------------

class GetSpectrumTest : public ::testing::Test {};

TEST_F(GetSpectrumTest, ZeroInput_ZeroSpectrum)
{
    MultidimArray<RFLOAT> v, spectrum;
    v.initZeros(8, 8);
    getSpectrum(v, spectrum, POWER_SPECTRUM);

    double maxVal = 0.0;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(spectrum)
        if (DIRECT_MULTIDIM_ELEM(spectrum, n) > maxVal)
            maxVal = DIRECT_MULTIDIM_ELEM(spectrum, n);
    EXPECT_NEAR(maxVal, 0.0, 1e-15);
}

TEST_F(GetSpectrumTest, PowerSpectrum_NonNegative)
{
    MultidimArray<RFLOAT> v, spectrum;
    fillRamp2D(v, 8, 8);
    getSpectrum(v, spectrum, POWER_SPECTRUM);

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(spectrum)
        EXPECT_GE(DIRECT_MULTIDIM_ELEM(spectrum, n), 0.0);
}

TEST_F(GetSpectrumTest, AmplitudeSpectrum_NonNegative)
{
    MultidimArray<RFLOAT> v, spectrum;
    fillRamp2D(v, 8, 8);
    getSpectrum(v, spectrum, AMPLITUDE_SPECTRUM);

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(spectrum)
        EXPECT_GE(DIRECT_MULTIDIM_ELEM(spectrum, n), 0.0);
}

TEST_F(GetSpectrumTest, AmplitudeSpectrum_AtMostSqrtOfPower)
{
    // Each shell averages independently: power = mean(|F|²), amplitude = mean(|F|).
    // By Jensen's inequality: mean(|F|)² ≤ mean(|F|²), so amplitude² ≤ power.
    MultidimArray<RFLOAT> v, power, amplitude;
    fillRamp2D(v, 8, 8);
    getSpectrum(v, power, POWER_SPECTRUM);
    getSpectrum(v, amplitude, AMPLITUDE_SPECTRUM);

    ASSERT_EQ(XSIZE(power), XSIZE(amplitude));
    for (long int n = 0; n < XSIZE(power); n++)
    {
        double p = DIRECT_A1D_ELEM(power, n);
        double a = DIRECT_A1D_ELEM(amplitude, n);
        EXPECT_LE(a * a, p + 1e-10) << "at shell " << n;
    }
}

TEST_F(GetSpectrumTest, ConstantImage_PowerConcentratedAtDC)
{
    // A constant image should have all power in the DC shell (shell 0)
    MultidimArray<RFLOAT> v, spectrum;
    v.resize(8, 8);
    v.initConstant(5.0);
    getSpectrum(v, spectrum, POWER_SPECTRUM);

    double dc = DIRECT_A1D_ELEM(spectrum, 0);
    EXPECT_GT(dc, 0.0);

    // Higher shells should be near zero
    for (long int i = 1; i < XSIZE(spectrum); i++)
        EXPECT_NEAR(DIRECT_A1D_ELEM(spectrum, i), 0.0, 1e-10);
}

// ---------------------------------------------------------------------------
// getFSC tests
// ---------------------------------------------------------------------------

class GetFSCTest : public ::testing::Test {};

TEST_F(GetFSCTest, IdenticalImages_FSC_AllOne)
{
    MultidimArray<RFLOAT> m1, fsc;
    fillRamp2D(m1, 8, 8);
    MultidimArray<RFLOAT> m2 = m1;

    getFSC(m1, m2, fsc);

    // FSC between identical images should be 1 everywhere (where there's signal)
    // Shell 0 (DC) is often defined as 1 by convention; check shells 1+
    for (long int i = 1; i < XSIZE(fsc); i++)
    {
        double val = DIRECT_A1D_ELEM(fsc, i);
        EXPECT_NEAR(val, 1.0, 1e-6) << "shell " << i;
    }
}

TEST_F(GetFSCTest, NegatedImages_FSC_AllMinusOne)
{
    MultidimArray<RFLOAT> m1, m2, fsc;
    fillRamp2D(m1, 8, 8);
    m2 = m1;
    m2 *= -1.0;

    getFSC(m1, m2, fsc);

    for (long int i = 1; i < XSIZE(fsc); i++)
    {
        double val = DIRECT_A1D_ELEM(fsc, i);
        EXPECT_NEAR(val, -1.0, 1e-6) << "shell " << i;
    }
}

TEST_F(GetFSCTest, OutputSize_HalfNyquist)
{
    MultidimArray<RFLOAT> m1, m2, fsc;
    fillRamp2D(m1, 8, 8);
    m2 = m1;

    getFSC(m1, m2, fsc);

    // FSC array should have size N/2 + 1 = 5
    EXPECT_EQ(XSIZE(fsc), 5);
}

// ---------------------------------------------------------------------------
// shiftImageInFourierTransform tests
// ---------------------------------------------------------------------------

class ShiftFourierTest : public ::testing::Test {};

TEST_F(ShiftFourierTest, ZeroShift_NoChange)
{
    MultidimArray<RFLOAT> v;
    fillRamp2D(v, 8, 8);
    MultidimArray<Complex> Fin, Fout;
    FourierTransformer ft;
    ft.FourierTransform(v, Fin, true);
    Fout = Fin;

    shiftImageInFourierTransform(Fin, Fout, 8, 0.0, 0.0);

    double maxd = 0.0;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fin)
    {
        double dr = std::abs(DIRECT_MULTIDIM_ELEM(Fout, n).real -
                             DIRECT_MULTIDIM_ELEM(Fin, n).real);
        double di = std::abs(DIRECT_MULTIDIM_ELEM(Fout, n).imag -
                             DIRECT_MULTIDIM_ELEM(Fin, n).imag);
        if (dr > maxd) maxd = dr;
        if (di > maxd) maxd = di;
    }
    EXPECT_LT(maxd, 1e-10);
}

TEST_F(ShiftFourierTest, ShiftAndUnshift_RoundTrip)
{
    MultidimArray<RFLOAT> orig, result;
    fillRamp2D(orig, 8, 8);
    result = orig;

    MultidimArray<Complex> F, Fshifted, Funshifted;
    FourierTransformer ft;
    ft.FourierTransform(result, F, true);

    shiftImageInFourierTransform(F, Fshifted, 8, 2.0, 3.0);
    shiftImageInFourierTransform(Fshifted, Funshifted, 8, -2.0, -3.0);

    // IFFT of Funshifted should match original
    MultidimArray<RFLOAT> recovered;
    recovered.initZeros(8, 8);
    ft.inverseFourierTransform(Funshifted, recovered);

    EXPECT_LT(maxDiff(recovered, orig), 1e-9);
}

TEST_F(ShiftFourierTest, DC_Unaffected_By_Shift)
{
    // Shifting should not change the DC amplitude
    MultidimArray<RFLOAT> v;
    v.resize(8, 8);
    v.initConstant(2.0);

    MultidimArray<Complex> Fin, Fout;
    FourierTransformer ft;
    ft.FourierTransform(v, Fin, true);

    shiftImageInFourierTransform(Fin, Fout, 8, 3.0, 1.0);

    double dcIn  = std::sqrt(DIRECT_A2D_ELEM(Fin, 0, 0).real  * DIRECT_A2D_ELEM(Fin, 0, 0).real  +
                             DIRECT_A2D_ELEM(Fin, 0, 0).imag  * DIRECT_A2D_ELEM(Fin, 0, 0).imag);
    double dcOut = std::sqrt(DIRECT_A2D_ELEM(Fout, 0, 0).real * DIRECT_A2D_ELEM(Fout, 0, 0).real +
                             DIRECT_A2D_ELEM(Fout, 0, 0).imag * DIRECT_A2D_ELEM(Fout, 0, 0).imag);
    EXPECT_NEAR(dcIn, dcOut, 1e-10);
}

// ---------------------------------------------------------------------------
// FourierTransformer copy constructor / getReal / getComplex / cleanup
// ---------------------------------------------------------------------------

TEST_F(FourierTransformerTest, CopyConstructor_DoesNotCrash)
{
    // Test that copy-constructing an uninitialized FourierTransformer does not crash.
    // (Copying a transformer that has active FFTW plans shares plan pointers and is
    // not safe to use, so we only verify the constructor itself doesn't crash.)
    FourierTransformer ft1;
    FourierTransformer ft2(ft1);
    SUCCEED();
}

TEST_F(FourierTransformerTest, GetReal_ReturnsRealArray)
{
    MultidimArray<RFLOAT> v;
    fillRamp2D(v, 4, 4);
    MultidimArray<Complex> F;
    FourierTransformer ft;
    ft.FourierTransform(v, F, false);

    const MultidimArray<RFLOAT>& r = ft.getReal();
    EXPECT_EQ(XSIZE(r), XSIZE(v));
    EXPECT_EQ(YSIZE(r), YSIZE(v));
}

TEST_F(FourierTransformerTest, GetComplex_AfterComplexSetReal)
{
    MultidimArray<Complex> cplx;
    cplx.resize(4, 4);
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(cplx)
        DIRECT_MULTIDIM_ELEM(cplx, n) = Complex(1.0, 0.0);

    FourierTransformer ft;
    ft.setReal(cplx);

    const MultidimArray<Complex>& c = ft.getComplex();
    EXPECT_EQ(XSIZE(c), XSIZE(cplx));
    EXPECT_EQ(YSIZE(c), YSIZE(cplx));
}

TEST_F(FourierTransformerTest, Cleanup_DoesNotCrash)
{
    MultidimArray<RFLOAT> v;
    fillRamp2D(v, 4, 4);
    MultidimArray<Complex> F;
    FourierTransformer ft;
    ft.FourierTransform(v, F, false);
    ft.cleanup();  // should not crash
    SUCCEED();
}

TEST_F(FourierTransformerTest, EnforceHermitianSymmetry_2D)
{
    MultidimArray<RFLOAT> v;
    fillRamp2D(v, 8, 8);
    MultidimArray<Complex> F;
    FourierTransformer ft;
    ft.FourierTransform(v, F, false);
    ft.enforceHermitianSymmetry();  // must not crash
    SUCCEED();
}

TEST_F(FourierTransformerTest, FourierTransformThenInverseNoArg)
{
    // Test the no-argument FourierTransform() and inverseFourierTransform() methods
    MultidimArray<RFLOAT> orig, work;
    fillRamp2D(orig, 4, 4);
    work = orig;

    FourierTransformer ft;
    ft.setReal(work);
    ft.FourierTransform();      // no-arg version
    ft.inverseFourierTransform(); // no-arg version

    EXPECT_LT(maxDiff(work, orig), 1e-9);
}

// ---------------------------------------------------------------------------
// getFSC with pre-computed Fourier transforms
// ---------------------------------------------------------------------------

TEST(GetFSCComplexTest, IdenticalFTs_AllOne)
{
    MultidimArray<RFLOAT> m1, fsc;
    fillRamp2D(m1, 8, 8);
    MultidimArray<Complex> FT1, FT2;
    FourierTransformer ft;
    ft.FourierTransform(m1, FT1, true);
    FT2 = FT1;

    getFSC(FT1, FT2, fsc);

    for (long int i = 1; i < XSIZE(fsc); i++)
        EXPECT_NEAR(DIRECT_A1D_ELEM(fsc, i), 1.0, 1e-6) << "shell " << i;
}

// ---------------------------------------------------------------------------
// getAmplitudeCorrelationAndDifferentialPhaseResidual
// ---------------------------------------------------------------------------

TEST(AcorrDprTest, IdenticalImages_AcorrAllOne)
{
    MultidimArray<RFLOAT> m1, acorr, dpr;
    fillRamp2D(m1, 8, 8);
    MultidimArray<RFLOAT> m2 = m1;

    getAmplitudeCorrelationAndDifferentialPhaseResidual(m1, m2, acorr, dpr);

    // Amplitude correlation between identical images should be 1 everywhere
    for (long int i = 1; i < XSIZE(acorr); i++)
        EXPECT_NEAR(DIRECT_A1D_ELEM(acorr, i), 1.0, 1e-5) << "shell " << i;
}

TEST(AcorrDprTest, IdenticalImages_DPRIsZero)
{
    MultidimArray<RFLOAT> m1, acorr, dpr;
    fillRamp2D(m1, 8, 8);
    MultidimArray<RFLOAT> m2 = m1;

    getAmplitudeCorrelationAndDifferentialPhaseResidual(m1, m2, acorr, dpr);

    for (long int i = 0; i < XSIZE(dpr); i++)
        EXPECT_NEAR(DIRECT_A1D_ELEM(dpr, i), 0.0, 1e-5) << "shell " << i;
}

// ---------------------------------------------------------------------------
// getCosDeltaPhase
// ---------------------------------------------------------------------------

TEST(CosDeltaPhaseTest, IdenticalFTs_AllOne)
{
    MultidimArray<RFLOAT> m1;
    fillRamp2D(m1, 8, 8);
    MultidimArray<Complex> FT1, FT2;
    FourierTransformer ft;
    ft.FourierTransform(m1, FT1, true);
    FT2 = FT1;

    MultidimArray<RFLOAT> cosPhi;
    getCosDeltaPhase(FT1, FT2, cosPhi);

    for (long int i = 0; i < XSIZE(cosPhi); i++)
        EXPECT_NEAR(DIRECT_A1D_ELEM(cosPhi, i), 1.0, 1e-5) << "shell " << i;
}

// ---------------------------------------------------------------------------
// getAbMatricesForShiftImageInFourierTransform
// ---------------------------------------------------------------------------

TEST(AbMatricesTest, ZeroShift_RealPartNearOne)
{
    MultidimArray<RFLOAT> v;
    fillRamp2D(v, 8, 8);
    MultidimArray<Complex> Fin, ab;
    FourierTransformer ft;
    ft.FourierTransform(v, Fin, true);

    getAbMatricesForShiftImageInFourierTransform(Fin, ab, 8.0, 0.0, 0.0, 0.0);

    // With zero shift, exp(i*0) = (1,0), so real part should be 1, imag 0
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(ab)
    {
        EXPECT_NEAR(DIRECT_MULTIDIM_ELEM(ab, n).real, 1.0, 1e-10);
        EXPECT_NEAR(DIRECT_MULTIDIM_ELEM(ab, n).imag, 0.0, 1e-10);
    }
}

// ---------------------------------------------------------------------------
// divideBySpectrum / multiplyBySpectrum
// ---------------------------------------------------------------------------

TEST(DivideBySpectrumTest, UniformSpectrum_PreservesImage)
{
    MultidimArray<RFLOAT> v;
    fillRamp2D(v, 8, 8);
    MultidimArray<RFLOAT> orig = v;

    // Build a uniform spectrum of ones
    MultidimArray<RFLOAT> spectrum;
    spectrum.initZeros(5);  // N/2+1 = 5
    spectrum.initConstant(1.0);

    divideBySpectrum(v, spectrum);

    EXPECT_LT(maxDiff(v, orig), 1e-9);
}

TEST(MultiplyBySpectrumTest, ZeroSpectrum_ZerosImage)
{
    MultidimArray<RFLOAT> v;
    fillRamp2D(v, 8, 8);

    MultidimArray<RFLOAT> spectrum;
    spectrum.initZeros(5);

    multiplyBySpectrum(v, spectrum);

    // Image should be all zeros after multiply by zero spectrum
    double maxVal = 0.0;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(v)
        maxVal = std::max(maxVal, std::abs((double)DIRECT_MULTIDIM_ELEM(v, n)));
    EXPECT_NEAR(maxVal, 0.0, 1e-10);
}

// ---------------------------------------------------------------------------
// whitenSpectrum
// ---------------------------------------------------------------------------

TEST(WhitenSpectrumTest, OutputSameShape)
{
    MultidimArray<RFLOAT> in, out;
    fillRamp2D(in, 8, 8);
    whitenSpectrum(in, out, POWER_SPECTRUM);
    EXPECT_EQ(XSIZE(out), XSIZE(in));
    EXPECT_EQ(YSIZE(out), YSIZE(in));
}

// ---------------------------------------------------------------------------
// adaptSpectrum
// ---------------------------------------------------------------------------

TEST(AdaptSpectrumTest, OutputSameShape)
{
    MultidimArray<RFLOAT> in, out;
    fillRamp2D(in, 8, 8);

    // Use a flat reference spectrum
    MultidimArray<RFLOAT> ref;
    ref.initZeros(5);
    ref.initConstant(1.0);

    adaptSpectrum(in, out, ref, POWER_SPECTRUM);
    EXPECT_EQ(XSIZE(out), XSIZE(in));
    EXPECT_EQ(YSIZE(out), YSIZE(in));
}

// ---------------------------------------------------------------------------
// resizeMap
// ---------------------------------------------------------------------------

TEST(ResizeMapTest, DownsampleByHalf_2D)
{
    MultidimArray<RFLOAT> v;
    fillRamp2D(v, 8, 8);

    resizeMap(v, 4);

    EXPECT_EQ(XSIZE(v), 4);
    EXPECT_EQ(YSIZE(v), 4);
}

TEST(ResizeMapTest, UpsampleByTwo_2D)
{
    MultidimArray<RFLOAT> v;
    fillRamp2D(v, 4, 4);

    resizeMap(v, 8);

    EXPECT_EQ(XSIZE(v), 8);
    EXPECT_EQ(YSIZE(v), 8);
}

// ---------------------------------------------------------------------------
// applyBFactorToMap
// ---------------------------------------------------------------------------

TEST(ApplyBFactorTest, ZeroBfactor_NoChange_RFLOAT)
{
    MultidimArray<RFLOAT> v;
    fillRamp2D(v, 8, 8);
    MultidimArray<RFLOAT> orig = v;

    applyBFactorToMap(v, 0.0, 1.0);

    EXPECT_LT(maxDiff(v, orig), 1e-6);
}

TEST(ApplyBFactorTest, PositiveBfactor_ReducesPower_RFLOAT)
{
    MultidimArray<RFLOAT> v;
    fillRamp2D(v, 8, 8);

    // Sum of squares before and after should differ with nonzero bfactor
    double before = 0.0;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(v)
        before += DIRECT_MULTIDIM_ELEM(v, n) * DIRECT_MULTIDIM_ELEM(v, n);

    applyBFactorToMap(v, 100.0, 1.0);

    double after = 0.0;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(v)
        after += DIRECT_MULTIDIM_ELEM(v, n) * DIRECT_MULTIDIM_ELEM(v, n);

    EXPECT_LT(after, before);
}

TEST(ApplyBFactorTest, FourierVariant_DoesNotCrash)
{
    MultidimArray<RFLOAT> v;
    fillRamp2D(v, 8, 8);
    MultidimArray<Complex> FT;
    FourierTransformer ft;
    ft.FourierTransform(v, FT, true);

    applyBFactorToMap(FT, 8, 0.0, 1.0);  // zero bfactor
    SUCCEED();
}

// ---------------------------------------------------------------------------
// randomizePhasesBeyond
// ---------------------------------------------------------------------------

TEST(RandomizePhasesBeyondTest, OutputSameShape)
{
    MultidimArray<RFLOAT> v;
    fillRamp2D(v, 8, 8);
    MultidimArray<RFLOAT> orig = v;

    randomizePhasesBeyond(v, 4);

    EXPECT_EQ(XSIZE(v), XSIZE(orig));
    EXPECT_EQ(YSIZE(v), YSIZE(orig));
}

// ---------------------------------------------------------------------------
// shiftImageInFourierTransformWithTabSincos
// ---------------------------------------------------------------------------

TEST(TabSincosShiftTest, ZeroShift_MatchesWindowFourierTransform)
{
    MultidimArray<RFLOAT> v;
    fillRamp2D(v, 8, 8);
    MultidimArray<Complex> Fin, Fout_tab, Fout_ref;
    FourierTransformer ft;
    ft.FourierTransform(v, Fin, true);

    TabSine   tabsin;
    TabCosine tabcos;
    tabsin.initialise(10000);
    tabcos.initialise(10000);

    // Zero shift should produce a windowed (same-size) output
    shiftImageInFourierTransformWithTabSincos(Fin, Fout_tab, 8.0, 8, tabsin, tabcos, 0.0, 0.0, 0.0);

    // With zero shift the output should match a plain windowFourierTransform
    windowFourierTransform(Fin, Fout_ref, 8);

    double maxd = 0.0;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fout_tab)
    {
        double dr = std::abs(DIRECT_MULTIDIM_ELEM(Fout_tab, n).real - DIRECT_MULTIDIM_ELEM(Fout_ref, n).real);
        double di = std::abs(DIRECT_MULTIDIM_ELEM(Fout_tab, n).imag - DIRECT_MULTIDIM_ELEM(Fout_ref, n).imag);
        if (dr > maxd) maxd = dr;
        if (di > maxd) maxd = di;
    }
    EXPECT_LT(maxd, 1e-9);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
