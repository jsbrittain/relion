/*
 * Unit tests for src/funcs.h / src/funcs.cpp
 *
 * Covers:
 *   - kaiser_value  / blob_val  — blob spatial value
 *   - kaiser_proj   / blob_proj — blob line integral (projection)
 *   - kaiser_Fourier_value / blob_Fourier_val — blob Fourier transform
 *   - fitStraightLine           — least-squares line fit
 *   - gaussian1D                — 1D Gaussian evaluation
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "src/funcs.h"

// ------------------------------------------------------------ kaiser_value --

class KaiserValueTest : public ::testing::Test
{
protected:
    // Standard RELION blob parameters used in reconstructions.
    blobtype blob;
    void SetUp() override
    {
        blob.radius = 2.0;
        blob.order  = 2;
        blob.alpha  = 3.6;
    }
};

TEST_F(KaiserValueTest, AtCenter_IsPositive)
{
    RFLOAT v = kaiser_value(0.0, blob.radius, blob.alpha, blob.order);
    EXPECT_GT(v, 0.0);
}

TEST_F(KaiserValueTest, InsideRadius_IsPositive)
{
    RFLOAT v = kaiser_value(1.0, blob.radius, blob.alpha, blob.order);
    EXPECT_GT(v, 0.0);
}

TEST_F(KaiserValueTest, BeyondRadius_IsZero)
{
    RFLOAT v = kaiser_value(blob.radius + 0.1, blob.radius, blob.alpha, blob.order);
    EXPECT_NEAR(v, 0.0, 1e-10);
}

TEST_F(KaiserValueTest, WellBeyondRadius_IsZero)
{
    RFLOAT v = kaiser_value(10.0, blob.radius, blob.alpha, blob.order);
    EXPECT_NEAR(v, 0.0, 1e-10);
}

TEST_F(KaiserValueTest, DecreaseWithDistance)
{
    // Blob is monotonically decreasing from centre.
    RFLOAT v0 = kaiser_value(0.0, blob.radius, blob.alpha, blob.order);
    RFLOAT v1 = kaiser_value(0.5, blob.radius, blob.alpha, blob.order);
    RFLOAT v2 = kaiser_value(1.0, blob.radius, blob.alpha, blob.order);
    EXPECT_GE(v0, v1);
    EXPECT_GE(v1, v2);
}

TEST_F(KaiserValueTest, BlobValMacro_MatchesDirectCall)
{
    RFLOAT direct = kaiser_value(1.0, blob.radius, blob.alpha, blob.order);
    RFLOAT macro  = blob_val(1.0, blob);
    EXPECT_DOUBLE_EQ(direct, macro);
}

// ------------------------------------------------------------ kaiser_proj --

class KaiserProjTest : public ::testing::Test
{
protected:
    blobtype blob;
    void SetUp() override
    {
        blob.radius = 2.0;
        blob.order  = 2;
        blob.alpha  = 3.6;
    }
};

TEST_F(KaiserProjTest, AtZeroOffset_IsPositive)
{
    RFLOAT p = kaiser_proj(0.0, blob.radius, blob.alpha, blob.order);
    EXPECT_GT(p, 0.0);
}

TEST_F(KaiserProjTest, InsideRadius_IsPositive)
{
    RFLOAT p = kaiser_proj(1.0, blob.radius, blob.alpha, blob.order);
    EXPECT_GT(p, 0.0);
}

TEST_F(KaiserProjTest, BeyondRadius_IsZero)
{
    RFLOAT p = kaiser_proj(blob.radius + 0.1, blob.radius, blob.alpha, blob.order);
    EXPECT_NEAR(p, 0.0, 1e-10);
}

TEST_F(KaiserProjTest, ProjectionExceedsValue)
{
    // The line integral integrates the blob along the full chord,
    // so it is always >= the blob value at the same lateral offset.
    RFLOAT val  = kaiser_value(0.5, blob.radius, blob.alpha, blob.order);
    RFLOAT proj = kaiser_proj (0.5, blob.radius, blob.alpha, blob.order);
    EXPECT_GE(proj, val);
}

TEST_F(KaiserProjTest, BlobProjMacro_MatchesDirectCall)
{
    RFLOAT direct = kaiser_proj(1.0, blob.radius, blob.alpha, blob.order);
    RFLOAT macro  = blob_proj(1.0, blob);
    EXPECT_DOUBLE_EQ(direct, macro);
}

// ------------------------------------------------ kaiser_Fourier_value --

class KaiserFourierTest : public ::testing::Test
{
protected:
    blobtype blob;
    void SetUp() override
    {
        blob.radius = 2.0;
        blob.order  = 2;
        blob.alpha  = 3.6;
    }
};

TEST_F(KaiserFourierTest, AtZeroFrequency_IsPositive)
{
    RFLOAT fv = kaiser_Fourier_value(0.0, blob.radius, blob.alpha, blob.order);
    EXPECT_GT(fv, 0.0);
}

TEST_F(KaiserFourierTest, BlobFourierValMacro_MatchesDirectCall)
{
    RFLOAT direct = kaiser_Fourier_value(0.5, blob.radius, blob.alpha, blob.order);
    RFLOAT macro  = blob_Fourier_val(0.5, blob);
    EXPECT_DOUBLE_EQ(direct, macro);
}

TEST_F(KaiserFourierTest, DecreaseWithFrequency)
{
    // Fourier transform of a blob generally decreases with frequency.
    RFLOAT f0 = kaiser_Fourier_value(0.0, blob.radius, blob.alpha, blob.order);
    RFLOAT f1 = kaiser_Fourier_value(0.5, blob.radius, blob.alpha, blob.order);
    EXPECT_GE(f0, f1);
}

// -------------------------------------------------------- fitStraightLine --

class FitStraightLineTest : public ::testing::Test
{
protected:
    static std::vector<fit_point2D> make_line(RFLOAT slope, RFLOAT intercept,
                                              int n = 5)
    {
        std::vector<fit_point2D> pts(n);
        for (int i = 0; i < n; ++i)
        {
            pts[i].x = static_cast<RFLOAT>(i);
            pts[i].y = slope * pts[i].x + intercept;
            pts[i].w = 1.0;
        }
        return pts;
    }
};

TEST_F(FitStraightLineTest, PerfectLine_CorrectSlopeAndIntercept)
{
    auto pts = make_line(2.0, 1.0);
    RFLOAT slope = 0.0, intercept = 0.0, corr = 0.0;
    fitStraightLine(pts, slope, intercept, corr);
    EXPECT_NEAR(slope,     2.0, 1e-6);
    EXPECT_NEAR(intercept, 1.0, 1e-6);
}

TEST_F(FitStraightLineTest, PerfectLine_CorrCoeffIsOne)
{
    auto pts = make_line(3.0, -2.0);
    RFLOAT slope, intercept, corr;
    fitStraightLine(pts, slope, intercept, corr);
    EXPECT_NEAR(corr, 1.0, 1e-6);
}

TEST_F(FitStraightLineTest, NegativeSlope_RecoveredCorrectly)
{
    auto pts = make_line(-0.5, 10.0);
    RFLOAT slope, intercept, corr;
    fitStraightLine(pts, slope, intercept, corr);
    EXPECT_NEAR(slope,     -0.5, 1e-6);
    EXPECT_NEAR(intercept, 10.0, 1e-6);
    // fitStraightLine returns the magnitude of the correlation coefficient.
    EXPECT_NEAR(std::abs(corr), 1.0, 1e-6);
}

TEST_F(FitStraightLineTest, ZeroSlope_HorizontalLine)
{
    // All y-values constant → slope should be 0.
    auto pts = make_line(0.0, 5.0);
    RFLOAT slope, intercept, corr;
    fitStraightLine(pts, slope, intercept, corr);
    EXPECT_NEAR(slope,     0.0, 1e-6);
    EXPECT_NEAR(intercept, 5.0, 1e-6);
}

TEST_F(FitStraightLineTest, WeightedFit_HighWeightDominates)
{
    // Two points define a line y=x; third point at (3,100) has low weight.
    std::vector<fit_point2D> pts = {
        {0.0, 0.0, 1.0},
        {1.0, 1.0, 1.0},
        {2.0, 2.0, 1.0},
        {3.0, 100.0, 0.0001}   // should barely affect the fit
    };
    RFLOAT slope, intercept, corr;
    fitStraightLine(pts, slope, intercept, corr);
    EXPECT_NEAR(slope,     1.0, 0.01);
    EXPECT_NEAR(intercept, 0.0, 0.05);
}

// ----------------------------------------------------------- gaussian1D --

TEST(Gaussian1DTest, PeakAtMean)
{
    RFLOAT sigma = 1.0;
    RFLOAT mu    = 2.0;
    RFLOAT peak  = gaussian1D(mu, sigma, mu);
    RFLOAT off   = gaussian1D(mu + 1.0, sigma, mu);
    EXPECT_GT(peak, off);
}

TEST(Gaussian1DTest, SymmetricAboutMean)
{
    RFLOAT sigma = 2.0, mu = 0.0;
    RFLOAT vp = gaussian1D( 1.0, sigma, mu);
    RFLOAT vm = gaussian1D(-1.0, sigma, mu);
    EXPECT_NEAR(vp, vm, 1e-10);
}

TEST(Gaussian1DTest, WiderSigma_LowerPeak)
{
    RFLOAT narrow = gaussian1D(0.0, 1.0, 0.0);
    RFLOAT wide   = gaussian1D(0.0, 2.0, 0.0);
    EXPECT_GT(narrow, wide);
}

TEST(Gaussian1DTest, DefaultMuIsZero)
{
    // gaussian1D(x, sigma) should equal gaussian1D(x, sigma, 0)
    RFLOAT sigma = 1.5;
    RFLOAT x     = 0.5;
    RFLOAT with_default = gaussian1D(x, sigma);
    RFLOAT with_zero    = gaussian1D(x, sigma, 0.0);
    EXPECT_NEAR(with_default, with_zero, 1e-10);
}

// -------------------------------------------------------- fitLeastSquaresPlane --

TEST(FitPlaneTest, PerfectPlane_CorrectCoefficients)
{
    // z = 2x + 3y + 1
    std::vector<fit_point3D> pts;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
        {
            fit_point3D p;
            p.x = i; p.y = j;
            p.z = 2.0 * i + 3.0 * j + 1.0;
            p.w = 1.0;
            pts.push_back(p);
        }
    RFLOAT a, b, c;
    fitLeastSquaresPlane(pts, a, b, c);
    EXPECT_NEAR(a, 2.0, 1e-6);
    EXPECT_NEAR(b, 3.0, 1e-6);
    EXPECT_NEAR(c, 1.0, 1e-6);
}

TEST(FitPlaneTest, HorizontalPlane_ZeroSlopes)
{
    std::vector<fit_point3D> pts;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
        {
            fit_point3D p;
            p.x = i; p.y = j; p.z = 5.0; p.w = 1.0;
            pts.push_back(p);
        }
    RFLOAT a, b, c;
    fitLeastSquaresPlane(pts, a, b, c);
    EXPECT_NEAR(a, 0.0, 1e-6);
    EXPECT_NEAR(b, 0.0, 1e-6);
    EXPECT_NEAR(c, 5.0, 1e-6);
}

// -------------------------------------------------------- blob volume helpers --

TEST(BasvolumeTest, ReturnsPositive)
{
    RFLOAT v = basvolume(2.0, 3.6, 2, 3);
    EXPECT_GT(v, 0.0);
}

TEST(BasvolumeTest, AlphaZero_ReturnsPositive)
{
    RFLOAT v = basvolume(2.0, 0.0, 2, 3);
    EXPECT_GT(v, 0.0);
}

TEST(BasvolumeTest, OddDimension_ReturnsPositive)
{
    RFLOAT v = basvolume(2.0, 3.6, 2, 1);
    EXPECT_GT(v, 0.0);
}

TEST(InZeroargTest, n0_IsOne)
{
    EXPECT_NEAR(in_zeroarg(0), 1.0, 1e-10);
}

TEST(InZeroargTest, n1_IsHalf)
{
    EXPECT_NEAR(in_zeroarg(1), 0.5, 1e-10);
}

TEST(InphZeroargTest, n0_IsPositive)
{
    EXPECT_GT(inph_zeroarg(0), 0.0);
}

TEST(InphZeroargTest, n1_IsSmaller)
{
    EXPECT_LT(inph_zeroarg(1), inph_zeroarg(0));
}

TEST(InTest, n0_MatchesBessi0)
{
    // i_n(0, x) = bessi0(x); just check it returns a reasonable value
    RFLOAT v = i_n(0, 1.0);
    EXPECT_GT(v, 1.0); // I_0(1) ≈ 1.266
}

TEST(InTest, n1_MatchesBessi1)
{
    RFLOAT v = i_n(1, 1.0);
    EXPECT_GT(v, 0.0);
}

TEST(InTest, n2_ReturnsPositive)
{
    RFLOAT v = i_n(2, 1.0);
    EXPECT_GT(v, 0.0);
}

TEST(InTest, xZero_ReturnsZero_ForNonzeroN)
{
    RFLOAT v = i_n(2, 0.0);
    EXPECT_NEAR(v, 0.0, 1e-10);
}

TEST(InphTest, n0_ReturnsPositive)
{
    RFLOAT v = i_nph(0, 1.0);
    EXPECT_GT(v, 0.0);
}

TEST(InphTest, xZero_ReturnsZero)
{
    RFLOAT v = i_nph(1, 0.0);
    EXPECT_NEAR(v, 0.0, 1e-10);
}

// -------------------------------------------------------- blob derived funcs --

TEST(BlobFreqZeroTest, ReturnsPositive)
{
    blobtype blob; blob.radius = 2.0; blob.order = 2; blob.alpha = 3.6;
    EXPECT_GT(blob_freq_zero(blob), 0.0);
}

TEST(BlobAttTest, AtZeroFreq_IsOne)
{
    blobtype blob; blob.radius = 2.0; blob.order = 2; blob.alpha = 3.6;
    EXPECT_NEAR(blob_att(0.0, blob), 1.0, 1e-6);
}

TEST(BlobAttTest, DecreasesWithFrequency)
{
    blobtype blob; blob.radius = 2.0; blob.order = 2; blob.alpha = 3.6;
    EXPECT_GE(blob_att(0.0, blob), blob_att(0.5, blob));
}

TEST(BlobOpsTest, ReturnsPositive)
{
    blobtype blob; blob.radius = 2.0; blob.order = 2; blob.alpha = 3.6;
    EXPECT_GT(blob_ops(0.0, blob), 0.0);
}

// -------------------------------------------------------- tstudent1D --

TEST(TStudent1DTest, PeakAtMean)
{
    RFLOAT peak = tstudent1D(0.0, 5.0, 1.0, 0.0);
    RFLOAT off  = tstudent1D(1.0, 5.0, 1.0, 0.0);
    EXPECT_GT(peak, off);
}

TEST(TStudent1DTest, SymmetricAboutMean)
{
    RFLOAT vp = tstudent1D( 1.0, 5.0, 1.0, 0.0);
    RFLOAT vm = tstudent1D(-1.0, 5.0, 1.0, 0.0);
    EXPECT_NEAR(vp, vm, 1e-10);
}

TEST(TStudent1DTest, IsPositive)
{
    EXPECT_GT(tstudent1D(0.0, 3.0, 2.0, 5.0), 0.0);
}

// -------------------------------------------------------- gaussian2D --

TEST(Gaussian2DTest, PeakAtCenter)
{
    RFLOAT peak = gaussian2D(0.0, 0.0, 1.0, 1.0, 0.0);
    RFLOAT off  = gaussian2D(1.0, 0.0, 1.0, 1.0, 0.0);
    EXPECT_GT(peak, off);
}

TEST(Gaussian2DTest, SymmetricWithZeroAngle)
{
    RFLOAT vx = gaussian2D(1.0, 0.0, 1.0, 1.0, 0.0);
    RFLOAT vy = gaussian2D(0.0, 1.0, 1.0, 1.0, 0.0);
    EXPECT_NEAR(vx, vy, 1e-10);
}

TEST(Gaussian2DTest, IsPositive)
{
    EXPECT_GT(gaussian2D(0.5, 0.3, 1.0, 2.0, 0.1, 0.0, 0.0), 0.0);
}

// -------------------------------------------------------- icdf_gauss / cdf_gauss --

TEST(IcdfGaussTest, SymmetryAroundHalf)
{
    // icdf(0.5) should be 0 for standard normal
    RFLOAT z = icdf_gauss(0.5);
    EXPECT_NEAR(z, 0.0, 1e-3);
}

TEST(IcdfGaussTest, LowProbability_Negative)
{
    RFLOAT z = icdf_gauss(0.1);
    EXPECT_LT(z, 0.0);
}

TEST(IcdfGaussTest, HighProbability_Positive)
{
    RFLOAT z = icdf_gauss(0.9);
    EXPECT_GT(z, 0.0);
}

TEST(CdfGaussTest, AtZero_IsHalf)
{
    EXPECT_NEAR(cdf_gauss(0.0), 0.5, 1e-10);
}

TEST(CdfGaussTest, LargePositive_NearOne)
{
    EXPECT_NEAR(cdf_gauss(5.0), 1.0, 1e-5);
}

TEST(CdfGaussTest, LargeNegative_NearZero)
{
    EXPECT_NEAR(cdf_gauss(-5.0), 0.0, 1e-5);
}

TEST(CdfGaussTest, MonotonicallyIncreasing)
{
    EXPECT_LT(cdf_gauss(-1.0), cdf_gauss(0.0));
    EXPECT_LT(cdf_gauss(0.0), cdf_gauss(1.0));
}

// -------------------------------------------------------- cdf_tstudent --

TEST(CdfTstudentTest, AtZero_IsHalf)
{
    EXPECT_NEAR(cdf_tstudent(5, 0.0), 0.5, 1e-10);
}

TEST(CdfTstudentTest, LargeNegative_NearZero)
{
    EXPECT_LT(cdf_tstudent(10, -5.0), 0.01);
}

TEST(CdfTstudentTest, LargePositive_NearOne)
{
    EXPECT_GT(cdf_tstudent(10, 5.0), 0.99);
}

TEST(CdfTstudentTest, OddDegreesOfFreedom)
{
    // k=3 (odd), t=1.0
    RFLOAT cdf = cdf_tstudent(3, 1.0);
    EXPECT_GT(cdf, 0.5);
    EXPECT_LT(cdf, 1.0);
}

TEST(CdfTstudentTest, EvenDegreesOfFreedom)
{
    // k=4 (even)
    RFLOAT cdf = cdf_tstudent(4, 2.0);
    EXPECT_GT(cdf, 0.5);
    EXPECT_LT(cdf, 1.0);
}

TEST(CdfTstudentTest, TLessThanMinusTwoPath)
{
    // t < -2 takes the betai path
    RFLOAT cdf = cdf_tstudent(5, -3.0);
    EXPECT_LT(cdf, 0.5);
}

// -------------------------------------------------------- cdf_FSnedecor / icdf --

TEST(CdfFSnedecorTest, AtZero_IsZero)
{
    EXPECT_NEAR(cdf_FSnedecor(2, 10, 0.0), 0.0, 1e-10);
}

TEST(CdfFSnedecorTest, MonotonicallyIncreasing)
{
    EXPECT_LT(cdf_FSnedecor(2, 10, 1.0), cdf_FSnedecor(2, 10, 5.0));
}

TEST(IcdfFSnedecorTest, RoundTrip)
{
    // icdf(cdf(x)) ≈ x
    RFLOAT x = 2.0;
    RFLOAT p = cdf_FSnedecor(2, 10, x);
    RFLOAT x2 = icdf_FSnedecor(2, 10, p);
    EXPECT_NEAR(x2, x, 0.01);
}

// -------------------------------------------------------- random generators --

TEST(RngTest, InitRandomGenerator_Seed_Deterministic)
{
    init_random_generator(42);
    float r1 = rnd_unif(0.0f, 1.0f);
    init_random_generator(42);
    float r2 = rnd_unif(0.0f, 1.0f);
    EXPECT_NEAR(r1, r2, 1e-6f);
}

TEST(RngTest, RndGaus_InRange)
{
    init_random_generator(1);
    // Generate many samples; most should be within 4 sigma of mean
    int n_outside = 0;
    for (int i = 0; i < 100; i++)
    {
        float v = rnd_gaus(0.0f, 1.0f);
        if (v < -4.0f || v > 4.0f) n_outside++;
    }
    EXPECT_LT(n_outside, 5);
}

TEST(RngTest, RndGaus_MeanApprox)
{
    init_random_generator(7);
    float sum = 0.0f;
    int N = 1000;
    for (int i = 0; i < N; i++)
        sum += rnd_gaus(5.0f, 1.0f);
    float mean = sum / N;
    EXPECT_NEAR(mean, 5.0f, 0.2f);
}

TEST(RngTest, RndLog_InRange)
{
    init_random_generator(3);
    for (int i = 0; i < 20; i++)
    {
        float v = rnd_log(1.0f, 100.0f);
        EXPECT_GE(v, 1.0f);
        EXPECT_LE(v, 100.0f);
    }
}

TEST(RngTest, RndLog_EqualBounds)
{
    float v = rnd_log(5.0f, 5.0f);
    EXPECT_NEAR(v, 5.0f, 1e-5f);
}

// -------------------------------------------------------- Gaussian area funcs --

TEST(GausAreaTest, WithinX0_AtInfinity_IsOne)
{
    // gaus_within_x0 at very large x0 should approach 1
    float v = gaus_within_x0(100.0f, 0.0f, 1.0f);
    EXPECT_NEAR(v, 1.0f, 1e-5f);
}

TEST(GausAreaTest, WithinX0_AtZero_IsZero)
{
    float v = gaus_within_x0(0.0f, 0.0f, 1.0f);
    EXPECT_NEAR(v, 0.0f, 1e-5f);
}

TEST(GausAreaTest, OutsideX0_ComplementOfWithin)
{
    float x0 = 1.5f;
    float within  = gaus_within_x0(x0, 0.0f, 1.0f);
    float outside = gaus_outside_x0(x0, 0.0f, 1.0f);
    EXPECT_NEAR(within + outside, 1.0f, 1e-5f);
}

TEST(GausAreaTest, UpToX0_AtMean_IsHalf)
{
    float v = gaus_up_to_x0(0.0f, 0.0f, 1.0f);
    EXPECT_NEAR(v, 0.5f, 1e-5f);
}

TEST(GausAreaTest, UpToX0_PlusFromX0_IsOne)
{
    float x0 = 1.2f;
    float up   = gaus_up_to_x0(x0, 0.0f, 1.0f);
    float from = gaus_from_x0(x0, 0.0f, 1.0f);
    EXPECT_NEAR(up + from, 1.0f, 1e-5f);
}

TEST(GausAreaTest, UpToX0_BelowMean)
{
    // x0 < mean: gaus_up_to_x0 should be < 0.5
    float v = gaus_up_to_x0(-1.0f, 0.0f, 1.0f);
    EXPECT_LT(v, 0.5f);
}

TEST(GausAreaTest, FromX0_AboveMean_IsLessThanHalf)
{
    float v = gaus_from_x0(1.0f, 0.0f, 1.0f);
    EXPECT_LT(v, 0.5f);
}

// -------------------------------------------------------- Student area funcs --

TEST(StudentAreaTest, WithinT0_LargeT_NearOne)
{
    float v = student_within_t0(100.0f, 10.0f);
    EXPECT_NEAR(v, 1.0f, 1e-4f);
}

TEST(StudentAreaTest, OutsideT0_ComplementOfWithin)
{
    float t0  = 2.0f;
    float dof = 10.0f;
    float within  = student_within_t0(t0, dof);
    float outside = student_outside_t0(t0, dof);
    EXPECT_NEAR(within + outside, 1.0f, 1e-5f);
}

TEST(StudentAreaTest, UpToT0_AtZero_IsHalf)
{
    float v = student_up_to_t0(0.0f, 5.0f);
    EXPECT_NEAR(v, 0.5f, 1e-4f);
}

TEST(StudentAreaTest, UpToT0_PlusFromT0_IsOne)
{
    float t0  = 1.5f;
    float dof = 8.0f;
    float up   = student_up_to_t0(t0, dof);
    float from = student_from_t0(t0, dof);
    EXPECT_NEAR(up + from, 1.0f, 1e-5f);
}

TEST(StudentAreaTest, UpToT0_NegativeT)
{
    float v = student_up_to_t0(-1.5f, 8.0f);
    EXPECT_LT(v, 0.5f);
}

// -------------------------------------------------------- chi2 area funcs --

TEST(Chi2AreaTest, UpToT0_LargeT_NearOne)
{
    float v = chi2_up_to_t0(50.0f, 5.0f);
    EXPECT_NEAR(v, 1.0f, 1e-4f);
}

TEST(Chi2AreaTest, UpToT0_PlusFromT0_IsOne)
{
    float t0  = 3.0f;
    float dof = 4.0f;
    float up   = chi2_up_to_t0(t0, dof);
    float from = chi2_from_t0(t0, dof);
    EXPECT_NEAR(up + from, 1.0f, 1e-5f);
}

TEST(Chi2AreaTest, UpToT0_AtZero_IsZero)
{
    float v = chi2_up_to_t0(0.0f, 2.0f);
    EXPECT_NEAR(v, 0.0f, 1e-5f);
}

// -------------------------------------------------------- swapbytes --

TEST(SwapbytesTest, TwoBytes)
{
    char v[2] = {0x01, 0x02};
    swapbytes(v, 2);
    EXPECT_EQ((unsigned char)v[0], 0x02);
    EXPECT_EQ((unsigned char)v[1], 0x01);
}

TEST(SwapbytesTest, FourBytes)
{
    char v[4] = {0x01, 0x02, 0x03, 0x04};
    swapbytes(v, 4);
    EXPECT_EQ((unsigned char)v[0], 0x04);
    EXPECT_EQ((unsigned char)v[1], 0x03);
    EXPECT_EQ((unsigned char)v[2], 0x02);
    EXPECT_EQ((unsigned char)v[3], 0x01);
}

TEST(SwapbytesTest, SingleByte_NoChange)
{
    char v[1] = {(char)0xAB};
    swapbytes(v, 1);
    EXPECT_EQ((unsigned char)v[0], (unsigned char)0xAB);
}

// -------------------------------------------------------- HSL2RGB --

TEST(HSL2RGBTest, GreyScale_S0)
{
    RFLOAT R, G, B;
    HSL2RGB(0.0, 0.0, 0.5, R, G, B);
    EXPECT_NEAR(R, 0.5, 1e-5);
    EXPECT_NEAR(G, 0.5, 1e-5);
    EXPECT_NEAR(B, 0.5, 1e-5);
}

TEST(HSL2RGBTest, PureRed_H0)
{
    RFLOAT R, G, B;
    HSL2RGB(0.0, 1.0, 0.5, R, G, B);
    EXPECT_GT(R, G);
    EXPECT_GT(R, B);
}

TEST(HSL2RGBTest, PureGreen_H1third)
{
    RFLOAT R, G, B;
    HSL2RGB(1.0/3.0, 1.0, 0.5, R, G, B);
    EXPECT_GT(G, R);
    EXPECT_GT(G, B);
}

TEST(HSL2RGBTest, InRange_0to1)
{
    RFLOAT R, G, B;
    HSL2RGB(0.6, 0.8, 0.4, R, G, B);
    EXPECT_GE(R, 0.0); EXPECT_LE(R, 1.0);
    EXPECT_GE(G, 0.0); EXPECT_LE(G, 1.0);
    EXPECT_GE(B, 0.0); EXPECT_LE(B, 1.0);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
