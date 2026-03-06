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

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
