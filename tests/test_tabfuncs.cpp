/*
 * Unit tests for src/tabfuncs.h / src/tabfuncs.cpp
 *
 * Covers:
 *   - TabSine    — tabulated sine function
 *   - TabCosine  — tabulated cosine function
 *   - TabBlob    — tabulated blob spatial value
 *   - TabFtBlob  — tabulated blob Fourier transform value
 */

#include <gtest/gtest.h>
#include <cmath>
#include "src/tabfuncs.h"
#include "src/funcs.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Accuracy tolerance: 5000-element table covering [0, 2π],
// sampling ≈ 2π/5000 ≈ 0.00126 rad; max error ≈ sampling ≈ 0.0013.
static constexpr double kTrigTol = 0.002;

// ---------------------------------------------------------------- TabSine --

class TabSineTest : public ::testing::Test
{
protected:
    TabSine ts;
    void SetUp() override { ts.initialise(5000); }
};

TEST_F(TabSineTest, Zero_IsZero)
{
    EXPECT_NEAR(ts(0.0), 0.0, kTrigTol);
}

TEST_F(TabSineTest, HalfPi_IsOne)
{
    EXPECT_NEAR(ts(M_PI / 2.0), 1.0, kTrigTol);
}

TEST_F(TabSineTest, Pi_IsZero)
{
    EXPECT_NEAR(ts(M_PI), 0.0, kTrigTol);
}

TEST_F(TabSineTest, NegativeArgument_IsOddFunction)
{
    // sin(-x) == -sin(x)
    RFLOAT x = M_PI / 3.0;
    EXPECT_NEAR(ts(-x), -ts(x), kTrigTol);
}

TEST_F(TabSineTest, MatchesStdSin_AtSeveralPoints)
{
    for (double x : {0.1, 0.5, 1.0, 1.5, 2.0, 2.5})
        EXPECT_NEAR(ts(x), std::sin(x), kTrigTol) << "x=" << x;
}

TEST_F(TabSineTest, PeriodWrapsCorrectly)
{
    // sin(x) == sin(x + 2π)
    RFLOAT x = 1.0;
    EXPECT_NEAR(ts(x), ts(x + 2.0 * M_PI), kTrigTol);
}

// -------------------------------------------------------------- TabCosine --

class TabCosineTest : public ::testing::Test
{
protected:
    TabCosine tc;
    void SetUp() override { tc.initialise(5000); }
};

TEST_F(TabCosineTest, Zero_IsOne)
{
    EXPECT_NEAR(tc(0.0), 1.0, kTrigTol);
}

TEST_F(TabCosineTest, HalfPi_IsZero)
{
    EXPECT_NEAR(tc(M_PI / 2.0), 0.0, kTrigTol);
}

TEST_F(TabCosineTest, Pi_IsMinusOne)
{
    EXPECT_NEAR(tc(M_PI), -1.0, kTrigTol);
}

TEST_F(TabCosineTest, NegativeArgument_IsEvenFunction)
{
    // cos(-x) == cos(x)
    RFLOAT x = M_PI / 4.0;
    EXPECT_NEAR(tc(-x), tc(x), kTrigTol);
}

TEST_F(TabCosineTest, MatchesStdCos_AtSeveralPoints)
{
    for (double x : {0.1, 0.5, 1.0, 1.5, 2.0, 2.5})
        EXPECT_NEAR(tc(x), std::cos(x), kTrigTol) << "x=" << x;
}

TEST_F(TabCosineTest, PeriodWrapsCorrectly)
{
    RFLOAT x = 1.0;
    EXPECT_NEAR(tc(x), tc(x + 2.0 * M_PI), kTrigTol);
}

// ---------------------------------------------------------------- TabBlob --

class TabBlobTest : public ::testing::Test
{
protected:
    TabBlob tb;
    RFLOAT radius = 2.0;
    RFLOAT alpha  = 3.6;
    int    order  = 2;

    void SetUp() override { tb.initialise(radius, alpha, order, 10000); }
};

TEST_F(TabBlobTest, AtCenter_IsPositive)
{
    EXPECT_GT(tb(0.0), 0.0);
}

TEST_F(TabBlobTest, InsideRadius_IsPositive)
{
    EXPECT_GT(tb(1.0), 0.0);
}

TEST_F(TabBlobTest, BeyondRadius_IsZero)
{
    EXPECT_NEAR(tb(radius + 0.1), 0.0, 1e-10);
}

TEST_F(TabBlobTest, WellBeyond_IsZero)
{
    EXPECT_NEAR(tb(10.0), 0.0, 1e-10);
}

TEST_F(TabBlobTest, ApproximatesKaiserValue)
{
    // Table approximation should be close to the direct computation.
    for (RFLOAT r : {0.0, 0.5, 1.0, 1.5})
    {
        RFLOAT exact = kaiser_value(r, radius, alpha, order);
        RFLOAT tabulated = tb(r);
        EXPECT_NEAR(tabulated, exact, exact * 0.01 + 1e-6)
            << "r=" << r;
    }
}

TEST_F(TabBlobTest, MonotoneDecreasing)
{
    EXPECT_GE(tb(0.0), tb(0.5));
    EXPECT_GE(tb(0.5), tb(1.0));
    EXPECT_GE(tb(1.0), tb(1.5));
}

// -------------------------------------------------------------- TabFtBlob --

class TabFtBlobTest : public ::testing::Test
{
protected:
    TabFtBlob tfb;
    RFLOAT radius = 2.0;
    RFLOAT alpha  = 3.6;
    int    order  = 2;

    void SetUp() override { tfb.initialise(radius, alpha, order, 10000); }
};

TEST_F(TabFtBlobTest, AtZeroFrequency_IsPositive)
{
    EXPECT_GT(tfb(0.0), 0.0);
}

TEST_F(TabFtBlobTest, ApproximatesKaiserFourierValue)
{
    for (RFLOAT w : {0.0, 0.1, 0.2, 0.3})
    {
        RFLOAT exact = kaiser_Fourier_value(w, radius, alpha, order);
        RFLOAT tabulated = tfb(w);
        EXPECT_NEAR(tabulated, exact, exact * 0.01 + 1e-6)
            << "w=" << w;
    }
}

TEST_F(TabFtBlobTest, BeyondNyquist_IsZero)
{
    // Frequencies beyond 0.5 (Nyquist) should fall off to zero.
    // The table covers [0, 0.5] with 10000 elements; beyond → 0.
    EXPECT_NEAR(tfb(1.0), 0.0, 1e-10);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
