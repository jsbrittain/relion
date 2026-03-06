/*
 * Unit tests for src/numerical_recipes.h / src/numerical_recipes.cpp
 *
 * Covers:
 *   - bessj0   — zeroth-order Bessel function of the first kind J0
 *   - bessi0   — zeroth-order modified Bessel function I0
 *   - gammln   — natural log of the Gamma function
 *   - gammp    — incomplete gamma function P(a,x)
 *   - betai    — incomplete beta function I_x(a,b)
 *   - ludcmp / lubksb — LU decomposition and back-substitution (template)
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "src/numerical_recipes.h"

static constexpr double EPS = 1e-4;

// ---------------------------------------------------------------- bessj0 --

TEST(Bessj0Test, AtZero_IsOne)
{
    EXPECT_NEAR(bessj0(0.0), 1.0, EPS);
}

TEST(Bessj0Test, FirstZeroAt2_4048)
{
    // J0 first zero ≈ 2.4048
    EXPECT_NEAR(bessj0(2.4048), 0.0, 0.001);
}

TEST(Bessj0Test, SecondZeroAt5_5201)
{
    // J0 second zero ≈ 5.5201
    EXPECT_NEAR(bessj0(5.5201), 0.0, 0.001);
}

TEST(Bessj0Test, AtPi_NearMinusHalf)
{
    // J0(π) ≈ -0.3042
    RFLOAT val = bessj0(M_PI);
    EXPECT_LT(val, 0.0);  // negative between first two zeros
}

TEST(Bessj0Test, LargeArgument_SmallMagnitude)
{
    // J0 decays for large arguments
    RFLOAT val = bessj0(20.0);
    EXPECT_LT(std::abs(val), 0.2);
}

// ---------------------------------------------------------------- bessi0 --

TEST(Bessi0Test, AtZero_IsOne)
{
    EXPECT_NEAR(bessi0(0.0), 1.0, EPS);
}

TEST(Bessi0Test, MonotoneIncreasing)
{
    // I0 is monotonically increasing for x >= 0
    EXPECT_LT(bessi0(0.0), bessi0(1.0));
    EXPECT_LT(bessi0(1.0), bessi0(2.0));
    EXPECT_LT(bessi0(2.0), bessi0(5.0));
}

TEST(Bessi0Test, AtOne_KnownValue)
{
    // I0(1) ≈ 1.2661
    EXPECT_NEAR(bessi0(1.0), 1.2661, 0.001);
}

TEST(Bessi0Test, EvenFunction_SameAtNegative)
{
    // I0(-x) = I0(x) for all x
    EXPECT_NEAR(bessi0(-1.0), bessi0(1.0), EPS);
    EXPECT_NEAR(bessi0(-3.0), bessi0(3.0), EPS);
}

// ---------------------------------------------------------------- gammln --

TEST(GammlnTest, GammaOne_IsZero)
{
    // Gamma(1) = 1 → ln(1) = 0
    EXPECT_NEAR(gammln(1.0), 0.0, EPS);
}

TEST(GammlnTest, GammaTwo_IsZero)
{
    // Gamma(2) = 1! = 1 → ln(1) = 0
    EXPECT_NEAR(gammln(2.0), 0.0, EPS);
}

TEST(GammlnTest, GammaThree_IsLnTwo)
{
    // Gamma(3) = 2! = 2 → ln(2)
    EXPECT_NEAR(gammln(3.0), std::log(2.0), EPS);
}

TEST(GammlnTest, GammaFour_IsLnSix)
{
    // Gamma(4) = 3! = 6 → ln(6)
    EXPECT_NEAR(gammln(4.0), std::log(6.0), EPS);
}

TEST(GammlnTest, GammaFive_IsLnTwentyFour)
{
    // Gamma(5) = 4! = 24 → ln(24)
    EXPECT_NEAR(gammln(5.0), std::log(24.0), EPS);
}

TEST(GammlnTest, GammaHalf_IsLnSqrtPi)
{
    // Gamma(0.5) = sqrt(π) → ln(sqrt(π)) = 0.5*ln(π)
    EXPECT_NEAR(gammln(0.5), 0.5 * std::log(M_PI), EPS);
}

// ---------------------------------------------------------------- gammp --

TEST(GammpTest, AtZero_IsZero)
{
    // P(a, 0) = 0 for all a > 0
    EXPECT_NEAR(gammp(1.0, 0.0), 0.0, EPS);
    EXPECT_NEAR(gammp(2.0, 0.0), 0.0, EPS);
}

TEST(GammpTest, LargeX_IsOne)
{
    // P(a, x→∞) → 1
    EXPECT_NEAR(gammp(1.0, 100.0), 1.0, EPS);
    EXPECT_NEAR(gammp(2.0, 100.0), 1.0, EPS);
}

TEST(GammpTest, MonotoneIncreasingInX)
{
    // P(a, x) is monotonically increasing in x
    RFLOAT a = 2.0;
    EXPECT_LT(gammp(a, 1.0), gammp(a, 2.0));
    EXPECT_LT(gammp(a, 2.0), gammp(a, 5.0));
}

TEST(GammpTest, InRange01)
{
    // P must be in [0, 1]
    for (double x : {0.5, 1.0, 2.0, 5.0, 10.0})
    {
        RFLOAT p = gammp(2.0, x);
        EXPECT_GE(p, 0.0) << "x=" << x;
        EXPECT_LE(p, 1.0) << "x=" << x;
    }
}

// ---------------------------------------------------------------- betai --

TEST(BetaiTest, AtZero_IsZero)
{
    // I_0(a,b) = 0 for all a,b > 0
    EXPECT_NEAR(betai(1.0, 1.0, 0.0), 0.0, EPS);
    EXPECT_NEAR(betai(2.0, 3.0, 0.0), 0.0, EPS);
}

TEST(BetaiTest, AtOne_IsOne)
{
    // I_1(a,b) = 1 for all a,b > 0
    EXPECT_NEAR(betai(1.0, 1.0, 1.0), 1.0, EPS);
    EXPECT_NEAR(betai(2.0, 3.0, 1.0), 1.0, EPS);
}

TEST(BetaiTest, SymmetricUniform_IsHalf)
{
    // For Uniform distribution (a=1, b=1): I_0.5(1,1) = 0.5
    EXPECT_NEAR(betai(1.0, 1.0, 0.5), 0.5, EPS);
}

TEST(BetaiTest, Symmetry_I_x_ab_Plus_I_1minusx_ba_Equals1)
{
    // I_x(a,b) + I_{1-x}(b,a) = 1
    RFLOAT a = 2.0, b = 3.0, x = 0.4;
    EXPECT_NEAR(betai(a, b, x) + betai(b, a, 1.0 - x), 1.0, EPS);
}

TEST(BetaiTest, MonotoneIncreasingInX)
{
    RFLOAT a = 2.0, b = 3.0;
    EXPECT_LT(betai(a, b, 0.2), betai(a, b, 0.5));
    EXPECT_LT(betai(a, b, 0.5), betai(a, b, 0.8));
}

// --------------------------------------------------- ludcmp / lubksb --

// Solve the 2×2 system  [[2, 1], [1, 3]] * x = [5, 10]
// Solution: x = [1, 3]
// NB: ludcmp uses 1-indexed storage: a[i*n+j] for i,j in [1..n].
//     Array must be sized at least n*(n+1) to hold indices [1..n][1..n].
TEST(LudcmpTest, Solve2x2System)
{
    const int n = 2;
    // stride = n; indices [1..n], so max index = n*n+n = 6 → size 7
    double a[7] = {0};
    a[1*n+1] = 2.0;  a[1*n+2] = 1.0;
    a[2*n+1] = 1.0;  a[2*n+2] = 3.0;

    int indx[n+1] = {0};
    double d = 0;
    ludcmp(a, n, indx, &d);

    double b[n+1] = {0, 5.0, 10.0};
    lubksb(a, n, indx, b);

    EXPECT_NEAR(b[1], 1.0, 1e-9);
    EXPECT_NEAR(b[2], 3.0, 1e-9);
}

TEST(LudcmpTest, Solve3x3System)
{
    // 3×3 diagonal system: diag(2,3,4) * [1,2,3] = [2,6,12]
    const int n = 3;
    // stride = n = 3; max index = n*n+n = 12 → size 13
    double a[13] = {0};
    a[1*n+1] = 2.0;
    a[2*n+2] = 3.0;
    a[3*n+3] = 4.0;

    int indx[n+1] = {0};
    double d = 0;
    ludcmp(a, n, indx, &d);

    double b[n+1] = {0, 2.0, 6.0, 12.0};
    lubksb(a, n, indx, b);

    EXPECT_NEAR(b[1], 1.0, 1e-9);
    EXPECT_NEAR(b[2], 2.0, 1e-9);
    EXPECT_NEAR(b[3], 3.0, 1e-9);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
