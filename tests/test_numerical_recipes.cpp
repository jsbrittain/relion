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

// ---------------------------------------------------------------- bessi1 --

TEST(Bessi1Test, AtZero_IsZero)
{
    EXPECT_NEAR(bessi1(0.0), 0.0, EPS);
}

TEST(Bessi1Test, AtOne_KnownValue)
{
    // I1(1) ≈ 0.5652
    EXPECT_NEAR(bessi1(1.0), 0.5652, 0.001);
}

TEST(Bessi1Test, PositiveForPositiveX)
{
    EXPECT_GT(bessi1(0.5), 0.0);
    EXPECT_GT(bessi1(2.0), 0.0);
    EXPECT_GT(bessi1(5.0), 0.0);
}

TEST(Bessi1Test, MonotoneIncreasing)
{
    EXPECT_LT(bessi1(0.5), bessi1(1.0));
    EXPECT_LT(bessi1(1.0), bessi1(2.0));
}

// ----------------------------------------------- bessi0_5 / bessi1_5 --

TEST(Bessi0_5Test, AtZero_IsZero)
{
    EXPECT_NEAR(bessi0_5(0.0), 0.0, EPS);
}

TEST(Bessi0_5Test, PositiveForPositiveX)
{
    // bessi0_5(x) = sqrt(2/(pi*x)) * sinh(x) > 0 for x > 0
    EXPECT_GT(bessi0_5(1.0), 0.0);
    EXPECT_GT(bessi0_5(2.0), 0.0);
}

TEST(Bessi1_5Test, AtZero_IsZero)
{
    EXPECT_NEAR(bessi1_5(0.0), 0.0, EPS);
}

TEST(Bessi1_5Test, PositiveForPositiveX)
{
    EXPECT_GT(bessi1_5(1.0), 0.0);
}

// ---------------------------------------------------------------- bessi2 --

TEST(Bessi2Test, AtZero_IsZero)
{
    EXPECT_NEAR(bessi2(0.0), 0.0, EPS);
}

TEST(Bessi2Test, AtOne_KnownValue)
{
    // I2(1) ≈ 0.1357
    EXPECT_NEAR(bessi2(1.0), 0.1357, 0.002);
}

TEST(Bessi2Test, PositiveForPositiveX)
{
    EXPECT_GT(bessi2(1.0), 0.0);
    EXPECT_GT(bessi2(3.0), 0.0);
}

// ---------------------------------------------------------------- bessi3 --

TEST(Bessi3Test, AtZero_IsZero)
{
    EXPECT_NEAR(bessi3(0.0), 0.0, EPS);
}

TEST(Bessi3Test, PositiveForPositiveX)
{
    EXPECT_GT(bessi3(1.0), 0.0);
    EXPECT_GT(bessi3(2.0), 0.0);
}

TEST(Bessi3Test, SmallerThanBessi2_AtSmallX)
{
    // For small x, I_n(x) < I_{n-1}(x)
    EXPECT_LT(bessi3(1.0), bessi2(1.0));
}

// ---------------------------------------------------------------- bessi4 --

TEST(Bessi4Test, AtZero_IsZero)
{
    EXPECT_NEAR(bessi4(0.0), 0.0, EPS);
}

TEST(Bessi4Test, PositiveForPositiveX)
{
    EXPECT_GT(bessi4(1.0), 0.0);
}

TEST(Bessi4Test, SmallerThanBessi3_AtSmallX)
{
    EXPECT_LT(bessi4(1.0), bessi3(1.0));
}

// ----------------------------------------- bessi2_5 / bessi3_5 --

TEST(Bessi2_5Test, AtZero_IsZero)
{
    EXPECT_NEAR(bessi2_5(0.0), 0.0, EPS);
}

TEST(Bessi2_5Test, PositiveForPositiveX)
{
    EXPECT_GT(bessi2_5(1.0), 0.0);
}

TEST(Bessi3_5Test, AtZero_IsZero)
{
    EXPECT_NEAR(bessi3_5(0.0), 0.0, EPS);
}

TEST(Bessi3_5Test, PositiveForPositiveX)
{
    EXPECT_GT(bessi3_5(1.0), 0.0);
}

// ---------------------------------------------------------- bessj1_5 --

TEST(Bessj1_5Test, PositiveForSmallX)
{
    // J_{3/2}(x) = sqrt(2/(pi*x)) * (sin(x)/x - cos(x))
    // For small x > 0, this is positive (sin dominates)
    EXPECT_GT(bessj1_5(1.0), 0.0);
}

TEST(Bessj1_5Test, SmallMagnitudeForLargeX)
{
    // Bessel functions decay as 1/sqrt(x)
    EXPECT_LT(std::abs(bessj1_5(20.0)), 0.2);
}

// ---------------------------------------------------------- bessj3_5 --

TEST(Bessj3_5Test, PositiveForSmallX)
{
    EXPECT_GT(bessj3_5(2.0), 0.0);
}

TEST(Bessj3_5Test, SmallMagnitudeForLargeX)
{
    EXPECT_LT(std::abs(bessj3_5(20.0)), 0.2);
}

// ----------------------------------------------- recurrence relation --

TEST(BesselRecurrenceTest, Bessi2_SatisfiesRecurrence)
{
    // I_{n+1}(x) = I_{n-1}(x) - (2n/x) * I_n(x)
    // => I_2(x) = I_0(x) - (2/x) * I_1(x)
    RFLOAT x = 2.0;
    RFLOAT expected = bessi0(x) - (2.0 / x) * bessi1(x);
    EXPECT_NEAR(bessi2(x), expected, EPS);
}

TEST(BesselRecurrenceTest, Bessi3_SatisfiesRecurrence)
{
    // I_3(x) = I_1(x) - (4/x) * I_2(x)
    RFLOAT x = 3.0;
    RFLOAT expected = bessi1(x) - (4.0 / x) * bessi2(x);
    EXPECT_NEAR(bessi3(x), expected, EPS);
}

// --------------------------------- ludcmp: singular (zero-row) → exit(1) --
// nrerror() calls exit(1); use EXPECT_EXIT to catch the process death.

TEST(LudcmpTest, ZeroRowMatrix_ExitsWithError)
{
    // Row 2 is all zeros → nrerror("Singular matrix in routine LUDCMP") → exit(1)
    EXPECT_EXIT(
    {
        const int n = 2;
        double a[7] = {0};
        a[1*n+1] = 2.0; a[1*n+2] = 1.0;
        // row 2 stays zero
        int indx[n+1] = {0};
        double d = 0;
        ludcmp(a, n, indx, &d);
    },
    ::testing::ExitedWithCode(1),
    "Singular matrix");
}

// --------------------------------- ludcmp: rank-deficient → TINY fix -------
// A proportional-row matrix (rank < n) does NOT have a zero row, so the
// "Singular matrix" nrerror is NOT triggered.  Instead, the diagonal becomes
// exactly 0 during elimination and the TINY substitution prevents a
// division-by-zero.  The call must complete without crashing.

TEST(LudcmpTest, RankDeficientMatrix_TinyFix_NoCrash)
{
    // [[2, 2], [1, 1]]: rows are proportional → during LU elimination the
    // (2,2) pivot reduces to 0 and is replaced by TINY.
    const int n = 2;
    double a[7] = {0};
    a[1*n+1] = 2.0; a[1*n+2] = 2.0;
    a[2*n+1] = 1.0; a[2*n+2] = 1.0;

    int indx[n+1] = {0};
    double d = 0;
    // Must complete without crash or exit; the TINY path keeps a[j*n+j] != 0.
    ludcmp(a, n, indx, &d);

    // After ludcmp the diagonal entry a[2*n+2] must equal TINY (≈ 1e-20),
    // confirming the branch was taken.
    EXPECT_NEAR(a[2*n+2], 1.0e-20, 1.0e-25);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
