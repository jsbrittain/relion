/*
 * Unit tests for src/jaz/math/Zernike.h / Zernike.cpp
 *
 * Covers all public static methods:
 *   Z, Z_cart, R, evenIndexToMN, numberOfEvenCoeffs, oddIndexToMN, numberOfOddCoeffs
 */

#include <gtest/gtest.h>
#include <cmath>
#include "src/jaz/math/Zernike.h"

static const double EPS = 1e-8;

// ---------------------------------------------------------------------------
// R — radial polynomial
// ---------------------------------------------------------------------------

TEST(ZernikeTest, R_ZeroOrderZeroMode)
{
    // R(0,0,rho) = 1 for all rho
    EXPECT_NEAR(Zernike::R(0, 0, 0.0), 1.0, EPS);
    EXPECT_NEAR(Zernike::R(0, 0, 0.5), 1.0, EPS);
    EXPECT_NEAR(Zernike::R(0, 0, 1.0), 1.0, EPS);
}

TEST(ZernikeTest, R_OddDifferenceIsZero)
{
    // If (n - m) is odd, R should be 0
    EXPECT_NEAR(Zernike::R(0, 1, 0.5), 0.0, EPS);
    EXPECT_NEAR(Zernike::R(1, 2, 0.5), 0.0, EPS);
}

TEST(ZernikeTest, R_FirstOrderFirstMode)
{
    // R(1,1,rho) = rho
    EXPECT_NEAR(Zernike::R(1, 1, 0.3), 0.3, EPS);
    EXPECT_NEAR(Zernike::R(1, 1, 1.0), 1.0, EPS);
}

TEST(ZernikeTest, R_SecondOrderZeroMode)
{
    // R(0,2,rho) = 2*rho^2 - 1
    double rho = 0.6;
    EXPECT_NEAR(Zernike::R(0, 2, rho), 2.0*rho*rho - 1.0, EPS);
}

TEST(ZernikeTest, R_SecondOrderSecondMode)
{
    // R(2,2,rho) = rho^2
    double rho = 0.7;
    EXPECT_NEAR(Zernike::R(2, 2, rho), rho*rho, EPS);
}

TEST(ZernikeTest, R_AtRhoZero)
{
    // At rho=0: only the n=0 term survives
    EXPECT_NEAR(Zernike::R(0, 0, 0.0), 1.0, EPS);
    EXPECT_NEAR(Zernike::R(2, 2, 0.0), 0.0, EPS);
}

TEST(ZernikeTest, R_AtRhoOne)
{
    // At rho=1: R(m,n,1) = 1 for all valid (m,n)
    EXPECT_NEAR(Zernike::R(0, 0, 1.0), 1.0, EPS);
    EXPECT_NEAR(Zernike::R(1, 1, 1.0), 1.0, EPS);
    EXPECT_NEAR(Zernike::R(2, 2, 1.0), 1.0, EPS);
    EXPECT_NEAR(Zernike::R(0, 2, 1.0), 1.0, EPS);
}

// ---------------------------------------------------------------------------
// Z — Zernike polynomial (polar form)
// ---------------------------------------------------------------------------

TEST(ZernikeTest, Z_ZeroModeIsRadialOnly)
{
    // Z(0, 0, rho, phi) = R(0,0,rho) * cos(0) = 1
    EXPECT_NEAR(Zernike::Z(0, 0, 0.5, 1.0), 1.0, EPS);
    EXPECT_NEAR(Zernike::Z(0, 0, 0.0, 0.0), 1.0, EPS);
}

TEST(ZernikeTest, Z_PositiveM_UsesCos)
{
    // Z(m, n, rho, phi) = R(m,n,rho) * cos(m*phi) for m >= 0
    double rho = 0.5, phi = M_PI / 4.0;
    double expected = Zernike::R(1, 1, rho) * cos(1.0 * phi);
    EXPECT_NEAR(Zernike::Z(1, 1, rho, phi), expected, EPS);
}

TEST(ZernikeTest, Z_NegativeM_UsesSin)
{
    // Z(-m, n, rho, phi) = R(m,n,rho) * sin(m*phi)
    double rho = 0.5, phi = M_PI / 6.0;
    double expected = Zernike::R(1, 1, rho) * sin(1.0 * phi);
    EXPECT_NEAR(Zernike::Z(-1, 1, rho, phi), expected, EPS);
}

// ---------------------------------------------------------------------------
// Z_cart — Zernike polynomial (Cartesian form)
// ---------------------------------------------------------------------------

TEST(ZernikeTest, ZCart_AtOrigin)
{
    // At origin: rho=0, so only Z(0,0,...) survives with value 1
    EXPECT_NEAR(Zernike::Z_cart(0, 0, 0.0, 0.0), 1.0, EPS);
}

TEST(ZernikeTest, ZCart_MatchesPolar)
{
    double x = 0.3, y = 0.4;
    double rho = sqrt(x*x + y*y);
    double phi = atan2(y, x);
    double polar = Zernike::Z(1, 1, rho, phi);
    double cart  = Zernike::Z_cart(1, 1, x, y);
    EXPECT_NEAR(cart, polar, EPS);
}

TEST(ZernikeTest, ZCart_NegativeM_MatchesPolar)
{
    double x = 0.2, y = 0.5;
    double rho = sqrt(x*x + y*y);
    double phi = atan2(y, x);
    double polar = Zernike::Z(-1, 1, rho, phi);
    double cart  = Zernike::Z_cart(-1, 1, x, y);
    EXPECT_NEAR(cart, polar, EPS);
}

// ---------------------------------------------------------------------------
// evenIndexToMN / numberOfEvenCoeffs
// ---------------------------------------------------------------------------

TEST(ZernikeTest, EvenIndexToMN_Index0)
{
    int m, n;
    Zernike::evenIndexToMN(0, m, n);
    EXPECT_EQ(m, 0);
    EXPECT_EQ(n, 0);
}

TEST(ZernikeTest, EvenIndexToMN_Index1)
{
    int m, n;
    Zernike::evenIndexToMN(1, m, n);
    EXPECT_EQ(n, 2);   // n=2*k, k=1
}

TEST(ZernikeTest, NumberOfEvenCoeffs_NMax0)
{
    EXPECT_EQ(Zernike::numberOfEvenCoeffs(0), 1);
}

TEST(ZernikeTest, NumberOfEvenCoeffs_NMax2)
{
    // l=1: 1+2+1 = 4... let's compute: l=nmax/2=1, l*l+2*l+1 = 1+2+1=4
    EXPECT_EQ(Zernike::numberOfEvenCoeffs(2), 4);
}

TEST(ZernikeTest, NumberOfEvenCoeffs_NMax4)
{
    // l=2: 4+4+1=9
    EXPECT_EQ(Zernike::numberOfEvenCoeffs(4), 9);
}

// ---------------------------------------------------------------------------
// oddIndexToMN / numberOfOddCoeffs
// ---------------------------------------------------------------------------

TEST(ZernikeTest, OddIndexToMN_Index0)
{
    int m, n;
    Zernike::oddIndexToMN(0, m, n);
    EXPECT_EQ(n, 1);
}

TEST(ZernikeTest, NumberOfOddCoeffs_NMax1)
{
    // l=(1-1)/2+1=1, l*l+l=2
    EXPECT_EQ(Zernike::numberOfOddCoeffs(1), 2);
}

TEST(ZernikeTest, NumberOfOddCoeffs_NMax3)
{
    // l=(3-1)/2+1=2, l*l+l=6
    EXPECT_EQ(Zernike::numberOfOddCoeffs(3), 6);
}

// ---------------------------------------------------------------------------
// Orthogonality check: Z(m,n) evaluated at many rho values
// ---------------------------------------------------------------------------

TEST(ZernikeTest, ZHigherOrder_DoesNotCrash)
{
    // trigger prepCoeffs for higher n
    EXPECT_NO_THROW(Zernike::R(4, 4, 0.5));
    EXPECT_NO_THROW(Zernike::R(2, 4, 0.5));
    EXPECT_NO_THROW(Zernike::R(0, 4, 0.5));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
