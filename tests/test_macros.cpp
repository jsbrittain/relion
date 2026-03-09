/*
 * Unit tests for src/macros.h
 *
 * Covers: ABS, SGN, SGN0, XMIPP_MIN, XMIPP_MAX, ROUND, CEIL, FLOOR,
 *         FRACTION, CLIP, intWRAP, realWRAP, DEG2RAD, RAD2DEG,
 *         COSD, SIND, SINC, NEXT_POWER_OF_2, LIN_INTERP, XOR,
 *         SWAP, FIRST_XMIPP_INDEX, LAST_XMIPP_INDEX
 */

#include <gtest/gtest.h>
#include <cmath>
#include "src/macros.h"

// ---------------------------------------------------------------------------
// ABS
// ---------------------------------------------------------------------------

TEST(MacrosTest, Abs_Positive) { EXPECT_EQ(ABS(5), 5); }
TEST(MacrosTest, Abs_Negative) { EXPECT_EQ(ABS(-7), 7); }
TEST(MacrosTest, Abs_Zero)     { EXPECT_EQ(ABS(0), 0); }
TEST(MacrosTest, Abs_Float)    { EXPECT_NEAR(ABS(-3.5), 3.5, 1e-10); }

// ---------------------------------------------------------------------------
// SGN / SGN0
// ---------------------------------------------------------------------------

TEST(MacrosTest, SGN_Positive)  { EXPECT_EQ(SGN(10),  1); }
TEST(MacrosTest, SGN_Negative)  { EXPECT_EQ(SGN(-10), -1); }
TEST(MacrosTest, SGN_Zero)      { EXPECT_EQ(SGN(0),   1); } // zero maps to +1

TEST(MacrosTest, SGN0_Positive) { EXPECT_EQ(SGN0(5),  1); }
TEST(MacrosTest, SGN0_Negative) { EXPECT_EQ(SGN0(-5), -1); }
TEST(MacrosTest, SGN0_Zero)     { EXPECT_EQ(SGN0(0),  0); }

// ---------------------------------------------------------------------------
// XMIPP_MIN / XMIPP_MAX
// ---------------------------------------------------------------------------

TEST(MacrosTest, Min_FirstSmaller)  { EXPECT_EQ(XMIPP_MIN(2, 5), 2); }
TEST(MacrosTest, Min_SecondSmaller) { EXPECT_EQ(XMIPP_MIN(7, 3), 3); }
TEST(MacrosTest, Min_Equal)         { EXPECT_EQ(XMIPP_MIN(4, 4), 4); }

TEST(MacrosTest, Max_FirstLarger)   { EXPECT_EQ(XMIPP_MAX(9, 3), 9); }
TEST(MacrosTest, Max_SecondLarger)  { EXPECT_EQ(XMIPP_MAX(1, 6), 6); }
TEST(MacrosTest, Max_Equal)         { EXPECT_EQ(XMIPP_MAX(5, 5), 5); }

// ---------------------------------------------------------------------------
// ROUND
// ---------------------------------------------------------------------------

TEST(MacrosTest, Round_PositiveUp)   { EXPECT_EQ(ROUND(0.8),  1); }
TEST(MacrosTest, Round_PositiveDown) { EXPECT_EQ(ROUND(0.2),  0); }
TEST(MacrosTest, Round_NegativeUp)   { EXPECT_EQ(ROUND(-0.2), 0); }
TEST(MacrosTest, Round_NegativeDown) { EXPECT_EQ(ROUND(-0.8), -1); }
TEST(MacrosTest, Round_Half)         { EXPECT_EQ(ROUND(0.5),  1); }
TEST(MacrosTest, Round_NegHalf)      { EXPECT_EQ(ROUND(-0.5), -1); }

// ---------------------------------------------------------------------------
// CEIL
// ---------------------------------------------------------------------------

TEST(MacrosTest, Ceil_PositiveFrac)  { EXPECT_EQ(CEIL(0.2),  1); }
TEST(MacrosTest, Ceil_NegativeFrac)  { EXPECT_EQ(CEIL(-0.8), 0); }
TEST(MacrosTest, Ceil_Exact)         { EXPECT_EQ(CEIL(3.0),  3); }
TEST(MacrosTest, Ceil_NegExact)      { EXPECT_EQ(CEIL(-3.0), -3); }

// ---------------------------------------------------------------------------
// FLOOR
// ---------------------------------------------------------------------------

TEST(MacrosTest, Floor_PositiveFrac)  { EXPECT_EQ(FLOOR(0.8),  0); }
TEST(MacrosTest, Floor_NegativeFrac)  { EXPECT_EQ(FLOOR(-0.2), -1); }
TEST(MacrosTest, Floor_Exact)         { EXPECT_EQ(FLOOR(4.0),  4); }
TEST(MacrosTest, Floor_NegExact)      { EXPECT_EQ(FLOOR(-4.0), -4); }

// ---------------------------------------------------------------------------
// FRACTION
// ---------------------------------------------------------------------------

TEST(MacrosTest, Fraction_Positive) { EXPECT_NEAR(FRACTION(3.7),  0.7, 1e-9); }
TEST(MacrosTest, Fraction_Negative) { EXPECT_NEAR(FRACTION(-3.7), -0.7, 1e-9); }
TEST(MacrosTest, Fraction_Whole)    { EXPECT_NEAR(FRACTION(5.0),  0.0, 1e-9); }

// ---------------------------------------------------------------------------
// CLIP
// ---------------------------------------------------------------------------

TEST(MacrosTest, Clip_BelowMin)    { EXPECT_EQ(CLIP(-5, -2, 2), -2); }
TEST(MacrosTest, Clip_AboveMax)    { EXPECT_EQ(CLIP(10, -2, 2),  2); }
TEST(MacrosTest, Clip_InRange)     { EXPECT_EQ(CLIP(1,  -2, 2),  1); }
TEST(MacrosTest, Clip_AtMin)       { EXPECT_EQ(CLIP(-2, -2, 2), -2); }
TEST(MacrosTest, Clip_AtMax)       { EXPECT_EQ(CLIP(2,  -2, 2),  2); }

// ---------------------------------------------------------------------------
// intWRAP
// ---------------------------------------------------------------------------
// intWRAP(x,-2,2): cycle length = 5 elements: -2,-1,0,1,2

TEST(MacrosTest, IntWrap_InRange)  { EXPECT_EQ(intWRAP(0,  -2, 2),  0); }
TEST(MacrosTest, IntWrap_AtLow)    { EXPECT_EQ(intWRAP(-2, -2, 2), -2); }
TEST(MacrosTest, IntWrap_AtHigh)   { EXPECT_EQ(intWRAP(2,  -2, 2),  2); }
TEST(MacrosTest, IntWrap_AboveHigh){ EXPECT_EQ(intWRAP(3,  -2, 2), -2); }
TEST(MacrosTest, IntWrap_BelowLow) { EXPECT_EQ(intWRAP(-3, -2, 2),  2); }

// ---------------------------------------------------------------------------
// realWRAP
// ---------------------------------------------------------------------------
// realWRAP(angle, 0, 360): angles map into [0, 360)

TEST(MacrosTest, RealWrap_InRange)
{
    EXPECT_NEAR(realWRAP(180.0, 0.0, 360.0), 180.0, 1e-9);
}
TEST(MacrosTest, RealWrap_Above)
{
    EXPECT_NEAR(realWRAP(400.0, 0.0, 360.0), 40.0, 1e-6);
}
TEST(MacrosTest, RealWrap_Below)
{
    EXPECT_NEAR(realWRAP(-10.0, 0.0, 360.0), 350.0, 1e-6);
}

// ---------------------------------------------------------------------------
// DEG2RAD / RAD2DEG
// ---------------------------------------------------------------------------

TEST(MacrosTest, Deg2Rad_90)   { EXPECT_NEAR(DEG2RAD(90.0),  PI/2,   1e-10); }
TEST(MacrosTest, Deg2Rad_180)  { EXPECT_NEAR(DEG2RAD(180.0), PI,     1e-10); }
TEST(MacrosTest, Rad2Deg_Pi)   { EXPECT_NEAR(RAD2DEG(PI),    180.0,  1e-9);  }
TEST(MacrosTest, Rad2Deg_PiH)  { EXPECT_NEAR(RAD2DEG(PI/2),  90.0,   1e-9);  }

TEST(MacrosTest, Deg2Rad_Rad2Deg_RoundTrip)
{
    EXPECT_NEAR(RAD2DEG(DEG2RAD(45.0)), 45.0, 1e-10);
}

// ---------------------------------------------------------------------------
// COSD / SIND
// ---------------------------------------------------------------------------

TEST(MacrosTest, CosDeg_0)   { EXPECT_NEAR(COSD(0.0),   1.0, 1e-10); }
TEST(MacrosTest, CosDeg_90)  { EXPECT_NEAR(COSD(90.0),  0.0, 1e-10); }
TEST(MacrosTest, CosDeg_180) { EXPECT_NEAR(COSD(180.0), -1.0, 1e-10); }

TEST(MacrosTest, SinDeg_0)   { EXPECT_NEAR(SIND(0.0),   0.0, 1e-10); }
TEST(MacrosTest, SinDeg_90)  { EXPECT_NEAR(SIND(90.0),  1.0, 1e-10); }
TEST(MacrosTest, SinDeg_180) { EXPECT_NEAR(SIND(180.0), 0.0, 1e-10); }

// ---------------------------------------------------------------------------
// SINC
// ---------------------------------------------------------------------------

TEST(MacrosTest, Sinc_Zero)      { EXPECT_NEAR(SINC(0.0),  1.0,  1e-10); }
TEST(MacrosTest, Sinc_One)       { EXPECT_NEAR(SINC(1.0),  0.0,  1e-10); }
TEST(MacrosTest, Sinc_Half)      { EXPECT_NEAR(SINC(0.5),  2.0/PI, 1e-10); }
TEST(MacrosTest, Sinc_Negative)  { EXPECT_NEAR(SINC(-1.0), 0.0,  1e-10); }

// ---------------------------------------------------------------------------
// NEXT_POWER_OF_2
// ---------------------------------------------------------------------------

TEST(MacrosTest, NextPow2_Exact)   { EXPECT_NEAR(NEXT_POWER_OF_2(8),    8.0,    1e-6); }
TEST(MacrosTest, NextPow2_Between) { EXPECT_NEAR(NEXT_POWER_OF_2(1000), 1024.0, 1.0); }
TEST(MacrosTest, NextPow2_Two)     { EXPECT_NEAR(NEXT_POWER_OF_2(2),    2.0,    1e-6); }

// ---------------------------------------------------------------------------
// LIN_INTERP
// ---------------------------------------------------------------------------

TEST(MacrosTest, LinInterp_ZeroAlpha) { EXPECT_NEAR(LIN_INTERP(0.0, 10.0, 20.0), 10.0, 1e-10); }
TEST(MacrosTest, LinInterp_OneAlpha)  { EXPECT_NEAR(LIN_INTERP(1.0, 10.0, 20.0), 20.0, 1e-10); }
TEST(MacrosTest, LinInterp_Half)      { EXPECT_NEAR(LIN_INTERP(0.5, 10.0, 20.0), 15.0, 1e-10); }

// ---------------------------------------------------------------------------
// XOR
// ---------------------------------------------------------------------------

TEST(MacrosTest, XOR_TT) { EXPECT_FALSE(XOR(true,  true));  }
TEST(MacrosTest, XOR_FF) { EXPECT_FALSE(XOR(false, false)); }
TEST(MacrosTest, XOR_TF) { EXPECT_TRUE (XOR(true,  false)); }
TEST(MacrosTest, XOR_FT) { EXPECT_TRUE (XOR(false, true));  }

// ---------------------------------------------------------------------------
// SWAP
// ---------------------------------------------------------------------------

TEST(MacrosTest, Swap_Integers)
{
    int a = 3, b = 7, tmp;
    SWAP(a, b, tmp);
    EXPECT_EQ(a, 7);
    EXPECT_EQ(b, 3);
}

TEST(MacrosTest, Swap_Doubles)
{
    double a = 1.5, b = 2.5, tmp;
    SWAP(a, b, tmp);
    EXPECT_NEAR(a, 2.5, 1e-10);
    EXPECT_NEAR(b, 1.5, 1e-10);
}

// ---------------------------------------------------------------------------
// FIRST_XMIPP_INDEX / LAST_XMIPP_INDEX
// ---------------------------------------------------------------------------

TEST(MacrosTest, FirstXmippIndex_Even) { EXPECT_EQ(FIRST_XMIPP_INDEX(10), -5); }
TEST(MacrosTest, FirstXmippIndex_Odd)  { EXPECT_EQ(FIRST_XMIPP_INDEX(9),  -4); }
TEST(MacrosTest, LastXmippIndex_Even)  { EXPECT_EQ(LAST_XMIPP_INDEX(10),   4); }
TEST(MacrosTest, LastXmippIndex_Odd)   { EXPECT_EQ(LAST_XMIPP_INDEX(9),    4); }

TEST(MacrosTest, XmippIndex_RangeSize)
{
    int sz = 8;
    long int first = FIRST_XMIPP_INDEX(sz);
    long int last  = LAST_XMIPP_INDEX(sz);
    EXPECT_EQ(last - first + 1, sz);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
