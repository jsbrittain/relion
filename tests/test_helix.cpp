/*
 * Unit tests for src/helix.h / src/helix.cpp
 *
 * Covers: HermiteInterpolate1D,
 *         getHelicalSigma2Rot,
 *         makeHelicalSymmetryList,
 *         transformCartesianAndHelicalCoords (scalar overload),
 *         flipPsiTiltForHelicalSegment,
 *         checkParametersFor3DHelicalReconstruction
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "src/helix.h"
#include "src/matrix1d.h"

static const double TOL = 1e-9;
static const double LOOSE = 1e-5;

// ---------------------------------------------------------------------------
// HermiteInterpolate1D
// ---------------------------------------------------------------------------

TEST(HelixTest, HermiteInterp_AtZero_ReturnsY1)
{
    // mu = 0 → result should equal y1
    RFLOAT r = HermiteInterpolate1D(0.0, 1.0, 2.0, 3.0, 0.0);
    EXPECT_NEAR(r, 1.0, TOL);
}

TEST(HelixTest, HermiteInterp_AtOne_ReturnsY2)
{
    // mu = 1 → result should equal y2
    RFLOAT r = HermiteInterpolate1D(0.0, 1.0, 2.0, 3.0, 1.0);
    EXPECT_NEAR(r, 2.0, TOL);
}

TEST(HelixTest, HermiteInterp_LinearFlatData)
{
    // All four points equal → interpolation always returns that value
    RFLOAT r = HermiteInterpolate1D(5.0, 5.0, 5.0, 5.0, 0.5);
    EXPECT_NEAR(r, 5.0, TOL);
}

TEST(HelixTest, HermiteInterp_Midpoint_Between_Y1_Y2)
{
    // For uniform linear data (0,1,2,3) and zero tension/bias,
    // the midpoint should be (1 + 2) / 2 = 1.5
    RFLOAT r = HermiteInterpolate1D(0.0, 1.0, 2.0, 3.0, 0.5, 0.0, 0.0);
    EXPECT_NEAR(r, 1.5, LOOSE);
}

TEST(HelixTest, HermiteInterp_SymmetryAroundMidpoint)
{
    // Symmetry: f(mu) and f(1-mu) should be symmetric about the midpoint
    RFLOAT y0 = 1.0, y1 = 2.0, y2 = 3.0, y3 = 4.0;
    RFLOAT r1 = HermiteInterpolate1D(y0, y1, y2, y3, 0.25);
    RFLOAT r2 = HermiteInterpolate1D(y3, y2, y1, y0, 0.75);
    EXPECT_NEAR(r1, r2, LOOSE);
}

// ---------------------------------------------------------------------------
// getHelicalSigma2Rot
// ---------------------------------------------------------------------------

TEST(HelixTest, GetHelicalSigma2Rot_ZeroRise)
{
    // When rise is zero the result depends on formula; just check non-NaN
    RFLOAT s = getHelicalSigma2Rot(0.0, 30.0, 1.0, 1.0, 1.0);
    EXPECT_FALSE(std::isnan(s));
}

TEST(HelixTest, GetHelicalSigma2Rot_NonzeroRise_PositiveResult)
{
    // Physical parameters: rise=4.75 Å, twist=−1.0 deg, offset step=1.0 Å, rot_step=1.0 deg
    RFLOAT s = getHelicalSigma2Rot(4.75, -1.0, 1.0, 1.0, 1.0);
    EXPECT_GT(s, 0.0);
    EXPECT_FALSE(std::isnan(s));
}

TEST(HelixTest, GetHelicalSigma2Rot_LargerOffsetStep_LargerSigma)
{
    // Larger offset step should increase sigma (more uncertainty)
    RFLOAT s_small = getHelicalSigma2Rot(5.0, 30.0, 0.5, 1.0, 1.0);
    RFLOAT s_large = getHelicalSigma2Rot(5.0, 30.0, 2.0, 1.0, 1.0);
    EXPECT_GT(s_large, s_small);
}

// ---------------------------------------------------------------------------
// makeHelicalSymmetryList
// ---------------------------------------------------------------------------

TEST(HelixTest, MakeHelicalSymmetryList_SingleEntry)
{
    std::vector<HelicalSymmetryItem> list;
    // No search: single rise and twist → exactly one entry
    makeHelicalSymmetryList(list, 5.0, 5.0, 0.5, false, 30.0, 30.0, 1.0, false);
    EXPECT_EQ(list.size(), (size_t)1);
    EXPECT_NEAR(list[0].rise_pix, 5.0, TOL);
    EXPECT_NEAR(list[0].twist_deg, 30.0, TOL);
}

TEST(HelixTest, MakeHelicalSymmetryList_SearchTwist)
{
    std::vector<HelicalSymmetryItem> list;
    // Search twist from 28 to 32 in steps of 1 → 5 entries
    makeHelicalSymmetryList(list, 5.0, 5.0, 0.5, false, 28.0, 32.0, 1.0, true);
    EXPECT_EQ(list.size(), (size_t)5);
}

TEST(HelixTest, MakeHelicalSymmetryList_SearchRise)
{
    std::vector<HelicalSymmetryItem> list;
    // Search rise from 4 to 6 in steps of 0.5 → 5 entries (4.0, 4.5, 5.0, 5.5, 6.0)
    makeHelicalSymmetryList(list, 4.0, 6.0, 0.5, true, 30.0, 30.0, 1.0, false);
    EXPECT_EQ(list.size(), (size_t)5);
}

TEST(HelixTest, MakeHelicalSymmetryList_SearchBoth)
{
    std::vector<HelicalSymmetryItem> list;
    makeHelicalSymmetryList(list, 4.0, 6.0, 1.0, true, 28.0, 30.0, 1.0, true);
    // 3 rises × 3 twists = 9 entries
    EXPECT_EQ(list.size(), (size_t)9);
}

// ---------------------------------------------------------------------------
// transformCartesianAndHelicalCoords (scalar overload)
// ---------------------------------------------------------------------------

TEST(HelixTest, TransformCartesianHelical_ZeroAngles_2D_Identity)
{
    // With zero rotation angles and 2D mode, coords should be unchanged
    RFLOAT xout, yout, zout;
    transformCartesianAndHelicalCoords(1.0, 2.0, 0.0,
                                       xout, yout, zout,
                                       0.0, 0.0, 0.0, 2, true);
    EXPECT_NEAR(xout, 1.0, LOOSE);
    EXPECT_NEAR(yout, 2.0, LOOSE);
}

TEST(HelixTest, TransformCartesianHelical_ZeroAngles_3D_Identity)
{
    RFLOAT xout, yout, zout;
    transformCartesianAndHelicalCoords(1.0, 2.0, 3.0,
                                       xout, yout, zout,
                                       0.0, 0.0, 0.0, 3, true);
    EXPECT_NEAR(xout, 1.0, LOOSE);
    EXPECT_NEAR(yout, 2.0, LOOSE);
    EXPECT_NEAR(zout, 3.0, LOOSE);
}

// ---------------------------------------------------------------------------
// flipPsiTiltForHelicalSegment
// ---------------------------------------------------------------------------

TEST(HelixTest, FlipPsiTilt_ZeroAngles)
{
    RFLOAT new_psi, new_tilt;
    flipPsiTiltForHelicalSegment(0.0, 0.0, new_psi, new_tilt);
    EXPECT_FALSE(std::isnan(new_psi));
    EXPECT_FALSE(std::isnan(new_tilt));
}

TEST(HelixTest, FlipPsiTilt_DoubleFl_Involution)
{
    // Applying flip twice should give back original angles
    RFLOAT psi = 30.0, tilt = 45.0;
    RFLOAT psi1, tilt1, psi2, tilt2;
    flipPsiTiltForHelicalSegment(psi,  tilt,  psi1, tilt1);
    flipPsiTiltForHelicalSegment(psi1, tilt1, psi2, tilt2);
    // Normalise to [-180, 180] before comparing
    auto norm = [](RFLOAT a) {
        while (a >  180.0) a -= 360.0;
        while (a < -180.0) a += 360.0;
        return a;
    };
    EXPECT_NEAR(norm(psi2),  norm(psi),  1e-4);
    EXPECT_NEAR(norm(tilt2), norm(tilt), 1e-4);
}

// ---------------------------------------------------------------------------
// checkParametersFor3DHelicalReconstruction
// ---------------------------------------------------------------------------

TEST(HelixTest, CheckParameters_ValidInputs)
{
    // Reasonable physical parameters: should return true (valid)
    bool ok = checkParametersFor3DHelicalReconstruction(
        false,  // ignore_symmetry
        false,  // do_symmetry_local_refinement
        1,      // nr_asu
        4.75,   // rise_initial_A
        3.0,    // rise_min_A
        7.0,    // rise_max_A
        -1.0,   // twist_initial_deg
        -5.0,   // twist_min_deg
        5.0,    // twist_max_deg
        200,    // box_len
        1.35,   // pixel_size_A
        0.3,    // z_percentage
        200.0,  // particle_diameter_A
        0.0,    // tube_inner_diameter_A
        180.0,  // tube_outer_diameter_A
        false   // verboseOutput
    );
    EXPECT_TRUE(ok);
}

TEST(HelixTest, CheckParameters_IgnoreSymmetry)
{
    // When ignore_symmetry=true, parameter checks are relaxed
    bool ok = checkParametersFor3DHelicalReconstruction(
        true, false, 1, 4.75, 3.0, 7.0, -1.0, -5.0, 5.0,
        200, 1.35, 0.3, 200.0, 0.0, 180.0, false);
    EXPECT_TRUE(ok);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
