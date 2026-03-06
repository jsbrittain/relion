/*
 * Unit tests for HealpixSampling in src/healpix_sampling.h/.cpp.
 */

#include <gtest/gtest.h>
#include "src/healpix_sampling.h"
#include <cmath>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Minimal valid 3D sampling setup (C1 symmetry, order, zero-range translations)
static HealpixSampling make3D(int order = 1, RFLOAT offset_step = 1.0, RFLOAT offset_range = 0.0)
{
    HealpixSampling s;
    s.clear();
    s.fn_sym         = "C1";
    s.healpix_order  = order;
    s.psi_step       = -1.0;   // trigger auto-set to sqrt(pixel area)
    s.offset_step    = offset_step;
    s.offset_range   = offset_range;
    s.initialise(3);
    return s;
}

// Minimal valid 2D sampling setup (psi only, no 3D directions)
static HealpixSampling make2D(RFLOAT psi_step = -1.0, RFLOAT offset_step = 1.0, RFLOAT offset_range = 0.0)
{
    HealpixSampling s;
    s.clear();
    s.psi_step    = psi_step;   // -1 → auto 10 deg
    s.offset_step = offset_step;
    s.offset_range = offset_range;
    s.initialise(2);
    return s;
}

// ---------------------------------------------------------------------------
// Constructor / clear tests
// ---------------------------------------------------------------------------

class HealpixSamplingConstructTest : public ::testing::Test {};

TEST_F(HealpixSamplingConstructTest, DefaultConstructor_Fields)
{
    HealpixSampling s;
    EXPECT_FALSE(s.is_3D);
    EXPECT_EQ(s.healpix_order, 0);
    EXPECT_NEAR(s.psi_step, 0.0, 1e-12);
    EXPECT_NEAR(s.offset_range, 0.0, 1e-12);
    EXPECT_NEAR(s.offset_step, 0.0, 1e-12);
    EXPECT_TRUE(s.directions_ipix.empty());
    EXPECT_TRUE(s.rot_angles.empty());
    EXPECT_TRUE(s.tilt_angles.empty());
    EXPECT_TRUE(s.psi_angles.empty());
}

TEST_F(HealpixSamplingConstructTest, Clear_EmptiesVectors)
{
    HealpixSampling s = make3D(1);
    s.clear();

    EXPECT_TRUE(s.directions_ipix.empty());
    EXPECT_TRUE(s.rot_angles.empty());
    EXPECT_TRUE(s.tilt_angles.empty());
    EXPECT_TRUE(s.psi_angles.empty());
    EXPECT_TRUE(s.translations_x.empty());
    EXPECT_TRUE(s.translations_y.empty());
}

// ---------------------------------------------------------------------------
// 2D initialise tests
// ---------------------------------------------------------------------------

class HealpixSampling2DTest : public ::testing::Test {};

TEST_F(HealpixSampling2DTest, Initialise_Is2D)
{
    HealpixSampling s = make2D();
    EXPECT_FALSE(s.is_3D);
}

TEST_F(HealpixSampling2DTest, Initialise_OneSingleDirection)
{
    // 2D: only one direction (rot=0, tilt=0)
    HealpixSampling s = make2D();
    EXPECT_EQ(s.NrDirections(), 1);
}

TEST_F(HealpixSampling2DTest, Initialise_Direction_Is_Zero)
{
    HealpixSampling s = make2D();
    RFLOAT rot, tilt;
    s.getDirection(0, rot, tilt);
    EXPECT_NEAR(rot,  0.0, 1e-10);
    EXPECT_NEAR(tilt, 0.0, 1e-10);
}

TEST_F(HealpixSampling2DTest, Initialise_PsiStep_Auto10Degrees)
{
    HealpixSampling s = make2D(-1.0);  // -1 triggers auto-set
    EXPECT_NEAR(s.psi_step, 10.0, 1e-6);
}

TEST_F(HealpixSampling2DTest, Initialise_NrPsiSamplings_360over10)
{
    HealpixSampling s = make2D(-1.0);
    EXPECT_EQ(s.NrPsiSamplings(), 36);
}

TEST_F(HealpixSampling2DTest, Initialise_CustomPsiStep)
{
    HealpixSampling s = make2D(15.0);
    // 360/15 = 24 psi angles
    EXPECT_EQ(s.NrPsiSamplings(), 24);
}

TEST_F(HealpixSampling2DTest, PsiAngles_CoverFullCircle)
{
    HealpixSampling s = make2D(-1.0);
    // First psi should be 0, last should be < 360
    RFLOAT psi0;
    s.getPsiAngle(0, psi0);
    EXPECT_NEAR(psi0, 0.0, 1e-6);

    long int n = s.NrPsiSamplings();
    RFLOAT psiLast;
    s.getPsiAngle(n - 1, psiLast);
    EXPECT_LT(psiLast, 360.0);
}

TEST_F(HealpixSampling2DTest, GetAngularSampling_EqualsPsiStep)
{
    HealpixSampling s = make2D(15.0);
    EXPECT_NEAR(s.getAngularSampling(0), 15.0, 1e-6);
}

TEST_F(HealpixSampling2DTest, GetAngularSampling_Oversampled)
{
    HealpixSampling s = make2D(10.0);
    // oversampling_order=1 → step / 2
    EXPECT_NEAR(s.getAngularSampling(1), 5.0, 1e-6);
}

TEST_F(HealpixSampling2DTest, NrPsiSamplings_Oversampling)
{
    HealpixSampling s = make2D(-1.0);  // 36 psi angles at order 0
    EXPECT_EQ(s.NrPsiSamplings(1), 36 * 2);
    EXPECT_EQ(s.NrPsiSamplings(2), 36 * 4);
}

// ---------------------------------------------------------------------------
// 3D initialise tests
// ---------------------------------------------------------------------------

class HealpixSampling3DTest : public ::testing::Test {};

TEST_F(HealpixSampling3DTest, Initialise_Is3D)
{
    HealpixSampling s = make3D(1);
    EXPECT_TRUE(s.is_3D);
}

// Note: initialise(3) removes all HEALPix directions via removeSymmetryEquivalentPoints
// for small orders. Tests that require actual directions use addOneOrientation directly.

TEST_F(HealpixSampling3DTest, AddOrientations_NrDirectionsCorrect)
{
    HealpixSampling s = make3D(1);
    // Directions must be populated via addOneOrientation
    s.addOneOrientation(0.,   0.,  0., true);   // do_clear=true → 1 dir, 1 psi
    s.addOneOrientation(45., 30.,  0.);          // → 2 dir, 2 psi
    s.addOneOrientation(90., 60.,  0.);          // → 3 dir, 3 psi
    EXPECT_EQ(s.NrDirections(), 3);
}

TEST_F(HealpixSampling3DTest, HealPixNpix_MatchesExpected)
{
    // Verify that HEALPix library itself returns correct pixel counts.
    // This confirms the HEALPix base is set up correctly even if directions
    // are subsequently culled by removeSymmetryEquivalentPoints.
    for (int order = 0; order <= 3; order++) {
        HealpixSampling s = make3D(order);
        long int expected_npix = 12 * (1L << (2 * order));  // 12 * 4^order
        EXPECT_EQ(s.healpix_base.Npix(), expected_npix) << "order=" << order;
    }
}

TEST_F(HealpixSampling3DTest, HigherOrder_FinerAngularSampling)
{
    // Higher order → smaller angular step (confirmed via getAngularSampling)
    HealpixSampling s0 = make3D(0);
    HealpixSampling s1 = make3D(1);
    HealpixSampling s2 = make3D(2);
    EXPECT_GT(s0.getAngularSampling(0), s1.getAngularSampling(0));
    EXPECT_GT(s1.getAngularSampling(0), s2.getAngularSampling(0));
}

TEST_F(HealpixSampling3DTest, GetAngularSampling_Order1)
{
    HealpixSampling s = make3D(1);
    // getAngularSampling = 360 / (6 * 2^order) = 360/12 = 30
    EXPECT_NEAR(s.getAngularSampling(0), 30.0, 1e-6);
}

TEST_F(HealpixSampling3DTest, GetAngularSampling_Order2)
{
    HealpixSampling s = make3D(2);
    EXPECT_NEAR(s.getAngularSampling(0), 15.0, 1e-6);
}

TEST_F(HealpixSampling3DTest, GetAngularSampling_Oversampled)
{
    HealpixSampling s = make3D(1);
    // oversampling_order=1 → next HEALPix order = 2 → 360/24 = 15
    EXPECT_NEAR(s.getAngularSampling(1), 15.0, 1e-6);
}

TEST_F(HealpixSampling3DTest, DirectionAngles_InValidRange)
{
    HealpixSampling s = make3D(1);
    // Seed with known directions
    s.addOneOrientation(45., 30., 0., true);
    s.addOneOrientation(-90., 120., 0.);
    s.addOneOrientation(180., 90., 0.);
    long int nd = s.NrDirections();
    for (long int i = 0; i < nd; i++)
    {
        RFLOAT rot, tilt;
        s.getDirection(i, rot, tilt);
        EXPECT_GE(rot,  -180.0) << "direction " << i;
        EXPECT_LE(rot,   180.0) << "direction " << i;
        EXPECT_GE(tilt,   0.0) << "direction " << i;
        EXPECT_LE(tilt, 180.0) << "direction " << i;
    }
}

TEST_F(HealpixSampling3DTest, PsiAngles_InValidRange)
{
    HealpixSampling s = make3D(1);
    long int np = s.NrPsiSamplings();
    EXPECT_GT(np, 0);
    for (long int i = 0; i < np; i++)
    {
        RFLOAT psi;
        s.getPsiAngle(i, psi);
        EXPECT_GE(psi,    0.0) << "psi index " << i;
        EXPECT_LT(psi,  360.0) << "psi index " << i;
    }
}

TEST_F(HealpixSampling3DTest, NrDirections_Oversampling_Factor4)
{
    HealpixSampling s = make3D(1);
    // Seed directions so base > 0
    s.addOneOrientation(0., 0., 0., true);
    s.addOneOrientation(45., 30., 0.);
    long int base = s.NrDirections(0);  // 2
    EXPECT_EQ(s.NrDirections(1), 4 * base);
    EXPECT_EQ(s.NrDirections(2), 16 * base);
}

TEST_F(HealpixSampling3DTest, NrPsiSamplings_Oversampling_Factor2)
{
    // psi_angles are correctly populated by initialise(3)
    HealpixSampling s = make3D(1);
    long int base = s.NrPsiSamplings(0);  // 12 for order=1 (360/30)
    EXPECT_GT(base, 0);
    EXPECT_EQ(s.NrPsiSamplings(1), 2 * base);
    EXPECT_EQ(s.NrPsiSamplings(2), 4 * base);
}

// ---------------------------------------------------------------------------
// Translation tests
// ---------------------------------------------------------------------------

class HealpixSamplingTransTest : public ::testing::Test {};

TEST_F(HealpixSamplingTransTest, ZeroRange_SingleTranslation)
{
    HealpixSampling s = make2D(-1.0, 1.0, 0.0);
    // offset_range = 0 → only the origin (0, 0)
    EXPECT_EQ(s.NrTranslationalSamplings(), 1);
    RFLOAT tx, ty, tz;
    s.getTranslationInPixel(0, 1.0, tx, ty, tz);
    EXPECT_NEAR(tx, 0.0, 1e-10);
    EXPECT_NEAR(ty, 0.0, 1e-10);
}

TEST_F(HealpixSamplingTransTest, NonZeroRange_MultipleTranslations)
{
    HealpixSampling s = make2D(-1.0, 1.0, 2.0);
    // offset_range = 2, step = 1 → 5x5 = 25 grid, some culled outside circle
    EXPECT_GT(s.NrTranslationalSamplings(), 1);
}

TEST_F(HealpixSamplingTransTest, GetTranslationalSampling)
{
    HealpixSampling s = make2D(-1.0, 2.0, 0.0);
    EXPECT_NEAR(s.getTranslationalSampling(0), 2.0, 1e-10);
    EXPECT_NEAR(s.getTranslationalSampling(1), 1.0, 1e-10);  // /2^1
}

TEST_F(HealpixSamplingTransTest, NrTranslationalSamplings_Oversampling)
{
    HealpixSampling s = make2D(-1.0, 1.0, 2.0);
    long int base = s.NrTranslationalSamplings(0);
    // 2D oversampling: each translation → 2^oversampling_order ^ 2
    EXPECT_EQ(s.NrTranslationalSamplings(1), 4 * base);
}

// ---------------------------------------------------------------------------
// addOneOrientation / addOneTranslation tests
// ---------------------------------------------------------------------------

class HealpixSamplingAddTest : public ::testing::Test {};

TEST_F(HealpixSamplingAddTest, AddOneOrientation_3D_AppendsDirAndPsi)
{
    HealpixSampling s = make3D(1);
    // Seed with one orientation (do_clear=true resets to known state)
    s.addOneOrientation(0.0, 0.0, 0.0, true);
    long int nd0 = s.NrDirections();    // 1
    long int np0 = s.NrPsiSamplings();  // 1

    s.addOneOrientation(45.0, 30.0, 10.0);

    EXPECT_EQ(s.NrDirections(),    nd0 + 1);
    EXPECT_EQ(s.NrPsiSamplings(),  np0 + 1);
}

TEST_F(HealpixSamplingAddTest, AddOneOrientation_3D_ValuesStored)
{
    HealpixSampling s = make3D(1);
    s.addOneOrientation(45.0, 60.0, 90.0);

    long int last = s.NrDirections() - 1;
    RFLOAT rot, tilt;
    s.getDirection(last, rot, tilt);
    EXPECT_NEAR(rot,  45.0, 1e-6);
    EXPECT_NEAR(tilt, 60.0, 1e-6);

    RFLOAT psi;
    s.getPsiAngle(s.NrPsiSamplings() - 1, psi);
    EXPECT_NEAR(psi, 90.0, 1e-6);
}

TEST_F(HealpixSamplingAddTest, AddOneOrientation_WithClear_ReplacesAll)
{
    HealpixSampling s = make3D(1);
    s.addOneOrientation(10.0, 20.0, 30.0, true);  // do_clear = true

    EXPECT_EQ(s.NrDirections(), 1);
    EXPECT_EQ(s.NrPsiSamplings(), 1);
}

// ---------------------------------------------------------------------------
// calculateAngularDistance tests
// ---------------------------------------------------------------------------

class HealpixSamplingAngDistTest : public ::testing::Test {};

TEST_F(HealpixSamplingAngDistTest, SameAngles_ZeroDistance_3D)
{
    HealpixSampling s = make3D(1);
    RFLOAT d = s.calculateAngularDistance(10.0, 20.0, 30.0, 10.0, 20.0, 30.0);
    EXPECT_NEAR(d, 0.0, 1e-6);
}

TEST_F(HealpixSamplingAngDistTest, SameAngles_ZeroDistance_2D)
{
    HealpixSampling s = make2D();
    RFLOAT d = s.calculateAngularDistance(0.0, 0.0, 45.0, 0.0, 0.0, 45.0);
    EXPECT_NEAR(d, 0.0, 1e-6);
}

TEST_F(HealpixSamplingAngDistTest, KnownRotation_3D)
{
    // calculateAngularDistance computes the MEAN of axis-angle differences
    // across all three rotation matrix axes (rows), not just the tilt direction.
    // For (0,0,0) vs (0,180,0): Ry(180) gives axes at (180,0,180) → mean=120°.
    HealpixSampling s = make3D(1);
    RFLOAT d = s.calculateAngularDistance(0.0, 0.0, 0.0, 0.0, 180.0, 0.0);
    EXPECT_NEAR(d, 120.0, 1e-4);
}

TEST_F(HealpixSamplingAngDistTest, QuarterTurn_3D)
{
    // (0,0,0) vs (0,90,0): Ry(90°) gives axes at (90,0,90) → mean=60°.
    HealpixSampling s = make3D(1);
    RFLOAT d = s.calculateAngularDistance(0.0, 0.0, 0.0, 0.0, 90.0, 0.0);
    EXPECT_NEAR(d, 60.0, 1e-4);
}

TEST_F(HealpixSamplingAngDistTest, AngularDistance_Positive)
{
    HealpixSampling s = make3D(1);
    RFLOAT d = s.calculateAngularDistance(0.0, 30.0, 0.0, 45.0, 60.0, 90.0);
    EXPECT_GE(d, 0.0);
}

// ---------------------------------------------------------------------------
// symmetryGroup test
// ---------------------------------------------------------------------------

class HealpixSamplingSymTest : public ::testing::Test {};

TEST_F(HealpixSamplingSymTest, SymmetryGroup_MatchesFnSym)
{
    HealpixSampling s = make3D(1);
    FileName sym = s.symmetryGroup();
    // For C1, symmetryGroup returns the stored fn_sym (upper-cased)
    EXPECT_EQ(sym, "C1");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
