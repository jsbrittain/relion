/*
 * Unit tests for src/local_symmetry.h / src/local_symmetry.cpp
 *
 * Covers: Localsym_composeOperator / Localsym_decomposeOperator (round-trip),
 *         Localsym_scaleTranslations, Localsym_shiftTranslations,
 *         standardiseEulerAngles,
 *         sameLocalsymOperators,
 *         isMultidimArray3DCubic,
 *         truncateMultidimArray,
 *         similar3DCubicMasks
 */

#include <gtest/gtest.h>
#include <cmath>
#include "src/local_symmetry.h"
#include "src/matrix1d.h"
#include "src/multidim_array.h"

static const double TOL = 1e-9;

// ---------------------------------------------------------------------------
// Compose / Decompose round-trip
// ---------------------------------------------------------------------------

TEST(LocalSymTest, ComposeDecompose_RoundTrip)
{
    Matrix1D<RFLOAT> op;
    Localsym_composeOperator(op, 10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 0.9);

    RFLOAT aa, bb, gg, dx, dy, dz, cc;
    Localsym_decomposeOperator(op, aa, bb, gg, dx, dy, dz, cc);

    EXPECT_NEAR(aa, 10.0, TOL);
    EXPECT_NEAR(bb, 20.0, TOL);
    EXPECT_NEAR(gg, 30.0, TOL);
    EXPECT_NEAR(dx, 1.0,  TOL);
    EXPECT_NEAR(dy, 2.0,  TOL);
    EXPECT_NEAR(dz, 3.0,  TOL);
    EXPECT_NEAR(cc, 0.9,  TOL);
}

TEST(LocalSymTest, ComposeDecompose_Defaults)
{
    // All defaults: angles = 0, translations = 0, cc = 1e10
    Matrix1D<RFLOAT> op;
    Localsym_composeOperator(op);

    RFLOAT aa, bb, gg, dx, dy, dz, cc;
    Localsym_decomposeOperator(op, aa, bb, gg, dx, dy, dz, cc);

    EXPECT_NEAR(aa, 0.0, TOL);
    EXPECT_NEAR(bb, 0.0, TOL);
    EXPECT_NEAR(gg, 0.0, TOL);
    EXPECT_NEAR(dx, 0.0, TOL);
    EXPECT_NEAR(dy, 0.0, TOL);
    EXPECT_NEAR(dz, 0.0, TOL);
    EXPECT_NEAR(cc, 1e10, 1.0);
}

// ---------------------------------------------------------------------------
// Localsym_scaleTranslations
// ---------------------------------------------------------------------------

TEST(LocalSymTest, ScaleTranslations_ByTwo)
{
    Matrix1D<RFLOAT> op;
    Localsym_composeOperator(op, 0.0, 0.0, 0.0, 4.0, 6.0, 8.0, 0.5);
    Localsym_scaleTranslations(op, 2.0);

    RFLOAT aa, bb, gg, dx, dy, dz, cc;
    Localsym_decomposeOperator(op, aa, bb, gg, dx, dy, dz, cc);

    EXPECT_NEAR(dx, 8.0,  TOL);
    EXPECT_NEAR(dy, 12.0, TOL);
    EXPECT_NEAR(dz, 16.0, TOL);
    // Angles and cc unchanged
    EXPECT_NEAR(aa, 0.0, TOL);
    EXPECT_NEAR(cc, 0.5, TOL);
}

TEST(LocalSymTest, ScaleTranslations_ByZero)
{
    Matrix1D<RFLOAT> op;
    Localsym_composeOperator(op, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 0.8);
    Localsym_scaleTranslations(op, 0.0);

    RFLOAT aa, bb, gg, dx, dy, dz, cc;
    Localsym_decomposeOperator(op, aa, bb, gg, dx, dy, dz, cc);

    EXPECT_NEAR(dx, 0.0, TOL);
    EXPECT_NEAR(dy, 0.0, TOL);
    EXPECT_NEAR(dz, 0.0, TOL);
    // Angles unchanged
    EXPECT_NEAR(aa, 1.0, TOL);
}

// ---------------------------------------------------------------------------
// Localsym_shiftTranslations
// ---------------------------------------------------------------------------

TEST(LocalSymTest, ShiftTranslations)
{
    Matrix1D<RFLOAT> op;
    Localsym_composeOperator(op, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.5);

    // voffset must be a 3-element (vectorR3) vector
    Matrix1D<RFLOAT> voffset(3);
    voffset.initZeros();
    XX(voffset) = 10.0;
    YY(voffset) = 20.0;
    ZZ(voffset) = 30.0;

    Localsym_shiftTranslations(op, voffset);

    RFLOAT aa, bb, gg, dx, dy, dz, cc;
    Localsym_decomposeOperator(op, aa, bb, gg, dx, dy, dz, cc);

    EXPECT_NEAR(dx, 11.0, TOL);
    EXPECT_NEAR(dy, 22.0, TOL);
    EXPECT_NEAR(dz, 33.0, TOL);
}

// ---------------------------------------------------------------------------
// sameLocalsymOperators
// ---------------------------------------------------------------------------

TEST(LocalSymTest, SameOperators_Identical)
{
    Matrix1D<RFLOAT> op1, op2;
    Localsym_composeOperator(op1, 10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 0.9);
    Localsym_composeOperator(op2, 10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 0.9);
    EXPECT_TRUE(sameLocalsymOperators(op1, op2));
}

TEST(LocalSymTest, SameOperators_Different)
{
    Matrix1D<RFLOAT> op1, op2;
    Localsym_composeOperator(op1, 10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 0.9);
    Localsym_composeOperator(op2, 11.0, 20.0, 30.0, 1.0, 2.0, 3.0, 0.9);
    EXPECT_FALSE(sameLocalsymOperators(op1, op2));
}

// ---------------------------------------------------------------------------
// standardiseEulerAngles
// ---------------------------------------------------------------------------

TEST(LocalSymTest, StandardiseEulerAngles_Identity)
{
    RFLOAT aa, bb, gg;
    standardiseEulerAngles(0.0, 0.0, 0.0, aa, bb, gg);
    // Should remain in [0, 360) or the normalised form — key: no NaN/inf
    EXPECT_FALSE(std::isnan(aa));
    EXPECT_FALSE(std::isnan(bb));
    EXPECT_FALSE(std::isnan(gg));
}

TEST(LocalSymTest, StandardiseEulerAngles_InRange)
{
    RFLOAT aa, bb, gg;
    standardiseEulerAngles(45.0, 60.0, 90.0, aa, bb, gg);
    EXPECT_FALSE(std::isnan(aa));
    EXPECT_FALSE(std::isnan(bb));
    EXPECT_FALSE(std::isnan(gg));
}

// ---------------------------------------------------------------------------
// isMultidimArray3DCubic
// ---------------------------------------------------------------------------

TEST(LocalSymTest, IsMultidimArray3DCubic_ValidCubic)
{
    MultidimArray<RFLOAT> v;
    v.initZeros(1, 4, 4, 4); // NSIZE=1, Z=Y=X=4, even
    EXPECT_TRUE(isMultidimArray3DCubic(v));
}

TEST(LocalSymTest, IsMultidimArray3DCubic_NonCubic)
{
    MultidimArray<RFLOAT> v;
    v.initZeros(1, 4, 6, 4); // Y ≠ Z
    EXPECT_FALSE(isMultidimArray3DCubic(v));
}

TEST(LocalSymTest, IsMultidimArray3DCubic_OddSize)
{
    MultidimArray<RFLOAT> v;
    v.initZeros(1, 5, 5, 5); // odd size not allowed
    EXPECT_FALSE(isMultidimArray3DCubic(v));
}

TEST(LocalSymTest, IsMultidimArray3DCubic_2D)
{
    MultidimArray<RFLOAT> v;
    v.initZeros(1, 1, 4, 4); // ZSIZE = 1, not > 1
    EXPECT_FALSE(isMultidimArray3DCubic(v));
}

// ---------------------------------------------------------------------------
// truncateMultidimArray
// ---------------------------------------------------------------------------

TEST(LocalSymTest, TruncateMultidimArray_ClampToRange)
{
    MultidimArray<RFLOAT> v;
    v.initZeros(5);
    A1D_ELEM(v, 0) = -1.0;
    A1D_ELEM(v, 1) =  0.5;
    A1D_ELEM(v, 2) =  1.5;
    A1D_ELEM(v, 3) =  2.5;
    A1D_ELEM(v, 4) = -0.5;

    truncateMultidimArray(v, 0.0, 1.0);

    EXPECT_NEAR(A1D_ELEM(v, 0), 0.0, TOL);  // clamped to min
    EXPECT_NEAR(A1D_ELEM(v, 1), 0.5, TOL);  // unchanged
    EXPECT_NEAR(A1D_ELEM(v, 2), 1.0, TOL);  // clamped to max
    EXPECT_NEAR(A1D_ELEM(v, 3), 1.0, TOL);  // clamped to max
    EXPECT_NEAR(A1D_ELEM(v, 4), 0.0, TOL);  // clamped to min
}

// ---------------------------------------------------------------------------
// similar3DCubicMasks
// ---------------------------------------------------------------------------

TEST(LocalSymTest, Similar3DCubicMasks_SameSums)
{
    // When ratios match exactly, masks are similar
    EXPECT_TRUE(similar3DCubicMasks(100.0, 50.0, 100.0, 50.0));
}

TEST(LocalSymTest, Similar3DCubicMasks_VeryDifferent)
{
    // Drastically different sums → not similar
    EXPECT_FALSE(similar3DCubicMasks(100.0, 1.0, 100.0, 90.0));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
