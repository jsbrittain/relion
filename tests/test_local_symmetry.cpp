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
// sum3DCubicMask
// ---------------------------------------------------------------------------

TEST(LocalSymTest, Sum3DCubicMask_AllOnes)
{
    // 4x4x4 volume of all-ones: sum = 64, ctr = 64
    MultidimArray<RFLOAT> v;
    v.initZeros(4, 4, 4);
    v.initConstant(1.0);
    RFLOAT val_sum = 0., val_ctr = 0.;
    sum3DCubicMask(v, val_sum, val_ctr);
    EXPECT_NEAR(val_sum, 64.0, 1e-6);
    EXPECT_NEAR(val_ctr, 64.0, 1e-6);
}

TEST(LocalSymTest, Sum3DCubicMask_HalfOnes)
{
    // 4x4x4 volume: half voxels = 0.5
    MultidimArray<RFLOAT> v;
    v.initZeros(4, 4, 4);
    v.initConstant(0.5);
    RFLOAT val_sum = 0., val_ctr = 0.;
    sum3DCubicMask(v, val_sum, val_ctr);
    EXPECT_NEAR(val_sum, 32.0, 1e-6); // 64 * 0.5
    EXPECT_NEAR(val_ctr, 64.0, 1e-6); // all 64 voxels > 0
}

// ---------------------------------------------------------------------------
// Localsym_translations2vector
// ---------------------------------------------------------------------------

TEST(LocalSymTest, Translations2vector_NoInvert)
{
    Matrix1D<RFLOAT> op;
    Localsym_composeOperator(op, 10.0, 20.0, 30.0, 5.0, 6.0, 7.0, 0.8);

    Matrix1D<RFLOAT> tvec;
    Localsym_translations2vector(op, tvec, LOCALSYM_OP_DONT_INVERT);

    EXPECT_NEAR(XX(tvec),  5.0, TOL);
    EXPECT_NEAR(YY(tvec),  6.0, TOL);
    EXPECT_NEAR(ZZ(tvec),  7.0, TOL);
}

TEST(LocalSymTest, Translations2vector_WithInvert)
{
    Matrix1D<RFLOAT> op;
    Localsym_composeOperator(op, 0.0, 0.0, 0.0, 3.0, -4.0, 5.0, 1.0);

    Matrix1D<RFLOAT> tvec;
    Localsym_translations2vector(op, tvec, LOCALSYM_OP_DO_INVERT);

    EXPECT_NEAR(XX(tvec), -3.0, TOL);
    EXPECT_NEAR(YY(tvec),  4.0, TOL);
    EXPECT_NEAR(ZZ(tvec), -5.0, TOL);
}

// ---------------------------------------------------------------------------
// Localsym_angles2matrix
// ---------------------------------------------------------------------------

TEST(LocalSymTest, Angles2Matrix_IdentityAngles_IsIdentity)
{
    // Zero Euler angles → identity-like rotation in 4x4
    Matrix1D<RFLOAT> op;
    Localsym_composeOperator(op, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    Matrix2D<RFLOAT> mat;
    Localsym_angles2matrix(op, mat, LOCALSYM_OP_DONT_INVERT);

    ASSERT_EQ(MAT_XSIZE(mat), 4);
    ASSERT_EQ(MAT_YSIZE(mat), 4);

    // Upper-left 3×3 should be identity
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_NEAR(MAT_ELEM(mat, i, j), (i == j) ? 1.0 : 0.0, 1e-9);
    // Bottom-right corner = 1
    EXPECT_NEAR(MAT_ELEM(mat, 3, 3), 1.0, TOL);
}

TEST(LocalSymTest, Angles2Matrix_Invert_TransposeRelation)
{
    Matrix1D<RFLOAT> op;
    Localsym_composeOperator(op, 30.0, 45.0, 60.0, 0.0, 0.0, 0.0, 1.0);

    Matrix2D<RFLOAT> mat, mat_inv;
    Localsym_angles2matrix(op, mat,     LOCALSYM_OP_DONT_INVERT);
    Localsym_angles2matrix(op, mat_inv, LOCALSYM_OP_DO_INVERT);

    // For a rotation matrix R^{-1} = R^T.
    // Check upper-left 3×3 block: mat_inv ≈ mat^T
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_NEAR(MAT_ELEM(mat_inv, i, j), MAT_ELEM(mat, j, i), 1e-9);
}

// ---------------------------------------------------------------------------
// Localsym_operator2matrix
// ---------------------------------------------------------------------------

TEST(LocalSymTest, Operator2Matrix_IdentityOperator_IsIdentityLike)
{
    Matrix1D<RFLOAT> op;
    Localsym_composeOperator(op, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    Matrix2D<RFLOAT> mat;
    Localsym_operator2matrix(op, mat, LOCALSYM_OP_DONT_INVERT);

    ASSERT_EQ(MAT_XSIZE(mat), 4);
    // Translation columns should be zero for zero translations
    EXPECT_NEAR(MAT_ELEM(mat, 0, 3), 0.0, TOL);
    EXPECT_NEAR(MAT_ELEM(mat, 1, 3), 0.0, TOL);
    EXPECT_NEAR(MAT_ELEM(mat, 2, 3), 0.0, TOL);
}

TEST(LocalSymTest, Operator2Matrix_NoInvert_TranslationInColumn3)
{
    Matrix1D<RFLOAT> op;
    Localsym_composeOperator(op, 0.0, 0.0, 0.0, 2.0, 3.0, 4.0, 0.9);

    Matrix2D<RFLOAT> mat;
    Localsym_operator2matrix(op, mat, LOCALSYM_OP_DONT_INVERT);

    EXPECT_NEAR(MAT_ELEM(mat, 0, 3), 2.0, TOL);
    EXPECT_NEAR(MAT_ELEM(mat, 1, 3), 3.0, TOL);
    EXPECT_NEAR(MAT_ELEM(mat, 2, 3), 4.0, TOL);
}

// ---------------------------------------------------------------------------
// compareOperatorsByCC
// ---------------------------------------------------------------------------

TEST(LocalSymTest, CompareOperatorsByCC_LowerCCComesFirst)
{
    Matrix1D<RFLOAT> op_low, op_high;
    // CC is stored at position CC_POS (index 6)
    Localsym_composeOperator(op_low,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2);
    Localsym_composeOperator(op_high, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9);

    // compareOperatorsByCC is a strict weak ordering — lower CC should sort first
    EXPECT_TRUE( compareOperatorsByCC(op_low,  op_high));
    EXPECT_FALSE(compareOperatorsByCC(op_high, op_low));
}

TEST(LocalSymTest, CompareOperatorsByCC_EqualCC_NeitherSmaller)
{
    Matrix1D<RFLOAT> op1, op2;
    Localsym_composeOperator(op1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5);
    Localsym_composeOperator(op2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5);

    EXPECT_FALSE(compareOperatorsByCC(op1, op2));
    EXPECT_FALSE(compareOperatorsByCC(op2, op1));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
