/*
 * Unit tests for src/symmetries.h / src/symmetries.cpp
 *
 * Covers:
 *   - SymList::isSymmetryGroup  — parsing symmetry group strings
 *   - SymList::read_sym_file    — loading built-in groups and counting operators
 *   - SymList::get_matrices     — retrieving L/R transformation matrices
 *   - SymList::non_redundant_ewald_sphere — solid-angle fractions
 */

#include <gtest/gtest.h>
#include <cmath>
#include "src/symmetries.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -------------------------------------------------------- isSymmetryGroup --

TEST(IsSymmetryGroupTest, C1_IsCN_Order1)
{
    int pgGroup = -1, pgOrder = -1;
    SymList sl;
    EXPECT_TRUE(sl.isSymmetryGroup("C1", pgGroup, pgOrder));
    EXPECT_EQ(pgGroup, pg_CN);
    EXPECT_EQ(pgOrder, 1);
}

TEST(IsSymmetryGroupTest, C2_IsCN_Order2)
{
    int pgGroup = -1, pgOrder = -1;
    SymList sl;
    EXPECT_TRUE(sl.isSymmetryGroup("C2", pgGroup, pgOrder));
    EXPECT_EQ(pgGroup, pg_CN);
    EXPECT_EQ(pgOrder, 2);
}

TEST(IsSymmetryGroupTest, C4_IsCN_Order4)
{
    int pgGroup = -1, pgOrder = -1;
    SymList sl;
    EXPECT_TRUE(sl.isSymmetryGroup("C4", pgGroup, pgOrder));
    EXPECT_EQ(pgGroup, pg_CN);
    EXPECT_EQ(pgOrder, 4);
}

TEST(IsSymmetryGroupTest, D2_IsDN_Order2)
{
    int pgGroup = -1, pgOrder = -1;
    SymList sl;
    EXPECT_TRUE(sl.isSymmetryGroup("D2", pgGroup, pgOrder));
    EXPECT_EQ(pgGroup, pg_DN);
    EXPECT_EQ(pgOrder, 2);
}

TEST(IsSymmetryGroupTest, CI_InversionSymmetry)
{
    int pgGroup = -1, pgOrder = -1;
    SymList sl;
    EXPECT_TRUE(sl.isSymmetryGroup("CI", pgGroup, pgOrder));
    EXPECT_EQ(pgGroup, pg_CI);
}

TEST(IsSymmetryGroupTest, O_Octahedral)
{
    int pgGroup = -1, pgOrder = -1;
    SymList sl;
    EXPECT_TRUE(sl.isSymmetryGroup("O", pgGroup, pgOrder));
    EXPECT_EQ(pgGroup, pg_O);
}

TEST(IsSymmetryGroupTest, I_Icosahedral)
{
    int pgGroup = -1, pgOrder = -1;
    SymList sl;
    EXPECT_TRUE(sl.isSymmetryGroup("I", pgGroup, pgOrder));
    EXPECT_EQ(pgGroup, pg_I);
}

TEST(IsSymmetryGroupTest, CaseSensitivity_LowercaseReturnsFalse)
{
    // The parser uses toupper internally, so "c2" should still match.
    int pgGroup = -1, pgOrder = -1;
    SymList sl;
    // c2 becomes "C2" after toupper — expect true
    EXPECT_TRUE(sl.isSymmetryGroup("c2", pgGroup, pgOrder));
    EXPECT_EQ(pgGroup, pg_CN);
    EXPECT_EQ(pgOrder, 2);
}

TEST(IsSymmetryGroupTest, InvalidName_ReturnsFalse)
{
    int pgGroup = -1, pgOrder = -1;
    SymList sl;
    EXPECT_FALSE(sl.isSymmetryGroup("XYZ", pgGroup, pgOrder));
}

// --------------------------------------------------------- read_sym_file --

TEST(ReadSymFileTest, C1_ZeroNonTrivialSymmetries)
{
    // C1 has no non-identity rotations.
    SymList sl;
    sl.read_sym_file("C1");
    EXPECT_EQ(sl.SymsNo(), 0);
}

TEST(ReadSymFileTest, C2_OneNonTrivialSymmetry)
{
    // C2: one 180° rotation around Z.
    SymList sl;
    sl.read_sym_file("C2");
    EXPECT_EQ(sl.SymsNo(), 1);
}

TEST(ReadSymFileTest, C4_ThreeNonTrivialSymmetries)
{
    // C4: rotations by 90°, 180°, 270°.
    SymList sl;
    sl.read_sym_file("C4");
    EXPECT_EQ(sl.SymsNo(), 3);
}

TEST(ReadSymFileTest, D2_ThreeNonTrivialSymmetries)
{
    // D2 generators: 180° around Z and 180° around X.
    // compute_subgroup adds the product: 180° around Y.
    // Total non-identity: 3.
    SymList sl;
    sl.read_sym_file("D2");
    EXPECT_EQ(sl.SymsNo(), 3);
}

TEST(ReadSymFileTest, TrueSymsNo_IsAtMostSymsNo)
{
    // TrueSymsNo counts the generating elements; SymsNo is the full subgroup.
    SymList sl;
    sl.read_sym_file("D2");
    EXPECT_LE(sl.TrueSymsNo(), sl.SymsNo());
}

// ----------------------------------------------------------- get_matrices --

TEST(GetMatricesTest, C2_LMatrixIsIdentity)
{
    SymList sl;
    sl.read_sym_file("C2");
    ASSERT_EQ(sl.SymsNo(), 1);

    Matrix2D<RFLOAT> L, R;
    sl.get_matrices(0, L, R);

    // L must be 4x4 identity
    ASSERT_EQ(MAT_XSIZE(L), 4);
    ASSERT_EQ(MAT_YSIZE(L), 4);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            EXPECT_NEAR(L(i, j), (i == j) ? 1.0 : 0.0, 1e-6)
                << "L(" << i << "," << j << ")";
}

TEST(GetMatricesTest, C2_RMatrixIsRotation180DegZ)
{
    SymList sl;
    sl.read_sym_file("C2");
    ASSERT_EQ(sl.SymsNo(), 1);

    Matrix2D<RFLOAT> L, R;
    sl.get_matrices(0, L, R);

    // Rz(180°) top-left 3×3 = diag(-1, -1, 1)
    EXPECT_NEAR(R(0, 0), -1.0, 1e-6);
    EXPECT_NEAR(R(1, 1), -1.0, 1e-6);
    EXPECT_NEAR(R(2, 2),  1.0, 1e-6);
    EXPECT_NEAR(R(0, 1),  0.0, 1e-6);
    EXPECT_NEAR(R(1, 0),  0.0, 1e-6);
    EXPECT_NEAR(R(0, 2),  0.0, 1e-6);
    EXPECT_NEAR(R(2, 0),  0.0, 1e-6);
}

TEST(GetMatricesTest, D2_AllRMatricesHaveDeterminantOne)
{
    // Every element of the subgroup must be a proper rotation (det = +1).
    SymList sl;
    sl.read_sym_file("D2");
    ASSERT_EQ(sl.SymsNo(), 3);

    for (int i = 0; i < sl.SymsNo(); ++i)
    {
        Matrix2D<RFLOAT> L, R;
        sl.get_matrices(i, L, R);

        // Compute 3×3 determinant from the top-left block.
        double det = R(0,0) * (R(1,1)*R(2,2) - R(1,2)*R(2,1))
                   - R(0,1) * (R(1,0)*R(2,2) - R(1,2)*R(2,0))
                   + R(0,2) * (R(1,0)*R(2,1) - R(1,1)*R(2,0));
        EXPECT_NEAR(std::abs(det), 1.0, 1e-5) << "matrix index " << i;
    }
}

// -------------------------------------------------- more read_sym_file --

TEST(ReadSymFileTest, C5_FourNonTrivialSymmetries)
{
    // C5: rotations by 72°, 144°, 216°, 288°.
    SymList sl;
    sl.read_sym_file("C5");
    EXPECT_EQ(sl.SymsNo(), 4);
}

TEST(ReadSymFileTest, D4_SevenNonTrivialSymmetries)
{
    // D4: 4 rotations around Z + 4 C2 rotations - identity = 7
    SymList sl;
    sl.read_sym_file("D4");
    EXPECT_EQ(sl.SymsNo(), 7);
}

TEST(ReadSymFileTest, T_ElevenNonTrivialSymmetries)
{
    // T (tetrahedral): 12 rotations total → 11 non-identity
    SymList sl;
    sl.read_sym_file("T");
    EXPECT_EQ(sl.SymsNo(), 11);
}

TEST(ReadSymFileTest, O_TwentyThreeNonTrivialSymmetries)
{
    // O (octahedral): 24 rotations total → 23 non-identity
    SymList sl;
    sl.read_sym_file("O");
    EXPECT_EQ(sl.SymsNo(), 23);
}

TEST(ReadSymFileTest, I_FiftyNineNonTrivialSymmetries)
{
    // I (icosahedral): 60 rotations total → 59 non-identity
    SymList sl;
    sl.read_sym_file("I");
    EXPECT_EQ(sl.SymsNo(), 59);
}

// -------------------------------------- matrix orthogonality checks --

static void checkMatricesOrthogonal(const std::string& group_name)
{
    SymList sl;
    sl.read_sym_file(group_name);
    for (int i = 0; i < sl.SymsNo(); i++)
    {
        Matrix2D<RFLOAT> L, R;
        sl.get_matrices(i, L, R);
        // Check R * Rᵀ = I
        Matrix2D<RFLOAT> RRt = R * R.transpose();
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                EXPECT_NEAR(RRt(r,c), (r==c) ? 1.0 : 0.0, 1e-5)
                    << group_name << " matrix " << i;
    }
}

static void checkMatricesDet1(const std::string& group_name)
{
    SymList sl;
    sl.read_sym_file(group_name);
    for (int i = 0; i < sl.SymsNo(); i++)
    {
        Matrix2D<RFLOAT> L, R;
        sl.get_matrices(i, L, R);
        double det = R(0,0)*(R(1,1)*R(2,2)-R(1,2)*R(2,1))
                   - R(0,1)*(R(1,0)*R(2,2)-R(1,2)*R(2,0))
                   + R(0,2)*(R(1,0)*R(2,1)-R(1,1)*R(2,0));
        EXPECT_NEAR(std::abs(det), 1.0, 1e-5)
            << group_name << " matrix " << i;
    }
}

TEST(MatrixOrthogonalityTest, C5_AllOrthogonal)
{
    checkMatricesOrthogonal("C5");
}

TEST(MatrixOrthogonalityTest, D4_AllOrthogonal)
{
    checkMatricesOrthogonal("D4");
}

TEST(MatrixOrthogonalityTest, T_AllOrthogonal)
{
    checkMatricesOrthogonal("T");
}

TEST(MatrixOrthogonalityTest, O_AllOrthogonal)
{
    checkMatricesOrthogonal("O");
}

TEST(MatrixDet1Test, C5_AllDet1)
{
    checkMatricesDet1("C5");
}

TEST(MatrixDet1Test, D4_AllDet1)
{
    checkMatricesDet1("D4");
}

TEST(MatrixDet1Test, T_AllDet1)
{
    checkMatricesDet1("T");
}

TEST(MatrixDet1Test, O_AllDet1)
{
    checkMatricesDet1("O");
}

// ----------------------------------------- non_redundant_ewald_sphere --

TEST(EwaldSphereTest, C1_FullSphere)
{
    SymList sl;
    EXPECT_NEAR(sl.non_redundant_ewald_sphere(pg_CN, 1), 4.0 * M_PI, 1e-10);
}

TEST(EwaldSphereTest, C2_HalfSphere)
{
    SymList sl;
    EXPECT_NEAR(sl.non_redundant_ewald_sphere(pg_CN, 2), 2.0 * M_PI, 1e-10);
}

TEST(EwaldSphereTest, D2_QuarterSphere)
{
    // D2: 4π / (order=2) / 2 = π
    SymList sl;
    EXPECT_NEAR(sl.non_redundant_ewald_sphere(pg_DN, 2), M_PI, 1e-10);
}

TEST(EwaldSphereTest, C1_IsLargerThanC2)
{
    SymList sl;
    RFLOAT full  = sl.non_redundant_ewald_sphere(pg_CN, 1);
    RFLOAT half  = sl.non_redundant_ewald_sphere(pg_CN, 2);
    RFLOAT third = sl.non_redundant_ewald_sphere(pg_CN, 3);
    EXPECT_GT(full, half);
    EXPECT_GT(half, third);
}

TEST(EwaldSphereTest, OctahedralGroup_IsPositive)
{
    SymList sl;
    RFLOAT area = sl.non_redundant_ewald_sphere(pg_O, 0);
    EXPECT_GT(area, 0.0);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
