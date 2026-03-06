/*
 * GoogleTest unit tests for Euler angle functions (src/euler.h/.cpp).
 *
 * Build and run:
 *   cmake -DBUILD_TESTS=ON ...
 *   make test_euler
 *   ./build/bin/test_euler
 *
 * No MPI required; pure CPU unit tests.
 *
 * What is tested:
 *   1.  Euler_angles2matrix(0,0,0) → 3×3 identity.
 *   2.  Euler_angles2matrix at known angles → known matrix entries.
 *   3.  Euler_matrix2angles round-trip: angles → matrix → angles.
 *   4.  Euler_angles2direction at known angles.
 *   5.  Euler_direction2angles round-trip: direction → angles → direction.
 *   6.  Euler_up_down: arithmetic relationships verified.
 *   7.  Euler_another_set: arithmetic relationships verified.
 *   8.  Euler_mirrorY / mirrorX / mirrorXY: arithmetic relationships.
 *   9.  Euler_apply_transf with identity L and R → same angles.
 *  10.  Euler_rotation3DMatrix returns 4×4 with correct 3×3 submatrix.
 */

#include <gtest/gtest.h>
#include <cmath>
#include "src/euler.h"
#include "src/matrix2d.h"
#include "src/matrix1d.h"

static constexpr double EPS = 1e-5;

// Convenience: check that a 3×3 matrix is approximately the identity.
static void expectIdentity3(const Matrix2D<RFLOAT>& A)
{
    ASSERT_EQ(A.Ydim(), 3);
    ASSERT_EQ(A.Xdim(), 3);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_NEAR(A(i,j), (i == j) ? 1.0 : 0.0, EPS)
                << "at (" << i << "," << j << ")";
}

// Convenience: build a direction vector from (alpha, beta) degrees and check
// it matches the output of Euler_angles2direction.
static Matrix1D<RFLOAT> directionFromAngles(RFLOAT alpha, RFLOAT beta)
{
    Matrix1D<RFLOAT> v;
    Euler_angles2direction(alpha, beta, v);
    return v;
}

// ---------------------------------------------------------------------------
// 1. angles2matrix(0,0,0) → identity
// ---------------------------------------------------------------------------
TEST(EulerTest, AnglesZeroGivesIdentity)
{
    Matrix2D<RFLOAT> A;
    Euler_angles2matrix(0, 0, 0, A);
    expectIdentity3(A);
}

// ---------------------------------------------------------------------------
// 2. angles2matrix at known angles
// ---------------------------------------------------------------------------
// Euler_angles2matrix(0, 90, 0) should give:
//   [[0, 0, -1],
//    [0, 1,  0],
//    [1, 0,  0]]
TEST(EulerTest, Angles0_90_0KnownMatrix)
{
    Matrix2D<RFLOAT> A;
    Euler_angles2matrix(0, 90, 0, A);

    EXPECT_NEAR(A(0,0),  0.0, EPS);
    EXPECT_NEAR(A(0,1),  0.0, EPS);
    EXPECT_NEAR(A(0,2), -1.0, EPS);
    EXPECT_NEAR(A(1,0),  0.0, EPS);
    EXPECT_NEAR(A(1,1),  1.0, EPS);
    EXPECT_NEAR(A(1,2),  0.0, EPS);
    EXPECT_NEAR(A(2,0),  1.0, EPS);
    EXPECT_NEAR(A(2,1),  0.0, EPS);
    EXPECT_NEAR(A(2,2),  0.0, EPS);
}

// A rotation matrix must be orthogonal: A * A^T = I
TEST(EulerTest, MatrixIsOrthogonal)
{
    Matrix2D<RFLOAT> A;
    Euler_angles2matrix(30, 45, 60, A);

    Matrix2D<RFLOAT> product = A * A.transpose();

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_NEAR(product(i,j), (i == j) ? 1.0 : 0.0, EPS)
                << "at (" << i << "," << j << ")";
}

// Determinant of a proper rotation matrix must be +1.
TEST(EulerTest, MatrixDetIsOne)
{
    Matrix2D<RFLOAT> A;
    Euler_angles2matrix(30, 45, 60, A);
    EXPECT_NEAR(A.det(), 1.0, EPS);
}

// homogeneous=true produces a 4×4 matrix whose top-left 3×3 matches the 3×3.
TEST(EulerTest, HomogeneousFlagGives4x4)
{
    Matrix2D<RFLOAT> A3, A4;
    Euler_angles2matrix(30, 45, 60, A3, false);
    Euler_angles2matrix(30, 45, 60, A4, true);

    EXPECT_EQ(A4.Ydim(), 4);
    EXPECT_EQ(A4.Xdim(), 4);
    EXPECT_NEAR(A4(3,3), 1.0, EPS);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_NEAR(A4(i,j), A3(i,j), EPS)
                << "at (" << i << "," << j << ")";
}

// ---------------------------------------------------------------------------
// 3. matrix2angles round-trip
// ---------------------------------------------------------------------------
TEST(EulerTest, AnglesMatrixAnglesRoundTrip)
{
    // Use generic angles where tilt != 0 and tilt != 180 for a clean round-trip.
    const RFLOAT rot = 30, tilt = 45, psi = 60;

    Matrix2D<RFLOAT> A;
    Euler_angles2matrix(rot, tilt, psi, A);

    RFLOAT rrot, rtilt, rpsi;
    Euler_matrix2angles(A, rrot, rtilt, rpsi);

    EXPECT_NEAR(rrot,  rot,  EPS);
    EXPECT_NEAR(rtilt, tilt, EPS);
    EXPECT_NEAR(rpsi,  psi,  EPS);
}

TEST(EulerTest, AnglesMatrixAnglesRoundTripNegativeTilt)
{
    const RFLOAT rot = -45, tilt = 60, psi = 10;

    Matrix2D<RFLOAT> A;
    Euler_angles2matrix(rot, tilt, psi, A);

    RFLOAT rrot, rtilt, rpsi;
    Euler_matrix2angles(A, rrot, rtilt, rpsi);

    // The round-trip matrix must reproduce the same rotation.
    Matrix2D<RFLOAT> B;
    Euler_angles2matrix(rrot, rtilt, rpsi, B);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_NEAR(B(i,j), A(i,j), EPS);
}

// ---------------------------------------------------------------------------
// 4. angles2direction at known angles
// ---------------------------------------------------------------------------
// alpha=0, beta=0 → view along +Z: direction = (0, 0, 1)
TEST(EulerTest, Direction_Alpha0_Beta0)
{
    Matrix1D<RFLOAT> v = directionFromAngles(0, 0);
    EXPECT_NEAR(v(0), 0.0, EPS);
    EXPECT_NEAR(v(1), 0.0, EPS);
    EXPECT_NEAR(v(2), 1.0, EPS);
}

// alpha=0, beta=90 → direction = (1, 0, 0)
TEST(EulerTest, Direction_Alpha0_Beta90)
{
    Matrix1D<RFLOAT> v = directionFromAngles(0, 90);
    EXPECT_NEAR(v(0), 1.0, EPS);
    EXPECT_NEAR(v(1), 0.0, EPS);
    EXPECT_NEAR(v(2), 0.0, EPS);
}

// alpha=90, beta=90 → direction = (0, 1, 0)
TEST(EulerTest, Direction_Alpha90_Beta90)
{
    Matrix1D<RFLOAT> v = directionFromAngles(90, 90);
    EXPECT_NEAR(v(0), 0.0, EPS);
    EXPECT_NEAR(v(1), 1.0, EPS);
    EXPECT_NEAR(v(2), 0.0, EPS);
}

// Direction vector is always a unit vector.
TEST(EulerTest, DirectionIsUnitVector)
{
    Matrix1D<RFLOAT> v = directionFromAngles(37, 52);
    EXPECT_NEAR(v.module(), 1.0, EPS);
}

// ---------------------------------------------------------------------------
// 5. direction2angles round-trip
// ---------------------------------------------------------------------------
TEST(EulerTest, Direction2AnglesRoundTrip)
{
    const RFLOAT alpha_in = 37, beta_in = 52;
    Matrix1D<RFLOAT> v = directionFromAngles(alpha_in, beta_in);

    RFLOAT alpha_out, beta_out;
    Euler_direction2angles(v, alpha_out, beta_out);

    // Recover direction from reconstructed angles and compare vectors.
    Matrix1D<RFLOAT> v2 = directionFromAngles(alpha_out, beta_out);
    for (int k = 0; k < 3; k++)
        EXPECT_NEAR(v2(k), v(k), EPS);
}

// For the degenerate case (tilt=0) alpha is set to 0 by convention.
TEST(EulerTest, Direction2AnglesZeroTiltAlphaIsZero)
{
    Matrix1D<RFLOAT> v = directionFromAngles(45, 0);  // (0,0,1) regardless of alpha
    RFLOAT alpha_out, beta_out;
    Euler_direction2angles(v, alpha_out, beta_out);
    EXPECT_NEAR(alpha_out, 0.0, EPS);
    EXPECT_NEAR(beta_out,  0.0, EPS);
}

// ---------------------------------------------------------------------------
// 6. Euler_up_down
// ---------------------------------------------------------------------------
TEST(EulerTest, UpDownArithmetic)
{
    const RFLOAT rot = 30, tilt = 45, psi = 60;
    RFLOAT nr, nt, np;
    Euler_up_down(rot, tilt, psi, nr, nt, np);

    EXPECT_NEAR(nr, rot,           EPS);
    EXPECT_NEAR(nt, tilt + 180,    EPS);
    EXPECT_NEAR(np, -(180 + psi),  EPS);
}

// ---------------------------------------------------------------------------
// 7. Euler_another_set
// ---------------------------------------------------------------------------
TEST(EulerTest, AnotherSetArithmetic)
{
    const RFLOAT rot = 30, tilt = 45, psi = 60;
    RFLOAT nr, nt, np;
    Euler_another_set(rot, tilt, psi, nr, nt, np);

    EXPECT_NEAR(nr, rot + 180,   EPS);
    EXPECT_NEAR(nt, -tilt,       EPS);
    EXPECT_NEAR(np, -180 + psi,  EPS);
}

// ---------------------------------------------------------------------------
// 8. Mirror operations
// ---------------------------------------------------------------------------
TEST(EulerTest, MirrorYArithmetic)
{
    const RFLOAT rot = 10, tilt = 20, psi = 30;
    RFLOAT nr, nt, np;
    Euler_mirrorY(rot, tilt, psi, nr, nt, np);

    EXPECT_NEAR(nr, rot,        EPS);
    EXPECT_NEAR(nt, tilt + 180, EPS);
    EXPECT_NEAR(np, -psi,       EPS);
}

TEST(EulerTest, MirrorXArithmetic)
{
    const RFLOAT rot = 10, tilt = 20, psi = 30;
    RFLOAT nr, nt, np;
    Euler_mirrorX(rot, tilt, psi, nr, nt, np);

    EXPECT_NEAR(nr, rot,        EPS);
    EXPECT_NEAR(nt, tilt + 180, EPS);
    EXPECT_NEAR(np, 180 - psi,  EPS);
}

TEST(EulerTest, MirrorXYArithmetic)
{
    const RFLOAT rot = 10, tilt = 20, psi = 30;
    RFLOAT nr, nt, np;
    Euler_mirrorXY(rot, tilt, psi, nr, nt, np);

    EXPECT_NEAR(nr, rot,        EPS);
    EXPECT_NEAR(nt, tilt,       EPS);
    EXPECT_NEAR(np, 180 + psi,  EPS);
}

// ---------------------------------------------------------------------------
// 9. Euler_apply_transf with identity L and R
// ---------------------------------------------------------------------------
TEST(EulerTest, ApplyTransfWithIdentityIsNoop)
{
    Matrix2D<RFLOAT> I;
    I.initIdentity(3);

    const RFLOAT rot = 30, tilt = 45, psi = 60;
    RFLOAT nr, nt, np;
    Euler_apply_transf(I, I, rot, tilt, psi, nr, nt, np);

    // Resulting matrix must equal the original matrix.
    Matrix2D<RFLOAT> A_in, A_out;
    Euler_angles2matrix(rot, tilt, psi, A_in);
    Euler_angles2matrix(nr, nt, np, A_out);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_NEAR(A_out(i,j), A_in(i,j), EPS)
                << "at (" << i << "," << j << ")";
}

// ---------------------------------------------------------------------------
// 10. Euler_rotation3DMatrix returns 4×4 with correct 3×3 submatrix
// ---------------------------------------------------------------------------
TEST(EulerTest, Rotation3DMatrixIs4x4)
{
    Matrix2D<RFLOAT> A4;
    Euler_rotation3DMatrix(30, 45, 60, A4);

    EXPECT_EQ(A4.Ydim(), 4);
    EXPECT_EQ(A4.Xdim(), 4);
    EXPECT_NEAR(A4(3,3), 1.0, EPS);
}

TEST(EulerTest, Rotation3DMatrixMatchesAngles2matrix)
{
    Matrix2D<RFLOAT> A3, A4;
    Euler_angles2matrix(30, 45, 60, A3, false);
    Euler_rotation3DMatrix(30, 45, 60, A4);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_NEAR(A4(i,j), A3(i,j), EPS)
                << "at (" << i << "," << j << ")";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
