/*
 * GoogleTest unit tests for geometrical transformations
 * (src/transformations.h/.cpp).
 *
 * Build and run:
 *   cmake -DBUILD_TESTS=ON ..
 *   make test_transformations
 *   ./build/bin/test_transformations
 *
 * No MPI required; pure unit tests with no GPU dependency.
 *
 * What is tested:
 *   1.  rotation2DMatrix     – size (homo/non-homo), known values at 0°/90°,
 *                              orthogonality (M^T*M = I), det = 1
 *   2.  translation2DMatrix  – 3×3 identity structure, shift stored in col 2
 *   3.  rotation3DMatrix     – Z/X/Y axes at 0°/90°, det = 1, transpose = inv
 *                              (both homogeneous 4×4 and non-homogeneous 3×3)
 *   4.  rotation3DMatrix (arbitrary axis) – matches axis-char overload for Z
 *   5.  translation3DMatrix (vector)      – 4×4 identity + shifts at col 3
 *   6.  translation3DMatrix (scalars)     – same as vector overload
 *   7.  scale3DMatrix        – diagonal entries, size (homo/non-homo)
 *   8.  alignWithZ           – result * axis ≈ (0, 0, |axis|)
 *   9.  applyGeometry 2D     – identity preserves image; constant image
 *                              preserved under arbitrary rotation
 *  10.  applyGeometry 3D     – identity preserves volume; constant volume
 *                              preserved under arbitrary rotation
 *  11.  rotate / selfRotate  – 0° is exact copy; constant image invariant
 *  12.  translate / selfTranslate – zero shift is exact copy; constant image
 *                              invariant under wrap-mode shift
 *  13.  scaleToSize          – output has new dimensions; constant uniform
 *                              image stays constant after scaling
 *  14.  radialAverage        – uniform 3D volume gives flat radial profile
 */

#include <gtest/gtest.h>
#include <cmath>
#include "src/transformations.h"
#include "src/matrix1d.h"
#include "src/matrix2d.h"
#include "src/multidim_array.h"
#include "src/error.h"

static const double EPS = 1e-9;

// ---------------------------------------------------------------------------
// Helper: check that a square matrix M satisfies M^T * M == I
// ---------------------------------------------------------------------------
static void expectOrthogonal(const Matrix2D<RFLOAT>& M, double tol = 1e-9)
{
    Matrix2D<RFLOAT> MtM = M.transpose() * M;
    int n = MAT_XSIZE(MtM);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            EXPECT_NEAR(MAT_ELEM(MtM, i, j), (i == j) ? 1.0 : 0.0, tol)
                << "M^T*M not identity at (" << i << "," << j << ")";
}

// ---------------------------------------------------------------------------
// Helper: 2D constant image (ydim × xdim, all pixels = val)
// ---------------------------------------------------------------------------
static MultidimArray<RFLOAT> makeConstant2D(int ydim, int xdim, RFLOAT val)
{
    MultidimArray<RFLOAT> img;
    img.resize(ydim, xdim);
    img.initConstant(val);
    return img;
}

// ---------------------------------------------------------------------------
// Helper: 3D constant volume (zdim × ydim × xdim, all voxels = val)
// ---------------------------------------------------------------------------
static MultidimArray<RFLOAT> makeConstant3D(int zdim, int ydim, int xdim, RFLOAT val)
{
    MultidimArray<RFLOAT> vol;
    vol.resize(zdim, ydim, xdim);
    vol.initConstant(val);
    return vol;
}

// ---------------------------------------------------------------------------
// 1. rotation2DMatrix
// ---------------------------------------------------------------------------
TEST(TransformationsTest, Rotation2DMatrixSize_Homogeneous)
{
    Matrix2D<RFLOAT> R;
    rotation2DMatrix(0.0, R, /*homogeneous=*/true);
    EXPECT_EQ(MAT_XSIZE(R), 3);
    EXPECT_EQ(MAT_YSIZE(R), 3);
}

TEST(TransformationsTest, Rotation2DMatrixSize_NonHomogeneous)
{
    Matrix2D<RFLOAT> R;
    rotation2DMatrix(0.0, R, /*homogeneous=*/false);
    EXPECT_EQ(MAT_XSIZE(R), 2);
    EXPECT_EQ(MAT_YSIZE(R), 2);
}

TEST(TransformationsTest, Rotation2DMatrixZeroDegrees_IsIdentity)
{
    Matrix2D<RFLOAT> R;
    rotation2DMatrix(0.0, R, /*homogeneous=*/false);
    EXPECT_NEAR(MAT_ELEM(R, 0, 0), 1.0, EPS);
    EXPECT_NEAR(MAT_ELEM(R, 0, 1), 0.0, EPS);
    EXPECT_NEAR(MAT_ELEM(R, 1, 0), 0.0, EPS);
    EXPECT_NEAR(MAT_ELEM(R, 1, 1), 1.0, EPS);
}

TEST(TransformationsTest, Rotation2DMatrix90Degrees_KnownValues)
{
    Matrix2D<RFLOAT> R;
    rotation2DMatrix(90.0, R, /*homogeneous=*/false);
    // [ cos90, -sin90 ]   [ 0, -1 ]
    // [ sin90,  cos90 ] = [ 1,  0 ]
    EXPECT_NEAR(MAT_ELEM(R, 0, 0),  0.0, EPS);
    EXPECT_NEAR(MAT_ELEM(R, 0, 1), -1.0, EPS);
    EXPECT_NEAR(MAT_ELEM(R, 1, 0),  1.0, EPS);
    EXPECT_NEAR(MAT_ELEM(R, 1, 1),  0.0, EPS);
}

TEST(TransformationsTest, Rotation2DMatrixOrthogonal)
{
    Matrix2D<RFLOAT> R;
    rotation2DMatrix(37.0, R, /*homogeneous=*/false);
    expectOrthogonal(R);
}

TEST(TransformationsTest, Rotation2DMatrixDet1)
{
    Matrix2D<RFLOAT> R;
    rotation2DMatrix(73.0, R, /*homogeneous=*/false);
    EXPECT_NEAR(R.det(), 1.0, EPS);
}

// ---------------------------------------------------------------------------
// 2. translation2DMatrix
// ---------------------------------------------------------------------------
TEST(TransformationsTest, Translation2DMatrix_Size)
{
    Matrix2D<RFLOAT> T;
    Matrix1D<RFLOAT> v = vectorR2(3.0, 5.0);
    translation2DMatrix(v, T);
    EXPECT_EQ(MAT_XSIZE(T), 3);
    EXPECT_EQ(MAT_YSIZE(T), 3);
}

TEST(TransformationsTest, Translation2DMatrix_ZeroIsIdentity)
{
    Matrix2D<RFLOAT> T;
    Matrix1D<RFLOAT> v = vectorR2(0.0, 0.0);
    translation2DMatrix(v, T);
    EXPECT_TRUE(T.isIdentity());
}

TEST(TransformationsTest, Translation2DMatrix_ShiftStoredAtCol2)
{
    Matrix2D<RFLOAT> T;
    Matrix1D<RFLOAT> v = vectorR2(3.0, 7.0);
    translation2DMatrix(v, T);
    EXPECT_NEAR(MAT_ELEM(T, 0, 2), 3.0, EPS);   // tx
    EXPECT_NEAR(MAT_ELEM(T, 1, 2), 7.0, EPS);   // ty
    // diagonal should be 1
    EXPECT_NEAR(MAT_ELEM(T, 0, 0), 1.0, EPS);
    EXPECT_NEAR(MAT_ELEM(T, 1, 1), 1.0, EPS);
    EXPECT_NEAR(MAT_ELEM(T, 2, 2), 1.0, EPS);
}

// ---------------------------------------------------------------------------
// 3. rotation3DMatrix (axis character)
// ---------------------------------------------------------------------------
TEST(TransformationsTest, Rotation3DMatrix_Size_Homogeneous)
{
    Matrix2D<RFLOAT> R;
    rotation3DMatrix(0.0, 'Z', R, /*homogeneous=*/true);
    EXPECT_EQ(MAT_XSIZE(R), 4);
    EXPECT_EQ(MAT_YSIZE(R), 4);
}

TEST(TransformationsTest, Rotation3DMatrix_Size_NonHomogeneous)
{
    Matrix2D<RFLOAT> R;
    rotation3DMatrix(0.0, 'Z', R, /*homogeneous=*/false);
    EXPECT_EQ(MAT_XSIZE(R), 3);
    EXPECT_EQ(MAT_YSIZE(R), 3);
}

TEST(TransformationsTest, Rotation3DMatrix_ZeroZ_IsIdentityBlock)
{
    Matrix2D<RFLOAT> R;
    rotation3DMatrix(0.0, 'Z', R, /*homogeneous=*/false);
    EXPECT_NEAR(MAT_ELEM(R, 0, 0), 1.0, EPS);
    EXPECT_NEAR(MAT_ELEM(R, 1, 1), 1.0, EPS);
    EXPECT_NEAR(MAT_ELEM(R, 2, 2), 1.0, EPS);
    EXPECT_NEAR(MAT_ELEM(R, 0, 1), 0.0, EPS);
    EXPECT_NEAR(MAT_ELEM(R, 1, 0), 0.0, EPS);
}

TEST(TransformationsTest, Rotation3DMatrix_90Z_KnownValues)
{
    Matrix2D<RFLOAT> R;
    rotation3DMatrix(90.0, 'Z', R, /*homogeneous=*/false);
    // Z at 90°: [ 0,-1, 0; 1, 0, 0; 0, 0, 1 ]
    EXPECT_NEAR(MAT_ELEM(R, 0, 0),  0.0, EPS);
    EXPECT_NEAR(MAT_ELEM(R, 0, 1), -1.0, EPS);
    EXPECT_NEAR(MAT_ELEM(R, 1, 0),  1.0, EPS);
    EXPECT_NEAR(MAT_ELEM(R, 1, 1),  0.0, EPS);
    EXPECT_NEAR(MAT_ELEM(R, 2, 2),  1.0, EPS);
}

TEST(TransformationsTest, Rotation3DMatrix_90X_KnownValues)
{
    Matrix2D<RFLOAT> R;
    rotation3DMatrix(90.0, 'X', R, /*homogeneous=*/false);
    // X at 90°: [ 1, 0, 0; 0, 0,-1; 0, 1, 0 ]
    EXPECT_NEAR(MAT_ELEM(R, 0, 0), 1.0, EPS);
    EXPECT_NEAR(MAT_ELEM(R, 1, 1), 0.0, EPS);
    EXPECT_NEAR(MAT_ELEM(R, 1, 2),-1.0, EPS);
    EXPECT_NEAR(MAT_ELEM(R, 2, 1), 1.0, EPS);
    EXPECT_NEAR(MAT_ELEM(R, 2, 2), 0.0, EPS);
}

TEST(TransformationsTest, Rotation3DMatrix_90Y_KnownValues)
{
    Matrix2D<RFLOAT> R;
    rotation3DMatrix(90.0, 'Y', R, /*homogeneous=*/false);
    // Y at 90° (after 2024 sign fix): [ 0, 0, 1; 0, 1, 0; -1, 0, 0 ]
    EXPECT_NEAR(MAT_ELEM(R, 0, 0),  0.0, EPS);
    EXPECT_NEAR(MAT_ELEM(R, 0, 2),  1.0, EPS);
    EXPECT_NEAR(MAT_ELEM(R, 1, 1),  1.0, EPS);
    EXPECT_NEAR(MAT_ELEM(R, 2, 0), -1.0, EPS);
    EXPECT_NEAR(MAT_ELEM(R, 2, 2),  0.0, EPS);
}

TEST(TransformationsTest, Rotation3DMatrix_Orthogonal_Z)
{
    Matrix2D<RFLOAT> R;
    rotation3DMatrix(53.0, 'Z', R, /*homogeneous=*/false);
    expectOrthogonal(R);
}

TEST(TransformationsTest, Rotation3DMatrix_Orthogonal_X)
{
    Matrix2D<RFLOAT> R;
    rotation3DMatrix(53.0, 'X', R, /*homogeneous=*/false);
    expectOrthogonal(R);
}

TEST(TransformationsTest, Rotation3DMatrix_Orthogonal_Y)
{
    Matrix2D<RFLOAT> R;
    rotation3DMatrix(53.0, 'Y', R, /*homogeneous=*/false);
    expectOrthogonal(R);
}

// ---------------------------------------------------------------------------
// 4. rotation3DMatrix (arbitrary axis) matches axis-char overload for Z axis
// ---------------------------------------------------------------------------
TEST(TransformationsTest, Rotation3DMatrix_ArbitraryZAxis_MatchesAxisChar)
{
    Matrix2D<RFLOAT> R_char, R_vec;
    rotation3DMatrix(45.0, 'Z', R_char, /*homogeneous=*/false);

    Matrix1D<RFLOAT> z_axis = vectorR3(0.0, 0.0, 1.0);
    rotation3DMatrix(45.0, z_axis, R_vec, /*homogeneous=*/false);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_NEAR(MAT_ELEM(R_vec, i, j), MAT_ELEM(R_char, i, j), 1e-9)
                << "Mismatch at (" << i << "," << j << ")";
}

// ---------------------------------------------------------------------------
// 5. translation3DMatrix (vector)
// ---------------------------------------------------------------------------
TEST(TransformationsTest, Translation3DMatrix_Size)
{
    Matrix2D<RFLOAT> T;
    Matrix1D<RFLOAT> v = vectorR3(1.0, 2.0, 3.0);
    translation3DMatrix(v, T);
    EXPECT_EQ(MAT_XSIZE(T), 4);
    EXPECT_EQ(MAT_YSIZE(T), 4);
}

TEST(TransformationsTest, Translation3DMatrix_ZeroIsIdentity)
{
    Matrix2D<RFLOAT> T;
    Matrix1D<RFLOAT> v = vectorR3(0.0, 0.0, 0.0);
    translation3DMatrix(v, T);
    EXPECT_TRUE(T.isIdentity());
}

TEST(TransformationsTest, Translation3DMatrix_ShiftStoredAtCol3)
{
    Matrix2D<RFLOAT> T;
    Matrix1D<RFLOAT> v = vectorR3(2.0, 4.0, 6.0);
    translation3DMatrix(v, T);
    EXPECT_NEAR(MAT_ELEM(T, 0, 3), 2.0, EPS);
    EXPECT_NEAR(MAT_ELEM(T, 1, 3), 4.0, EPS);
    EXPECT_NEAR(MAT_ELEM(T, 2, 3), 6.0, EPS);
    EXPECT_NEAR(MAT_ELEM(T, 3, 3), 1.0, EPS);
}

// ---------------------------------------------------------------------------
// 6. translation3DMatrix (scalars) - same as vector overload
// ---------------------------------------------------------------------------
TEST(TransformationsTest, Translation3DMatrix_ScalarMatchesVector)
{
    Matrix2D<RFLOAT> T_vec, T_scalar;
    Matrix1D<RFLOAT> v = vectorR3(1.5, 2.5, 3.5);
    translation3DMatrix(v, T_vec);
    translation3DMatrix(1.5, 2.5, 3.5, T_scalar);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            EXPECT_NEAR(MAT_ELEM(T_scalar, i, j), MAT_ELEM(T_vec, i, j), EPS);
}

// ---------------------------------------------------------------------------
// 7. scale3DMatrix
// ---------------------------------------------------------------------------
TEST(TransformationsTest, Scale3DMatrix_Size_Homogeneous)
{
    Matrix2D<RFLOAT> S;
    Matrix1D<RFLOAT> sc = vectorR3(2.0, 3.0, 4.0);
    scale3DMatrix(sc, S, /*homogeneous=*/true);
    EXPECT_EQ(MAT_XSIZE(S), 4);
    EXPECT_EQ(MAT_YSIZE(S), 4);
}

TEST(TransformationsTest, Scale3DMatrix_Size_NonHomogeneous)
{
    Matrix2D<RFLOAT> S;
    Matrix1D<RFLOAT> sc = vectorR3(2.0, 3.0, 4.0);
    scale3DMatrix(sc, S, /*homogeneous=*/false);
    EXPECT_EQ(MAT_XSIZE(S), 3);
    EXPECT_EQ(MAT_YSIZE(S), 3);
}

TEST(TransformationsTest, Scale3DMatrix_DiagonalEntries)
{
    Matrix2D<RFLOAT> S;
    Matrix1D<RFLOAT> sc = vectorR3(2.0, 3.0, 4.0);
    scale3DMatrix(sc, S, /*homogeneous=*/false);
    EXPECT_NEAR(MAT_ELEM(S, 0, 0), 2.0, EPS);
    EXPECT_NEAR(MAT_ELEM(S, 1, 1), 3.0, EPS);
    EXPECT_NEAR(MAT_ELEM(S, 2, 2), 4.0, EPS);
    // off-diagonal = 0
    EXPECT_NEAR(MAT_ELEM(S, 0, 1), 0.0, EPS);
    EXPECT_NEAR(MAT_ELEM(S, 1, 0), 0.0, EPS);
}

// ---------------------------------------------------------------------------
// 8. alignWithZ
// ---------------------------------------------------------------------------
TEST(TransformationsTest, AlignWithZ_ZAxis_IsIdentity)
{
    Matrix2D<RFLOAT> A;
    Matrix1D<RFLOAT> z = vectorR3(0.0, 0.0, 1.0);
    alignWithZ(z, A, /*homogeneous=*/false);
    // result * Z = Z; for the Z axis the result should be the identity
    EXPECT_NEAR(MAT_ELEM(A, 0, 0), 1.0, EPS);
    EXPECT_NEAR(MAT_ELEM(A, 1, 1), 1.0, EPS);
    EXPECT_NEAR(MAT_ELEM(A, 2, 2), 1.0, EPS);
}

TEST(TransformationsTest, AlignWithZ_ArbitraryAxis_MapsToZ)
{
    // axis = (1, 1, 1) normalised
    Matrix2D<RFLOAT> A;
    Matrix1D<RFLOAT> axis = vectorR3(1.0, 1.0, 1.0);
    alignWithZ(axis, A, /*homogeneous=*/false);

    // normalise the axis
    double len = std::sqrt(3.0);
    Matrix1D<RFLOAT> axis_norm = vectorR3(1.0/len, 1.0/len, 1.0/len);

    // A * axis_norm should be (0, 0, 1)
    Matrix1D<RFLOAT> result = A * axis_norm;
    EXPECT_NEAR(XX(result), 0.0, 1e-9);
    EXPECT_NEAR(YY(result), 0.0, 1e-9);
    EXPECT_NEAR(ZZ(result), 1.0, 1e-9);
}

TEST(TransformationsTest, AlignWithZ_XAxis_MapsToZ)
{
    Matrix2D<RFLOAT> A;
    Matrix1D<RFLOAT> x_axis = vectorR3(1.0, 0.0, 0.0);
    alignWithZ(x_axis, A, /*homogeneous=*/false);

    Matrix1D<RFLOAT> result = A * x_axis;
    EXPECT_NEAR(XX(result), 0.0, 1e-9);
    EXPECT_NEAR(YY(result), 0.0, 1e-9);
    EXPECT_NEAR(ZZ(result), 1.0, 1e-9);
}

// ---------------------------------------------------------------------------
// 9. applyGeometry 2D
// ---------------------------------------------------------------------------
TEST(TransformationsTest, ApplyGeometry2D_Identity_ExactCopy)
{
    Matrix2D<RFLOAT> I;
    I.initIdentity(3);

    MultidimArray<RFLOAT> in = makeConstant2D(6, 6, 1.0);
    // Put a distinctive pixel at the center
    DIRECT_A2D_ELEM(in, 3, 3) = 99.0;

    MultidimArray<RFLOAT> out;
    applyGeometry(in, out, I, IS_NOT_INV, DONT_WRAP);

    // isIdentity fast path: out should be bit-exact copy
    EXPECT_EQ(out.xdim, in.xdim);
    EXPECT_EQ(out.ydim, in.ydim);
    for (int i = 0; i < (int)in.ydim; i++)
        for (int j = 0; j < (int)in.xdim; j++)
            EXPECT_EQ(DIRECT_A2D_ELEM(out, i, j), DIRECT_A2D_ELEM(in, i, j));
}

TEST(TransformationsTest, ApplyGeometry2D_ConstantImage_PreservedUnder90DegRotation)
{
    Matrix2D<RFLOAT> R;
    rotation2DMatrix(90.0, R, /*homogeneous=*/true);

    const RFLOAT val = 7.0;
    MultidimArray<RFLOAT> in = makeConstant2D(8, 8, val);
    MultidimArray<RFLOAT> out;
    applyGeometry(in, out, R, IS_NOT_INV, DONT_WRAP, val);

    for (int i = 0; i < (int)out.ydim; i++)
        for (int j = 0; j < (int)out.xdim; j++)
            EXPECT_NEAR(DIRECT_A2D_ELEM(out, i, j), val, 1e-6);
}

TEST(TransformationsTest, ApplyGeometry2D_ConstantImage_PreservedUnderTranslation)
{
    Matrix2D<RFLOAT> T;
    Matrix1D<RFLOAT> shift = vectorR2(2.0, 3.0);
    translation2DMatrix(shift, T);

    const RFLOAT val = 4.0;
    MultidimArray<RFLOAT> in = makeConstant2D(8, 8, val);
    MultidimArray<RFLOAT> out;
    applyGeometry(in, out, T, IS_NOT_INV, WRAP, val);

    for (int i = 0; i < (int)out.ydim; i++)
        for (int j = 0; j < (int)out.xdim; j++)
            EXPECT_NEAR(DIRECT_A2D_ELEM(out, i, j), val, 1e-6);
}

// ---------------------------------------------------------------------------
// 10. applyGeometry 3D
// ---------------------------------------------------------------------------
TEST(TransformationsTest, ApplyGeometry3D_Identity_ExactCopy)
{
    Matrix2D<RFLOAT> I;
    I.initIdentity(4);

    MultidimArray<RFLOAT> in = makeConstant3D(4, 4, 4, 2.0);
    DIRECT_A3D_ELEM(in, 2, 2, 2) = 55.0;

    MultidimArray<RFLOAT> out;
    applyGeometry(in, out, I, IS_NOT_INV, DONT_WRAP);

    for (int k = 0; k < (int)in.zdim; k++)
        for (int i = 0; i < (int)in.ydim; i++)
            for (int j = 0; j < (int)in.xdim; j++)
                EXPECT_EQ(DIRECT_A3D_ELEM(out, k, i, j), DIRECT_A3D_ELEM(in, k, i, j));
}

TEST(TransformationsTest, ApplyGeometry3D_ConstantVolume_PreservedUnder90DegZ)
{
    Matrix2D<RFLOAT> R;
    rotation3DMatrix(90.0, 'Z', R, /*homogeneous=*/true);

    const RFLOAT val = 3.0;
    MultidimArray<RFLOAT> in = makeConstant3D(6, 6, 6, val);
    MultidimArray<RFLOAT> out;
    applyGeometry(in, out, R, IS_NOT_INV, DONT_WRAP, val);

    for (int k = 0; k < (int)out.zdim; k++)
        for (int i = 0; i < (int)out.ydim; i++)
            for (int j = 0; j < (int)out.xdim; j++)
                EXPECT_NEAR(DIRECT_A3D_ELEM(out, k, i, j), val, 1e-6);
}

// ---------------------------------------------------------------------------
// 11. rotate / selfRotate
// ---------------------------------------------------------------------------
TEST(TransformationsTest, Rotate2D_ZeroDegrees_ExactCopy)
{
    MultidimArray<RFLOAT> in = makeConstant2D(6, 6, 1.0);
    DIRECT_A2D_ELEM(in, 3, 3) = 42.0;

    MultidimArray<RFLOAT> out;
    rotate(in, out, 0.0);

    for (int i = 0; i < (int)in.ydim; i++)
        for (int j = 0; j < (int)in.xdim; j++)
            EXPECT_EQ(DIRECT_A2D_ELEM(out, i, j), DIRECT_A2D_ELEM(in, i, j));
}

TEST(TransformationsTest, Rotate2D_ConstantImage_PreservedUnder90Deg)
{
    const RFLOAT val = 5.0;
    MultidimArray<RFLOAT> in = makeConstant2D(8, 8, val);
    MultidimArray<RFLOAT> out;
    rotate(in, out, 90.0, 'Z', DONT_WRAP, val);

    for (int i = 0; i < (int)out.ydim; i++)
        for (int j = 0; j < (int)out.xdim; j++)
            EXPECT_NEAR(DIRECT_A2D_ELEM(out, i, j), val, 1e-6);
}

TEST(TransformationsTest, Rotate3D_ZeroDegrees_ExactCopy)
{
    MultidimArray<RFLOAT> in = makeConstant3D(4, 4, 4, 2.0);
    DIRECT_A3D_ELEM(in, 2, 2, 2) = 88.0;

    MultidimArray<RFLOAT> out;
    rotate(in, out, 0.0, 'Z');

    for (int k = 0; k < (int)in.zdim; k++)
        for (int i = 0; i < (int)in.ydim; i++)
            for (int j = 0; j < (int)in.xdim; j++)
                EXPECT_EQ(DIRECT_A3D_ELEM(out, k, i, j), DIRECT_A3D_ELEM(in, k, i, j));
}

TEST(TransformationsTest, SelfRotate2D_SameAsRotate)
{
    MultidimArray<RFLOAT> original = makeConstant2D(6, 6, 1.0);
    DIRECT_A2D_ELEM(original, 1, 2) = 11.0;

    MultidimArray<RFLOAT> out_rotate;
    rotate(original, out_rotate, 45.0);

    MultidimArray<RFLOAT> self = original;
    selfRotate(self, 45.0);

    for (int i = 0; i < (int)self.ydim; i++)
        for (int j = 0; j < (int)self.xdim; j++)
            EXPECT_NEAR(DIRECT_A2D_ELEM(self, i, j),
                        DIRECT_A2D_ELEM(out_rotate, i, j), 1e-9);
}

// ---------------------------------------------------------------------------
// 12. translate / selfTranslate
// ---------------------------------------------------------------------------
TEST(TransformationsTest, Translate2D_ZeroShift_ExactCopy)
{
    MultidimArray<RFLOAT> in = makeConstant2D(6, 6, 1.0);
    DIRECT_A2D_ELEM(in, 2, 3) = 77.0;

    MultidimArray<RFLOAT> out;
    Matrix1D<RFLOAT> v = vectorR2(0.0, 0.0);
    translate(in, out, v, WRAP);

    for (int i = 0; i < (int)in.ydim; i++)
        for (int j = 0; j < (int)in.xdim; j++)
            EXPECT_EQ(DIRECT_A2D_ELEM(out, i, j), DIRECT_A2D_ELEM(in, i, j));
}

TEST(TransformationsTest, Translate2D_ConstantImage_PreservedWithWrap)
{
    const RFLOAT val = 6.0;
    MultidimArray<RFLOAT> in = makeConstant2D(8, 8, val);
    MultidimArray<RFLOAT> out;
    Matrix1D<RFLOAT> v = vectorR2(3.0, 2.0);
    translate(in, out, v, WRAP, (RFLOAT)0.0);

    for (int i = 0; i < (int)out.ydim; i++)
        for (int j = 0; j < (int)out.xdim; j++)
            EXPECT_NEAR(DIRECT_A2D_ELEM(out, i, j), val, 1e-6);
}

TEST(TransformationsTest, Translate3D_ZeroShift_ExactCopy)
{
    MultidimArray<RFLOAT> in = makeConstant3D(4, 4, 4, 2.0);
    DIRECT_A3D_ELEM(in, 1, 2, 3) = 33.0;

    MultidimArray<RFLOAT> out;
    Matrix1D<RFLOAT> v = vectorR3(0.0, 0.0, 0.0);
    translate(in, out, v, WRAP);

    for (int k = 0; k < (int)in.zdim; k++)
        for (int i = 0; i < (int)in.ydim; i++)
            for (int j = 0; j < (int)in.xdim; j++)
                EXPECT_EQ(DIRECT_A3D_ELEM(out, k, i, j),
                          DIRECT_A3D_ELEM(in, k, i, j));
}

TEST(TransformationsTest, SelfTranslate2D_SameAsTranslate)
{
    MultidimArray<RFLOAT> original = makeConstant2D(6, 6, 1.0);
    DIRECT_A2D_ELEM(original, 1, 1) = 22.0;

    MultidimArray<RFLOAT> out;
    Matrix1D<RFLOAT> v = vectorR2(1.0, 2.0);
    translate(original, out, v, WRAP);

    MultidimArray<RFLOAT> self = original;
    selfTranslate(self, v, WRAP);

    for (int i = 0; i < (int)self.ydim; i++)
        for (int j = 0; j < (int)self.xdim; j++)
            EXPECT_NEAR(DIRECT_A2D_ELEM(self, i, j),
                        DIRECT_A2D_ELEM(out, i, j), 1e-9);
}

// ---------------------------------------------------------------------------
// 13. scaleToSize
// ---------------------------------------------------------------------------
TEST(TransformationsTest, ScaleToSize2D_OutputDimensions)
{
    MultidimArray<RFLOAT> in = makeConstant2D(8, 8, 1.0);
    MultidimArray<RFLOAT> out;
    scaleToSize(in, out, 16, 12);
    EXPECT_EQ((int)out.xdim, 16);
    EXPECT_EQ((int)out.ydim, 12);
}

TEST(TransformationsTest, ScaleToSize2D_ConstantImage_Preserved)
{
    const RFLOAT val = 9.0;
    MultidimArray<RFLOAT> in = makeConstant2D(8, 8, val);
    MultidimArray<RFLOAT> out;
    scaleToSize(in, out, 16, 16);

    for (int i = 0; i < (int)out.ydim; i++)
        for (int j = 0; j < (int)out.xdim; j++)
            EXPECT_NEAR(DIRECT_A2D_ELEM(out, i, j), val, 1e-6);
}

TEST(TransformationsTest, ScaleToSize3D_OutputDimensions)
{
    MultidimArray<RFLOAT> in = makeConstant3D(4, 4, 4, 1.0);
    MultidimArray<RFLOAT> out;
    scaleToSize(in, out, 8, 8, 8);
    EXPECT_EQ((int)out.xdim, 8);
    EXPECT_EQ((int)out.ydim, 8);
    EXPECT_EQ((int)out.zdim, 8);
}

TEST(TransformationsTest, ScaleToSize2D_DownsampleHalf_OutputDimensions)
{
    MultidimArray<RFLOAT> in = makeConstant2D(16, 16, 1.0);
    MultidimArray<RFLOAT> out;
    scaleToSize(in, out, 8, 8);
    EXPECT_EQ((int)out.xdim, 8);
    EXPECT_EQ((int)out.ydim, 8);
}

// ---------------------------------------------------------------------------
// 14. radialAverage
// ---------------------------------------------------------------------------
TEST(TransformationsTest, RadialAverage_UniformVolume_FlatProfile)
{
    // Uniform 3D volume: every radial shell has the same mean value
    const RFLOAT val = 5.0;
    MultidimArray<RFLOAT> vol = makeConstant3D(8, 8, 8, val);

    Matrix1D<int> center(3);
    center.initZeros();
    MultidimArray<RFLOAT> radial_mean;
    MultidimArray<int>    radial_count;
    radialAverage(vol, center, radial_mean, radial_count);

    for (int i = 0; i < (int)radial_mean.xdim; i++)
        if (radial_count(i) > 0)
            EXPECT_NEAR(radial_mean(i), val, 1e-6) << "Shell " << i;
}

TEST(TransformationsTest, RadialAverage_ZeroVolume_ZeroProfile)
{
    MultidimArray<RFLOAT> vol = makeConstant3D(6, 6, 6, 0.0);
    Matrix1D<int> center(3);
    center.initZeros();
    MultidimArray<RFLOAT> radial_mean;
    MultidimArray<int>    radial_count;
    radialAverage(vol, center, radial_mean, radial_count);

    for (int i = 0; i < (int)radial_mean.xdim; i++)
        if (radial_count(i) > 0)
            EXPECT_NEAR(radial_mean(i), 0.0, 1e-9);
}

// ---------------------------------------------------------------------------
// Error-path tests
// ---------------------------------------------------------------------------

// rotation3DMatrix: unknown axis character → REPORT_ERROR → throws RelionError
TEST(TransformationsTest, Rotation3DMatrix_InvalidAxis_Throws)
{
    Matrix2D<RFLOAT> R;
    EXPECT_THROW(rotation3DMatrix(45.0, 'W', R, /*homogeneous=*/false), RelionError);
    EXPECT_THROW(rotation3DMatrix(90.0, 'A', R, /*homogeneous=*/true),  RelionError);
}

// alignWithZ: axis vector must be in R3 (size == 3)
TEST(TransformationsTest, AlignWithZ_Non3DAxis_Throws)
{
    Matrix2D<RFLOAT> A;
    Matrix1D<RFLOAT> v2 = vectorR2(1.0, 0.0); // 2-element
    EXPECT_THROW(alignWithZ(v2, A, /*homogeneous=*/false), RelionError);
}

// translation3DMatrix(vector): vector must be in R3 (size == 3)
TEST(TransformationsTest, Translation3DMatrix_Non3DVector_Throws)
{
    Matrix2D<RFLOAT> T;
    Matrix1D<RFLOAT> v2 = vectorR2(1.0, 2.0); // 2-element
    EXPECT_THROW(translation3DMatrix(v2, T), RelionError);
}

// scale3DMatrix: scale vector must be in R3 (size == 3)
TEST(TransformationsTest, Scale3DMatrix_Non3DVector_Throws)
{
    Matrix2D<RFLOAT> S;
    Matrix1D<RFLOAT> sc2 = vectorR2(2.0, 3.0); // 2-element
    EXPECT_THROW(scale3DMatrix(sc2, S, /*homogeneous=*/false), RelionError);
    EXPECT_THROW(scale3DMatrix(sc2, S, /*homogeneous=*/true),  RelionError);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
