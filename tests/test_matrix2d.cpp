/*
 * GoogleTest unit tests for Matrix2D<T>.
 *
 * Build and run:
 *   cmake -DBUILD_TESTS=ON ...
 *   make test_matrix2d
 *   ./build/bin/test_matrix2d
 *
 * No MPI required; pure CPU unit tests.
 *
 * What is tested:
 *   1.  Default constructor – empty (mdimx=mdimy=0).
 *   2.  Dimension constructor – correct Xdim/Ydim.
 *   3.  Copy constructor – deep copy, independent mutation.
 *   4.  Assignment operator – deep copy.
 *   5.  clear() – returns to empty state.
 *   6.  resize() – expands with zeros, preserves existing data.
 *   7.  initZeros() – all elements zero.
 *   8.  initConstant() – all elements equal supplied value.
 *   9.  initIdentity() – produces identity matrix.
 *  10.  isIdentity() – true for identity, false otherwise.
 *  11.  operator()(i,j) – element read/write.
 *  12.  MAT_ELEM macro – element access.
 *  13.  MAT_XSIZE / MAT_YSIZE / Xdim / Ydim.
 *  14.  sameShape() – matching and mismatched sizes.
 *  15.  equal() – element-wise equality with tolerance.
 *  16.  Scalar arithmetic: *, / (and in-place *=, /=).
 *  17.  Matrix addition / subtraction (+, -, +=, -=).
 *  18.  Matrix-by-matrix multiplication – known result.
 *  19.  Matrix-by-vector multiplication.
 *  20.  Vector-by-matrix multiplication.
 *  21.  transpose() – correct transposition.
 *  22.  inv() – M * inv(M) ≈ I for 2×2 and 3×3.
 *  23.  det() – identity det=1, known 2×2 value.
 *  24.  computeMax() / computeMin().
 *  25.  getRow() / getCol() / setRow() / setCol().
 *  26.  fromVector() / toVector().
 *  27.  submatrix() extraction.
 *  28.  typeCast() – int → double.
 */

#include <gtest/gtest.h>
#include <cmath>
#include "src/matrix2d.h"
#include "src/error.h"

using Mat = Matrix2D<double>;
using Vec = Matrix1D<double>;

static constexpr double EPS = 1e-9;
static constexpr double NEAR = 1e-6;

// Helper: build a 2×2 [[a,b],[c,d]]
static Mat make2x2(double a, double b, double c, double d)
{
    Mat m(2, 2);
    m(0,0) = a; m(0,1) = b;
    m(1,0) = c; m(1,1) = d;
    return m;
}

// Helper: build a 3×3 identity
static Mat identity3()
{
    Mat m(3, 3);
    m.initIdentity();
    return m;
}

// ---------------------------------------------------------------------------
// 1. Default constructor
// ---------------------------------------------------------------------------
TEST(Matrix2DTest, DefaultConstructorIsEmpty)
{
    Mat m;
    EXPECT_EQ(m.mdimx, 0);
    EXPECT_EQ(m.mdimy, 0);
    EXPECT_EQ(m.mdata, nullptr);
}

// ---------------------------------------------------------------------------
// 2. Dimension constructor
// ---------------------------------------------------------------------------
TEST(Matrix2DTest, DimensionConstructorSetsSize)
{
    Mat m(3, 4);
    EXPECT_EQ(m.Ydim(), 3);
    EXPECT_EQ(m.Xdim(), 4);
    EXPECT_NE(m.mdata, nullptr);
}

// ---------------------------------------------------------------------------
// 3. Copy constructor
// ---------------------------------------------------------------------------
TEST(Matrix2DTest, CopyConstructorIsDeepCopy)
{
    Mat a(2, 2);
    a(0,0) = 1; a(0,1) = 2;
    a(1,0) = 3; a(1,1) = 4;

    Mat b(a);
    EXPECT_NEAR(b(0,0), 1.0, EPS);

    b(0,0) = 99.0;
    EXPECT_NEAR(a(0,0), 1.0, EPS);  // a unaffected
}

// ---------------------------------------------------------------------------
// 4. Assignment operator
// ---------------------------------------------------------------------------
TEST(Matrix2DTest, AssignmentIsDeepCopy)
{
    Mat a = make2x2(5, 6, 7, 8);
    Mat b;
    b = a;
    EXPECT_NEAR(b(1,1), 8.0, EPS);
    b(1,1) = 0;
    EXPECT_NEAR(a(1,1), 8.0, EPS);
}

// ---------------------------------------------------------------------------
// 5. clear()
// ---------------------------------------------------------------------------
TEST(Matrix2DTest, ClearReturnsToEmpty)
{
    Mat m(3, 3);
    m.clear();
    EXPECT_EQ(m.mdimx, 0);
    EXPECT_EQ(m.mdimy, 0);
    EXPECT_EQ(m.mdata, nullptr);
}

// ---------------------------------------------------------------------------
// 6. resize()
// ---------------------------------------------------------------------------
TEST(Matrix2DTest, ResizeExpandsWithZeros)
{
    Mat m(2, 2);
    m(0,0) = 1; m(0,1) = 2;
    m(1,0) = 3; m(1,1) = 4;

    m.resize(3, 3);
    EXPECT_EQ(m.Ydim(), 3);
    EXPECT_EQ(m.Xdim(), 3);
    EXPECT_NEAR(m(0,0), 1.0, EPS);
    EXPECT_NEAR(m(0,2), 0.0, EPS);  // new element is zero
    EXPECT_NEAR(m(2,2), 0.0, EPS);
}

// ---------------------------------------------------------------------------
// 7. initZeros()
// ---------------------------------------------------------------------------
TEST(Matrix2DTest, InitZerosAllZero)
{
    Mat m(3, 3);
    m.initConstant(5.0);
    m.initZeros();
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_NEAR(m(i,j), 0.0, EPS);
}

TEST(Matrix2DTest, InitZerosWithSize)
{
    Mat m;
    m.initZeros(2, 4);
    EXPECT_EQ(m.Ydim(), 2);
    EXPECT_EQ(m.Xdim(), 4);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++)
            EXPECT_NEAR(m(i,j), 0.0, EPS);
}

// ---------------------------------------------------------------------------
// 8. initConstant()
// ---------------------------------------------------------------------------
TEST(Matrix2DTest, InitConstant)
{
    Mat m(2, 3);
    m.initConstant(7.0);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_NEAR(m(i,j), 7.0, EPS);
}

// ---------------------------------------------------------------------------
// 9. initIdentity()
// ---------------------------------------------------------------------------
TEST(Matrix2DTest, InitIdentity3x3)
{
    Mat m(3, 3);
    m.initIdentity();
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_NEAR(m(i,j), (i == j) ? 1.0 : 0.0, EPS);
}

// ---------------------------------------------------------------------------
// 10. isIdentity()
// ---------------------------------------------------------------------------
TEST(Matrix2DTest, IsIdentityTrue)
{
    EXPECT_TRUE(identity3().isIdentity());
}

TEST(Matrix2DTest, IsIdentityFalse)
{
    Mat m = make2x2(1, 0, 1, 1);  // lower-triangular, not identity
    EXPECT_FALSE(m.isIdentity());
}

// ---------------------------------------------------------------------------
// 11. operator()(i,j) element access
// ---------------------------------------------------------------------------
TEST(Matrix2DTest, ElementAccessRoundTrip)
{
    Mat m(3, 3);
    m(1, 2) = 42.0;
    EXPECT_NEAR(m(1, 2), 42.0, EPS);
}

// ---------------------------------------------------------------------------
// 12. MAT_ELEM macro
// ---------------------------------------------------------------------------
TEST(Matrix2DTest, MatElemMacro)
{
    Mat m(2, 2);
    MAT_ELEM(m, 0, 1) = 9.0;
    EXPECT_NEAR(MAT_ELEM(m, 0, 1), 9.0, EPS);
}

// ---------------------------------------------------------------------------
// 13. MAT_XSIZE / MAT_YSIZE / Xdim / Ydim
// ---------------------------------------------------------------------------
TEST(Matrix2DTest, DimensionMacrosAndMethods)
{
    Mat m(4, 5);
    EXPECT_EQ(MAT_XSIZE(m), 5);
    EXPECT_EQ(MAT_YSIZE(m), 4);
    EXPECT_EQ(m.Xdim(), 5);
    EXPECT_EQ(m.Ydim(), 4);
}

// ---------------------------------------------------------------------------
// 14. sameShape()
// ---------------------------------------------------------------------------
TEST(Matrix2DTest, SameShapeMatchingDimensions)
{
    Mat a(3, 3), b(3, 3);
    EXPECT_TRUE(a.sameShape(b));
}

TEST(Matrix2DTest, SameShapeMismatch)
{
    Mat a(3, 3), b(2, 3);
    EXPECT_FALSE(a.sameShape(b));
}

// ---------------------------------------------------------------------------
// 15. equal()
// ---------------------------------------------------------------------------
TEST(Matrix2DTest, EqualWithinTolerance)
{
    Mat a = make2x2(1, 2, 3, 4);
    Mat b = make2x2(1, 2, 3, 4);
    EXPECT_TRUE(a.equal(b));
}

TEST(Matrix2DTest, EqualOutsideTolerance)
{
    Mat a = make2x2(1, 2, 3, 4);
    Mat b = make2x2(1, 2, 3, 5);
    EXPECT_FALSE(a.equal(b));
}

// ---------------------------------------------------------------------------
// 16. Scalar arithmetic
// ---------------------------------------------------------------------------
TEST(Matrix2DTest, ScalarMultiply)
{
    Mat m = make2x2(1, 2, 3, 4);
    Mat r = m * 2.0;
    EXPECT_NEAR(r(0,0), 2.0, EPS);
    EXPECT_NEAR(r(1,1), 8.0, EPS);
}

TEST(Matrix2DTest, ScalarDivide)
{
    Mat m = make2x2(4, 6, 8, 10);
    Mat r = m / 2.0;
    EXPECT_NEAR(r(0,0), 2.0, EPS);
    EXPECT_NEAR(r(1,1), 5.0, EPS);
}

TEST(Matrix2DTest, InPlaceScalarMultiply)
{
    Mat m = make2x2(1, 2, 3, 4);
    m *= 3.0;
    EXPECT_NEAR(m(0,0), 3.0, EPS);
    EXPECT_NEAR(m(1,1), 12.0, EPS);
}

TEST(Matrix2DTest, InPlaceScalarDivide)
{
    Mat m = make2x2(6, 8, 10, 12);
    m /= 2.0;
    EXPECT_NEAR(m(0,0), 3.0, EPS);
    EXPECT_NEAR(m(1,1), 6.0, EPS);
}

// ---------------------------------------------------------------------------
// 17. Matrix addition / subtraction
// ---------------------------------------------------------------------------
TEST(Matrix2DTest, MatrixAdd)
{
    Mat a = make2x2(1, 2, 3, 4);
    Mat b = make2x2(5, 6, 7, 8);
    Mat c = a + b;
    EXPECT_NEAR(c(0,0), 6.0, EPS);
    EXPECT_NEAR(c(1,1), 12.0, EPS);
}

TEST(Matrix2DTest, MatrixSubtract)
{
    Mat a = make2x2(5, 6, 7, 8);
    Mat b = make2x2(1, 2, 3, 4);
    Mat c = a - b;
    EXPECT_NEAR(c(0,0), 4.0, EPS);
    EXPECT_NEAR(c(1,1), 4.0, EPS);
}

TEST(Matrix2DTest, InPlaceMatrixAdd)
{
    Mat a = make2x2(1, 2, 3, 4);
    Mat b = make2x2(1, 1, 1, 1);
    a += b;
    EXPECT_NEAR(a(0,0), 2.0, EPS);
    EXPECT_NEAR(a(1,1), 5.0, EPS);
}

TEST(Matrix2DTest, InPlaceMatrixSubtract)
{
    Mat a = make2x2(5, 6, 7, 8);
    Mat b = make2x2(1, 1, 1, 1);
    a -= b;
    EXPECT_NEAR(a(0,0), 4.0, EPS);
    EXPECT_NEAR(a(1,1), 7.0, EPS);
}

// ---------------------------------------------------------------------------
// 18. Matrix-by-matrix multiplication
// ---------------------------------------------------------------------------
TEST(Matrix2DTest, MatMulIdentityIsNoop)
{
    Mat a = make2x2(2, 3, 4, 5);
    Mat I(2, 2);
    I.initIdentity();
    Mat r = a * I;
    EXPECT_TRUE(r.equal(a));
}

TEST(Matrix2DTest, MatMulKnownResult)
{
    // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
    Mat a = make2x2(1, 2, 3, 4);
    Mat b = make2x2(5, 6, 7, 8);
    Mat c = a * b;
    EXPECT_NEAR(c(0,0), 19.0, EPS);
    EXPECT_NEAR(c(0,1), 22.0, EPS);
    EXPECT_NEAR(c(1,0), 43.0, EPS);
    EXPECT_NEAR(c(1,1), 50.0, EPS);
}

// ---------------------------------------------------------------------------
// 19. Matrix-by-vector multiplication
// ---------------------------------------------------------------------------
TEST(Matrix2DTest, MatVecMultiply)
{
    // [[1,0],[0,2]] * [3,4]^T = [3,8]^T
    Mat m = make2x2(1, 0, 0, 2);
    Vec v(2);
    v(0) = 3; v(1) = 4;
    v.setCol();

    Vec r = m * v;
    EXPECT_NEAR(r(0), 3.0, EPS);
    EXPECT_NEAR(r(1), 8.0, EPS);
}

// ---------------------------------------------------------------------------
// 20. Vector-by-matrix multiplication
// ---------------------------------------------------------------------------
TEST(Matrix2DTest, VecMatMultiply)
{
    // [1, 2] * [[1,0],[0,2]] = [1, 4]
    Vec v(2);
    v(0) = 1; v(1) = 2;
    v.setRow();

    Mat m = make2x2(1, 0, 0, 2);
    Vec r = v * m;
    EXPECT_NEAR(r(0), 1.0, EPS);
    EXPECT_NEAR(r(1), 4.0, EPS);
}

// ---------------------------------------------------------------------------
// 21. transpose()
// ---------------------------------------------------------------------------
TEST(Matrix2DTest, TransposeSquare)
{
    Mat m = make2x2(1, 2, 3, 4);
    Mat t = m.transpose();
    EXPECT_NEAR(t(0,0), 1.0, EPS);
    EXPECT_NEAR(t(0,1), 3.0, EPS);
    EXPECT_NEAR(t(1,0), 2.0, EPS);
    EXPECT_NEAR(t(1,1), 4.0, EPS);
}

TEST(Matrix2DTest, TransposeNonSquare)
{
    // 2×3 matrix
    Mat m(2, 3);
    m(0,0) = 1; m(0,1) = 2; m(0,2) = 3;
    m(1,0) = 4; m(1,1) = 5; m(1,2) = 6;
    Mat t = m.transpose();
    EXPECT_EQ(t.Ydim(), 3);
    EXPECT_EQ(t.Xdim(), 2);
    EXPECT_NEAR(t(2,1), 6.0, EPS);
}

TEST(Matrix2DTest, DoubleTransposeIsOriginal)
{
    Mat m = make2x2(1, 2, 3, 4);
    EXPECT_TRUE(m.transpose().transpose().equal(m));
}

// ---------------------------------------------------------------------------
// 22. inv() — M * M_inv ≈ I
// ---------------------------------------------------------------------------
TEST(Matrix2DTest, Inverse2x2)
{
    Mat m = make2x2(2, 1, 5, 3);
    Mat inv_m = m.inv();
    Mat product = m * inv_m;
    Mat I(2, 2);
    I.initIdentity();
    EXPECT_TRUE(product.equal(I, NEAR));
}

TEST(Matrix2DTest, Inverse3x3)
{
    Mat m(3, 3);
    m(0,0) = 1; m(0,1) = 2; m(0,2) = 0;
    m(1,0) = 0; m(1,1) = 1; m(1,2) = 0;
    m(2,0) = 0; m(2,1) = 0; m(2,2) = 1;

    Mat inv_m = m.inv();
    Mat product = m * inv_m;
    Mat I = identity3();
    EXPECT_TRUE(product.equal(I, NEAR));
}

// ---------------------------------------------------------------------------
// 23. det()
// ---------------------------------------------------------------------------
TEST(Matrix2DTest, DetIdentityIsOne)
{
    Mat I(3, 3);
    I.initIdentity();
    EXPECT_NEAR(I.det(), 1.0, NEAR);
}

TEST(Matrix2DTest, Det2x2KnownValue)
{
    // det([[3,4],[2,1]]) = 3*1 - 4*2 = -5
    Mat m = make2x2(3, 4, 2, 1);
    EXPECT_NEAR(m.det(), -5.0, NEAR);
}

// ---------------------------------------------------------------------------
// 24. computeMax() / computeMin()
// ---------------------------------------------------------------------------
TEST(Matrix2DTest, ComputeMax)
{
    Mat m = make2x2(1, 7, 3, 5);
    EXPECT_NEAR(m.computeMax(), 7.0, EPS);
}

TEST(Matrix2DTest, ComputeMin)
{
    Mat m = make2x2(4, 7, 2, 9);
    EXPECT_NEAR(m.computeMin(), 2.0, EPS);
}

// ---------------------------------------------------------------------------
// 25. getRow() / getCol() / setRow() / setCol()
// ---------------------------------------------------------------------------
TEST(Matrix2DTest, GetRow)
{
    Mat m = make2x2(1, 2, 3, 4);
    Vec row;
    m.getRow(1, row);
    EXPECT_NEAR(row(0), 3.0, EPS);
    EXPECT_NEAR(row(1), 4.0, EPS);
    EXPECT_TRUE(row.isRow());
}

TEST(Matrix2DTest, GetCol)
{
    Mat m = make2x2(1, 2, 3, 4);
    Vec col;
    m.getCol(0, col);
    EXPECT_NEAR(col(0), 1.0, EPS);
    EXPECT_NEAR(col(1), 3.0, EPS);
    EXPECT_TRUE(col.isCol());
}

TEST(Matrix2DTest, SetRow)
{
    Mat m = make2x2(1, 2, 3, 4);
    Vec row(2);
    row(0) = 9; row(1) = 8;
    row.setRow();
    m.setRow(0, row);
    EXPECT_NEAR(m(0,0), 9.0, EPS);
    EXPECT_NEAR(m(0,1), 8.0, EPS);
}

TEST(Matrix2DTest, SetCol)
{
    Mat m = make2x2(1, 2, 3, 4);
    Vec col(2);
    col(0) = 7; col(1) = 6;
    col.setCol();
    m.setCol(1, col);
    EXPECT_NEAR(m(0,1), 7.0, EPS);
    EXPECT_NEAR(m(1,1), 6.0, EPS);
}

// ---------------------------------------------------------------------------
// 26. fromVector() / toVector()
// ---------------------------------------------------------------------------
TEST(Matrix2DTest, FromVectorColumnAndBack)
{
    Vec v(3);
    v(0) = 1; v(1) = 2; v(2) = 3;
    v.setCol();

    Mat m;
    m.fromVector(v);
    EXPECT_EQ(m.Ydim(), 3);
    EXPECT_EQ(m.Xdim(), 1);
    EXPECT_NEAR(m(1, 0), 2.0, EPS);

    Vec out;
    m.toVector(out);
    EXPECT_EQ(out.vdim, 3);
    EXPECT_NEAR(out(2), 3.0, EPS);
}

TEST(Matrix2DTest, FromVectorRowAndBack)
{
    Vec v(3);
    v(0) = 4; v(1) = 5; v(2) = 6;
    v.setRow();

    Mat m;
    m.fromVector(v);
    EXPECT_EQ(m.Ydim(), 1);
    EXPECT_EQ(m.Xdim(), 3);
    EXPECT_NEAR(m(0, 2), 6.0, EPS);

    Vec out;
    m.toVector(out);
    EXPECT_NEAR(out(0), 4.0, EPS);
}

// ---------------------------------------------------------------------------
// 27. submatrix()
// ---------------------------------------------------------------------------
TEST(Matrix2DTest, SubmatrixExtraction)
{
    Mat m(3, 3);
    m.initIdentity();
    // Extract bottom-right 2×2
    m.submatrix(1, 1, 2, 2);
    EXPECT_EQ(m.Ydim(), 2);
    EXPECT_EQ(m.Xdim(), 2);
    EXPECT_NEAR(m(0,0), 1.0, EPS);
    EXPECT_NEAR(m(0,1), 0.0, EPS);
    EXPECT_NEAR(m(1,1), 1.0, EPS);
}

// ---------------------------------------------------------------------------
// 28. typeCast()
// ---------------------------------------------------------------------------
TEST(Matrix2DTest, TypeCastIntToDouble)
{
    Matrix2D<int> iv(2, 2);
    iv(0,0) = 1; iv(0,1) = 2;
    iv(1,0) = 3; iv(1,1) = 4;

    Mat dv;
    typeCast(iv, dv);
    EXPECT_NEAR(dv(0,0), 1.0, EPS);
    EXPECT_NEAR(dv(1,1), 4.0, EPS);
}

// ---------------------------------------------------------------------------
// 29. svdcmp — SVD decomposition: A = U * diag(W) * Vᵀ
// ---------------------------------------------------------------------------

TEST(Matrix2DSvdTest, Reconstruction_3x3)
{
    // Create a simple 3×3 matrix
    Mat A(3, 3);
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=4; A(1,1)=5; A(1,2)=6;
    A(2,0)=7; A(2,1)=8; A(2,2)=9;

    Matrix2D<RFLOAT> U, V;
    Matrix1D<RFLOAT> W;
    svdcmp(A, U, W, V);

    // Reconstruct A = U * diag(W) * Vᵀ
    Matrix2D<RFLOAT> Vt = V.transpose();
    // Build diag(W) as matrix
    Matrix2D<RFLOAT> DW(3, 3);
    DW.initZeros();
    for (int i = 0; i < 3; i++)
        DW(i, i) = W(i);

    Matrix2D<RFLOAT> Final = U * DW * Vt;

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_NEAR(Final(i,j), (RFLOAT)A(i,j), 1e-6) << "i=" << i << " j=" << j;
}

TEST(Matrix2DSvdTest, SingularValues_NonNegative)
{
    Mat A(3, 3);
    A.initIdentity();
    A(0,0) = 2; A(1,1) = 3; A(2,2) = 5;

    Matrix2D<RFLOAT> U, V;
    Matrix1D<RFLOAT> W;
    svdcmp(A, U, W, V);

    FOR_ALL_ELEMENTS_IN_MATRIX1D(W)
        EXPECT_GE(W(i), 0.0);
}

TEST(Matrix2DSvdTest, U_IsOrthogonal)
{
    Mat A(3, 3);
    A(0,0)=2; A(0,1)=1; A(0,2)=0;
    A(1,0)=1; A(1,1)=3; A(1,2)=1;
    A(2,0)=0; A(2,1)=1; A(2,2)=4;

    Matrix2D<RFLOAT> U, V;
    Matrix1D<RFLOAT> W;
    svdcmp(A, U, W, V);

    // Uᵀ * U should be identity
    Matrix2D<RFLOAT> UtU = U.transpose() * U;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_NEAR(UtU(i,j), (i==j) ? 1.0 : 0.0, 1e-6);
}

// ---------------------------------------------------------------------------
// 30. solve(A, b, result, tol) — SVD-based linear solver
// ---------------------------------------------------------------------------

TEST(Matrix2DSolveTest, Solve2x2_SVD)
{
    // [[2,1],[1,3]] * [1,3] = [5,10]
    Mat A(2, 2);
    A(0,0)=2; A(0,1)=1;
    A(1,0)=1; A(1,1)=3;

    Matrix1D<RFLOAT> b(2), result;
    b(0)=5; b(1)=10;

    solve(A, b, result, 1e-10);

    EXPECT_NEAR(result(0), 1.0, 1e-6);
    EXPECT_NEAR(result(1), 3.0, 1e-6);
}

TEST(Matrix2DSolveTest, Solve3x3_SVD)
{
    // Diagonal system: diag(2,3,4) * [1,2,3] = [2,6,12]
    Mat A(3, 3);
    A.initZeros();
    A(0,0)=2; A(1,1)=3; A(2,2)=4;

    Matrix1D<RFLOAT> b(3), result;
    b(0)=2; b(1)=6; b(2)=12;

    solve(A, b, result, 1e-10);

    EXPECT_NEAR(result(0), 1.0, 1e-6);
    EXPECT_NEAR(result(1), 2.0, 1e-6);
    EXPECT_NEAR(result(2), 3.0, 1e-6);
}

TEST(Matrix2DSolveTest, EmptyMatrix_Throws)
{
    Mat A;
    Matrix1D<RFLOAT> b, result;
    EXPECT_THROW(solve(A, b, result, 1e-10), RelionError);
}

TEST(Matrix2DSolveTest, MismatchedSizes_Throws)
{
    Mat A(2, 2);
    A.initIdentity();
    Matrix1D<RFLOAT> b(3), result;
    EXPECT_THROW(solve(A, b, result, 1e-10), RelionError);
}

// ---------------------------------------------------------------------------
// 31. ludcmp / lubksb — Matrix2D LU wrappers
// ---------------------------------------------------------------------------

TEST(Matrix2DLUTest, Solve2x2)
{
    // [[2,1],[1,3]] * [1,3] = [5,10]
    Mat A(2, 2);
    A(0,0)=2; A(0,1)=1;
    A(1,0)=1; A(1,1)=3;

    Matrix2D<RFLOAT> LU;
    Matrix1D<int> indx;
    RFLOAT d = 0;
    ludcmp(A, LU, indx, d);

    Matrix1D<RFLOAT> b(2);
    b(0)=5; b(1)=10;
    lubksb(LU, indx, b);

    EXPECT_NEAR(b(0), 1.0, 1e-9);
    EXPECT_NEAR(b(1), 3.0, 1e-9);
}

TEST(Matrix2DLUTest, Solve3x3_Diagonal)
{
    Mat A(3, 3);
    A.initZeros();
    A(0,0)=2; A(1,1)=3; A(2,2)=4;

    Matrix2D<RFLOAT> LU;
    Matrix1D<int> indx;
    RFLOAT d = 0;
    ludcmp(A, LU, indx, d);

    Matrix1D<RFLOAT> b(3);
    b(0)=2; b(1)=6; b(2)=12;
    lubksb(LU, indx, b);

    EXPECT_NEAR(b(0), 1.0, 1e-9);
    EXPECT_NEAR(b(1), 2.0, 1e-9);
    EXPECT_NEAR(b(2), 3.0, 1e-9);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
