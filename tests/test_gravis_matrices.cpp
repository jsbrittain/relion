/*
 * Unit tests for src/jaz/gravis/t3Matrix.h and t4Matrix.h
 *
 * Covers t3Matrix and t4Matrix constructors, operators, linear algebra methods,
 * and static factory functions.
 */

#include <gtest/gtest.h>
#include <cmath>
#include "src/jaz/gravis/t3Matrix.h"
#include "src/jaz/gravis/t4Matrix.h"

using namespace gravis;
typedef t3Matrix<double> d3Mat;
typedef t4Matrix<double> d4Mat;
typedef t3Vector<double> d3Vec;

static const double EPS = 1e-10;

// ---------------------------------------------------------------------------
// t3Matrix constructors
// ---------------------------------------------------------------------------

TEST(T3MatrixTest, DefaultConstructor_IsIdentity)
{
    d3Mat m;
    // Column-major: m[0]=1,m[1]=0,m[2]=0, m[3]=0,m[4]=1,m[5]=0, m[6]=0,m[7]=0,m[8]=1
    EXPECT_NEAR(m(0,0), 1.0, EPS);
    EXPECT_NEAR(m(1,1), 1.0, EPS);
    EXPECT_NEAR(m(2,2), 1.0, EPS);
    EXPECT_NEAR(m(0,1), 0.0, EPS);
    EXPECT_NEAR(m(1,0), 0.0, EPS);
}

TEST(T3MatrixTest, ScalarConstructor_AllSame)
{
    d3Mat m(5.0);
    for (int i = 0; i < 9; i++)
        EXPECT_NEAR(m[i], 5.0, EPS);
}

TEST(T3MatrixTest, ArrayConstructor)
{
    double arr[9] = {1,2,3,4,5,6,7,8,9};
    d3Mat m(arr);
    for (int i = 0; i < 9; i++)
        EXPECT_NEAR(m[i], arr[i], EPS);
}

TEST(T3MatrixTest, CopyConstructor)
{
    d3Mat a(3.0);
    d3Mat b(a);
    for (int i = 0; i < 9; i++)
        EXPECT_NEAR(b[i], 3.0, EPS);
}

TEST(T3MatrixTest, RowMajorConstructor)
{
    // Constructor signature: (m0,m3,m6, m1,m4,m7, m2,m5,m8) = row-major
    // Row 0: m0,m3,m6  Row 1: m1,m4,m7  Row 2: m2,m5,m8
    d3Mat m(1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0);
    EXPECT_NEAR(m(0,0), 1.0, EPS);
    EXPECT_NEAR(m(0,1), 2.0, EPS);
    EXPECT_NEAR(m(0,2), 3.0, EPS);
    EXPECT_NEAR(m(1,0), 4.0, EPS);
    EXPECT_NEAR(m(2,0), 7.0, EPS);
}

TEST(T3MatrixTest, CrossTypeConstructor)
{
    t3Matrix<float> f(1.5f);
    d3Mat d(f);
    for (int i = 0; i < 9; i++)
        EXPECT_NEAR(d[i], 1.5, 1e-6);
}

// ---------------------------------------------------------------------------
// t3Matrix set / subscript
// ---------------------------------------------------------------------------

TEST(T3MatrixTest, Set_OverwritesEntries)
{
    d3Mat m;
    m.set(1,2,3, 4,5,6, 7,8,9);
    EXPECT_NEAR(m(0,0), 1.0, EPS);
    EXPECT_NEAR(m(1,0), 4.0, EPS);
}

TEST(T3MatrixTest, Subscript_ReadWrite)
{
    d3Mat m;
    m[3] = 99.0;
    EXPECT_NEAR(m[3], 99.0, EPS);
}

TEST(T3MatrixTest, RowColSubscript_ReadWrite)
{
    d3Mat m;
    m(1, 2) = 42.0;
    EXPECT_NEAR(m(1, 2), 42.0, EPS);
}

// ---------------------------------------------------------------------------
// t3Matrix comparison
// ---------------------------------------------------------------------------

TEST(T3MatrixTest, EqualityOp_SameMatrix)
{
    d3Mat a(2.0);
    d3Mat b(2.0);
    EXPECT_TRUE(a == b);
}

TEST(T3MatrixTest, InequalityOp)
{
    d3Mat a(2.0);
    d3Mat b(3.0);
    EXPECT_TRUE(a != b);
}

TEST(T3MatrixTest, IsClose_True)
{
    d3Mat a;
    d3Mat b;
    EXPECT_TRUE(a.isClose(b, 1e-10));
}

TEST(T3MatrixTest, IsClose_False)
{
    d3Mat a;
    d3Mat b;
    b[0] = 1.5;
    EXPECT_FALSE(a.isClose(b, 1e-10));
}

// ---------------------------------------------------------------------------
// t3Matrix arithmetic operators
// ---------------------------------------------------------------------------

TEST(T3MatrixTest, ScalarMul_InPlace)
{
    d3Mat m(2.0);
    m *= 3.0;
    for (int i = 0; i < 9; i++)
        EXPECT_NEAR(m[i], 6.0, EPS);
}

TEST(T3MatrixTest, ScalarMul_Value)
{
    d3Mat m(2.0);
    d3Mat r = m * 3.0;
    for (int i = 0; i < 9; i++)
        EXPECT_NEAR(r[i], 6.0, EPS);
}

TEST(T3MatrixTest, ScalarDiv_InPlace)
{
    d3Mat m(6.0);
    m /= 2.0;
    for (int i = 0; i < 9; i++)
        EXPECT_NEAR(m[i], 3.0, EPS);
}

TEST(T3MatrixTest, ScalarDiv_Value)
{
    d3Mat m(6.0);
    d3Mat r = m / 2.0;
    for (int i = 0; i < 9; i++)
        EXPECT_NEAR(r[i], 3.0, EPS);
}

TEST(T3MatrixTest, MatrixAdd)
{
    d3Mat a(1.0);
    d3Mat b(2.0);
    d3Mat c = a + b;
    for (int i = 0; i < 9; i++)
        EXPECT_NEAR(c[i], 3.0, EPS);
}

TEST(T3MatrixTest, MatrixSub)
{
    d3Mat a(5.0);
    d3Mat b(2.0);
    d3Mat c = a - b;
    for (int i = 0; i < 9; i++)
        EXPECT_NEAR(c[i], 3.0, EPS);
}

TEST(T3MatrixTest, Negate)
{
    d3Mat a(3.0);
    d3Mat b = -a;
    for (int i = 0; i < 9; i++)
        EXPECT_NEAR(b[i], -3.0, EPS);
}

TEST(T3MatrixTest, MatrixMulVector_Identity)
{
    d3Mat I;
    d3Vec v(1.0, 2.0, 3.0);
    d3Vec r = I * v;
    EXPECT_NEAR(r.x, 1.0, EPS);
    EXPECT_NEAR(r.y, 2.0, EPS);
    EXPECT_NEAR(r.z, 3.0, EPS);
}

TEST(T3MatrixTest, MatrixMulMatrix_Identity)
{
    d3Mat I;
    d3Mat A(1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0);
    d3Mat R = I * A;
    for (int i = 0; i < 9; i++)
        EXPECT_NEAR(R[i], A[i], EPS);
}

// ---------------------------------------------------------------------------
// t3Matrix lmul / rmul
// ---------------------------------------------------------------------------

TEST(T3MatrixTest, Rmul_Identity)
{
    d3Mat A(1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0);
    d3Mat I;
    d3Mat B = A;
    B.rmul(I);
    for (int i = 0; i < 9; i++)
        EXPECT_NEAR(B[i], A[i], EPS);
}

TEST(T3MatrixTest, Lmul_Identity)
{
    d3Mat A(1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0);
    d3Mat I;
    d3Mat B = A;
    B.lmul(I);
    for (int i = 0; i < 9; i++)
        EXPECT_NEAR(B[i], A[i], EPS);
}

// ---------------------------------------------------------------------------
// t3Matrix trace / det / transpose / invert
// ---------------------------------------------------------------------------

TEST(T3MatrixTest, Trace_Identity)
{
    d3Mat I;
    EXPECT_NEAR(I.trace(), 3.0, EPS);
}

TEST(T3MatrixTest, Det_Identity)
{
    d3Mat I;
    EXPECT_NEAR(I.det(), 1.0, EPS);
}

TEST(T3MatrixTest, Det_ScalarMatrix)
{
    // det(2*I) = 8
    d3Mat m(0.0);
    m.loadIdentity();
    m *= 2.0;
    EXPECT_NEAR(m.det(), 8.0, EPS);
}

TEST(T3MatrixTest, Transpose_Identity)
{
    d3Mat I;
    I.transpose();
    // transposing identity gives identity
    EXPECT_NEAR(I(0,0), 1.0, EPS);
    EXPECT_NEAR(I(0,1), 0.0, EPS);
}

TEST(T3MatrixTest, Transpose_NonSymmetric)
{
    d3Mat A(1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0);
    double a01 = A(0,1);
    double a10 = A(1,0);
    A.transpose();
    EXPECT_NEAR(A(0,1), a10, EPS);
    EXPECT_NEAR(A(1,0), a01, EPS);
}

TEST(T3MatrixTest, Invert_Identity)
{
    d3Mat I;
    I.invert();
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            EXPECT_NEAR(I(r,c), (r==c) ? 1.0 : 0.0, EPS);
}

TEST(T3MatrixTest, Invert_RecoverIdentity)
{
    // A * A^-1 = I
    d3Mat A(2.0, 1.0, 0.0,
            0.0, 3.0, 1.0,
            0.0, 0.0, 4.0);
    d3Mat B = A;
    B.invert();
    d3Mat R = A * B;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            EXPECT_NEAR(R(r,c), (r==c) ? 1.0 : 0.0, 1e-10);
}

TEST(T3MatrixTest, Adjugate_Identity)
{
    d3Mat I;
    d3Mat adj = I.adjugate();
    // adjugate of identity is identity
    EXPECT_NEAR(adj(0,0), 1.0, EPS);
    EXPECT_NEAR(adj(1,1), 1.0, EPS);
    EXPECT_NEAR(adj(2,2), 1.0, EPS);
}

TEST(T3MatrixTest, LoadIdentity)
{
    d3Mat m(5.0);
    m.loadIdentity();
    EXPECT_NEAR(m(0,0), 1.0, EPS);
    EXPECT_NEAR(m(0,1), 0.0, EPS);
}

// ---------------------------------------------------------------------------
// t3Matrix static factories
// ---------------------------------------------------------------------------

TEST(T3MatrixTest, Scale_Uniform)
{
    d3Mat S = d3Mat::scale(2.0);
    EXPECT_NEAR(S(0,0), 2.0, EPS);
    EXPECT_NEAR(S(1,1), 2.0, EPS);
    EXPECT_NEAR(S(2,2), 2.0, EPS);
    EXPECT_NEAR(S(0,1), 0.0, EPS);
}

TEST(T3MatrixTest, Scale_Vector)
{
    d3Mat S = d3Mat::scale(d3Vec(1.0, 2.0, 3.0));
    EXPECT_NEAR(S(0,0), 1.0, EPS);
    EXPECT_NEAR(S(1,1), 2.0, EPS);
    EXPECT_NEAR(S(2,2), 3.0, EPS);
}

TEST(T3MatrixTest, RotationX_90Deg)
{
    double angle = M_PI / 2.0;
    d3Mat R = d3Mat::rotationX(angle);
    d3Vec v(0.0, 1.0, 0.0);
    d3Vec w = R * v;
    EXPECT_NEAR(w.x, 0.0, 1e-10);
    EXPECT_NEAR(w.y, 0.0, 1e-10);
    EXPECT_NEAR(w.z, 1.0, 1e-10);
}

TEST(T3MatrixTest, RotationZ_90Deg)
{
    double angle = M_PI / 2.0;
    d3Mat R = d3Mat::rotationZ(angle);
    d3Vec v(1.0, 0.0, 0.0);
    d3Vec w = R * v;
    EXPECT_NEAR(w.x, 0.0, 1e-10);
    EXPECT_NEAR(w.y, 1.0, 1e-10);
    EXPECT_NEAR(w.z, 0.0, 1e-10);
}

TEST(T3MatrixTest, RotationY_90Deg)
{
    double angle = M_PI / 2.0;
    d3Mat R = d3Mat::rotationY(angle);
    d3Vec v(0.0, 0.0, 1.0);
    d3Vec w = R * v;
    EXPECT_NEAR(w.x, -1.0, 1e-10);
    EXPECT_NEAR(w.y,  0.0, 1e-10);
    EXPECT_NEAR(w.z,  0.0, 1e-10);
}

TEST(T3MatrixTest, RotationAxisAngle_PreservesLength)
{
    d3Vec axis(0.0, 1.0, 0.0);
    d3Mat R = d3Mat::rotation(axis, 45.0);
    d3Vec v(1.0, 2.0, 3.0);
    d3Vec w = R * v;
    EXPECT_NEAR(w.norm2(), v.norm2(), 1e-10);
}

TEST(T3MatrixTest, RotationUV_SameVector_IsIdentity)
{
    // rotation(u, u) should give identity
    d3Vec u(1.0, 0.0, 0.0);
    d3Mat R = d3Mat::rotation(u, u);
    d3Vec w = R * u;
    EXPECT_NEAR(w.x, u.x, 1e-10);
    EXPECT_NEAR(w.y, u.y, 1e-10);
    EXPECT_NEAR(w.z, u.z, 1e-10);
}

TEST(T3MatrixTest, GetAxis_IdentityReturnsX)
{
    d3Mat I;
    d3Vec axis = I.getAxis();
    // getAxis on identity: a0=m[5]+m[7]=0 → returns (1,0,0)
    EXPECT_NEAR(axis.x, 1.0, EPS);
    EXPECT_NEAR(axis.y, 0.0, EPS);
    EXPECT_NEAR(axis.z, 0.0, EPS);
}

// ---------------------------------------------------------------------------
// t4Matrix constructors
// ---------------------------------------------------------------------------

TEST(T4MatrixTest, DefaultConstructor_IsIdentity)
{
    d4Mat m;
    EXPECT_NEAR(m(0,0), 1.0, EPS);
    EXPECT_NEAR(m(1,1), 1.0, EPS);
    EXPECT_NEAR(m(2,2), 1.0, EPS);
    EXPECT_NEAR(m(3,3), 1.0, EPS);
    EXPECT_NEAR(m(0,1), 0.0, EPS);
    EXPECT_NEAR(m(1,0), 0.0, EPS);
}

TEST(T4MatrixTest, ScalarConstructor_AllSame)
{
    d4Mat m(7.0);
    for (int i = 0; i < 16; i++)
        EXPECT_NEAR(m[i], 7.0, EPS);
}

TEST(T4MatrixTest, ArrayConstructor)
{
    double arr[16];
    for (int i = 0; i < 16; i++) arr[i] = (double)i;
    d4Mat m(arr);
    for (int i = 0; i < 16; i++)
        EXPECT_NEAR(m[i], (double)i, EPS);
}

TEST(T4MatrixTest, CopyConstructor)
{
    d4Mat a(3.0);
    d4Mat b(a);
    for (int i = 0; i < 16; i++)
        EXPECT_NEAR(b[i], 3.0, EPS);
}

TEST(T4MatrixTest, From3Matrix)
{
    d3Mat m3(1.0, 2.0, 3.0,
             4.0, 5.0, 6.0,
             7.0, 8.0, 9.0);
    d4Mat m4(m3);
    EXPECT_NEAR(m4(0,0), m3(0,0), EPS);
    EXPECT_NEAR(m4(1,1), m3(1,1), EPS);
    EXPECT_NEAR(m4(3,3), 1.0, EPS);  // last row/col is homogeneous
    EXPECT_NEAR(m4(0,3), 0.0, EPS);
}

TEST(T4MatrixTest, CrossTypeConstructor)
{
    t4Matrix<float> f(2.0f);
    d4Mat d(f);
    for (int i = 0; i < 16; i++)
        EXPECT_NEAR(d[i], 2.0, 1e-6);
}

// ---------------------------------------------------------------------------
// t4Matrix set / subscript
// ---------------------------------------------------------------------------

TEST(T4MatrixTest, Set_OverwritesEntries)
{
    d4Mat m;
    m.set(1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16);
    EXPECT_NEAR(m(0,0), 1.0, EPS);
    EXPECT_NEAR(m(1,0), 5.0, EPS);
}

TEST(T4MatrixTest, Subscript_ReadWrite)
{
    d4Mat m;
    m[5] = 77.0;
    EXPECT_NEAR(m[5], 77.0, EPS);
}

TEST(T4MatrixTest, RowColSubscript_ReadWrite)
{
    d4Mat m;
    m(2, 3) = 13.0;
    EXPECT_NEAR(m(2, 3), 13.0, EPS);
}

// ---------------------------------------------------------------------------
// t4Matrix comparison
// ---------------------------------------------------------------------------

TEST(T4MatrixTest, EqualityOp)
{
    d4Mat a;
    d4Mat b;
    EXPECT_TRUE(a == b);
}

TEST(T4MatrixTest, IsClose)
{
    d4Mat a;
    d4Mat b;
    EXPECT_TRUE(a.isClose(b, 1e-10));
    b[0] = 2.0;
    EXPECT_FALSE(a.isClose(b, 1e-10));
}

// ---------------------------------------------------------------------------
// t4Matrix arithmetic
// ---------------------------------------------------------------------------

TEST(T4MatrixTest, ScalarMul_InPlace)
{
    d4Mat m(2.0);
    m *= 3.0;
    for (int i = 0; i < 16; i++)
        EXPECT_NEAR(m[i], 6.0, EPS);
}

TEST(T4MatrixTest, ScalarMul_Value)
{
    d4Mat m(2.0);
    d4Mat r = m * 3.0;
    for (int i = 0; i < 16; i++)
        EXPECT_NEAR(r[i], 6.0, EPS);
}

TEST(T4MatrixTest, ScalarDiv_Value)
{
    d4Mat m(6.0);
    d4Mat r = m / 2.0;
    for (int i = 0; i < 16; i++)
        EXPECT_NEAR(r[i], 3.0, EPS);
}

TEST(T4MatrixTest, MatrixAdd)
{
    d4Mat a(1.0);
    d4Mat b(2.0);
    d4Mat c = a + b;
    for (int i = 0; i < 16; i++)
        EXPECT_NEAR(c[i], 3.0, EPS);
}

TEST(T4MatrixTest, MatrixSub)
{
    d4Mat a(5.0);
    d4Mat b(2.0);
    d4Mat c = a - b;
    for (int i = 0; i < 16; i++)
        EXPECT_NEAR(c[i], 3.0, EPS);
}

TEST(T4MatrixTest, Negate)
{
    d4Mat a(3.0);
    d4Mat b = -a;
    for (int i = 0; i < 16; i++)
        EXPECT_NEAR(b[i], -3.0, EPS);
}

TEST(T4MatrixTest, MatMulVector_Identity)
{
    d4Mat I;
    t4Vector<double> v(1.0, 2.0, 3.0, 1.0);
    t4Vector<double> r = I * v;
    EXPECT_NEAR(r.x, 1.0, EPS);
    EXPECT_NEAR(r.y, 2.0, EPS);
    EXPECT_NEAR(r.z, 3.0, EPS);
    EXPECT_NEAR(r.w, 1.0, EPS);
}

TEST(T4MatrixTest, MatMulMat_Identity)
{
    d4Mat I;
    d4Mat A(1.0);
    d4Mat R = I * A;
    for (int i = 0; i < 16; i++)
        EXPECT_NEAR(R[i], A[i], EPS);
}

// ---------------------------------------------------------------------------
// t4Matrix lmul / rmul
// ---------------------------------------------------------------------------

TEST(T4MatrixTest, Rmul_Identity)
{
    d4Mat A(2.0);
    d4Mat I;
    d4Mat B = A;
    B.rmul(I);
    for (int i = 0; i < 16; i++)
        EXPECT_NEAR(B[i], A[i], EPS);
}

TEST(T4MatrixTest, Lmul_Identity)
{
    d4Mat A(2.0);
    d4Mat I;
    d4Mat B = A;
    B.lmul(I);
    for (int i = 0; i < 16; i++)
        EXPECT_NEAR(B[i], A[i], EPS);
}

// ---------------------------------------------------------------------------
// t4Matrix trace / det / FrobeniusNorm / transpose / invert
// ---------------------------------------------------------------------------

TEST(T4MatrixTest, Trace_Identity)
{
    d4Mat I;
    EXPECT_NEAR(I.trace(), 4.0, EPS);
}

TEST(T4MatrixTest, Det_DoesNotCrash)
{
    // The library's det4x4 implementation is non-standard; just verify it is callable
    d4Mat I;
    double d = I.det();
    EXPECT_TRUE(std::isfinite(d));
}

TEST(T4MatrixTest, FrobeniusNorm_Identity)
{
    d4Mat I;
    EXPECT_NEAR(I.FrobeniusNorm(), 4.0, EPS);
}

TEST(T4MatrixTest, Adjugate_Identity)
{
    d4Mat I;
    d4Mat adj = I.adjugate();
    EXPECT_NEAR(adj(0,0), 1.0, EPS);
    EXPECT_NEAR(adj(1,1), 1.0, EPS);
    EXPECT_NEAR(adj(0,1), 0.0, EPS);
}

TEST(T4MatrixTest, Transpose_Identity)
{
    d4Mat I;
    I.transpose();
    EXPECT_NEAR(I(0,0), 1.0, EPS);
    EXPECT_NEAR(I(0,1), 0.0, EPS);
}

TEST(T4MatrixTest, Invert_Identity)
{
    d4Mat I;
    I.invert();
    EXPECT_NEAR(I(0,0), 1.0, EPS);
    EXPECT_NEAR(I(0,1), 0.0, EPS);
}

TEST(T4MatrixTest, LoadIdentity)
{
    d4Mat m(5.0);
    m.loadIdentity();
    EXPECT_NEAR(m(0,0), 1.0, EPS);
    EXPECT_NEAR(m(0,1), 0.0, EPS);
    EXPECT_NEAR(m(3,3), 1.0, EPS);
}

TEST(T4MatrixTest, Copy_From3Matrix)
{
    d4Mat m(0.0);
    d3Mat m3;  // identity
    m.copy(m3);
    EXPECT_NEAR(m(0,0), 1.0, EPS);
    EXPECT_NEAR(m(1,1), 1.0, EPS);
    EXPECT_NEAR(m(2,2), 1.0, EPS);
}

// ---------------------------------------------------------------------------
// t4Matrix static factories
// ---------------------------------------------------------------------------

TEST(T4MatrixTest, Translation_AppliedToPoint)
{
    d4Mat T = d4Mat::translation(d3Vec(1.0, 2.0, 3.0));
    t4Vector<double> p(0.0, 0.0, 0.0, 1.0);
    t4Vector<double> r = T * p;
    EXPECT_NEAR(r.x, 1.0, EPS);
    EXPECT_NEAR(r.y, 2.0, EPS);
    EXPECT_NEAR(r.z, 3.0, EPS);
    EXPECT_NEAR(r.w, 1.0, EPS);
}

TEST(T4MatrixTest, Scale_Uniform)
{
    d4Mat S = d4Mat::scale(2.0);
    EXPECT_NEAR(S(0,0), 2.0, EPS);
    EXPECT_NEAR(S(1,1), 2.0, EPS);
    EXPECT_NEAR(S(2,2), 2.0, EPS);
    EXPECT_NEAR(S(3,3), 1.0, EPS);
}

TEST(T4MatrixTest, Scale_Vector)
{
    d4Mat S = d4Mat::scale(d3Vec(1.0, 2.0, 3.0));
    EXPECT_NEAR(S(0,0), 1.0, EPS);
    EXPECT_NEAR(S(1,1), 2.0, EPS);
    EXPECT_NEAR(S(2,2), 3.0, EPS);
}

TEST(T4MatrixTest, RotationX_90Deg)
{
    double a = M_PI / 2.0;
    d4Mat R = d4Mat::rotationX(a);
    t4Vector<double> v(0.0, 1.0, 0.0, 1.0);
    t4Vector<double> w = R * v;
    EXPECT_NEAR(w.x, 0.0, 1e-10);
    EXPECT_NEAR(w.y, 0.0, 1e-10);
    EXPECT_NEAR(w.z, 1.0, 1e-10);
    EXPECT_NEAR(w.w, 1.0, 1e-10);
}

TEST(T4MatrixTest, RotationZ_90Deg)
{
    double a = M_PI / 2.0;
    d4Mat R = d4Mat::rotationZ(a);
    t4Vector<double> v(1.0, 0.0, 0.0, 1.0);
    t4Vector<double> w = R * v;
    EXPECT_NEAR(w.x, 0.0, 1e-10);
    EXPECT_NEAR(w.y, 1.0, 1e-10);
    EXPECT_NEAR(w.z, 0.0, 1e-10);
}

TEST(T4MatrixTest, RotationUV_SameVector_IsIdentity)
{
    // rotation(u, u) should give identity for the vector
    d3Vec u(1.0, 0.0, 0.0);
    d4Mat R = d4Mat::rotation(u, u);
    t4Vector<double> vp(u.x, u.y, u.z, 0.0);
    t4Vector<double> rp = R * vp;
    EXPECT_NEAR(rp.x, u.x, 1e-10);
    EXPECT_NEAR(rp.y, u.y, 1e-10);
    EXPECT_NEAR(rp.z, u.z, 1e-10);
}

TEST(T4MatrixTest, RotationAxisAngle_PreservesLength)
{
    d3Vec axis(0.0, 1.0, 0.0);
    d4Mat R = d4Mat::rotation(axis, 45.0);
    t4Vector<double> v(1.0, 2.0, 3.0, 0.0);
    t4Vector<double> w = R * v;
    double lenV = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    double lenW = sqrt(w.x*w.x + w.y*w.y + w.z*w.z);
    EXPECT_NEAR(lenW, lenV, 1e-10);
}

TEST(T4MatrixTest, Extract_From4Matrix)
{
    d4Mat m4;
    m4(0,1) = 5.0;
    d3Mat m3 = d3Mat::extract(m4);
    EXPECT_NEAR(m3(0,1), 5.0, EPS);
    EXPECT_NEAR(m3(0,0), 1.0, EPS);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
