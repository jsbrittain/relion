/*
 * GoogleTest unit tests for Matrix1D<T>.
 *
 * Build and run:
 *   cmake -DBUILD_TESTS=ON ...
 *   make test_matrix1d
 *   ./build/bin/test_matrix1d
 *
 * No MPI required; pure CPU unit tests.
 *
 * What is tested:
 *   1.  Default constructor – empty (vdim=0).
 *   2.  Dimension constructor – correct vdim.
 *   3.  Copy constructor – deep copy, independent mutation.
 *   4.  Assignment operator – deep copy, independent mutation.
 *   5.  clear() – returns to empty state.
 *   6.  resize() – expands with zeros, truncates.
 *   7.  initZeros() – all elements zero.
 *   8.  initConstant() – all elements equal supplied value.
 *   9.  operator() – element read/write round-trip.
 *  10.  VEC_ELEM macro – element access.
 *  11.  XX / YY / ZZ macros – R3 component access.
 *  12.  isRow / isCol / setRow / setCol – orientation flag.
 *  13.  Scalar arithmetic: *, /, +, - (both v*k and k*v forms).
 *  14.  In-place scalar: *=, /=, +=, -=.
 *  15.  Vector arithmetic: +, -, *, / (element-wise).
 *  16.  In-place vector: +=, -=, *=, /=.
 *  17.  Unary minus.
 *  18.  sum() / sum2() / module().
 *  19.  selfNormalize() – unit vector.
 *  20.  dotProduct() – orthogonal and parallel vectors.
 *  21.  vectorProduct() – cross product, right-hand rule.
 *  22.  vectorR2 / vectorR3 factory helpers.
 *  23.  sameShape() – matching and mismatched sizes.
 *  24.  typeCast() – int → double conversion.
 *  25.  FOR_ALL_ELEMENTS_IN_MATRIX1D / VEC_XSIZE macro.
 *  26.  selfReverse() – reverses element order.
 *  27.  maxIndex / minIndex – correct positions.
 */

#include <gtest/gtest.h>
#include <cmath>
#include "src/matrix1d.h"
#include "src/error.h"

using Vec = Matrix1D<double>;

static constexpr double EPS = 1e-9;

// ---------------------------------------------------------------------------
// 1. Default constructor
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, DefaultConstructorIsEmpty)
{
    Vec v;
    EXPECT_EQ(v.vdim, 0);
    EXPECT_EQ(v.vdata, nullptr);
}

// ---------------------------------------------------------------------------
// 2. Dimension constructor
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, DimensionConstructorSetsSize)
{
    Vec v(5);
    EXPECT_EQ(v.vdim, 5);
    EXPECT_NE(v.vdata, nullptr);
}

// ---------------------------------------------------------------------------
// 3. Copy constructor
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, CopyConstructorIsDeepCopy)
{
    Vec a(3);
    a(0) = 1.0; a(1) = 2.0; a(2) = 3.0;

    Vec b(a);
    EXPECT_NEAR(b(0), 1.0, EPS);
    EXPECT_NEAR(b(1), 2.0, EPS);
    EXPECT_NEAR(b(2), 3.0, EPS);

    // Mutating b does not affect a
    b(0) = 99.0;
    EXPECT_NEAR(a(0), 1.0, EPS);
}

// ---------------------------------------------------------------------------
// 4. Assignment operator
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, AssignmentIsDeepCopy)
{
    Vec a(2);
    a(0) = 7.0; a(1) = 8.0;

    Vec b;
    b = a;
    EXPECT_NEAR(b(0), 7.0, EPS);
    EXPECT_NEAR(b(1), 8.0, EPS);

    b(1) = 0.0;
    EXPECT_NEAR(a(1), 8.0, EPS);
}

// ---------------------------------------------------------------------------
// 5. clear()
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, ClearReturnsToEmpty)
{
    Vec v(4);
    v.clear();
    EXPECT_EQ(v.vdim, 0);
    EXPECT_EQ(v.vdata, nullptr);
}

// ---------------------------------------------------------------------------
// 6. resize()
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, ResizeExpandsFillsZero)
{
    Vec v(2);
    v(0) = 1.0; v(1) = 2.0;
    v.resize(4);
    EXPECT_EQ(v.vdim, 4);
    EXPECT_NEAR(v(0), 1.0, EPS);
    EXPECT_NEAR(v(1), 2.0, EPS);
    EXPECT_NEAR(v(2), 0.0, EPS);
    EXPECT_NEAR(v(3), 0.0, EPS);
}

TEST(Matrix1DTest, ResizeTruncates)
{
    Vec v(4);
    v(0) = 1.0; v(1) = 2.0; v(2) = 3.0; v(3) = 4.0;
    v.resize(2);
    EXPECT_EQ(v.vdim, 2);
    EXPECT_NEAR(v(0), 1.0, EPS);
    EXPECT_NEAR(v(1), 2.0, EPS);
}

// ---------------------------------------------------------------------------
// 7. initZeros()
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, InitZerosAllZero)
{
    Vec v(5);
    for (int i = 0; i < 5; i++) v(i) = 1.0;
    v.initZeros();
    for (int i = 0; i < 5; i++)
        EXPECT_NEAR(v(i), 0.0, EPS);
}

TEST(Matrix1DTest, InitZerosWithSize)
{
    Vec v;
    v.initZeros(3);
    EXPECT_EQ(v.vdim, 3);
    for (int i = 0; i < 3; i++)
        EXPECT_NEAR(v(i), 0.0, EPS);
}

// ---------------------------------------------------------------------------
// 8. initConstant()
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, InitConstant)
{
    Vec v(4);
    v.initConstant(3.14);
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(v(i), 3.14, EPS);
}

// ---------------------------------------------------------------------------
// 9. operator() element access
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, ElementAccessRoundTrip)
{
    Vec v(3);
    v(0) = 10.0; v(1) = 20.0; v(2) = 30.0;
    EXPECT_NEAR(v(0), 10.0, EPS);
    EXPECT_NEAR(v(1), 20.0, EPS);
    EXPECT_NEAR(v(2), 30.0, EPS);
}

// ---------------------------------------------------------------------------
// 10. VEC_ELEM macro
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, VecElemMacro)
{
    Vec v(3);
    VEC_ELEM(v, 0) = 5.0;
    VEC_ELEM(v, 1) = 6.0;
    EXPECT_NEAR(VEC_ELEM(v, 0), 5.0, EPS);
    EXPECT_NEAR(VEC_ELEM(v, 1), 6.0, EPS);
}

// ---------------------------------------------------------------------------
// 11. XX / YY / ZZ macros
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, XxYyZzMacros)
{
    Vec v(3);
    XX(v) = 1.0; YY(v) = 2.0; ZZ(v) = 3.0;
    EXPECT_NEAR(XX(v), 1.0, EPS);
    EXPECT_NEAR(YY(v), 2.0, EPS);
    EXPECT_NEAR(ZZ(v), 3.0, EPS);
}

// ---------------------------------------------------------------------------
// 12. isRow / isCol / setRow / setCol
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, ColumnByDefault)
{
    Vec v(3);
    EXPECT_FALSE(v.isRow());
    EXPECT_TRUE(v.isCol());
}

TEST(Matrix1DTest, SetRowAndSetCol)
{
    Vec v(3);
    v.setRow();
    EXPECT_TRUE(v.isRow());
    EXPECT_FALSE(v.isCol());

    v.setCol();
    EXPECT_FALSE(v.isRow());
    EXPECT_TRUE(v.isCol());
}

// ---------------------------------------------------------------------------
// 13. Scalar arithmetic
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, ScalarMultiply)
{
    Vec v(3);
    v(0) = 1.0; v(1) = 2.0; v(2) = 3.0;
    Vec r = v * 2.0;
    EXPECT_NEAR(r(0), 2.0, EPS);
    EXPECT_NEAR(r(1), 4.0, EPS);
    EXPECT_NEAR(r(2), 6.0, EPS);
}

TEST(Matrix1DTest, ScalarMultiplyCommutative)
{
    Vec v(2);
    v(0) = 3.0; v(1) = 4.0;
    Vec r = 2.0 * v;
    EXPECT_NEAR(r(0), 6.0, EPS);
    EXPECT_NEAR(r(1), 8.0, EPS);
}

TEST(Matrix1DTest, ScalarDivide)
{
    Vec v(2);
    v(0) = 4.0; v(1) = 6.0;
    Vec r = v / 2.0;
    EXPECT_NEAR(r(0), 2.0, EPS);
    EXPECT_NEAR(r(1), 3.0, EPS);
}

TEST(Matrix1DTest, ScalarAdd)
{
    Vec v(2);
    v(0) = 1.0; v(1) = 2.0;
    Vec r = v + 10.0;
    EXPECT_NEAR(r(0), 11.0, EPS);
    EXPECT_NEAR(r(1), 12.0, EPS);
}

TEST(Matrix1DTest, ScalarSubtract)
{
    Vec v(2);
    v(0) = 5.0; v(1) = 8.0;
    Vec r = v - 3.0;
    EXPECT_NEAR(r(0), 2.0, EPS);
    EXPECT_NEAR(r(1), 5.0, EPS);
}

// ---------------------------------------------------------------------------
// 14. In-place scalar operators
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, InPlaceScalarOps)
{
    Vec v(2);
    v(0) = 4.0; v(1) = 6.0;

    v *= 2.0;
    EXPECT_NEAR(v(0), 8.0, EPS);

    v /= 4.0;
    EXPECT_NEAR(v(0), 2.0, EPS);

    v += 1.0;
    EXPECT_NEAR(v(0), 3.0, EPS);

    v -= 1.0;
    EXPECT_NEAR(v(0), 2.0, EPS);
}

// ---------------------------------------------------------------------------
// 15. Vector arithmetic (element-wise)
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, VectorAdd)
{
    Vec a(3), b(3);
    a(0) = 1; a(1) = 2; a(2) = 3;
    b(0) = 4; b(1) = 5; b(2) = 6;
    Vec c = a + b;
    EXPECT_NEAR(c(0), 5.0, EPS);
    EXPECT_NEAR(c(1), 7.0, EPS);
    EXPECT_NEAR(c(2), 9.0, EPS);
}

TEST(Matrix1DTest, VectorSubtract)
{
    Vec a(2), b(2);
    a(0) = 10; a(1) = 5;
    b(0) = 3;  b(1) = 2;
    Vec c = a - b;
    EXPECT_NEAR(c(0), 7.0, EPS);
    EXPECT_NEAR(c(1), 3.0, EPS);
}

TEST(Matrix1DTest, VectorElementWiseMultiply)
{
    Vec a(2), b(2);
    a(0) = 2; a(1) = 3;
    b(0) = 4; b(1) = 5;
    Vec c = a * b;
    EXPECT_NEAR(c(0), 8.0, EPS);
    EXPECT_NEAR(c(1), 15.0, EPS);
}

TEST(Matrix1DTest, VectorElementWiseDivide)
{
    Vec a(2), b(2);
    a(0) = 6; a(1) = 8;
    b(0) = 2; b(1) = 4;
    Vec c = a / b;
    EXPECT_NEAR(c(0), 3.0, EPS);
    EXPECT_NEAR(c(1), 2.0, EPS);
}

// ---------------------------------------------------------------------------
// 16. In-place vector operators
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, InPlaceVectorOps)
{
    Vec a(2), b(2);
    a(0) = 1; a(1) = 2;
    b(0) = 3; b(1) = 4;

    a += b;
    EXPECT_NEAR(a(0), 4.0, EPS);
    EXPECT_NEAR(a(1), 6.0, EPS);

    a -= b;
    EXPECT_NEAR(a(0), 1.0, EPS);
    EXPECT_NEAR(a(1), 2.0, EPS);

    a *= b;
    EXPECT_NEAR(a(0), 3.0, EPS);
    EXPECT_NEAR(a(1), 8.0, EPS);

    a /= b;
    EXPECT_NEAR(a(0), 1.0, EPS);
    EXPECT_NEAR(a(1), 2.0, EPS);
}

// ---------------------------------------------------------------------------
// 17. Unary minus
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, UnaryMinus)
{
    Vec v(3);
    v(0) = 1.0; v(1) = -2.0; v(2) = 3.0;
    Vec r = -v;
    EXPECT_NEAR(r(0), -1.0, EPS);
    EXPECT_NEAR(r(1),  2.0, EPS);
    EXPECT_NEAR(r(2), -3.0, EPS);
}

// ---------------------------------------------------------------------------
// 18. sum() / sum2() / module()
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, Sum)
{
    Vec v(4);
    v(0) = 1; v(1) = 2; v(2) = 3; v(3) = 4;
    EXPECT_NEAR(v.sum(), 10.0, EPS);
}

TEST(Matrix1DTest, SumAverage)
{
    Vec v(4);
    v(0) = 1; v(1) = 2; v(2) = 3; v(3) = 4;
    EXPECT_NEAR(v.sum(true), 2.5, EPS);  // average
}

TEST(Matrix1DTest, Sum2)
{
    Vec v(3);
    v(0) = 1; v(1) = 2; v(2) = 2;
    EXPECT_NEAR(v.sum2(), 9.0, EPS);
}

TEST(Matrix1DTest, Module)
{
    Vec v(3);
    v(0) = 0; v(1) = 3; v(2) = 4;
    EXPECT_NEAR(v.module(), 5.0, EPS);
}

// ---------------------------------------------------------------------------
// 19. selfNormalize()
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, SelfNormalizeProducesUnitVector)
{
    Vec v(3);
    v(0) = 3; v(1) = 0; v(2) = 4;
    v.selfNormalize();
    EXPECT_NEAR(v.module(), 1.0, 1e-6);
    EXPECT_NEAR(v(0), 0.6, 1e-6);
    EXPECT_NEAR(v(2), 0.8, 1e-6);
}

TEST(Matrix1DTest, SelfNormalizeZeroVectorBecomesZero)
{
    Vec v(3);
    v.initZeros();
    v.selfNormalize();
    EXPECT_NEAR(v(0), 0.0, EPS);
    EXPECT_NEAR(v(1), 0.0, EPS);
    EXPECT_NEAR(v(2), 0.0, EPS);
}

// ---------------------------------------------------------------------------
// 20. dotProduct()
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, DotProductOrthogonal)
{
    Vec a(3), b(3);
    a(0) = 1; a(1) = 0; a(2) = 0;
    b(0) = 0; b(1) = 1; b(2) = 0;
    EXPECT_NEAR(dotProduct(a, b), 0.0, EPS);
}

TEST(Matrix1DTest, DotProductParallel)
{
    Vec a(3), b(3);
    a(0) = 2; a(1) = 3; a(2) = 4;
    b(0) = 2; b(1) = 3; b(2) = 4;
    EXPECT_NEAR(dotProduct(a, b), 4.0 + 9.0 + 16.0, EPS);
}

// ---------------------------------------------------------------------------
// 21. vectorProduct()
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, CrossProductXcrossYisZ)
{
    Vec x = vectorR3(1.0, 0.0, 0.0);
    Vec y = vectorR3(0.0, 1.0, 0.0);
    Vec z = vectorProduct(x, y);
    EXPECT_NEAR(z(0), 0.0, EPS);
    EXPECT_NEAR(z(1), 0.0, EPS);
    EXPECT_NEAR(z(2), 1.0, EPS);
}

TEST(Matrix1DTest, CrossProductAntiCommutative)
{
    Vec a = vectorR3(1.0, 2.0, 3.0);
    Vec b = vectorR3(4.0, 5.0, 6.0);
    Vec axb = vectorProduct(a, b);
    Vec bxa = vectorProduct(b, a);
    for (int i = 0; i < 3; i++)
        EXPECT_NEAR(axb(i), -bxa(i), EPS);
}

// ---------------------------------------------------------------------------
// 22. vectorR2 / vectorR3 factory functions
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, VectorR2)
{
    Vec v = vectorR2(3.0, 4.0);
    EXPECT_EQ(v.vdim, 2);
    EXPECT_NEAR(XX(v), 3.0, EPS);
    EXPECT_NEAR(YY(v), 4.0, EPS);
}

TEST(Matrix1DTest, VectorR3)
{
    Vec v = vectorR3(1.0, 2.0, 3.0);
    EXPECT_EQ(v.vdim, 3);
    EXPECT_NEAR(XX(v), 1.0, EPS);
    EXPECT_NEAR(YY(v), 2.0, EPS);
    EXPECT_NEAR(ZZ(v), 3.0, EPS);
}

// ---------------------------------------------------------------------------
// 23. sameShape()
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, SameShapeMatchingSize)
{
    Vec a(4), b(4);
    EXPECT_TRUE(a.sameShape(b));
}

TEST(Matrix1DTest, SameShapeMismatchedSize)
{
    Vec a(3), b(5);
    EXPECT_FALSE(a.sameShape(b));
}

// ---------------------------------------------------------------------------
// 24. typeCast()
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, TypeCastIntToDouble)
{
    Matrix1D<int> iv(3);
    iv(0) = 1; iv(1) = 2; iv(2) = 3;

    Vec dv;
    typeCast(iv, dv);
    EXPECT_EQ(dv.vdim, 3);
    EXPECT_NEAR(dv(0), 1.0, EPS);
    EXPECT_NEAR(dv(1), 2.0, EPS);
    EXPECT_NEAR(dv(2), 3.0, EPS);
}

// ---------------------------------------------------------------------------
// 25. FOR_ALL_ELEMENTS_IN_MATRIX1D / VEC_XSIZE
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, ForAllElementsMacro)
{
    Vec v(5);
    v.initZeros();
    int count = 0;
    FOR_ALL_ELEMENTS_IN_MATRIX1D(v)
    {
        v(i) = (double)i;
        count++;
    }
    EXPECT_EQ(count, 5);
    EXPECT_NEAR(v(3), 3.0, EPS);
}

TEST(Matrix1DTest, VecXsizeMacro)
{
    Vec v(7);
    EXPECT_EQ(VEC_XSIZE(v), 7);
}

// ---------------------------------------------------------------------------
// 26. selfReverse()
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, SelfReverse)
{
    Vec v(4);
    v(0) = 1; v(1) = 2; v(2) = 3; v(3) = 4;
    v.selfReverse();
    EXPECT_NEAR(v(0), 4.0, EPS);
    EXPECT_NEAR(v(1), 3.0, EPS);
    EXPECT_NEAR(v(2), 2.0, EPS);
    EXPECT_NEAR(v(3), 1.0, EPS);
}

// ---------------------------------------------------------------------------
// 27. maxIndex / minIndex
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, MaxIndex)
{
    Vec v(4);
    v(0) = 1; v(1) = 5; v(2) = 3; v(3) = 2;
    int idx;
    v.maxIndex(idx);
    EXPECT_EQ(idx, 1);
}

TEST(Matrix1DTest, MinIndex)
{
    Vec v(4);
    v(0) = 3; v(1) = 1; v(2) = 4; v(3) = 2;
    int idx;
    v.minIndex(idx);
    EXPECT_EQ(idx, 1);
}

TEST(Matrix1DTest, MaxIndexEmptyReturnsMinusOne)
{
    Vec v;
    int idx;
    v.maxIndex(idx);
    EXPECT_EQ(idx, -1);
}

// ---------------------------------------------------------------------------
// 28. minIndex when vdim==0 → returns -1
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, MinIndexEmptyReturnsMinusOne)
{
    Vec v;
    int idx;
    v.minIndex(idx);
    EXPECT_EQ(idx, -1);
}

// ---------------------------------------------------------------------------
// 29. resize(dim <= 0) → calls clear(), vdim becomes 0
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, ResizeZero_ClearsVector)
{
    Vec v(5);
    v(0) = 1.0;
    v.resize(0);
    EXPECT_EQ(v.vdim, 0);
    EXPECT_EQ(v.vdata, nullptr);
}

// ---------------------------------------------------------------------------
// 30. selfROUND() — rounds each element to nearest integer
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, SelfROUND_RoundsElements)
{
    Vec v(4);
    v(0) = 1.4; v(1) = 1.6; v(2) = -1.4; v(3) = -1.6;
    v.selfROUND();
    EXPECT_NEAR(v(0),  1.0, EPS);
    EXPECT_NEAR(v(1),  2.0, EPS);
    EXPECT_NEAR(v(2), -1.0, EPS);
    EXPECT_NEAR(v(3), -2.0, EPS);
}

// ---------------------------------------------------------------------------
// 31. typeCast when vdim==0 → output is cleared
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, TypeCast_EmptyInput_ClearsOutput)
{
    Matrix1D<int> empty_iv; // vdim == 0
    Vec dv(3);
    dv(0) = 1.0; dv(1) = 2.0; dv(2) = 3.0;
    typeCast(empty_iv, dv);
    EXPECT_EQ(dv.vdim, 0);
}

// ---------------------------------------------------------------------------
// 32. REPORT_ERROR: size-mismatch in binary vector operators
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, VectorAdd_SizeMismatch_Throws)
{
    Vec a(3), b(2);
    EXPECT_THROW(a + b, RelionError);
}

TEST(Matrix1DTest, VectorSubtract_SizeMismatch_Throws)
{
    Vec a(3), b(2);
    EXPECT_THROW(a - b, RelionError);
}

TEST(Matrix1DTest, VectorMultiply_SizeMismatch_Throws)
{
    Vec a(3), b(2);
    EXPECT_THROW(a * b, RelionError);
}

TEST(Matrix1DTest, VectorDivide_SizeMismatch_Throws)
{
    Vec a(3), b(2);
    EXPECT_THROW(a / b, RelionError);
}

TEST(Matrix1DTest, VectorPlusEquals_SizeMismatch_Throws)
{
    Vec a(3), b(2);
    EXPECT_THROW(a += b, RelionError);
}

TEST(Matrix1DTest, VectorMinusEquals_SizeMismatch_Throws)
{
    Vec a(3), b(2);
    EXPECT_THROW(a -= b, RelionError);
}

TEST(Matrix1DTest, VectorTimesEquals_SizeMismatch_Throws)
{
    Vec a(3), b(2);
    EXPECT_THROW(a *= b, RelionError);
}

TEST(Matrix1DTest, VectorDivideEquals_SizeMismatch_Throws)
{
    Vec a(3), b(2);
    EXPECT_THROW(a /= b, RelionError);
}

// ---------------------------------------------------------------------------
// 33. REPORT_ERROR: dotProduct different sizes/shapes
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, DotProduct_SizeMismatch_Throws)
{
    Vec a(3), b(2);
    EXPECT_THROW(dotProduct(a, b), RelionError);
}

// ---------------------------------------------------------------------------
// 34. REPORT_ERROR: vectorProduct non-R3 or different orientations
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, VectorProduct_NonR3_Throws)
{
    Vec a(2), b(2); // not R3
    EXPECT_THROW(vectorProduct(a, b), RelionError);
}

TEST(Matrix1DTest, VectorProduct_DifferentOrientation_Throws)
{
    Vec row = vectorR3(1.0, 0.0, 0.0);
    Vec col = vectorR3(0.0, 1.0, 0.0);
    row.setRow();
    col.setCol();
    EXPECT_THROW(vectorProduct(row, col), RelionError);
}

// ---------------------------------------------------------------------------
// 35. REPORT_ERROR: sortTwoVectors different shapes
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, SortTwoVectors_SizeMismatch_Throws)
{
    Vec a(3), b(2);
    EXPECT_THROW(sortTwoVectors(a, b), RelionError);
}

// ---------------------------------------------------------------------------
// 36. sortTwoVectors green path — for loop swaps element-wise so v1 ≤ v2
// ---------------------------------------------------------------------------
TEST(Matrix1DTest, SortTwoVectors_SortsElementWise)
{
    Vec a(3), b(3);
    a(0) = 3.0; a(1) = 1.0; a(2) = 4.0;
    b(0) = 1.0; b(1) = 5.0; b(2) = 1.0;

    sortTwoVectors(a, b);

    // After sorting: a[j] = min, b[j] = max for each j
    EXPECT_NEAR(a(0), 1.0, EPS);  EXPECT_NEAR(b(0), 3.0, EPS);
    EXPECT_NEAR(a(1), 1.0, EPS);  EXPECT_NEAR(b(1), 5.0, EPS);
    EXPECT_NEAR(a(2), 1.0, EPS);  EXPECT_NEAR(b(2), 4.0, EPS);
}

TEST(Matrix1DTest, SortTwoVectors_AlreadySorted_Unchanged)
{
    Vec a(2), b(2);
    a(0) = 1.0; a(1) = 2.0;
    b(0) = 3.0; b(1) = 4.0;
    sortTwoVectors(a, b);
    EXPECT_NEAR(a(0), 1.0, EPS);  EXPECT_NEAR(b(0), 3.0, EPS);
    EXPECT_NEAR(a(1), 2.0, EPS);  EXPECT_NEAR(b(1), 4.0, EPS);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
