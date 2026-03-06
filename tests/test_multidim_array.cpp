/*
 * GoogleTest unit tests for MultidimArray<T>.
 *
 * Build and run:
 *   cmake -DBUILD_TESTS=ON ...
 *   make test_multidim_array
 *   ./build/bin/test_multidim_array
 *
 * No MPI is required; these are pure CPU unit tests.
 *
 * What is tested:
 *   1.  Default construction – empty/zero dimensions.
 *   2.  1-D/2-D/3-D/4-D size constructors – correct dim fields.
 *   3.  initZeros(sizes) – all elements are zero after call.
 *   4.  initConstant      – every element equals the supplied value.
 *   5.  operator() 1-D logical access – read / write round-trip.
 *   6.  operator() 2-D logical access – read / write round-trip.
 *   7.  operator() 3-D logical access – read / write round-trip.
 *   8.  DIRECT_A1D_ELEM / DIRECT_A2D_ELEM / DIRECT_A3D_ELEM macros.
 *   9.  sameShape() – matching and mis-matched arrays.
 *  10.  Copy constructor – deep copy, independent mutation.
 *  11.  Assignment operator – deep copy, independent mutation.
 *  12.  clear() – returns array to empty state.
 *  13.  resize() – preserves existing data within new bounds.
 *  14.  sum() – returns correct element sum.
 *  15.  computeAvg() – returns correct mean.
 *  16.  computeStats() – avg, stddev, min, max.
 *  17.  Arithmetic operators +, -, *, / (array-array).
 *  18.  Scalar arithmetic operators +, *, / (array-scalar).
 *  19.  In-place operators +=, -=, *=, /=.
 *  20.  alias() – shared data, no independent copy.
 *  21.  moveFrom() – ownership transfer.
 *  22.  FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY iteration.
 *  23.  NZYXSIZE / XSIZE / YSIZE / ZSIZE / NSIZE macros.
 *  24.  Vector constructor from std::vector<T>.
 *  25.  initZeros(pattern) – shape is adopted from the template array.
 */

#include <gtest/gtest.h>
#include <sstream>
#include "src/multidim_array.h"
#include "src/complex.h"

// Convenience: use double throughout so arithmetic comparisons are clean.
using Array1D = MultidimArray<double>;

// ---------------------------------------------------------------------------
// 1. Default construction
// ---------------------------------------------------------------------------
TEST(MultidimArrayTest, DefaultConstructorIsEmpty)
{
    Array1D a;
    EXPECT_EQ(NZYXSIZE(a), 0);
    EXPECT_EQ(XSIZE(a), 0);
    EXPECT_EQ(a.data, nullptr);
}

// ---------------------------------------------------------------------------
// 2. Size constructors
// ---------------------------------------------------------------------------
TEST(MultidimArrayTest, Constructor1D)
{
    Array1D a(5);
    EXPECT_EQ(XSIZE(a), 5);
    EXPECT_EQ(YSIZE(a), 1);
    EXPECT_EQ(ZSIZE(a), 1);
    EXPECT_EQ(NSIZE(a), 1);
    EXPECT_EQ(NZYXSIZE(a), 5);
    EXPECT_NE(a.data, nullptr);
}

TEST(MultidimArrayTest, Constructor2D)
{
    Array1D a(3, 4);
    EXPECT_EQ(YSIZE(a), 3);
    EXPECT_EQ(XSIZE(a), 4);
    EXPECT_EQ(NZYXSIZE(a), 12);
}

TEST(MultidimArrayTest, Constructor3D)
{
    Array1D a(2, 3, 4);
    EXPECT_EQ(ZSIZE(a), 2);
    EXPECT_EQ(YSIZE(a), 3);
    EXPECT_EQ(XSIZE(a), 4);
    EXPECT_EQ(NZYXSIZE(a), 24);
}

TEST(MultidimArrayTest, Constructor4D)
{
    Array1D a(2, 3, 4, 5);
    EXPECT_EQ(NSIZE(a), 2);
    EXPECT_EQ(ZSIZE(a), 3);
    EXPECT_EQ(YSIZE(a), 4);
    EXPECT_EQ(XSIZE(a), 5);
    EXPECT_EQ(NZYXSIZE(a), 120);
}

// ---------------------------------------------------------------------------
// 3. initZeros with explicit sizes
// ---------------------------------------------------------------------------
TEST(MultidimArrayTest, InitZerosWithSizes)
{
    Array1D a;
    a.initZeros(6);
    EXPECT_EQ(XSIZE(a), 6);
    for (long int i = 0; i < 6; ++i)
        EXPECT_EQ(DIRECT_A1D_ELEM(a, i), 0.0);
}

TEST(MultidimArrayTest, InitZerosPreservesShape)
{
    Array1D a(4);
    a.initConstant(99.0);
    a.initZeros();
    for (long int i = 0; i < 4; ++i)
        EXPECT_EQ(DIRECT_A1D_ELEM(a, i), 0.0);
}

// ---------------------------------------------------------------------------
// 4. initConstant
// ---------------------------------------------------------------------------
TEST(MultidimArrayTest, InitConstant)
{
    Array1D a(5);
    a.initConstant(3.14);
    for (long int i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(DIRECT_A1D_ELEM(a, i), 3.14);
}

// ---------------------------------------------------------------------------
// 5. operator() 1-D logical access
// ---------------------------------------------------------------------------
TEST(MultidimArrayTest, Operator1DReadWrite)
{
    Array1D a(5);
    a.initZeros();
    a(2) = 42.0;
    EXPECT_DOUBLE_EQ(a(2), 42.0);
    EXPECT_DOUBLE_EQ(a(0), 0.0);
}

// ---------------------------------------------------------------------------
// 6. operator() 2-D logical access
// ---------------------------------------------------------------------------
TEST(MultidimArrayTest, Operator2DReadWrite)
{
    Array1D a(3, 4);
    a.initZeros();
    a(1, 2) = 7.5;
    EXPECT_DOUBLE_EQ(a(1, 2), 7.5);
    EXPECT_DOUBLE_EQ(a(0, 0), 0.0);
}

// ---------------------------------------------------------------------------
// 7. operator() 3-D logical access
// ---------------------------------------------------------------------------
TEST(MultidimArrayTest, Operator3DReadWrite)
{
    Array1D a(2, 3, 4);
    a.initZeros();
    a(1, 2, 3) = -1.5;
    EXPECT_DOUBLE_EQ(a(1, 2, 3), -1.5);
    EXPECT_DOUBLE_EQ(a(0, 0, 0), 0.0);
}

// ---------------------------------------------------------------------------
// 8. Direct element access macros
// ---------------------------------------------------------------------------
TEST(MultidimArrayTest, DirectA1DElem)
{
    Array1D a(4);
    a.initZeros();
    DIRECT_A1D_ELEM(a, 3) = 5.0;
    EXPECT_DOUBLE_EQ(DIRECT_A1D_ELEM(a, 3), 5.0);
}

TEST(MultidimArrayTest, DirectA2DElem)
{
    Array1D a(3, 4);
    a.initZeros();
    DIRECT_A2D_ELEM(a, 2, 1) = 8.0;
    EXPECT_DOUBLE_EQ(DIRECT_A2D_ELEM(a, 2, 1), 8.0);
}

TEST(MultidimArrayTest, DirectA3DElem)
{
    Array1D a(2, 3, 4);
    a.initZeros();
    DIRECT_A3D_ELEM(a, 1, 2, 3) = 9.0;
    EXPECT_DOUBLE_EQ(DIRECT_A3D_ELEM(a, 1, 2, 3), 9.0);
}

// ---------------------------------------------------------------------------
// 9. sameShape()
// ---------------------------------------------------------------------------
TEST(MultidimArrayTest, SameShapeTrue)
{
    Array1D a(3, 4), b(3, 4);
    EXPECT_TRUE(a.sameShape(b));
}

TEST(MultidimArrayTest, SameShapeFalseDifferentSize)
{
    Array1D a(3, 4), b(3, 5);
    EXPECT_FALSE(a.sameShape(b));
}

TEST(MultidimArrayTest, SameShapeEmptyBoth)
{
    // Two empty (0-element) arrays should report same shape.
    Array1D a, b;
    EXPECT_TRUE(a.sameShape(b));
}

// ---------------------------------------------------------------------------
// 10. Copy constructor – deep copy
// ---------------------------------------------------------------------------
TEST(MultidimArrayTest, CopyConstructorIsDeep)
{
    Array1D a(4);
    a.initConstant(1.0);

    Array1D b(a);  // copy
    b(0) = 99.0;

    EXPECT_DOUBLE_EQ(a(0), 1.0)  // original unchanged
        << "Copy constructor must not share data with the original";
}

// ---------------------------------------------------------------------------
// 11. Assignment operator – deep copy
// ---------------------------------------------------------------------------
TEST(MultidimArrayTest, AssignmentIsDeep)
{
    Array1D a(4);
    a.initConstant(2.0);

    Array1D b;
    b = a;
    b(1) = 77.0;

    EXPECT_DOUBLE_EQ(a(1), 2.0)
        << "Assignment must not share data with the original";
}

// ---------------------------------------------------------------------------
// 12. clear() – returns to empty state
// ---------------------------------------------------------------------------
TEST(MultidimArrayTest, ClearReturnsToEmpty)
{
    Array1D a(5);
    a.initConstant(1.0);
    a.clear();

    EXPECT_EQ(XSIZE(a), 0);
    EXPECT_EQ(NZYXSIZE(a), 0);
    EXPECT_EQ(a.data, nullptr);
}

// ---------------------------------------------------------------------------
// 13. resize() – preserves existing data within new bounds
// ---------------------------------------------------------------------------
TEST(MultidimArrayTest, ResizePreservesData)
{
    Array1D a(4);
    for (long int i = 0; i < 4; ++i)
        DIRECT_A1D_ELEM(a, i) = static_cast<double>(i);

    a.resize(1, 1, 1, 8);  // expand
    EXPECT_EQ(XSIZE(a), 8);
    // Original four elements should be present in the expanded array.
    for (long int i = 0; i < 4; ++i)
        EXPECT_DOUBLE_EQ(DIRECT_A1D_ELEM(a, i), static_cast<double>(i));
}

// ---------------------------------------------------------------------------
// 14. sum()
// ---------------------------------------------------------------------------
TEST(MultidimArrayTest, SumIsCorrect)
{
    Array1D a(5);
    for (long int i = 0; i < 5; ++i)
        DIRECT_A1D_ELEM(a, i) = static_cast<double>(i + 1);  // 1,2,3,4,5

    EXPECT_DOUBLE_EQ(a.sum(), 15.0);
}

TEST(MultidimArrayTest, SumOfZerosIsZero)
{
    Array1D a(10);
    a.initZeros();
    EXPECT_DOUBLE_EQ(a.sum(), 0.0);
}

// ---------------------------------------------------------------------------
// 15. computeAvg()
// ---------------------------------------------------------------------------
TEST(MultidimArrayTest, ComputeAvgIsCorrect)
{
    Array1D a(4);
    DIRECT_A1D_ELEM(a, 0) = 1.0;
    DIRECT_A1D_ELEM(a, 1) = 2.0;
    DIRECT_A1D_ELEM(a, 2) = 3.0;
    DIRECT_A1D_ELEM(a, 3) = 4.0;

    EXPECT_DOUBLE_EQ(a.computeAvg(), 2.5);
}

// ---------------------------------------------------------------------------
// 16. computeStats() – avg, stddev, min, max
// ---------------------------------------------------------------------------
TEST(MultidimArrayTest, ComputeStats)
{
    Array1D a(4);
    DIRECT_A1D_ELEM(a, 0) = 1.0;
    DIRECT_A1D_ELEM(a, 1) = 2.0;
    DIRECT_A1D_ELEM(a, 2) = 3.0;
    DIRECT_A1D_ELEM(a, 3) = 4.0;

    RFLOAT avg, stddev;
    double minval, maxval;
    a.computeStats(avg, stddev, minval, maxval);

    EXPECT_DOUBLE_EQ(avg, 2.5);
    EXPECT_DOUBLE_EQ(minval, 1.0);
    EXPECT_DOUBLE_EQ(maxval, 4.0);
    // stddev of {1,2,3,4}: variance = 1.25, stddev ≈ 1.118034
    EXPECT_NEAR(stddev, 1.118034, 1e-5);
}

// ---------------------------------------------------------------------------
// 17. Array-array arithmetic operators
// ---------------------------------------------------------------------------
TEST(MultidimArrayTest, ArrayAddition)
{
    Array1D a(3), b(3);
    a.initConstant(2.0);
    b.initConstant(3.0);
    Array1D c = a + b;
    for (long int i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(DIRECT_A1D_ELEM(c, i), 5.0);
}

TEST(MultidimArrayTest, ArraySubtraction)
{
    Array1D a(3), b(3);
    a.initConstant(5.0);
    b.initConstant(2.0);
    Array1D c = a - b;
    for (long int i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(DIRECT_A1D_ELEM(c, i), 3.0);
}

TEST(MultidimArrayTest, ArrayMultiplication)
{
    Array1D a(3), b(3);
    a.initConstant(4.0);
    b.initConstant(0.5);
    Array1D c = a * b;
    for (long int i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(DIRECT_A1D_ELEM(c, i), 2.0);
}

TEST(MultidimArrayTest, ArrayDivision)
{
    Array1D a(3), b(3);
    a.initConstant(9.0);
    b.initConstant(3.0);
    Array1D c = a / b;
    for (long int i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(DIRECT_A1D_ELEM(c, i), 3.0);
}

// ---------------------------------------------------------------------------
// 18. Scalar arithmetic operators (array op scalar)
// ---------------------------------------------------------------------------
TEST(MultidimArrayTest, ScalarAddition)
{
    Array1D a(3);
    a.initConstant(1.0);
    Array1D b = a + 4.0;
    for (long int i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(DIRECT_A1D_ELEM(b, i), 5.0);
}

TEST(MultidimArrayTest, ScalarMultiplication)
{
    Array1D a(3);
    a.initConstant(3.0);
    Array1D b = a * 4.0;
    for (long int i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(DIRECT_A1D_ELEM(b, i), 12.0);
}

TEST(MultidimArrayTest, ScalarDivision)
{
    Array1D a(3);
    a.initConstant(6.0);
    Array1D b = a / 2.0;
    for (long int i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(DIRECT_A1D_ELEM(b, i), 3.0);
}

// ---------------------------------------------------------------------------
// 19. In-place operators +=, -=, *=, /=
// ---------------------------------------------------------------------------
TEST(MultidimArrayTest, InPlaceAddition)
{
    Array1D a(3), b(3);
    a.initConstant(1.0);
    b.initConstant(2.0);
    a += b;
    for (long int i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(DIRECT_A1D_ELEM(a, i), 3.0);
}

TEST(MultidimArrayTest, InPlaceSubtraction)
{
    Array1D a(3), b(3);
    a.initConstant(5.0);
    b.initConstant(3.0);
    a -= b;
    for (long int i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(DIRECT_A1D_ELEM(a, i), 2.0);
}

TEST(MultidimArrayTest, InPlaceMultiplication)
{
    Array1D a(3), b(3);
    a.initConstant(3.0);
    b.initConstant(4.0);
    a *= b;
    for (long int i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(DIRECT_A1D_ELEM(a, i), 12.0);
}

TEST(MultidimArrayTest, InPlaceDivision)
{
    Array1D a(3), b(3);
    a.initConstant(8.0);
    b.initConstant(2.0);
    a /= b;
    for (long int i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(DIRECT_A1D_ELEM(a, i), 4.0);
}

// ---------------------------------------------------------------------------
// 20. alias() – shared data pointer; mutation via alias changes original
// ---------------------------------------------------------------------------
TEST(MultidimArrayTest, AliasSharesData)
{
    Array1D a(4);
    a.initConstant(1.0);

    Array1D b;
    b.alias(a);

    EXPECT_EQ(b.data, a.data) << "alias() must share the same data pointer";

    DIRECT_A1D_ELEM(b, 0) = 99.0;
    EXPECT_DOUBLE_EQ(DIRECT_A1D_ELEM(a, 0), 99.0)
        << "Mutation via alias must be visible in the original";
}

// ---------------------------------------------------------------------------
// 21. moveFrom() – ownership transfer; original becomes alias
// ---------------------------------------------------------------------------
TEST(MultidimArrayTest, MoveFromTransfersOwnership)
{
    Array1D src(4);
    src.initConstant(7.0);
    double* original_ptr = src.data;

    Array1D dst;
    dst.moveFrom(src);

    EXPECT_EQ(dst.data, original_ptr)   << "moveFrom must transfer the pointer";
    EXPECT_TRUE(dst.destroyData)        << "destination must own the data";
    EXPECT_FALSE(src.destroyData)       << "source must no longer own the data";
    EXPECT_DOUBLE_EQ(DIRECT_A1D_ELEM(dst, 0), 7.0);
}

// ---------------------------------------------------------------------------
// 22. FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY iteration
// ---------------------------------------------------------------------------
TEST(MultidimArrayTest, ForAllDirectElementsIteration)
{
    const int N = 6;
    Array1D a(N);
    a.initZeros();

    long int n = 0;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(a)
    {
        DIRECT_MULTIDIM_ELEM(a, n) = static_cast<double>(n);
    }

    for (long int i = 0; i < N; ++i)
        EXPECT_DOUBLE_EQ(DIRECT_A1D_ELEM(a, i), static_cast<double>(i));
}

// ---------------------------------------------------------------------------
// 23. Size macro accessors
// ---------------------------------------------------------------------------
TEST(MultidimArrayTest, SizeMacros)
{
    Array1D a(2, 3, 4, 5);
    EXPECT_EQ(NSIZE(a),    2);
    EXPECT_EQ(ZSIZE(a),    3);
    EXPECT_EQ(YSIZE(a),    4);
    EXPECT_EQ(XSIZE(a),    5);
    EXPECT_EQ(NZYXSIZE(a), 2 * 3 * 4 * 5);
    EXPECT_EQ(YXSIZE(a),   4 * 5);
    EXPECT_EQ(ZYXSIZE(a),  3 * 4 * 5);
}

// ---------------------------------------------------------------------------
// 24. Constructor from std::vector<T>
// ---------------------------------------------------------------------------
TEST(MultidimArrayTest, VectorConstructor)
{
    std::vector<double> v = {10.0, 20.0, 30.0};
    Array1D a(v);

    ASSERT_EQ(XSIZE(a), 3);
    EXPECT_DOUBLE_EQ(a(0), 10.0);
    EXPECT_DOUBLE_EQ(a(1), 20.0);
    EXPECT_DOUBLE_EQ(a(2), 30.0);
}

// ---------------------------------------------------------------------------
// 25. initZeros(pattern) – shape copied from template array
// ---------------------------------------------------------------------------
TEST(MultidimArrayTest, InitZerosFromPattern)
{
    Array1D pattern(2, 3, 4);
    Array1D a;
    a.initZeros(pattern);

    EXPECT_TRUE(a.sameShape(pattern))
        << "initZeros(pattern) must adopt the shape of the template array";

    for (long int n = 0; n < NZYXSIZE(a); ++n)
        EXPECT_EQ(DIRECT_MULTIDIM_ELEM(a, n), 0.0);
}

// ---------------------------------------------------------------------------
// resize for non-double types
// ---------------------------------------------------------------------------

TEST(MultidimArrayTest, BoolResize1D)
{
    MultidimArray<bool> a;
    a.resize(5);
    EXPECT_EQ(XSIZE(a), 5);
    EXPECT_EQ(YSIZE(a), 1);
    EXPECT_EQ(ZSIZE(a), 1);
}

TEST(MultidimArrayTest, BoolResize4D)
{
    MultidimArray<bool> a;
    a.resize(2, 3, 4, 5);
    EXPECT_EQ(NSIZE(a), 2);
    EXPECT_EQ(ZSIZE(a), 3);
    EXPECT_EQ(YSIZE(a), 4);
    EXPECT_EQ(XSIZE(a), 5);
}

TEST(MultidimArrayTest, FloatResize4D)
{
    MultidimArray<float> a;
    a.resize(2, 3, 4, 5);
    EXPECT_EQ(NSIZE(a), 2);
    EXPECT_EQ(XSIZE(a), 5);
}

TEST(MultidimArrayTest, IntResize2D)
{
    MultidimArray<int> a;
    a.resize(3, 4);
    EXPECT_EQ(YSIZE(a), 3);
    EXPECT_EQ(XSIZE(a), 4);
}

TEST(MultidimArrayTest, IntResize3D)
{
    MultidimArray<int> a;
    a.resize(2, 3, 4);
    EXPECT_EQ(ZSIZE(a), 2);
    EXPECT_EQ(YSIZE(a), 3);
    EXPECT_EQ(XSIZE(a), 4);
}

TEST(MultidimArrayTest, IntResizeFromPattern)
{
    MultidimArray<int> src;
    src.resize(3, 4);
    MultidimArray<int> dst;
    dst.resize(src);
    EXPECT_EQ(YSIZE(dst), 3);
    EXPECT_EQ(XSIZE(dst), 4);
}

TEST(MultidimArrayTest, FloatResizeFromPattern)
{
    MultidimArray<float> src;
    src.resize(3, 4);
    MultidimArray<float> dst;
    dst.resize(src);
    EXPECT_EQ(YSIZE(dst), 3);
    EXPECT_EQ(XSIZE(dst), 4);
}

// ---------------------------------------------------------------------------
// window operations
// ---------------------------------------------------------------------------

TEST(MultidimArrayTest, Window2D_InPlace_CropsCorrectly)
{
    MultidimArray<RFLOAT> a;
    a.resize(8, 8);
    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(a)
        DIRECT_A2D_ELEM(a, i, j) = static_cast<RFLOAT>(i * 8 + j);

    // Pass explicit init_value to disambiguate from 1D overload
    a.window(2, 2, 5, 5, 0.0);
    EXPECT_EQ((int)YSIZE(a), 4);
    EXPECT_EQ((int)XSIZE(a), 4);
}

TEST(MultidimArrayTest, Window2D_IntoOutput_CorrectSize)
{
    MultidimArray<RFLOAT> src, dst;
    src.resize(8, 8);
    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(src)
        DIRECT_A2D_ELEM(src, i, j) = 1.0;

    src.window(dst, 1, 1, 4, 4, 0.0);
    EXPECT_EQ((int)YSIZE(dst), 4);
    EXPECT_EQ((int)XSIZE(dst), 4);
}

TEST(MultidimArrayTest, Window3D_InPlace_CropsCorrectly)
{
    MultidimArray<RFLOAT> a;
    a.resize(8, 8, 8);
    a.initConstant(1.0);

    // Pass explicit init_value to disambiguate from 2D overload
    a.window(1, 1, 1, 4, 4, 4, 0.0);
    EXPECT_EQ((int)ZSIZE(a), 4);
    EXPECT_EQ((int)YSIZE(a), 4);
    EXPECT_EQ((int)XSIZE(a), 4);
}

TEST(MultidimArrayTest, Window3D_IntoOutput_CorrectSize)
{
    MultidimArray<RFLOAT> src, dst;
    src.resize(8, 8, 8);
    src.initConstant(1.0);

    src.window(dst, 0, 0, 0, 3, 3, 3, 0.0);
    EXPECT_EQ((int)ZSIZE(dst), 4);
    EXPECT_EQ((int)YSIZE(dst), 4);
    EXPECT_EQ((int)XSIZE(dst), 4);
}

// ---------------------------------------------------------------------------
// printShape
// ---------------------------------------------------------------------------

TEST(MultidimArrayTest, PrintShape_DoesNotCrash)
{
    MultidimArray<RFLOAT> a;
    a.resize(4, 4);
    std::ostringstream oss;
    EXPECT_NO_THROW(a.printShape(oss));
}

TEST(MultidimArrayTest, PrintShape_Complex_DoesNotCrash)
{
    MultidimArray<Complex> a;
    a.resize(4, 4);
    std::ostringstream oss;
    EXPECT_NO_THROW(a.printShape(oss));
}

// ---------------------------------------------------------------------------
// checkDimension
// ---------------------------------------------------------------------------

TEST(MultidimArrayTest, CheckDimension_CorrectDimDoesNotThrow)
{
    MultidimArray<RFLOAT> a;
    a.resize(4, 4);
    EXPECT_NO_THROW(a.checkDimension(2));
}

TEST(MultidimArrayTest, CheckDimension_WrongDimExits)
{
    MultidimArray<RFLOAT> a;
    a.resize(4, 4);
    EXPECT_DEATH(a.checkDimension(3), "");
}

// ---------------------------------------------------------------------------
// rowNumber / colNumber
// ---------------------------------------------------------------------------

TEST(MultidimArrayTest, RowNumber_2D)
{
    MultidimArray<RFLOAT> a;
    a.resize(3, 5);
    EXPECT_EQ(a.rowNumber(), 3);
}

TEST(MultidimArrayTest, ColNumber_2D)
{
    MultidimArray<RFLOAT> a;
    a.resize(3, 5);
    EXPECT_EQ(a.colNumber(), 5);
}

// ---------------------------------------------------------------------------
// setXmippOrigin for float
// ---------------------------------------------------------------------------

TEST(MultidimArrayTest, FloatSetXmippOrigin_SetsNegativeStart)
{
    MultidimArray<float> a;
    a.resize(8, 8);
    a.setXmippOrigin();
    EXPECT_LT(STARTINGX(a), 0);
    EXPECT_LT(STARTINGY(a), 0);
}

// ---------------------------------------------------------------------------
// Complex operator()(long, long) const
// ---------------------------------------------------------------------------

TEST(MultidimArrayTest, ComplexTwoIndexAccess)
{
    MultidimArray<Complex> a;
    a.resize(4, 4);
    a.setXmippOrigin();
    A2D_ELEM(a, 1, 1) = Complex(2.5, 3.5);
    const MultidimArray<Complex>& ca = a;
    Complex val = ca(1, 1);
    EXPECT_NEAR(val.real, 2.5, 1e-6);
    EXPECT_NEAR(val.imag, 3.5, 1e-6);
}

// ---------------------------------------------------------------------------
// getImage / setSlice
// ---------------------------------------------------------------------------

TEST(MultidimArrayTest, GetImage_ExtractsCorrectSlice)
{
    MultidimArray<RFLOAT> vol4d;
    vol4d.resize(2, 1, 4, 4);
    // Set image 1 (second image) to 5.0
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            DIRECT_NZYX_ELEM(vol4d, 1, 0, i, j) = 5.0;

    MultidimArray<RFLOAT> img;
    vol4d.getImage(1, img);
    EXPECT_NEAR(DIRECT_A2D_ELEM(img, 0, 0), 5.0, 1e-6);
}

TEST(MultidimArrayTest, SetSlice_UpdatesCorrectLayer)
{
    MultidimArray<RFLOAT> vol;
    vol.resize(4, 4, 4);
    vol.initZeros();

    MultidimArray<RFLOAT> slice;
    slice.resize(4, 4);
    slice.initConstant(7.0);

    vol.setSlice(1, slice);
    EXPECT_NEAR(DIRECT_A3D_ELEM(vol, 1, 0, 0), 7.0, 1e-6);
    EXPECT_NEAR(DIRECT_A3D_ELEM(vol, 0, 0, 0), 0.0, 1e-6);
}

// ---------------------------------------------------------------------------
// computeDoubleMinMax / computeStddev
// ---------------------------------------------------------------------------

TEST(MultidimArrayTest, ComputeDoubleMinMax_CorrectRange)
{
    MultidimArray<RFLOAT> a;
    a.resize(4);
    DIRECT_A1D_ELEM(a, 0) = 3.0;
    DIRECT_A1D_ELEM(a, 1) = 1.0;
    DIRECT_A1D_ELEM(a, 2) = 4.0;
    DIRECT_A1D_ELEM(a, 3) = 1.5;

    double mn, mx;
    a.computeDoubleMinMax(mn, mx);
    EXPECT_NEAR(mn, 1.0, 1e-6);
    EXPECT_NEAR(mx, 4.0, 1e-6);
}

TEST(MultidimArrayTest, ComputeStddev_ConstantArray_IsZero)
{
    MultidimArray<RFLOAT> a;
    a.resize(4);
    a.initConstant(3.0);
    EXPECT_NEAR(a.computeStddev(), 0.0, 1e-6);
}

TEST(MultidimArrayTest, ComputeStddev_NonConstant_IsPositive)
{
    MultidimArray<RFLOAT> a;
    a.resize(4);
    DIRECT_A1D_ELEM(a, 0) = 1.0;
    DIRECT_A1D_ELEM(a, 1) = 2.0;
    DIRECT_A1D_ELEM(a, 2) = 3.0;
    DIRECT_A1D_ELEM(a, 3) = 4.0;
    EXPECT_GT(a.computeStddev(), 0.0);
}

// ---------------------------------------------------------------------------
// Complex in-place arithmetic
// ---------------------------------------------------------------------------

TEST(MultidimArrayTest, ComplexInPlaceAdd)
{
    MultidimArray<Complex> a, b;
    a.resize(4);
    b.resize(4);
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(a)
    {
        DIRECT_MULTIDIM_ELEM(a, n) = Complex(1.0, 0.0);
        DIRECT_MULTIDIM_ELEM(b, n) = Complex(2.0, 1.0);
    }
    a += b;
    EXPECT_NEAR(DIRECT_MULTIDIM_ELEM(a, 0).real, 3.0, 1e-6);
    EXPECT_NEAR(DIRECT_MULTIDIM_ELEM(a, 0).imag, 1.0, 1e-6);
}

TEST(MultidimArrayTest, ComplexInPlaceSubtract)
{
    MultidimArray<Complex> a, b;
    a.resize(4);
    b.resize(4);
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(a)
    {
        DIRECT_MULTIDIM_ELEM(a, n) = Complex(5.0, 3.0);
        DIRECT_MULTIDIM_ELEM(b, n) = Complex(1.0, 1.0);
    }
    a -= b;
    EXPECT_NEAR(DIRECT_MULTIDIM_ELEM(a, 0).real, 4.0, 1e-6);
    EXPECT_NEAR(DIRECT_MULTIDIM_ELEM(a, 0).imag, 2.0, 1e-6);
}

// ---------------------------------------------------------------------------
// RFLOAT in-place scalar addition
// ---------------------------------------------------------------------------

TEST(MultidimArrayTest, ScalarInPlaceAdd)
{
    MultidimArray<RFLOAT> a;
    a.resize(4);
    a.initConstant(2.0);
    a += 3.0;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(a)
        EXPECT_NEAR(DIRECT_MULTIDIM_ELEM(a, n), 5.0, 1e-6);
}

// ---------------------------------------------------------------------------
// cross-type resize (resize<T2>(MultidimArray<T2>))
// ---------------------------------------------------------------------------

TEST(MultidimArrayTest, ResizeFromBoolPattern)
{
    MultidimArray<bool> src;
    src.resize(3, 4);
    MultidimArray<RFLOAT> dst;
    dst.resize(src);  // cross-type: resize<bool>(MultidimArray<bool>)
    EXPECT_EQ((int)YSIZE(dst), 3);
    EXPECT_EQ((int)XSIZE(dst), 4);
}

TEST(MultidimArrayTest, ComplexResizeFromRfloatPattern)
{
    MultidimArray<RFLOAT> src;
    src.resize(3, 4);
    MultidimArray<Complex> dst;
    dst.resize(src);  // resize<RFLOAT>(MultidimArray<RFLOAT>)
    EXPECT_EQ((int)YSIZE(dst), 3);
    EXPECT_EQ((int)XSIZE(dst), 4);
}

// ---------------------------------------------------------------------------
// cross-type sameShape
// ---------------------------------------------------------------------------

TEST(MultidimArrayTest, SameShape_IntVsDouble_SameSize)
{
    MultidimArray<int> a;
    MultidimArray<RFLOAT> b;
    a.resize(4, 4);
    b.resize(4, 4);
    EXPECT_TRUE(a.sameShape(b));
}

TEST(MultidimArrayTest, SameShape_FloatVsFloat_SameSize)
{
    MultidimArray<float> a, b;
    a.resize(4, 4);
    b.resize(4, 4);
    EXPECT_TRUE(a.sameShape(b));
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
