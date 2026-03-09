/*
 * Unit tests for src/jaz/gravis/t3Vector.h and t4Vector.h
 */

#include <gtest/gtest.h>
#include <cmath>
#include "src/jaz/gravis/t2Vector.h"
#include "src/jaz/gravis/t3Vector.h"
#include "src/jaz/gravis/t4Vector.h"

using namespace gravis;
typedef t3Vector<double> d3Vec;
typedef t4Vector<double> d4Vec;

// ---------------------------------------------------------------------------
// t3Vector constructors
// ---------------------------------------------------------------------------

TEST(T3VectorTest, DefaultConstructor_IsZero)
{
    d3Vec v;
    EXPECT_EQ(v.x, 0.0);
    EXPECT_EQ(v.y, 0.0);
    EXPECT_EQ(v.z, 0.0);
}

TEST(T3VectorTest, ScalarConstructor_AllSame)
{
    d3Vec v(3.0);
    EXPECT_EQ(v.x, 3.0);
    EXPECT_EQ(v.y, 3.0);
    EXPECT_EQ(v.z, 3.0);
}

TEST(T3VectorTest, XYZConstructor)
{
    d3Vec v(1.0, 2.0, 3.0);
    EXPECT_EQ(v.x, 1.0);
    EXPECT_EQ(v.y, 2.0);
    EXPECT_EQ(v.z, 3.0);
}

TEST(T3VectorTest, CopyConstructor)
{
    d3Vec v(1.0, 2.0, 3.0);
    d3Vec w(v);
    EXPECT_EQ(w.x, 1.0);
    EXPECT_EQ(w.y, 2.0);
    EXPECT_EQ(w.z, 3.0);
}

TEST(T3VectorTest, ArrayConstructor)
{
    double arr[3] = {4.0, 5.0, 6.0};
    d3Vec v(arr);
    EXPECT_EQ(v.x, 4.0);
    EXPECT_EQ(v.y, 5.0);
    EXPECT_EQ(v.z, 6.0);
}

TEST(T3VectorTest, UnitX)
{
    d3Vec v = d3Vec::unitX();
    EXPECT_EQ(v.x, 1.0); EXPECT_EQ(v.y, 0.0); EXPECT_EQ(v.z, 0.0);
}

TEST(T3VectorTest, UnitY)
{
    d3Vec v = d3Vec::unitY();
    EXPECT_EQ(v.x, 0.0); EXPECT_EQ(v.y, 1.0); EXPECT_EQ(v.z, 0.0);
}

TEST(T3VectorTest, UnitZ)
{
    d3Vec v = d3Vec::unitZ();
    EXPECT_EQ(v.x, 0.0); EXPECT_EQ(v.y, 0.0); EXPECT_EQ(v.z, 1.0);
}

// ---------------------------------------------------------------------------
// t3Vector set
// ---------------------------------------------------------------------------

TEST(T3VectorTest, SetScalar)
{
    d3Vec v;
    v.set(7.0);
    EXPECT_EQ(v.x, 7.0); EXPECT_EQ(v.y, 7.0); EXPECT_EQ(v.z, 7.0);
}

TEST(T3VectorTest, SetXYZ)
{
    d3Vec v;
    v.set(1.0, 2.0, 3.0);
    EXPECT_EQ(v.x, 1.0); EXPECT_EQ(v.y, 2.0); EXPECT_EQ(v.z, 3.0);
}

// ---------------------------------------------------------------------------
// t3Vector norms / length
// ---------------------------------------------------------------------------

TEST(T3VectorTest, Length)
{
    d3Vec v(3.0, 4.0, 0.0);
    EXPECT_NEAR(v.length(), 5.0, 1e-10);
}

TEST(T3VectorTest, Norm2)
{
    d3Vec v(1.0, 2.0, 2.0);
    EXPECT_NEAR(v.norm2(), 9.0, 1e-10);
}

TEST(T3VectorTest, NormL1)
{
    d3Vec v(-1.0, 2.0, -3.0);
    EXPECT_NEAR(v.normL1(), 6.0, 1e-10);
}

TEST(T3VectorTest, NormL2)
{
    d3Vec v(0.0, 3.0, 4.0);
    EXPECT_NEAR(v.normL2(), 5.0, 1e-10);
}

TEST(T3VectorTest, NormLInf)
{
    d3Vec v(1.0, -5.0, 3.0);
    EXPECT_NEAR(v.normLInf(), 5.0, 1e-10);
}

TEST(T3VectorTest, NormL2sqr)
{
    d3Vec v(1.0, 2.0, 2.0);
    EXPECT_NEAR(v.normL2sqr(), 9.0, 1e-10);
}

TEST(T3VectorTest, Sum)
{
    d3Vec v(1.0, 2.0, 3.0);
    EXPECT_NEAR(v.sum(), 6.0, 1e-10);
}

// ---------------------------------------------------------------------------
// t3Vector dot / cross
// ---------------------------------------------------------------------------

TEST(T3VectorTest, DotProduct)
{
    d3Vec a(1.0, 0.0, 0.0);
    d3Vec b(0.0, 1.0, 0.0);
    EXPECT_NEAR(a.dot(b), 0.0, 1e-10);
    EXPECT_NEAR(d3Vec::dot(a, a), 1.0, 1e-10);
}

TEST(T3VectorTest, CrossProduct)
{
    d3Vec x = d3Vec::unitX();
    d3Vec y = d3Vec::unitY();
    d3Vec z = x.cross(y);
    EXPECT_NEAR(z.x, 0.0, 1e-10);
    EXPECT_NEAR(z.y, 0.0, 1e-10);
    EXPECT_NEAR(z.z, 1.0, 1e-10);
}

TEST(T3VectorTest, StaticCrossProduct)
{
    d3Vec x = d3Vec::unitX();
    d3Vec y = d3Vec::unitY();
    d3Vec z = d3Vec::cross(x, y);
    EXPECT_NEAR(z.z, 1.0, 1e-10);
}

// ---------------------------------------------------------------------------
// t3Vector normalize / invert
// ---------------------------------------------------------------------------

TEST(T3VectorTest, Normalize)
{
    d3Vec v(3.0, 0.0, 4.0);
    v.normalize();
    EXPECT_NEAR(v.length(), 1.0, 1e-10);
}

TEST(T3VectorTest, StaticNormalize)
{
    d3Vec v(0.0, 3.0, 4.0);
    d3Vec n = d3Vec::normalize(v);
    EXPECT_NEAR(n.length(), 1.0, 1e-10);
}

TEST(T3VectorTest, Invert)
{
    d3Vec v(1.0, -2.0, 3.0);
    v.invert();
    EXPECT_EQ(v.x, -1.0);
    EXPECT_EQ(v.y,  2.0);
    EXPECT_EQ(v.z, -3.0);
}

// ---------------------------------------------------------------------------
// t3Vector arithmetic operators
// ---------------------------------------------------------------------------

TEST(T3VectorTest, AddInPlace)
{
    d3Vec v(1.0, 2.0, 3.0);
    v += d3Vec(1.0, 1.0, 1.0);
    EXPECT_EQ(v.x, 2.0); EXPECT_EQ(v.y, 3.0); EXPECT_EQ(v.z, 4.0);
}

TEST(T3VectorTest, SubtractInPlace)
{
    d3Vec v(3.0, 3.0, 3.0);
    v -= d3Vec(1.0, 1.0, 1.0);
    EXPECT_EQ(v.x, 2.0); EXPECT_EQ(v.y, 2.0); EXPECT_EQ(v.z, 2.0);
}

TEST(T3VectorTest, ScalarAddInPlace)
{
    d3Vec v(1.0, 2.0, 3.0);
    v += 1.0;
    EXPECT_EQ(v.x, 2.0); EXPECT_EQ(v.y, 3.0); EXPECT_EQ(v.z, 4.0);
}

TEST(T3VectorTest, ScalarSubtractInPlace)
{
    d3Vec v(2.0, 3.0, 4.0);
    v -= 1.0;
    EXPECT_EQ(v.x, 1.0); EXPECT_EQ(v.y, 2.0); EXPECT_EQ(v.z, 3.0);
}

TEST(T3VectorTest, ScalarMultiplyInPlace)
{
    d3Vec v(1.0, 2.0, 3.0);
    v *= 2.0;
    EXPECT_EQ(v.x, 2.0); EXPECT_EQ(v.y, 4.0); EXPECT_EQ(v.z, 6.0);
}

TEST(T3VectorTest, ScalarMultiply)
{
    d3Vec v(1.0, 2.0, 3.0);
    d3Vec w = v * 3.0;
    EXPECT_EQ(w.x, 3.0); EXPECT_EQ(w.y, 6.0); EXPECT_EQ(w.z, 9.0);
}

TEST(T3VectorTest, ScalarDivideInPlace)
{
    d3Vec v(4.0, 6.0, 8.0);
    v /= 2.0;
    EXPECT_EQ(v.x, 2.0); EXPECT_EQ(v.y, 3.0); EXPECT_EQ(v.z, 4.0);
}

TEST(T3VectorTest, ScalarDivide)
{
    d3Vec v(6.0, 9.0, 12.0);
    d3Vec w = v / 3.0;
    EXPECT_EQ(w.x, 2.0); EXPECT_EQ(w.y, 3.0); EXPECT_EQ(w.z, 4.0);
}

// ---------------------------------------------------------------------------
// t3Vector comparison / distance
// ---------------------------------------------------------------------------

TEST(T3VectorTest, Equality)
{
    d3Vec a(1.0, 2.0, 3.0);
    d3Vec b(1.0, 2.0, 3.0);
    EXPECT_TRUE(a == b);
}

TEST(T3VectorTest, Inequality)
{
    d3Vec a(1.0, 2.0, 3.0);
    d3Vec b(1.0, 2.0, 4.0);
    EXPECT_TRUE(a != b);
}

TEST(T3VectorTest, IsClose)
{
    d3Vec a(1.0, 2.0, 3.0);
    d3Vec b(1.001, 2.001, 3.001);
    EXPECT_TRUE(a.isClose(b, 0.01));
    EXPECT_FALSE(a.isClose(b, 1e-6));
}

TEST(T3VectorTest, Dist2)
{
    d3Vec a(0.0, 0.0, 0.0);
    d3Vec b(3.0, 4.0, 0.0);
    EXPECT_NEAR(a.dist2(b), 25.0, 1e-10);
}

TEST(T3VectorTest, Dist)
{
    d3Vec a(0.0, 0.0, 0.0);
    d3Vec b(0.0, 3.0, 4.0);
    EXPECT_NEAR(a.dist(b), 5.0, 1e-10);
}

// ---------------------------------------------------------------------------
// t3Vector subscript
// ---------------------------------------------------------------------------

TEST(T3VectorTest, SubscriptRead)
{
    d3Vec v(1.0, 2.0, 3.0);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 2.0);
    EXPECT_EQ(v[2], 3.0);
}

TEST(T3VectorTest, SubscriptWrite)
{
    d3Vec v;
    v[0] = 5.0; v[1] = 6.0; v[2] = 7.0;
    EXPECT_EQ(v.x, 5.0); EXPECT_EQ(v.y, 6.0); EXPECT_EQ(v.z, 7.0);
}

// ---------------------------------------------------------------------------
// t3Vector cmul / cdiv / interpolate / findOrthogonal / xy
// ---------------------------------------------------------------------------

TEST(T3VectorTest, CMul)
{
    d3Vec a(2.0, 3.0, 4.0);
    d3Vec b(5.0, 6.0, 7.0);
    d3Vec c = a.cmul(b);
    EXPECT_EQ(c.x, 10.0); EXPECT_EQ(c.y, 18.0); EXPECT_EQ(c.z, 28.0);
}

TEST(T3VectorTest, CDiv)
{
    d3Vec a(10.0, 12.0, 14.0);
    d3Vec b(2.0, 3.0, 7.0);
    d3Vec c = a.cdiv(b);
    EXPECT_NEAR(c.x, 5.0, 1e-10);
    EXPECT_NEAR(c.y, 4.0, 1e-10);
    EXPECT_NEAR(c.z, 2.0, 1e-10);
}

TEST(T3VectorTest, Interpolate)
{
    d3Vec bary(0.5, 0.3, 0.2);
    double result = bary.interpolate(1.0, 2.0, 3.0);
    EXPECT_NEAR(result, 1.7, 1e-10);
}

TEST(T3VectorTest, FindOrthogonal_IsOrthogonal)
{
    d3Vec v(1.0, 0.0, 0.0);
    d3Vec orth = v.findOrthogonal();
    EXPECT_NEAR(v.dot(orth), 0.0, 1e-10);
}

TEST(T3VectorTest, XY)
{
    d3Vec v(1.0, 2.0, 3.0);
    auto xy = v.xy();
    EXPECT_EQ(xy.x, 1.0);
    EXPECT_EQ(xy.y, 2.0);
}

// ---------------------------------------------------------------------------
// t3Vector from t4Vector
// ---------------------------------------------------------------------------

TEST(T3VectorTest, FromT4VectorDivW)
{
    d4Vec v4(2.0, 4.0, 6.0, 2.0);
    d3Vec v3(v4);
    EXPECT_NEAR(v3.x, 1.0, 1e-10);
    EXPECT_NEAR(v3.y, 2.0, 1e-10);
    EXPECT_NEAR(v3.z, 3.0, 1e-10);
}

// ---------------------------------------------------------------------------
// t4Vector constructors
// ---------------------------------------------------------------------------

TEST(T4VectorTest, DefaultConstructor)
{
    d4Vec v;
    EXPECT_EQ(v.x, 0.0); EXPECT_EQ(v.y, 0.0);
    EXPECT_EQ(v.z, 0.0); EXPECT_EQ(v.w, 1.0);
}

TEST(T4VectorTest, ScalarConstructor)
{
    d4Vec v(5.0);
    EXPECT_EQ(v.x, 5.0); EXPECT_EQ(v.y, 5.0);
    EXPECT_EQ(v.z, 5.0); EXPECT_EQ(v.w, 5.0);
}

TEST(T4VectorTest, XYZWConstructor)
{
    d4Vec v(1.0, 2.0, 3.0, 4.0);
    EXPECT_EQ(v.x, 1.0); EXPECT_EQ(v.y, 2.0);
    EXPECT_EQ(v.z, 3.0); EXPECT_EQ(v.w, 4.0);
}

TEST(T4VectorTest, FromT3Vector)
{
    d3Vec v3(1.0, 2.0, 3.0);
    d4Vec v4(v3);
    EXPECT_EQ(v4.x, 1.0); EXPECT_EQ(v4.y, 2.0);
    EXPECT_EQ(v4.z, 3.0); EXPECT_EQ(v4.w, 1.0);
}

TEST(T4VectorTest, UnitX) { d4Vec v = d4Vec::unitX(); EXPECT_EQ(v.x, 1.0); EXPECT_EQ(v.y, 0.0); }
TEST(T4VectorTest, UnitY) { d4Vec v = d4Vec::unitY(); EXPECT_EQ(v.y, 1.0); EXPECT_EQ(v.x, 0.0); }
TEST(T4VectorTest, UnitZ) { d4Vec v = d4Vec::unitZ(); EXPECT_EQ(v.z, 1.0); EXPECT_EQ(v.w, 1.0); }

// ---------------------------------------------------------------------------
// t4Vector norms / length
// ---------------------------------------------------------------------------

TEST(T4VectorTest, Length)
{
    d4Vec v(1.0, 0.0, 0.0, 0.0);
    EXPECT_NEAR(v.length(), 1.0, 1e-10);
}

TEST(T4VectorTest, Norm2)
{
    d4Vec v(1.0, 1.0, 1.0, 1.0);
    EXPECT_NEAR(v.norm2(), 4.0, 1e-10);
}

TEST(T4VectorTest, NormL1)
{
    d4Vec v(1.0, -1.0, 1.0, -1.0);
    EXPECT_NEAR(v.normL1(), 4.0, 1e-10);
}

TEST(T4VectorTest, NormL2)
{
    d4Vec v(2.0, 0.0, 0.0, 0.0);
    EXPECT_NEAR(v.normL2(), 2.0, 1e-10);
}

TEST(T4VectorTest, NormLInf)
{
    d4Vec v(1.0, -3.0, 2.0, 0.5);
    EXPECT_NEAR(v.normLInf(), 3.0, 1e-10);
}

// ---------------------------------------------------------------------------
// t4Vector dot / divideW / toVector3 / xyz / xy
// ---------------------------------------------------------------------------

TEST(T4VectorTest, Dot)
{
    d4Vec a(1.0, 0.0, 0.0, 0.0);
    d4Vec b(0.0, 1.0, 0.0, 0.0);
    EXPECT_NEAR(d4Vec::dot(a, b), 0.0, 1e-10);
    EXPECT_NEAR(d4Vec::dot(a, a), 1.0, 1e-10);
}

TEST(T4VectorTest, DivideW)
{
    d4Vec v(4.0, 6.0, 8.0, 2.0);
    v.divideW();
    EXPECT_NEAR(v.x, 2.0, 1e-10);
    EXPECT_NEAR(v.y, 3.0, 1e-10);
    EXPECT_NEAR(v.z, 4.0, 1e-10);
    EXPECT_NEAR(v.w, 1.0, 1e-10);
}

TEST(T4VectorTest, ToVector3_PointCase)
{
    d4Vec v(4.0, 6.0, 8.0, 2.0);
    d3Vec v3 = v.toVector3();
    EXPECT_NEAR(v3.x, 2.0, 1e-10);
    EXPECT_NEAR(v3.y, 3.0, 1e-10);
    EXPECT_NEAR(v3.z, 4.0, 1e-10);
}

TEST(T4VectorTest, ToVector3_DirectionCase)
{
    d4Vec v(1.0, 2.0, 3.0, 0.0);
    d3Vec v3 = v.toVector3();
    EXPECT_NEAR(v3.x, 1.0, 1e-10);
    EXPECT_NEAR(v3.y, 2.0, 1e-10);
    EXPECT_NEAR(v3.z, 3.0, 1e-10);
}

TEST(T4VectorTest, XYZ)
{
    d4Vec v(1.0, 2.0, 3.0, 4.0);
    d3Vec xyz = v.xyz();
    EXPECT_EQ(xyz.x, 1.0); EXPECT_EQ(xyz.y, 2.0); EXPECT_EQ(xyz.z, 3.0);
}

TEST(T4VectorTest, XY)
{
    d4Vec v(1.0, 2.0, 3.0, 4.0);
    auto xy = v.xy();
    EXPECT_EQ(xy.x, 1.0); EXPECT_EQ(xy.y, 2.0);
}

// ---------------------------------------------------------------------------
// t4Vector arithmetic operators
// ---------------------------------------------------------------------------

TEST(T4VectorTest, AddInPlace)
{
    d4Vec v(1.0, 2.0, 3.0, 4.0);
    v += d4Vec(1.0, 1.0, 1.0, 1.0);
    EXPECT_EQ(v.x, 2.0); EXPECT_EQ(v.y, 3.0); EXPECT_EQ(v.z, 4.0); EXPECT_EQ(v.w, 5.0);
}

TEST(T4VectorTest, SubtractInPlace)
{
    d4Vec v(3.0, 3.0, 3.0, 3.0);
    v -= d4Vec(1.0, 1.0, 1.0, 1.0);
    EXPECT_EQ(v.x, 2.0);
}

TEST(T4VectorTest, ScalarAddInPlace)
{
    d4Vec v(1.0, 2.0, 3.0, 4.0);
    v += 1.0;
    EXPECT_EQ(v.x, 2.0); EXPECT_EQ(v.y, 3.0); EXPECT_EQ(v.z, 4.0); EXPECT_EQ(v.w, 5.0);
}

TEST(T4VectorTest, ScalarSubtractInPlace)
{
    d4Vec v(2.0, 3.0, 4.0, 5.0);
    v -= 1.0;
    EXPECT_EQ(v.x, 1.0); EXPECT_EQ(v.y, 2.0); EXPECT_EQ(v.z, 3.0); EXPECT_EQ(v.w, 4.0);
}

TEST(T4VectorTest, ScalarMultiplyInPlace)
{
    d4Vec v(1.0, 2.0, 3.0, 4.0);
    v *= 2.0;
    EXPECT_EQ(v.x, 2.0); EXPECT_EQ(v.y, 4.0); EXPECT_EQ(v.z, 6.0); EXPECT_EQ(v.w, 8.0);
}

TEST(T4VectorTest, ScalarDivideInPlace)
{
    d4Vec v(4.0, 6.0, 8.0, 10.0);
    v /= 2.0;
    EXPECT_EQ(v.x, 2.0); EXPECT_EQ(v.y, 3.0); EXPECT_EQ(v.z, 4.0); EXPECT_EQ(v.w, 5.0);
}

TEST(T4VectorTest, Add)
{
    d4Vec a(1.0, 2.0, 3.0, 4.0);
    d4Vec b(5.0, 6.0, 7.0, 8.0);
    d4Vec c = a + b;
    EXPECT_EQ(c.x, 6.0); EXPECT_EQ(c.w, 12.0);
}

TEST(T4VectorTest, Subtract)
{
    d4Vec a(5.0, 6.0, 7.0, 8.0);
    d4Vec b(1.0, 2.0, 3.0, 4.0);
    d4Vec c = a - b;
    EXPECT_EQ(c.x, 4.0); EXPECT_EQ(c.w, 4.0);
}

TEST(T4VectorTest, Negate)
{
    d4Vec v(1.0, -2.0, 3.0, -4.0);
    d4Vec n = -v;
    EXPECT_EQ(n.x, -1.0); EXPECT_EQ(n.y, 2.0); EXPECT_EQ(n.z, -3.0); EXPECT_EQ(n.w, 4.0);
}

TEST(T4VectorTest, ScalarMultiply)
{
    d4Vec v(1.0, 2.0, 3.0, 4.0);
    d4Vec w = v * 2.0;
    EXPECT_EQ(w.x, 2.0); EXPECT_EQ(w.y, 4.0); EXPECT_EQ(w.z, 6.0); EXPECT_EQ(w.w, 8.0);
}

TEST(T4VectorTest, ScalarMultiplyLeft)
{
    d4Vec v(1.0, 2.0, 3.0, 4.0);
    d4Vec w = 2.0 * v;
    EXPECT_EQ(w.x, 2.0); EXPECT_EQ(w.w, 8.0);
}

TEST(T4VectorTest, ScalarDivide)
{
    d4Vec v(4.0, 6.0, 8.0, 10.0);
    d4Vec w = v / 2.0;
    EXPECT_EQ(w.x, 2.0); EXPECT_EQ(w.w, 5.0);
}

TEST(T4VectorTest, ScalarAdd)
{
    d4Vec v(1.0, 2.0, 3.0, 4.0);
    d4Vec w = v + 1.0;
    EXPECT_EQ(w.x, 2.0); EXPECT_EQ(w.w, 5.0);
}

TEST(T4VectorTest, ScalarSubtract)
{
    d4Vec v(2.0, 3.0, 4.0, 5.0);
    d4Vec w = v - 1.0;
    EXPECT_EQ(w.x, 1.0); EXPECT_EQ(w.w, 4.0);
}

// ---------------------------------------------------------------------------
// t4Vector comparison / isClose / normalize / invert
// ---------------------------------------------------------------------------

TEST(T4VectorTest, Equality)
{
    d4Vec a(1.0, 2.0, 3.0, 4.0);
    d4Vec b(1.0, 2.0, 3.0, 4.0);
    EXPECT_TRUE(a == b);
}

TEST(T4VectorTest, Inequality)
{
    d4Vec a(1.0, 2.0, 3.0, 4.0);
    d4Vec b(1.0, 2.0, 3.0, 5.0);
    EXPECT_TRUE(a != b);
}

TEST(T4VectorTest, IsClose)
{
    d4Vec a(1.0, 2.0, 3.0, 4.0);
    d4Vec b(1.001, 2.001, 3.001, 4.001);
    EXPECT_TRUE(a.isClose(b, 0.01));
    EXPECT_FALSE(a.isClose(b, 1e-6));
}

TEST(T4VectorTest, Normalize)
{
    d4Vec v(3.0, 0.0, 0.0, 4.0);
    v.normalize();
    EXPECT_NEAR(v.length(), 1.0, 1e-10);
}

TEST(T4VectorTest, StaticNormalize)
{
    d4Vec v(0.0, 3.0, 0.0, 4.0);
    d4Vec n = d4Vec::normalize(v);
    EXPECT_NEAR(n.length(), 1.0, 1e-10);
}

TEST(T4VectorTest, Invert)
{
    d4Vec v(1.0, -2.0, 3.0, -4.0);
    v.invert();
    EXPECT_EQ(v.x, -1.0); EXPECT_EQ(v.y, 2.0);
    EXPECT_EQ(v.z, -3.0); EXPECT_EQ(v.w, 4.0);
}

TEST(T4VectorTest, SubscriptRead)
{
    d4Vec v(1.0, 2.0, 3.0, 4.0);
    EXPECT_EQ(v[0], 1.0); EXPECT_EQ(v[3], 4.0);
}

TEST(T4VectorTest, SubscriptWrite)
{
    d4Vec v;
    v[0] = 5.0; v[3] = 9.0;
    EXPECT_EQ(v.x, 5.0); EXPECT_EQ(v.w, 9.0);
}

// ---------------------------------------------------------------------------
// Cross-type construction
// ---------------------------------------------------------------------------

TEST(T3VectorTest, CrossTypeConstruct_FloatFromDouble)
{
    t3Vector<double> vd(1.5, 2.5, 3.5);
    t3Vector<float>  vf(vd);
    EXPECT_NEAR(vf.x, 1.5f, 1e-5f);
    EXPECT_NEAR(vf.y, 2.5f, 1e-5f);
}

TEST(T4VectorTest, CrossTypeConstruct_FloatFromDouble)
{
    t4Vector<double> vd(1.5, 2.5, 3.5, 4.5);
    t4Vector<float>  vf(vd);
    EXPECT_NEAR(vf.x, 1.5f, 1e-5f);
    EXPECT_NEAR(vf.w, 4.5f, 1e-5f);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
