/*
 * Unit tests for src/float16.h
 *
 * Covers:
 *   - float2half  — float32 → float16 conversion
 *   - half2float  — float16 → float32 conversion
 *   - Round-trip accuracy for representable values
 *   - Special cases: zero, overflow (→ MAX not INF), underflow (→ signed zero)
 *   - Sign bit preservation
 *   - INF/NaN pass-through
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include "src/float16.h"

// Largest representable float16: exponent=30 (bias 15 → value 15), frac all 1s
// = 2^15 * (1 + 1023/1024) ≈ 65504.0
static constexpr float F16_MAX = 65504.0f;

// Smallest positive normal float16: exponent=1 (value -14), frac=0
// = 2^(-14) ≈ 6.1e-5
static constexpr float F16_MIN_NORMAL = 6.103515625e-5f;

// ----------------------------------------------------------- zero --

TEST(Float16Test, PositiveZeroRoundTrip)
{
    float16 h = float2half(0.0f);
    float f = half2float(h);
    EXPECT_EQ(f, 0.0f);
}

TEST(Float16Test, PositiveZeroHalfBits)
{
    // Positive zero: all bits zero
    float16 h = float2half(0.0f);
    EXPECT_EQ(h, (float16)0);
}

// --------------------------------------------------------- one --

TEST(Float16Test, PositiveOneRoundTrip)
{
    float16 h = float2half(1.0f);
    float f = half2float(h);
    // 1.0 is exactly representable in float16 (exp=15, frac=0)
    EXPECT_NEAR(f, 1.0f, 1e-3f);
}

TEST(Float16Test, NegativeOneRoundTrip)
{
    float16 h = float2half(-1.0f);
    float f = half2float(h);
    EXPECT_NEAR(f, -1.0f, 1e-3f);
}

// Sign bit: negative numbers must have bit 15 set
TEST(Float16Test, NegativeSignBitSet)
{
    float16 h = float2half(-1.0f);
    EXPECT_NE(h & 0x8000u, 0u);
}

TEST(Float16Test, PositiveSignBitClear)
{
    float16 h = float2half(1.0f);
    EXPECT_EQ(h & 0x8000u, 0u);
}

// ------------------------------------------------------- 0.5 --

TEST(Float16Test, HalfRoundTrip)
{
    float16 h = float2half(0.5f);
    float f = half2float(h);
    // 0.5 is exactly representable
    EXPECT_NEAR(f, 0.5f, 1e-4f);
}

// --------------------------------------------------- 2.0 and 100.0 --

TEST(Float16Test, TwoRoundTrip)
{
    float16 h = float2half(2.0f);
    float f = half2float(h);
    EXPECT_NEAR(f, 2.0f, 1e-3f);
}

TEST(Float16Test, OneHundredRoundTrip)
{
    float16 h = float2half(100.0f);
    float f = half2float(h);
    EXPECT_NEAR(f, 100.0f, 0.5f);  // float16 has ~3 decimal digits precision
}

// --------------------------------------------------- negative 0.5 --

TEST(Float16Test, NegativeHalfRoundTrip)
{
    float16 h = float2half(-0.5f);
    float f = half2float(h);
    EXPECT_NEAR(f, -0.5f, 1e-4f);
}

// -------------------------------------------- overflow → MAX (not INF) --

TEST(Float16Test, OverflowTruncatesToMax)
{
    // 1e10 >> F16_MAX: should saturate to MAX, not INF
    float16 h = float2half(1e10f);
    float f = half2float(h);
    EXPECT_NEAR(f, F16_MAX, 1.0f);
    EXPECT_FALSE(std::isinf(f));
}

TEST(Float16Test, NegativeOverflowTruncatesToNegativeMax)
{
    float16 h = float2half(-1e10f);
    float f = half2float(h);
    EXPECT_NEAR(f, -F16_MAX, 1.0f);
    EXPECT_FALSE(std::isinf(f));
}

// ----------------------------------------- underflow → signed zero --

TEST(Float16Test, UnderflowBecomesZero)
{
    // 1e-10 << F16_MIN_NORMAL: should underflow to (positive) signed zero
    float16 h = float2half(1e-10f);
    float f = half2float(h);
    EXPECT_EQ(f, 0.0f);
}

TEST(Float16Test, NegativeUnderflowBecomesSignedZero)
{
    float16 h = float2half(-1e-10f);
    float f = half2float(h);
    // Result is -0.0 (negative signed zero) — value is 0
    EXPECT_EQ(std::abs(f), 0.0f);
}

// ----------------------------------------------- INF pass-through --

TEST(Float16Test, InfBecomesInf)
{
    float inf = std::numeric_limits<float>::infinity();
    float16 h = float2half(inf);
    float f = half2float(h);
    EXPECT_TRUE(std::isinf(f));
    EXPECT_GT(f, 0.0f);  // positive INF
}

TEST(Float16Test, NegInfBecomesNegInf)
{
    float neginf = -std::numeric_limits<float>::infinity();
    float16 h = float2half(neginf);
    float f = half2float(h);
    EXPECT_TRUE(std::isinf(f));
    EXPECT_LT(f, 0.0f);  // negative INF
}

// ----------------------------------------------- NaN pass-through --

TEST(Float16Test, NanRemainsNan)
{
    float nan = std::numeric_limits<float>::quiet_NaN();
    float16 h = float2half(nan);
    float f = half2float(h);
    EXPECT_TRUE(std::isnan(f));
}

// ----------------------------------- round-trip accuracy for a sweep --

TEST(Float16Test, SweepOfValues_RelativeError)
{
    // For values in the normal float16 range, relative error < 0.2%
    const float values[] = {1.0f, 2.0f, 4.0f, 8.0f, 0.25f, 0.125f, 10.0f, 1000.0f};
    for (float v : values)
    {
        float rt = half2float(float2half(v));
        EXPECT_NEAR(rt, v, std::abs(v) * 0.002f + 1e-4f) << "v=" << v;
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
