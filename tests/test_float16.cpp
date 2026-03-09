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

// ---- Rounding carry: fractional &= 0x007fffffu; exponent++ (float16.h:123-124) ----
//
// When the 23-bit float32 mantissa, after the rounding increment (+1<<12),
// overflows into bit 23, the code does two things:
//   (a) fractional &= 0x007fffffu  — clears the carry bit so it does not leak
//       into the float16 exponent field when the mantissa is later shifted by 13.
//   (b) exponent++                 — promotes the value to the next power of two,
//       as required by IEEE 754: 1.111...1 rounded up == 10.000...0.
//
// The carry fires when fractional >= 0x7FF000, because
//   0x7FF000 + 0x001000 (rounding constant) = 0x800000 (bit 23 set).
// After the carry the remaining mantissa bits always fall below bit 13,
// so (fractional & 0x007fffffu) >> 13 == 0, and the float16 fractional is zero.
// The result is therefore always an exact power of two (mantissa == 1.0).
//
// Reference input: 1.99951171875 = 4095/2048 = 1 + 2047/2048, which has
//   float32 biased exponent = 127 (actual = 0), fractional = 0x7FF000.

TEST(Float16Test, RoundingCarry_ExponentIncremented)
{
    // 1.99951171875f has float32 biased exp=127, frac=0x7FF000 (minimum
    // carry-triggering fractional: bits [22:12] all set).
    // Rounding: 0x7FF000 + 0x001000 = 0x800000 → bit 23 set → carry fires.
    //   frac &= 0x007fffff → 0x000000    (carry bit cleared)
    //   exponent++         → 128         (actual exponent 1)
    // float16 result: biased exp = 128+15-127 = 16, frac = 0 → value = 2.0.
    float16 h = float2half(1.99951171875f);
    EXPECT_EQ(h & 0x7C00u, 16u << 10); // exponent field incremented to 16
    EXPECT_EQ(h & 0x03FFu, 0u);         // fractional field zero after carry
    EXPECT_EQ(h & 0x8000u, 0u);         // positive
    EXPECT_NEAR(half2float(h), 2.0f, 1e-6f);
}

TEST(Float16Test, RoundingNoCarry_BelowCarryThreshold)
{
    // 1.9990234375f = 1 + 1023/1024 has float32 biased exp=127, frac=0x7FE000
    // — one step below the carry threshold.
    // Rounding: 0x7FE000 + 0x001000 = 0x7FF000 < 0x800000 → no carry.
    //   exponent stays at 127, float16 biased exp = 15.
    //   frac >> 13 = 0x7FF000 >> 13 = 0x3FF (all 10 float16 mantissa bits set).
    // Contrast with the carry case above: same input exponent, but the float16
    // exponent is 15 (not 16) and the fractional is 0x3FF (not 0).
    float16 h = float2half(1.9990234375f);
    EXPECT_EQ(h & 0x7C00u, 15u << 10); // exponent unchanged at 15
    EXPECT_EQ(h & 0x03FFu, 0x3FFu);    // fractional = all 10 bits set
    EXPECT_NEAR(half2float(h), 1.9990234375f, 1e-6f);
}

TEST(Float16Test, RoundingCarry_AllOnesManitssa)
{
    // Construct a float32 with biased exp=127 and fractional=0x7FFFFF
    // (all 23 mantissa bits set — the float32 value nearest to 2.0 from below).
    // Rounding: 0x7FFFFF + 0x001000 = 0x800FFF → bit 23 set → carry fires.
    //   frac &= 0x007fffff → 0x000FFF  (low 12 bits remain, but >> 13 gives 0)
    //   exponent++ → 128; float16 = 2.0.
    // This confirms that (a) the mask correctly isolates only bits [22:0]
    // and (b) the remaining fractional bits [11:0] do not contribute to the
    // float16 mantissa after the >> 13 shift.
    float32 f;
    f.i = (127u << 23) | 0x7FFFFFu;
    float16 h = float2half(f.f);
    EXPECT_EQ(h & 0x7C00u, 16u << 10); // exponent incremented to 16
    EXPECT_EQ(h & 0x03FFu, 0u);         // fractional zero despite non-zero low bits
    EXPECT_NEAR(half2float(h), 2.0f, 1e-6f);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
