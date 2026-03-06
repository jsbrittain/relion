/*
 * Unit tests for src/mask.h / src/mask.cpp
 *
 * Covers:
 *   - raisedCosineMask   — circular soft-edge mask generation
 *   - raisedCrownMask    — annular soft-edge mask generation
 *   - softMaskOutsideMap (radius overload)  — spherical soft masking in-place
 *   - softMaskOutsideMap (mask-array overload) — mask-driven solvent replacement
 */

#include <gtest/gtest.h>
#include <cmath>
#include "src/mask.h"
#include "src/multidim_array.h"

// --------------------------------------------------- raisedCosineMask --

class RaisedCosineMaskTest : public ::testing::Test
{
protected:
    static constexpr int N = 30;
    MultidimArray<RFLOAT> mask;

    void SetUp() override
    {
        mask.initZeros(N, N, N);
        // radius=5, radius_p=8, centred at origin
        raisedCosineMask(mask, 5.0, 8.0, 0, 0, 0);
    }
};

TEST_F(RaisedCosineMaskTest, CenterIsOne)
{
    EXPECT_NEAR(A3D_ELEM(mask, 0, 0, 0), 1.0, 1e-10);
}

TEST_F(RaisedCosineMaskTest, DeepInsideRadius_IsOne)
{
    // (0, 0, 2) → distance 2 < 5
    EXPECT_NEAR(A3D_ELEM(mask, 0, 0, 2), 1.0, 1e-10);
}

TEST_F(RaisedCosineMaskTest, FarCorner_IsZero)
{
    // Corner at Xmipp coords (-N/2, -N/2, -N/2) → distance ≈ 26 > 8
    RFLOAT corner = A3D_ELEM(mask, -N/2, -N/2, -N/2);
    EXPECT_NEAR(corner, 0.0, 1e-10);
}

TEST_F(RaisedCosineMaskTest, AllValuesInUnitInterval)
{
    FOR_ALL_ELEMENTS_IN_ARRAY3D(mask)
        EXPECT_GE(A3D_ELEM(mask, k, i, j), 0.0) << "k=" << k << " i=" << i << " j=" << j;
    FOR_ALL_ELEMENTS_IN_ARRAY3D(mask)
        EXPECT_LE(A3D_ELEM(mask, k, i, j), 1.0) << "k=" << k << " i=" << i << " j=" << j;
}

TEST_F(RaisedCosineMaskTest, InTransitionZone_IsInOpenInterval)
{
    // Point at distance ~6.5: between radius=5 and radius_p=8
    // (0, 0, 6) → distance = 6, in transition zone [5, 8]
    RFLOAT v = A3D_ELEM(mask, 0, 0, 6);
    EXPECT_GT(v, 0.0);
    EXPECT_LT(v, 1.0);
}

TEST_F(RaisedCosineMaskTest, RadiallySymmetric)
{
    // Points at the same distance from origin should have the same mask value.
    RFLOAT v1 = A3D_ELEM(mask,  3, 0, 0);
    RFLOAT v2 = A3D_ELEM(mask,  0, 3, 0);
    RFLOAT v3 = A3D_ELEM(mask,  0, 0, 3);
    RFLOAT v4 = A3D_ELEM(mask, -3, 0, 0);
    EXPECT_NEAR(v1, v2, 1e-10);
    EXPECT_NEAR(v1, v3, 1e-10);
    EXPECT_NEAR(v1, v4, 1e-10);
}

TEST_F(RaisedCosineMaskTest, DecreaseWithDistance)
{
    // Along x-axis: d=2 < d=6 → mask should decrease.
    RFLOAT v_inner = A3D_ELEM(mask, 0, 0, 2);
    RFLOAT v_outer = A3D_ELEM(mask, 0, 0, 6);
    EXPECT_GE(v_inner, v_outer);
}

// --------------------------------------------------- raisedCrownMask --

TEST(RaisedCrownMaskTest, InsideCrown_IsOne)
{
    // Crown centred at origin, inner_radius=4, outer_radius=8, width=2
    // → inner_border=2, outer_border=10
    // A point at r=6 (in [4,8]) should be 1.
    MultidimArray<RFLOAT> mask;
    mask.initZeros(30, 30, 30);
    raisedCrownMask(mask, 4.0, 8.0, 2.0, 0.0, 0.0, 0.0);

    RFLOAT v = A3D_ELEM(mask, 0, 0, 6);
    EXPECT_NEAR(v, 1.0, 1e-10);
}

TEST(RaisedCrownMaskTest, Center_IsZero)
{
    // Centre (r=0) is inside inner_border=2 → should be 0.
    MultidimArray<RFLOAT> mask;
    mask.initZeros(30, 30, 30);
    raisedCrownMask(mask, 4.0, 8.0, 2.0, 0.0, 0.0, 0.0);

    RFLOAT v = A3D_ELEM(mask, 0, 0, 0);
    EXPECT_NEAR(v, 0.0, 1e-10);
}

TEST(RaisedCrownMaskTest, FarOutside_IsZero)
{
    MultidimArray<RFLOAT> mask;
    mask.initZeros(30, 30, 30);
    raisedCrownMask(mask, 4.0, 8.0, 2.0, 0.0, 0.0, 0.0);

    // Corner far outside outer_border=10 → 0.
    RFLOAT corner = A3D_ELEM(mask, -14, -14, -14);
    EXPECT_NEAR(corner, 0.0, 1e-10);
}

TEST(RaisedCrownMaskTest, AllValuesInUnitInterval)
{
    MultidimArray<RFLOAT> mask;
    mask.initZeros(30, 30, 30);
    raisedCrownMask(mask, 4.0, 8.0, 2.0, 0.0, 0.0, 0.0);

    FOR_ALL_ELEMENTS_IN_ARRAY3D(mask)
    {
        EXPECT_GE(A3D_ELEM(mask, k, i, j), 0.0);
        EXPECT_LE(A3D_ELEM(mask, k, i, j), 1.0);
    }
}

// ---------------------------------------- softMaskOutsideMap (radius) --

TEST(SoftMaskRadiusTest, InnerValues_Preserved)
{
    // Create a uniform 20x20x20 volume with value 5.0.
    MultidimArray<RFLOAT> vol;
    vol.initZeros(20, 20, 20);
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(vol)
        DIRECT_MULTIDIM_ELEM(vol, n) = 5.0;

    // Apply mask: radius=3, cosine_width=2. Centre is within radius.
    softMaskOutsideMap(vol, 3.0, 2.0);

    // For a uniform volume the background average == 5.0,
    // so everything should still be 5.0 after masking.
    vol.setXmippOrigin();
    EXPECT_NEAR(A3D_ELEM(vol, 0, 0, 0), 5.0, 1e-6);
    EXPECT_NEAR(A3D_ELEM(vol, 0, 0, 2), 5.0, 1e-6);
}

TEST(SoftMaskRadiusTest, OuterValues_SetToBackground)
{
    // Inner sphere = 10.0, outer = 0.0; mask radius = 3, width = 1.
    // avg background (all outer r > 4) = 0.0 → outer stays 0.0 after mask.
    MultidimArray<RFLOAT> vol;
    vol.initZeros(20, 20, 20);
    vol.setXmippOrigin();
    FOR_ALL_ELEMENTS_IN_ARRAY3D(vol)
    {
        RFLOAT r = sqrt(RFLOAT(k*k + i*i + j*j));
        A3D_ELEM(vol, k, i, j) = (r < 3.0) ? 10.0 : 0.0;
    }

    softMaskOutsideMap(vol, 3.0, 1.0);

    // Far corner (r >> 4): should be ~avg background ≈ 0.
    EXPECT_NEAR(A3D_ELEM(vol, -9, -9, -9), 0.0, 0.1);
    // Centre (r=0 < 3): original 10.0 preserved.
    EXPECT_NEAR(A3D_ELEM(vol,  0,  0,  0), 10.0, 1e-6);
}

// -------------------------------------- softMaskOutsideMap (mask array) --

TEST(SoftMaskArrayTest, InsideMask_ValuesPreserved)
{
    // 10x10x10 volume: inside 5×10×10 slab = 7.0, outside = 2.0
    // Mask: inside slab = 1.0, outside = 0.0
    MultidimArray<RFLOAT> vol, msk;
    vol.initZeros(10, 10, 10);
    msk.initZeros(10, 10, 10);

    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(vol)
    {
        long int jj = j;  // physical x-index
        DIRECT_A3D_ELEM(vol, k, i, j) = (jj < 5) ? 7.0 : 2.0;
        DIRECT_A3D_ELEM(msk, k, i, j) = (jj < 5) ? 1.0 : 0.0;
    }

    softMaskOutsideMap(vol, msk);

    // Values inside mask (jj<5) should remain at 7.0.
    EXPECT_NEAR(DIRECT_A3D_ELEM(vol, 0, 0, 0), 7.0, 1e-6);
    EXPECT_NEAR(DIRECT_A3D_ELEM(vol, 3, 5, 2), 7.0, 1e-6);
}

TEST(SoftMaskArrayTest, OutsideMask_ValuesUniform)
{
    // 10x10x10 volume: first 5 columns (x-index) = 7.0, rest = varied
    // Mask: first 5 cols = 1.0, rest = 0.0
    // After masking, rest should all become avg(rest values in original vol).
    MultidimArray<RFLOAT> vol, msk;
    vol.initZeros(10, 10, 10);
    msk.initZeros(10, 10, 10);

    // Outside region: alternating 0.0 and 4.0 → avg = 2.0
    int count = 0;
    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(vol)
    {
        long int jj = j;
        if (jj < 5)
        {
            DIRECT_A3D_ELEM(vol, k, i, j) = 7.0;
            DIRECT_A3D_ELEM(msk, k, i, j) = 1.0;
        }
        else
        {
            DIRECT_A3D_ELEM(vol, k, i, j) = (count++ % 2 == 0) ? 0.0 : 4.0;
            DIRECT_A3D_ELEM(msk, k, i, j) = 0.0;
        }
    }

    softMaskOutsideMap(vol, msk);

    // Outside region: all values should now equal avg = 2.0.
    RFLOAT expected_bg = 2.0;
    for (long int jj = 5; jj < 10; ++jj)
        EXPECT_NEAR(DIRECT_A3D_ELEM(vol, 0, 0, jj), expected_bg, 1e-6)
            << "jj=" << jj;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
