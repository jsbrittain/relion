/*
 * Unit tests for src/backprojector.h / src/backprojector.cpp
 *
 * Tests in-memory operations only — no reconstruction or file I/O.
 *
 * Covers:
 *   - Default constructor: compiles and constructs
 *   - Full constructor: sets ori_size, ref_dim, padding_factor
 *   - initialiseDataAndWeight: resizes data and weight to matching dimensions
 *   - initZeros: all elements zero after call
 *   - Assignment operator: deep copy
 *   - clear(): data and weight become empty
 */

#include <gtest/gtest.h>
#include <cmath>
#include "src/backprojector.h"

// Helper: build a minimal 3D BackProjector with C1 symmetry.
static BackProjector makeC1_3D(int ori_size = 32)
{
    return BackProjector(ori_size, /*ref_dim=*/3, /*fn_sym=*/"c1");
}

// ------------------------------------------------------ constructors --

TEST(BackProjectorTest, DefaultConstructor_Compiles)
{
    BackProjector bp;
    // Should not throw; data and weight are empty
    EXPECT_EQ(NZYXSIZE(bp.weight), (size_t)0);
}

TEST(BackProjectorTest, FullConstructor_SetsOriSize)
{
    BackProjector bp = makeC1_3D(32);
    EXPECT_EQ(bp.ori_size, 32);
}

TEST(BackProjectorTest, FullConstructor_SetsRefDim)
{
    BackProjector bp = makeC1_3D(32);
    EXPECT_EQ(bp.ref_dim, 3);
}

TEST(BackProjectorTest, FullConstructor_SetsPaddingFactor)
{
    // Default padding_factor_3d = 2
    BackProjector bp = makeC1_3D(32);
    EXPECT_NEAR(bp.padding_factor, 2.0f, 1e-6f);
}

TEST(BackProjectorTest, FullConstructor_C2Symmetry)
{
    BackProjector bp(32, 3, "c2");
    EXPECT_EQ(bp.ori_size, 32);
    EXPECT_EQ(bp.SL.SymsNo(), 1);  // C2 has 1 non-identity symmetry
}

// ---------------------------------------- initialiseDataAndWeight --

TEST(BackProjectorTest, InitialiseDataAndWeight_DataNonEmpty)
{
    BackProjector bp = makeC1_3D(32);
    bp.initialiseDataAndWeight();

    EXPECT_GT(NZYXSIZE(bp.data), (size_t)0);
}

TEST(BackProjectorTest, InitialiseDataAndWeight_WeightNonEmpty)
{
    BackProjector bp = makeC1_3D(32);
    bp.initialiseDataAndWeight();

    EXPECT_GT(NZYXSIZE(bp.weight), (size_t)0);
}

TEST(BackProjectorTest, InitialiseDataAndWeight_DataAndWeightSameSize)
{
    BackProjector bp = makeC1_3D(32);
    bp.initialiseDataAndWeight();

    EXPECT_EQ(NZYXSIZE(bp.data), NZYXSIZE(bp.weight));
}

// ------------------------------------------------------- initZeros --

TEST(BackProjectorTest, InitZeros_DataAllZero)
{
    BackProjector bp = makeC1_3D(16);
    bp.initZeros();

    bool all_zero = true;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(bp.weight)
    {
        if (DIRECT_MULTIDIM_ELEM(bp.weight, n) != 0.0)
        {
            all_zero = false;
            break;
        }
    }
    EXPECT_TRUE(all_zero);
}

TEST(BackProjectorTest, InitZeros_WeightAllZero)
{
    BackProjector bp = makeC1_3D(16);
    bp.initZeros();

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(bp.weight)
        EXPECT_EQ(DIRECT_MULTIDIM_ELEM(bp.weight, n), 0.0)
            << "non-zero at element " << n;
}

TEST(BackProjectorTest, InitZeros_DataNonEmpty)
{
    BackProjector bp = makeC1_3D(16);
    bp.initZeros();

    EXPECT_GT(NZYXSIZE(bp.data), (size_t)0);
}

// ----------------------------------------------- assignment / copy --

TEST(BackProjectorTest, AssignmentOperator_CopiesOriSize)
{
    BackProjector bp1 = makeC1_3D(32);
    BackProjector bp2;
    bp2 = bp1;

    EXPECT_EQ(bp2.ori_size, bp1.ori_size);
    EXPECT_EQ(bp2.ref_dim, bp1.ref_dim);
    EXPECT_NEAR(bp2.padding_factor, bp1.padding_factor, 1e-6f);
}

TEST(BackProjectorTest, CopyConstructor_CopiesOriSize)
{
    BackProjector bp1 = makeC1_3D(32);
    bp1.initZeros();

    BackProjector bp2(bp1);
    EXPECT_EQ(bp2.ori_size, 32);
    EXPECT_EQ(NZYXSIZE(bp2.data), NZYXSIZE(bp1.data));
}

// ---------------------------------------------------------------- clear --

TEST(BackProjectorTest, Clear_WeightBecomesEmpty)
{
    BackProjector bp = makeC1_3D(32);
    bp.initialiseDataAndWeight();
    ASSERT_GT(NZYXSIZE(bp.weight), (size_t)0);

    bp.clear();
    EXPECT_EQ(NZYXSIZE(bp.weight), (size_t)0);
}

// ---------------------------------------------- 2D BackProjector --

TEST(BackProjectorTest, TwoDimensional_Constructor)
{
    BackProjector bp(32, 2, "c1");
    EXPECT_EQ(bp.ori_size, 32);
    EXPECT_EQ(bp.ref_dim, 2);
}

TEST(BackProjectorTest, TwoDimensional_InitZeros)
{
    BackProjector bp(16, 2, "c1");
    bp.initZeros();
    EXPECT_GT(NZYXSIZE(bp.data), (size_t)0);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
