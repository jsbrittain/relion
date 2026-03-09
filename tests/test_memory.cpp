/*
 * Unit tests for src/memory.h
 *
 * Covers: ask_Tvector / free_Tvector,
 *         ask_Tmatrix / free_Tmatrix,
 *         ask_Tvolume / free_Tvolume,
 *         askMemory / freeMemory
 */

#include <gtest/gtest.h>
#include "src/memory.h"

// ---------------------------------------------------------------------------
// ask_Tvector / free_Tvector
// ---------------------------------------------------------------------------

TEST(MemoryTest, AskFreeVector_Basic)
{
    double* v = nullptr;
    ask_Tvector(v, 0, 9);
    ASSERT_NE(v, nullptr);
    // Write and read back over valid range
    for (int i = 0; i <= 9; i++)
        v[i] = (double)i * 1.5;
    for (int i = 0; i <= 9; i++)
        EXPECT_NEAR(v[i], (double)i * 1.5, 1e-10);
    free_Tvector(v, 0, 9);
    EXPECT_EQ(v, nullptr);
}

TEST(MemoryTest, AskFreeVector_NegativeOffset)
{
    int* v = nullptr;
    // Valid range: v[-3] .. v[3]
    ask_Tvector(v, -3, 3);
    ASSERT_NE(v, nullptr);
    for (int i = -3; i <= 3; i++)
        v[i] = i * 2;
    for (int i = -3; i <= 3; i++)
        EXPECT_EQ(v[i], i * 2);
    free_Tvector(v, -3, 3);
    EXPECT_EQ(v, nullptr);
}

TEST(MemoryTest, AskVector_ZeroSize_ReturnsNull)
{
    float* v = nullptr;
    // nh - nl + 1 = 1, not > 1 → should return NULL
    ask_Tvector(v, 0, 0);
    EXPECT_EQ(v, nullptr);
}

TEST(MemoryTest, FreeVector_NullPointer_Safe)
{
    double* v = nullptr;
    // Should not crash
    free_Tvector(v, 0, 9);
    EXPECT_EQ(v, nullptr);
}

// ---------------------------------------------------------------------------
// ask_Tmatrix / free_Tmatrix
// ---------------------------------------------------------------------------

TEST(MemoryTest, AskFreeMatrix_Basic)
{
    double** m = nullptr;
    ask_Tmatrix(m, 0, 3, 0, 3);
    ASSERT_NE(m, nullptr);
    for (int r = 0; r <= 3; r++)
        for (int c = 0; c <= 3; c++)
            m[r][c] = (double)(r * 4 + c);
    for (int r = 0; r <= 3; r++)
        for (int c = 0; c <= 3; c++)
            EXPECT_NEAR(m[r][c], (double)(r * 4 + c), 1e-10);
    free_Tmatrix(m, 0, 3, 0, 3);
    EXPECT_EQ(m, nullptr);
}

TEST(MemoryTest, AskMatrix_ZeroSize_ReturnsNull)
{
    int** m = nullptr;
    // nrh - nrl + 1 = 1, not > 1
    ask_Tmatrix(m, 0, 0, 0, 5);
    EXPECT_EQ(m, nullptr);
}

TEST(MemoryTest, FreeMatrix_NullPointer_Safe)
{
    double** m = nullptr;
    free_Tmatrix(m, 0, 3, 0, 3);
    EXPECT_EQ(m, nullptr);
}

// ---------------------------------------------------------------------------
// ask_Tvolume / free_Tvolume
// ---------------------------------------------------------------------------

TEST(MemoryTest, AskFreeVolume_Basic)
{
    float*** vol = nullptr;
    ask_Tvolume(vol, 0, 2, 0, 2, 0, 2);
    ASSERT_NE(vol, nullptr);
    for (int k = 0; k <= 2; k++)
        for (int j = 0; j <= 2; j++)
            for (int i = 0; i <= 2; i++)
                vol[k][j][i] = (float)(k * 9 + j * 3 + i);
    for (int k = 0; k <= 2; k++)
        for (int j = 0; j <= 2; j++)
            for (int i = 0; i <= 2; i++)
                EXPECT_NEAR(vol[k][j][i], (float)(k * 9 + j * 3 + i), 1e-5f);
    free_Tvolume(vol, 0, 2, 0, 2, 0, 2);
    EXPECT_EQ(vol, nullptr);
}

TEST(MemoryTest, AskVolume_ZeroSize_ReturnsNull)
{
    int*** vol = nullptr;
    ask_Tvolume(vol, 0, 0, 0, 2, 0, 2);
    EXPECT_EQ(vol, nullptr);
}

TEST(MemoryTest, FreeVolume_NullPointer_Safe)
{
    double*** vol = nullptr;
    free_Tvolume(vol, 0, 2, 0, 2, 0, 2);
    EXPECT_EQ(vol, nullptr);
}

// ---------------------------------------------------------------------------
// askMemory / freeMemory
// ---------------------------------------------------------------------------

TEST(MemoryTest, AskFreeMemory_Basic)
{
    char* ptr = askMemory(64);
    ASSERT_NE(ptr, nullptr);
    // Write and read back
    for (int i = 0; i < 64; i++)
        ptr[i] = (char)i;
    for (int i = 0; i < 64; i++)
        EXPECT_EQ((unsigned char)ptr[i], (unsigned char)i);
    int ret = freeMemory(ptr, 64);
    EXPECT_EQ(ret, 0);
}

TEST(MemoryTest, AskMemory_LargeAllocation)
{
    // 1 MB allocation
    char* ptr = askMemory(1024 * 1024);
    ASSERT_NE(ptr, nullptr);
    ptr[0] = 42;
    ptr[1024 * 1024 - 1] = 99;
    EXPECT_EQ((int)(unsigned char)ptr[0], 42);
    EXPECT_EQ((int)(unsigned char)ptr[1024 * 1024 - 1], 99);
    freeMemory(ptr, 1024 * 1024);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
