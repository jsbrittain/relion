/*
 * Unit tests for src/jaz/util/zio.cpp
 *
 * Covers: ZIO::split, ZIO::itoa, ZIO::beginsWith, ZIO::endsWith,
 *         ZIO::ensureEndingSlash, ZIO::fileExists,
 *         ZIO::readDoubles, ZIO::readInts,
 *         ZIO::readFixedDoublesTable, ZIO::readDoublesTable
 */

#include <gtest/gtest.h>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include "src/jaz/util/zio.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string writeTempFile(const std::string& content)
{
    char path[] = "/tmp/test_zio_XXXXXX";
    int fd = mkstemp(path);
    FILE* f = fdopen(fd, "w");
    fputs(content.c_str(), f);
    fclose(f);
    return std::string(path);
}

// ---------------------------------------------------------------------------
// ZIO::split
// ---------------------------------------------------------------------------

TEST(ZIOTest, Split_BasicDelimiter)
{
    std::string s = "a,b,c";
    auto parts = ZIO::split(s, ",");
    ASSERT_EQ(parts.size(), (size_t)3);
    EXPECT_EQ(parts[0], "a");
    EXPECT_EQ(parts[1], "b");
    EXPECT_EQ(parts[2], "c");
}

TEST(ZIOTest, Split_SingleToken)
{
    std::string s = "hello";
    auto parts = ZIO::split(s, ",");
    ASSERT_EQ(parts.size(), (size_t)1);
    EXPECT_EQ(parts[0], "hello");
}

// ---------------------------------------------------------------------------
// ZIO::itoa
// ---------------------------------------------------------------------------

TEST(ZIOTest, Itoa_Integer)
{
    std::string s = ZIO::itoa(42.0);
    EXPECT_EQ(s, "42");
}

TEST(ZIOTest, Itoa_Double)
{
    std::string s = ZIO::itoa(3.14);
    EXPECT_FALSE(s.empty());
    EXPECT_NE(s.find('3'), std::string::npos);
}

// ---------------------------------------------------------------------------
// ZIO::beginsWith / endsWith
// ---------------------------------------------------------------------------

TEST(ZIOTest, BeginsWith_True)
{
    EXPECT_TRUE(ZIO::beginsWith("hello world", "hello"));
}

TEST(ZIOTest, BeginsWith_False)
{
    EXPECT_FALSE(ZIO::beginsWith("hello world", "world"));
}

TEST(ZIOTest, BeginsWith_EmptyPrefix)
{
    EXPECT_TRUE(ZIO::beginsWith("hello", ""));
}

TEST(ZIOTest, EndsWith_True)
{
    EXPECT_TRUE(ZIO::endsWith("hello world", "world"));
}

TEST(ZIOTest, EndsWith_False)
{
    EXPECT_FALSE(ZIO::endsWith("hello world", "hello"));
}

TEST(ZIOTest, EndsWith_EmptySuffix)
{
    EXPECT_TRUE(ZIO::endsWith("hello", ""));
}

// ---------------------------------------------------------------------------
// ZIO::ensureEndingSlash
// ---------------------------------------------------------------------------

TEST(ZIOTest, EnsureEndingSlash_AddsSlash)
{
    std::string r = ZIO::ensureEndingSlash("mydir");
    EXPECT_EQ(r, "mydir/");
}

TEST(ZIOTest, EnsureEndingSlash_AlreadyHasSlash)
{
    std::string r = ZIO::ensureEndingSlash("mydir/");
    EXPECT_EQ(r, "mydir/");
}

TEST(ZIOTest, EnsureEndingSlash_EmptyString)
{
    std::string r = ZIO::ensureEndingSlash("");
    EXPECT_EQ(r, "");
}

// ---------------------------------------------------------------------------
// ZIO::fileExists
// ---------------------------------------------------------------------------

TEST(ZIOTest, FileExists_ExistingFile)
{
    std::string path = writeTempFile("content");
    EXPECT_TRUE(ZIO::fileExists(path));
    std::remove(path.c_str());
}

TEST(ZIOTest, FileExists_NonExistentFile)
{
    EXPECT_FALSE(ZIO::fileExists("/tmp/this_file_does_not_exist_zio_test_xyz.txt"));
}

// ---------------------------------------------------------------------------
// ZIO::readDoubles
// ---------------------------------------------------------------------------

TEST(ZIOTest, ReadDoubles_BasicValues)
{
    std::string path = writeTempFile("1.5\n2.5\n3.5\n");
    auto v = ZIO::readDoubles(path);
    ASSERT_EQ(v.size(), (size_t)3);
    EXPECT_NEAR(v[0], 1.5, 1e-10);
    EXPECT_NEAR(v[1], 2.5, 1e-10);
    EXPECT_NEAR(v[2], 3.5, 1e-10);
    std::remove(path.c_str());
}

TEST(ZIOTest, ReadDoubles_SingleValue)
{
    std::string path = writeTempFile("42.0\n");
    auto v = ZIO::readDoubles(path);
    ASSERT_EQ(v.size(), (size_t)1);
    EXPECT_NEAR(v[0], 42.0, 1e-10);
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// ZIO::readInts
// ---------------------------------------------------------------------------

TEST(ZIOTest, ReadInts_BasicValues)
{
    std::string path = writeTempFile("10\n20\n30\n");
    auto v = ZIO::readInts(path);
    ASSERT_EQ(v.size(), (size_t)3);
    EXPECT_EQ(v[0], 10);
    EXPECT_EQ(v[1], 20);
    EXPECT_EQ(v[2], 30);
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// ZIO::readFixedDoublesTable
// ---------------------------------------------------------------------------

TEST(ZIOTest, ReadFixedDoublesTable_SpaceDelim)
{
    std::string path = writeTempFile("1.0 2.0 3.0\n4.0 5.0 6.0\n");
    auto t = ZIO::readFixedDoublesTable(path, 3, ' ');
    ASSERT_EQ(t.size(), (size_t)2);
    ASSERT_EQ(t[0].size(), (size_t)3);
    EXPECT_NEAR(t[0][0], 1.0, 1e-10);
    EXPECT_NEAR(t[0][2], 3.0, 1e-10);
    EXPECT_NEAR(t[1][1], 5.0, 1e-10);
    std::remove(path.c_str());
}

TEST(ZIOTest, ReadFixedDoublesTable_CommaDelim)
{
    std::string path = writeTempFile("1.0,2.0,3.0\n4.0,5.0,6.0\n");
    auto t = ZIO::readFixedDoublesTable(path, 3, ',');
    ASSERT_EQ(t.size(), (size_t)2);
    EXPECT_NEAR(t[0][0], 1.0, 1e-10);
    EXPECT_NEAR(t[1][2], 6.0, 1e-10);
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// ZIO::readDoublesTable
// ---------------------------------------------------------------------------

TEST(ZIOTest, ReadDoublesTable_VariableColumns)
{
    std::string path = writeTempFile("1.0 2.0\n3.0 4.0 5.0\n");
    auto t = ZIO::readDoublesTable(path, ' ');
    ASSERT_EQ(t.size(), (size_t)2);
    EXPECT_EQ(t[0].size(), (size_t)2);
    EXPECT_EQ(t[1].size(), (size_t)3);
    EXPECT_NEAR(t[0][0], 1.0, 1e-10);
    EXPECT_NEAR(t[1][2], 5.0, 1e-10);
    std::remove(path.c_str());
}

TEST(ZIOTest, ReadDoublesTable_CommaDelim)
{
    std::string path = writeTempFile("10.0,20.0\n30.0,40.0\n");
    auto t = ZIO::readDoublesTable(path, ',');
    ASSERT_EQ(t.size(), (size_t)2);
    EXPECT_NEAR(t[0][1], 20.0, 1e-10);
    EXPECT_NEAR(t[1][0], 30.0, 1e-10);
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
