/*
 * Unit tests for src/filename.h / src/filename.cpp
 *
 * Covers:
 *   - FileName constructors
 *   - compose(root, no, ext)  — zero-padded numbered filenames
 *   - compose(no, str)        — "@"-prefixed stack notation
 *   - decompose               — splitting "@"-prefixed filenames
 *   - getExtension            — extension without dot
 *   - getBaseName             — base name without directory or extension
 *   - isInStack               — detects "@" separator
 *   - addExtension / withoutExtension
 *   - beforeFirstOf / afterFirstOf
 *   - toLowercase / toUppercase
 *   - contains
 */

#include <gtest/gtest.h>
#include <string>
#include "src/filename.h"

// ---------------------------------------------------------- constructors --

TEST(FileNameConstructorTest, DefaultConstructor_IsEmpty)
{
    FileName fn;
    EXPECT_EQ(std::string(fn), "");
}

TEST(FileNameConstructorTest, FromString)
{
    FileName fn(std::string("myfile.mrc"));
    EXPECT_EQ(std::string(fn), "myfile.mrc");
}

TEST(FileNameConstructorTest, FromCharPtr)
{
    FileName fn("myfile.mrc");
    EXPECT_EQ(std::string(fn), "myfile.mrc");
}

TEST(FileNameConstructorTest, CopyConstructor)
{
    FileName fn1("myfile.mrc");
    FileName fn2(fn1);
    EXPECT_EQ(std::string(fn1), std::string(fn2));
}

// ---------------------------------------- compose(root, no, ext) --

class ComposeRootTest : public ::testing::Test
{
protected:
    FileName fn;
};

TEST_F(ComposeRootTest, NumberPaddedToEightDigits)
{
    fn.compose("img", 1, "mrc");
    EXPECT_EQ(std::string(fn), "img00000001.mrc");
}

TEST_F(ComposeRootTest, LargerNumber_CorrectPadding)
{
    fn.compose("img", 42, "mrc");
    EXPECT_EQ(std::string(fn), "img00000042.mrc");
}

TEST_F(ComposeRootTest, NoExtension_NoTrailingDot)
{
    fn.compose("img", 1, "");
    // Extension is empty → no dot appended
    EXPECT_EQ(std::string(fn), "img00000001");
}

TEST_F(ComposeRootTest, EmptyRoot_JustNumberAndExtension)
{
    fn.compose("", 5, "star");
    EXPECT_EQ(std::string(fn), "00000005.star");
}

// ------------------------------------- compose(no, str) — stack notation --

class ComposeStackTest : public ::testing::Test
{
protected:
    FileName fn;
};

TEST_F(ComposeStackTest, Number1_ProducesAtPrefix)
{
    fn.compose(1L, std::string("particles.mrcs"));
    EXPECT_EQ(std::string(fn), "00000001@particles.mrcs");
}

TEST_F(ComposeStackTest, NumberIsZeroPadded)
{
    fn.compose(7L, std::string("stack.mrcs"));
    EXPECT_EQ(std::string(fn), "00000007@stack.mrcs");
}

TEST_F(ComposeStackTest, ResultContainsAtSign)
{
    fn.compose(3L, std::string("imgs.mrcs"));
    EXPECT_NE(std::string(fn).find('@'), std::string::npos);
}

// ---------------------------------------------------------------- decompose --

TEST(DecomposeTest, SplitsNumberAndFilename)
{
    FileName fn;
    fn.compose(5L, std::string("movie.mrcs"));

    long int no = 0;
    std::string name;
    fn.decompose(no, name);

    EXPECT_EQ(no,   5);
    EXPECT_EQ(name, "movie.mrcs");
}

TEST(DecomposeTest, LargeNumber_RecoveredExactly)
{
    FileName fn;
    fn.compose(1234L, std::string("tomo.mrc"));

    long int no = 0;
    std::string name;
    fn.decompose(no, name);

    EXPECT_EQ(no,   1234);
    EXPECT_EQ(name, "tomo.mrc");
}

// ------------------------------------------------------------- isInStack --

TEST(IsInStackTest, PureFilename_NotInStack)
{
    FileName fn("particles.mrcs");
    EXPECT_FALSE(fn.isInStack());
}

TEST(IsInStackTest, AtPrefixed_IsInStack)
{
    FileName fn;
    fn.compose(1L, std::string("stack.mrcs"));
    EXPECT_TRUE(fn.isInStack());
}

TEST(IsInStackTest, ExplicitAtSign_IsInStack)
{
    FileName fn("000001@stack.mrcs");
    EXPECT_TRUE(fn.isInStack());
}

// ---------------------------------------------------------- getExtension --

TEST(GetExtensionTest, MrcExtension)
{
    FileName fn("volume.mrc");
    EXPECT_EQ(fn.getExtension(), "mrc");
}

TEST(GetExtensionTest, StarExtension)
{
    FileName fn("particles.star");
    EXPECT_EQ(fn.getExtension(), "star");
}

TEST(GetExtensionTest, NoExtension_ReturnsEmpty)
{
    FileName fn("noext");
    EXPECT_EQ(fn.getExtension(), "");
}

TEST(GetExtensionTest, MultipleExtensions_ReturnsLast)
{
    FileName fn("file.xmp.bak");
    EXPECT_EQ(fn.getExtension(), "bak");
}

// ----------------------------------------------------------- getBaseName --

TEST(GetBaseNameTest, SimpleFilename)
{
    FileName fn("myfile.mrc");
    EXPECT_EQ(fn.getBaseName(), "myfile");
}

TEST(GetBaseNameTest, FilenameWithDirectory)
{
    FileName fn("data/images/myfile.mrc");
    EXPECT_EQ(fn.getBaseName(), "myfile");
}

TEST(GetBaseNameTest, NoExtension)
{
    FileName fn("noext");
    EXPECT_EQ(fn.getBaseName(), "noext");
}

// ------------------------------------------------------- addExtension --

TEST(AddExtensionTest, AddsExtensionWithDot)
{
    FileName fn("file");
    FileName result = fn.addExtension("mrc");
    EXPECT_EQ(std::string(result), "file.mrc");
}

TEST(AddExtensionTest, EmptyExtension_NoChange)
{
    FileName fn("file.mrc");
    FileName result = fn.addExtension("");
    EXPECT_EQ(std::string(result), "file.mrc");
}

// --------------------------------------------------- withoutExtension --

TEST(WithoutExtensionTest, RemovesLastExtension)
{
    FileName fn("file.mrc");
    FileName result = fn.withoutExtension();
    EXPECT_EQ(std::string(result), "file");
}

TEST(WithoutExtensionTest, NoExtension_Unchanged)
{
    FileName fn("noext");
    FileName result = fn.withoutExtension();
    EXPECT_EQ(std::string(result), "noext");
}

TEST(WithoutExtensionTest, MultipleExtensions_RemovesLast)
{
    FileName fn("file.xmp.bak");
    FileName result = fn.withoutExtension();
    EXPECT_EQ(std::string(result), "file.xmp");
}

// ----------------------------------------- beforeFirstOf / afterFirstOf --

TEST(StringSlicingTest, BeforeFirstOf_AtSign)
{
    FileName fn("000001@particles.mrcs");
    FileName before = fn.beforeFirstOf("@");
    EXPECT_EQ(std::string(before), "000001");
}

TEST(StringSlicingTest, AfterFirstOf_AtSign)
{
    FileName fn("000001@particles.mrcs");
    FileName after = fn.afterFirstOf("@");
    EXPECT_EQ(std::string(after), "particles.mrcs");
}

TEST(StringSlicingTest, BeforeLastOf_Dot)
{
    FileName fn("file.xmp.bak");
    FileName before = fn.beforeLastOf(".");
    EXPECT_EQ(std::string(before), "file.xmp");
}

// -------------------------------------------- toLowercase / toUppercase --

TEST(CaseTest, ToLowercase)
{
    FileName fn("MyFile.MRC");
    EXPECT_EQ(std::string(fn.toLowercase()), "myfile.mrc");
}

TEST(CaseTest, ToUppercase)
{
    FileName fn("myfile.mrc");
    EXPECT_EQ(std::string(fn.toUppercase()), "MYFILE.MRC");
}

// ---------------------------------------------------------------- contains --

TEST(ContainsTest, PresentSubstring_ReturnsTrue)
{
    FileName fn("path/to/particles.star");
    EXPECT_TRUE(fn.contains("particles"));
}

TEST(ContainsTest, AbsentSubstring_ReturnsFalse)
{
    FileName fn("path/to/particles.star");
    EXPECT_FALSE(fn.contains("micrographs"));
}

// ----------------------------------------------- removeAllExtensions --

TEST(RemoveAllExtensionsTest, RemovesEverythingAfterFirstDot)
{
    FileName fn("file.xmp.bak");
    FileName result = fn.removeAllExtensions();
    EXPECT_EQ(std::string(result), "file");
}

TEST(RemoveAllExtensionsTest, NoExtension_Unchanged)
{
    FileName fn("noext");
    FileName result = fn.removeAllExtensions();
    EXPECT_EQ(std::string(result), "noext");
}

// ------------------------------------------ endsWith --

TEST(EndsWithTest, MatchingSuffix_ReturnsTrue)
{
    FileName fn("particles.star");
    EXPECT_TRUE(fn.endsWith(".star"));
}

TEST(EndsWithTest, NonMatchingSuffix_ReturnsFalse)
{
    FileName fn("particles.star");
    EXPECT_FALSE(fn.endsWith(".mrc"));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
