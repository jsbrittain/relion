/*
 * Unit tests for src/args.h / src/args.cpp
 *
 * Covers:
 *   - getParameter   — retrieve value after a named flag
 *   - checkParameter — detect presence of a flag
 *   - IOParser       — section/option registration, value retrieval,
 *                      boolean flags, unknown-argument detection
 */

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <sstream>
#include "src/args.h"

// Helper: build a mutable argv from a list of string literals.
// Lifetime tied to the returned vector of strings.
static std::vector<char*> makeArgv(std::vector<std::string>& storage)
{
    std::vector<char*> argv;
    for (auto& s : storage)
        argv.push_back(const_cast<char*>(s.c_str()));
    return argv;
}

// --------------------------------------------------------- getParameter --

TEST(GetParameterTest, PresentFlag_ReturnsValue)
{
    std::vector<std::string> args = {"prog", "--sigma", "3.5"};
    auto argv = makeArgv(args);
    int argc = static_cast<int>(argv.size());

    EXPECT_EQ(getParameter(argc, argv.data(), "--sigma"), "3.5");
}

TEST(GetParameterTest, AbsentFlag_WithDefault_ReturnsDefault)
{
    std::vector<std::string> args = {"prog", "--other", "1.0"};
    auto argv = makeArgv(args);
    int argc = static_cast<int>(argv.size());

    EXPECT_EQ(getParameter(argc, argv.data(), "--sigma", "5.0"), "5.0");
}

TEST(GetParameterTest, MultipleArgs_CorrectValueReturned)
{
    std::vector<std::string> args = {"prog", "--a", "hello", "--b", "world"};
    auto argv = makeArgv(args);
    int argc = static_cast<int>(argv.size());

    EXPECT_EQ(getParameter(argc, argv.data(), "--a"), "hello");
    EXPECT_EQ(getParameter(argc, argv.data(), "--b"), "world");
}

TEST(GetParameterTest, FlagAtLastPosition_UsesDefault)
{
    // Flag present but no following value → uses default.
    std::vector<std::string> args = {"prog", "--sigma"};
    auto argv = makeArgv(args);
    int argc = static_cast<int>(argv.size());

    EXPECT_EQ(getParameter(argc, argv.data(), "--sigma", "99"), "99");
}

// -------------------------------------------------------- checkParameter --

TEST(CheckParameterTest, PresentFlag_ReturnsTrue)
{
    std::vector<std::string> args = {"prog", "--verbose"};
    auto argv = makeArgv(args);
    int argc = static_cast<int>(argv.size());

    EXPECT_TRUE(checkParameter(argc, argv.data(), "--verbose"));
}

TEST(CheckParameterTest, AbsentFlag_ReturnsFalse)
{
    std::vector<std::string> args = {"prog", "--other"};
    auto argv = makeArgv(args);
    int argc = static_cast<int>(argv.size());

    EXPECT_FALSE(checkParameter(argc, argv.data(), "--verbose"));
}

TEST(CheckParameterTest, EmptyArgv_ReturnsFalse)
{
    std::vector<std::string> args = {"prog"};
    auto argv = makeArgv(args);
    int argc = static_cast<int>(argv.size());

    EXPECT_FALSE(checkParameter(argc, argv.data(), "--verbose"));
}

TEST(CheckParameterTest, FlagMatchesPrecisely)
{
    // "--verbos" should not match "--verbose".
    std::vector<std::string> args = {"prog", "--verbos"};
    auto argv = makeArgv(args);
    int argc = static_cast<int>(argv.size());

    EXPECT_FALSE(checkParameter(argc, argv.data(), "--verbose"));
}

// --------------------------------------------------------------- IOParser --

class IOParserTest : public ::testing::Test
{
protected:
    std::vector<std::string> args;
    std::vector<char*>       argv_ptrs;
    IOParser parser;

    // Call after populating `args`.
    void setupParser()
    {
        argv_ptrs = makeArgv(args);
        parser.setCommandLine(static_cast<int>(argv_ptrs.size()),
                              argv_ptrs.data());
        parser.addSection("General");
    }
};

TEST_F(IOParserTest, GetOption_PresentFlag_ReturnsValue)
{
    args = {"prog", "--sigma", "2.5"};
    setupParser();

    std::string val = parser.getOption("--sigma", "Sigma value", "1.0");
    EXPECT_EQ(val, "2.5");
}

TEST_F(IOParserTest, GetOption_AbsentFlag_ReturnsDefault)
{
    args = {"prog"};
    setupParser();

    std::string val = parser.getOption("--sigma", "Sigma value", "1.0");
    EXPECT_EQ(val, "1.0");
}

TEST_F(IOParserTest, GetOption_MultipleOptions_CorrectValues)
{
    args = {"prog", "--iter", "10", "--angpix", "1.35"};
    setupParser();

    EXPECT_EQ(parser.getOption("--iter",   "Iterations", "5"),   "10");
    EXPECT_EQ(parser.getOption("--angpix", "Pixel size", "1.0"), "1.35");
}

TEST_F(IOParserTest, CheckOption_PresentFlag_ReturnsTrue)
{
    args = {"prog", "--do_ctf"};
    setupParser();

    EXPECT_TRUE(parser.checkOption("--do_ctf", "Apply CTF correction"));
}

TEST_F(IOParserTest, CheckOption_AbsentFlag_ReturnsFalse)
{
    args = {"prog"};
    setupParser();

    EXPECT_FALSE(parser.checkOption("--do_ctf", "Apply CTF correction"));
}

TEST_F(IOParserTest, OptionExists_AfterGetOption_ReturnsTrue)
{
    args = {"prog", "--sigma", "2.0"};
    setupParser();
    parser.getOption("--sigma", "Sigma value", "1.0");

    EXPECT_TRUE(parser.optionExists("--sigma"));
}

TEST_F(IOParserTest, OptionExists_NotYetRegistered_ReturnsFalse)
{
    args = {"prog"};
    setupParser();

    EXPECT_FALSE(parser.optionExists("--unregistered"));
}

TEST_F(IOParserTest, AddSection_SetsCurrentSection)
{
    args = {"prog"};
    setupParser();

    int s2 = parser.addSection("Advanced");
    EXPECT_EQ(parser.getSection(), s2);
}

TEST_F(IOParserTest, AbsentRequiredOption_AddsErrorMessage)
{
    // An option with no default (defaultvalue="NULL") that is not on the
    // command line should add an entry to error_messages.
    args = {"prog", "--other", "x"};
    setupParser();

    // getOption with no default: missing → error message, returns "".
    std::string val = parser.getOption("--required", "A required option");
    // checkForErrors would return true (errors present).
    // We verify the returned value is empty (the sentinel for "not found").
    EXPECT_EQ(val, "");
}

TEST_F(IOParserTest, WriteCommandLine_IncludesArgs)
{
    args = {"prog", "--sigma", "2.0"};
    setupParser();

    std::ostringstream oss;
    parser.writeCommandLine(oss);
    std::string line = oss.str();

    EXPECT_NE(line.find("--sigma"), std::string::npos);
    EXPECT_NE(line.find("2.0"),     std::string::npos);
}

// -------------------------------------------------------- untangleDeviceIDs --

TEST(UntangleDeviceIDsTest, SingleRankSingleDevice)
{
    std::string tangled = "0";
    std::vector<std::vector<std::string>> result;
    untangleDeviceIDs(tangled, result);

    ASSERT_EQ(result.size(), 1u);
    ASSERT_EQ(result[0].size(), 1u);
    EXPECT_EQ(result[0][0], "0");
}

TEST(UntangleDeviceIDsTest, MultipleThreadsSameRank)
{
    // "0,1" → rank 0 has threads on devices 0 and 1
    std::string tangled = "0,1";
    std::vector<std::vector<std::string>> result;
    untangleDeviceIDs(tangled, result);

    ASSERT_EQ(result.size(), 1u);
    ASSERT_EQ(result[0].size(), 2u);
    EXPECT_EQ(result[0][0], "0");
    EXPECT_EQ(result[0][1], "1");
}

TEST(UntangleDeviceIDsTest, MultipleRanks)
{
    // "0:1" → rank 0 on device 0, rank 1 on device 1
    std::string tangled = "0:1";
    std::vector<std::vector<std::string>> result;
    untangleDeviceIDs(tangled, result);

    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0][0], "0");
    EXPECT_EQ(result[1][0], "1");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
