/*
 * GoogleTest unit tests for string utility functions (src/strings.h/.cpp).
 *
 * Build and run:
 *   cmake -DBUILD_TESTS=ON ...
 *   make test_strings
 *   ./build/bin/test_strings
 *
 * No MPI required; pure unit tests with no file I/O.
 *
 * What is tested:
 *   1.  removeChar()   – removes all occurrences of a character.
 *   2.  unescape()     – removes control characters; converts tab → space.
 *   3.  escapeStringForSTAR() – empty → "\"\""; whitespace/leading-quote →
 *                               wrapped in double-quotes.
 *   4.  bestPrecision() – returns correct precision for printf format.
 *   5.  isNumber()     – true for numeric strings, false otherwise.
 *   6.  textToDouble() – string to double conversion.
 *   7.  textToFloat()  – string to float conversion.
 *   8.  textToInteger() – string to int conversion.
 *   9.  textToBool()   – "true"/"yes" → true, "false"/"no" → false.
 *  10.  textToLongLong() – string to long long conversion.
 *  11.  floatToString() – float to formatted string.
 *  12.  integerToString() – int to string with optional zero-padding.
 *  13.  stringToString() – truncate or right-pad with spaces.
 *  14.  toLower()      – char* and std::string overloads.
 *  15.  simplify()     – collapses whitespace and strips leading/trailing.
 *  16.  trim()         – removes leading and trailing spaces in-place.
 *  17.  trim2()        – removes leading/trailing spaces and newlines.
 *  18.  removeSpaces() – collapses multiple spaces; strips leading/trailing.
 *  19.  splitString()  – splits by delimiter, excludes empties by default.
 *  20.  tokenize()     – splits by whitespace delimiters.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include "src/strings.h"
#include "src/error.h"

// ---------------------------------------------------------------------------
// 1. removeChar()
// ---------------------------------------------------------------------------
TEST(StringsTest, RemoveCharAll)
{
    EXPECT_EQ(removeChar("hello", 'l'), "heo");
}

TEST(StringsTest, RemoveCharNonePresent)
{
    EXPECT_EQ(removeChar("hello", 'z'), "hello");
}

TEST(StringsTest, RemoveCharAll2)
{
    EXPECT_EQ(removeChar("aaaa", 'a'), "");
}

TEST(StringsTest, RemoveCharEmpty)
{
    EXPECT_EQ(removeChar("", 'x'), "");
}

// ---------------------------------------------------------------------------
// 2. unescape()
// ---------------------------------------------------------------------------
TEST(StringsTest, UnescapeTabBecomesSpace)
{
    EXPECT_EQ(unescape("a\tb"), "a b");
}

TEST(StringsTest, UnescapeNewlineRemoved)
{
    EXPECT_EQ(unescape("a\nb"), "ab");
}

TEST(StringsTest, UnescapeCarriageReturnRemoved)
{
    EXPECT_EQ(unescape("a\rb"), "ab");
}

TEST(StringsTest, UnescapeNormalStringUnchanged)
{
    EXPECT_EQ(unescape("hello world"), "hello world");
}

TEST(StringsTest, UnescapeMultipleControlChars)
{
    // \t → space, \n removed, \r removed
    std::string result = unescape("a\tb\nc\rd");
    EXPECT_EQ(result, "a bcd");
}

// ---------------------------------------------------------------------------
// 3. escapeStringForSTAR()
// ---------------------------------------------------------------------------
TEST(StringsTest, EscapeEmptyStringBecomesQuotedEmpty)
{
    std::string s = "";
    escapeStringForSTAR(s);
    EXPECT_EQ(s, "\"\"");
}

TEST(StringsTest, EscapeStringWithoutSpaceUnchanged)
{
    std::string s = "hello";
    escapeStringForSTAR(s);
    EXPECT_EQ(s, "hello");
}

TEST(StringsTest, EscapeStringWithSpaceGetsWrapped)
{
    std::string s = "hello world";
    escapeStringForSTAR(s);
    EXPECT_EQ(s, "\"hello world\"");
}

TEST(StringsTest, EscapeStringStartingWithDoubleQuoteGetsWrapped)
{
    std::string s = "\"quoted";
    escapeStringForSTAR(s);
    // Starts with '"' → must be wrapped in outer double quotes
    EXPECT_EQ(s[0], '"');
    EXPECT_EQ(s.back(), '"');
    EXPECT_GT(s.size(), (size_t)2);  // at least the two wrapping quotes
}

// ---------------------------------------------------------------------------
// 4. bestPrecision()
// ---------------------------------------------------------------------------
TEST(StringsTest, BestPrecisionZeroReturnsOne)
{
    EXPECT_EQ(bestPrecision(0.0f, 10), 1);
}

TEST(StringsTest, BestPrecisionOneReturnsWidth_minus_2)
{
    // F=1.0, exp=0 <= width-3=7; advised_prec = width-2 = 8
    EXPECT_EQ(bestPrecision(1.0f, 10), 8);
}

TEST(StringsTest, BestPrecisionLargeNumberReturnsMinusOne)
{
    // F=1e8, exp=8 > width-3=7; advised_prec = -1
    EXPECT_EQ(bestPrecision(1e8f, 10), -1);
}

TEST(StringsTest, BestPrecisionVerySmallNumberReturnsMinusOne)
{
    // F=1e-10, exp=-10; advised = 10+(-11)-3 = -4 → clamped to -1
    EXPECT_EQ(bestPrecision(1e-10f, 10), -1);
}

// ---------------------------------------------------------------------------
// 5. isNumber()
// ---------------------------------------------------------------------------
TEST(StringsTest, IsNumberInteger)
{
    EXPECT_TRUE(isNumber("42"));
}

TEST(StringsTest, IsNumberFloat)
{
    EXPECT_TRUE(isNumber("3.14"));
}

TEST(StringsTest, IsNumberNegative)
{
    EXPECT_TRUE(isNumber("-7.5"));
}

TEST(StringsTest, IsNumberWord)
{
    EXPECT_FALSE(isNumber("hello"));
}

TEST(StringsTest, IsNumberShortAlpha)
{
    // Single non-numeric character is not a number
    EXPECT_FALSE(isNumber("x"));
}

// ---------------------------------------------------------------------------
// 6. textToDouble()
// ---------------------------------------------------------------------------
TEST(StringsTest, TextToDoublePositive)
{
    EXPECT_NEAR(textToDouble("3.14159"), 3.14159, 1e-5);
}

TEST(StringsTest, TextToDoubleNegative)
{
    EXPECT_NEAR(textToDouble("-2.5"), -2.5, 1e-9);
}

TEST(StringsTest, TextToDoubleInteger)
{
    EXPECT_NEAR(textToDouble("42"), 42.0, 1e-9);
}

TEST(StringsTest, TextToDoubleScientific)
{
    EXPECT_NEAR(textToDouble("1e-3"), 0.001, 1e-9);
}

// ---------------------------------------------------------------------------
// 7. textToFloat()
// ---------------------------------------------------------------------------
TEST(StringsTest, TextToFloat)
{
    EXPECT_NEAR(textToFloat("2.718"), 2.718f, 1e-3f);
}

// ---------------------------------------------------------------------------
// 8. textToInteger()
// ---------------------------------------------------------------------------
TEST(StringsTest, TextToIntegerPositive)
{
    EXPECT_EQ(textToInteger("123"), 123);
}

TEST(StringsTest, TextToIntegerNegative)
{
    EXPECT_EQ(textToInteger("-456"), -456);
}

TEST(StringsTest, TextToIntegerZero)
{
    EXPECT_EQ(textToInteger("0"), 0);
}

// ---------------------------------------------------------------------------
// 9. textToBool()
// ---------------------------------------------------------------------------
TEST(StringsTest, TextToBoolTrueLowercase)
{
    EXPECT_TRUE(textToBool("true"));
}

TEST(StringsTest, TextToBoolTrueUppercase)
{
    EXPECT_TRUE(textToBool("TRUE"));
}

TEST(StringsTest, TextToBoolYes)
{
    EXPECT_TRUE(textToBool("yes"));
}

TEST(StringsTest, TextToBoolFalse)
{
    EXPECT_FALSE(textToBool("false"));
}

TEST(StringsTest, TextToBoolNo)
{
    EXPECT_FALSE(textToBool("no"));
}

// ---------------------------------------------------------------------------
// 10. textToLongLong()
// ---------------------------------------------------------------------------
TEST(StringsTest, TextToLongLong)
{
    EXPECT_EQ(textToLongLong("9876543210"), 9876543210LL);
}

// ---------------------------------------------------------------------------
// 11. floatToString()
// ---------------------------------------------------------------------------
TEST(StringsTest, FloatToStringBasic)
{
    // With width=0 uses minimal representation.
    std::string s = floatToString(1.0f);
    EXPECT_FALSE(s.empty());
    EXPECT_NEAR(std::stof(s), 1.0f, 1e-5f);
}

TEST(StringsTest, FloatToStringZero)
{
    std::string s = floatToString(0.0f);
    EXPECT_FALSE(s.empty());
    EXPECT_NEAR(std::stof(s), 0.0f, 1e-5f);
}

// ---------------------------------------------------------------------------
// 12. integerToString()
// ---------------------------------------------------------------------------
TEST(StringsTest, IntegerToStringBasic)
{
    EXPECT_EQ(integerToString(42), "42");
}

TEST(StringsTest, IntegerToStringZero)
{
    EXPECT_EQ(integerToString(0), "0");
}

TEST(StringsTest, IntegerToStringZeroPadded)
{
    EXPECT_EQ(integerToString(42, 5), "00042");
}

TEST(StringsTest, IntegerToStringNegative)
{
    // Negative: sign prepended, digits zero-padded to (width-1) positions.
    std::string s = integerToString(-7, 3);
    EXPECT_EQ(s[0], '-');
    EXPECT_NE(s.find('7'), std::string::npos);
}

// ---------------------------------------------------------------------------
// 13. stringToString()
// ---------------------------------------------------------------------------
TEST(StringsTest, StringToStringZeroWidthUnchanged)
{
    EXPECT_EQ(stringToString("hello", 0), "hello");
}

TEST(StringsTest, StringToStringTruncates)
{
    EXPECT_EQ(stringToString("hello", 3), "hel");
}

TEST(StringsTest, StringToStringPadsWithSpaces)
{
    std::string r = stringToString("hi", 5);
    EXPECT_EQ(r.size(), (size_t)5);
    EXPECT_EQ(r.substr(0, 2), "hi");
    EXPECT_EQ(r.substr(2), "   ");
}

// ---------------------------------------------------------------------------
// 14. toLower()
// ---------------------------------------------------------------------------
TEST(StringsTest, ToLowerCString)
{
    char buf[] = "Hello World 123";
    toLower(buf);
    EXPECT_STREQ(buf, "hello world 123");
}

TEST(StringsTest, ToLowerStdString)
{
    std::string s = "Hello World";
    toLower(s);
    EXPECT_EQ(s, "hello world");
}

TEST(StringsTest, ToLowerAlreadyLower)
{
    std::string s = "already lower";
    toLower(s);
    EXPECT_EQ(s, "already lower");
}

// ---------------------------------------------------------------------------
// 15. simplify()
// ---------------------------------------------------------------------------
TEST(StringsTest, SimplifyCollapseSpaces)
{
    EXPECT_EQ(simplify("  hello   world  "), "hello world");
}

TEST(StringsTest, SimplifyStripsLeadingSpaces)
{
    EXPECT_EQ(simplify("   abc"), "abc");
}

TEST(StringsTest, SimplifyStripsTrailingSpaces)
{
    EXPECT_EQ(simplify("abc   "), "abc");
}

TEST(StringsTest, SimplifyConvertsTabToSpace)
{
    // unescape turns \t → space, then multiple spaces collapse to one
    std::string r = simplify("a\tb");
    EXPECT_EQ(r, "a b");
}

TEST(StringsTest, SimplifyEmptyString)
{
    EXPECT_EQ(simplify(""), "");
}

// ---------------------------------------------------------------------------
// 16. trim()
// ---------------------------------------------------------------------------
TEST(StringsTest, TrimLeadingTrailingSpaces)
{
    std::string s = "  hello  ";
    trim(s);
    EXPECT_EQ(s, "hello");
}

TEST(StringsTest, TrimNoSpaces)
{
    std::string s = "hello";
    trim(s);
    EXPECT_EQ(s, "hello");
}

TEST(StringsTest, TrimAllSpaces)
{
    std::string s = "     ";
    trim(s);
    EXPECT_EQ(s, "");
}

// ---------------------------------------------------------------------------
// 17. trim2()
// ---------------------------------------------------------------------------
TEST(StringsTest, Trim2RemovesSpaces)
{
    EXPECT_EQ(trim2("  hello  "), "hello");
}

TEST(StringsTest, Trim2RemovesNewlines)
{
    EXPECT_EQ(trim2("\nhello\n"), "hello");
}

TEST(StringsTest, Trim2RemovesMixedWhitespace)
{
    EXPECT_EQ(trim2("  \n  hello  \n  "), "hello");
}

// ---------------------------------------------------------------------------
// 18. removeSpaces()
// ---------------------------------------------------------------------------
TEST(StringsTest, RemoveSpacesCollapsesMultiple)
{
    EXPECT_EQ(removeSpaces("hello   world"), "hello world");
}

TEST(StringsTest, RemoveSpacesStripsLeadingTrailing)
{
    EXPECT_EQ(removeSpaces("  hello  "), "hello");
}

TEST(StringsTest, RemoveSpacesSingleWord)
{
    EXPECT_EQ(removeSpaces("hello"), "hello");
}

// ---------------------------------------------------------------------------
// 19. splitString()
// ---------------------------------------------------------------------------
TEST(StringsTest, SplitStringByComma)
{
    std::vector<std::string> parts;
    splitString("a,b,c", ",", parts);
    ASSERT_EQ(parts.size(), (size_t)3);
    EXPECT_EQ(parts[0], "a");
    EXPECT_EQ(parts[1], "b");
    EXPECT_EQ(parts[2], "c");
}

TEST(StringsTest, SplitStringNoDelimiter)
{
    std::vector<std::string> parts;
    splitString("hello", ",", parts);
    ASSERT_EQ(parts.size(), (size_t)1);
    EXPECT_EQ(parts[0], "hello");
}

TEST(StringsTest, SplitStringExcludesEmptiesByDefault)
{
    std::vector<std::string> parts;
    splitString("a,,b", ",", parts);
    // Empty token between the two commas is excluded
    ASSERT_EQ(parts.size(), (size_t)2);
    EXPECT_EQ(parts[0], "a");
    EXPECT_EQ(parts[1], "b");
}

TEST(StringsTest, SplitStringIncludeEmpties)
{
    std::vector<std::string> parts;
    splitString("a,,b", ",", parts, true);
    ASSERT_EQ(parts.size(), (size_t)3);
    EXPECT_EQ(parts[1], "");
}

TEST(StringsTest, SplitStringEmptyInput)
{
    std::vector<std::string> parts;
    splitString("", ",", parts);
    EXPECT_TRUE(parts.empty());
}

// ---------------------------------------------------------------------------
// 20. tokenize()
// ---------------------------------------------------------------------------
TEST(StringsTest, TokenizeSpaceDelimited)
{
    std::vector<std::string> tokens;
    tokenize("one two three", tokens);
    ASSERT_EQ(tokens.size(), (size_t)3);
    EXPECT_EQ(tokens[0], "one");
    EXPECT_EQ(tokens[1], "two");
    EXPECT_EQ(tokens[2], "three");
}

TEST(StringsTest, TokenizeTabDelimited)
{
    std::vector<std::string> tokens;
    tokenize("a\tb\tc", tokens);
    ASSERT_EQ(tokens.size(), (size_t)3);
    EXPECT_EQ(tokens[0], "a");
    EXPECT_EQ(tokens[2], "c");
}

TEST(StringsTest, TokenizeSkipsLeadingDelimiters)
{
    std::vector<std::string> tokens;
    tokenize("   hello", tokens);
    ASSERT_EQ(tokens.size(), (size_t)1);
    EXPECT_EQ(tokens[0], "hello");
}

TEST(StringsTest, TokenizeCustomDelimiter)
{
    std::vector<std::string> tokens;
    tokenize("a:b:c", tokens, ":");
    ASSERT_EQ(tokens.size(), (size_t)3);
    EXPECT_EQ(tokens[1], "b");
}

TEST(StringsTest, TokenizeEmptyString)
{
    std::vector<std::string> tokens;
    tokenize("", tokens);
    EXPECT_TRUE(tokens.empty());
}

// ---------------------------------------------------------------------------
// 21. textToInt()  — single char → int via ASCII offset (char - 48)
// ---------------------------------------------------------------------------
TEST(StringsTest, TextToIntDigitZero)
{
    EXPECT_EQ(textToInt("0"), 0);
}

TEST(StringsTest, TextToIntDigitFive)
{
    EXPECT_EQ(textToInt("5"), 5);
}

TEST(StringsTest, TextToIntDigitNine)
{
    EXPECT_EQ(textToInt("9"), 9);
}

// ---------------------------------------------------------------------------
// 22. checkAngle() — validates angle type string
// ---------------------------------------------------------------------------
TEST(StringsTest, CheckAngleRot_DoesNotThrow)
{
    EXPECT_NO_THROW(checkAngle("rot"));
}

TEST(StringsTest, CheckAngleTilt_DoesNotThrow)
{
    EXPECT_NO_THROW(checkAngle("tilt"));
}

TEST(StringsTest, CheckAnglePsi_DoesNotThrow)
{
    EXPECT_NO_THROW(checkAngle("psi"));
}

TEST(StringsTest, CheckAngleInvalid_Throws)
{
    EXPECT_THROW(checkAngle("xyz"), RelionError);
}

TEST(StringsTest, CheckAngleEmpty_Throws)
{
    EXPECT_THROW(checkAngle(""), RelionError);
}

// ---------------------------------------------------------------------------
// 23. firstToken / nextToken() — strtok-based C-API
// ---------------------------------------------------------------------------
TEST(StringsTest, FirstToken_ReturnsFirstWord)
{
    char buf[] = "hello world";
    char* tok = firstToken(buf);
    ASSERT_NE(tok, nullptr);
    EXPECT_STREQ(tok, "hello");
}

TEST(StringsTest, NextToken_ReturnsSecondWord)
{
    char buf[] = "hello world";
    firstToken(buf);
    char* tok = nextToken();
    ASSERT_NE(tok, nullptr);
    EXPECT_STREQ(tok, "world");
}

TEST(StringsTest, NextToken_ReturnsNullAtEnd)
{
    char buf[] = "hello";
    firstToken(buf);
    char* tok = nextToken();
    EXPECT_EQ(tok, nullptr);
}

// ---------------------------------------------------------------------------
// 24. nextToken(str, i) — index-advancing version
// ---------------------------------------------------------------------------
TEST(StringsTest, NextTokenByIndex_SingleWord)
{
    std::string s = "hello";
    int i = 0;
    std::string tok = nextToken(s, i);
    // Token must start with "hello"
    EXPECT_EQ(tok.find("hello"), (size_t)0);
    EXPECT_GT(i, 0);
}

TEST(StringsTest, NextTokenByIndex_TwoWords)
{
    std::string s = "alpha beta";
    int i = 0;
    std::string tok1 = nextToken(s, i);
    std::string tok2 = nextToken(s, i);
    EXPECT_EQ(tok1.find("alpha"), (size_t)0);
    EXPECT_EQ(tok2.find("beta"), (size_t)0);
}

TEST(StringsTest, NextTokenByIndex_EmptyAtEnd)
{
    std::string s = "x";
    int i = 0;
    nextToken(s, i);  // consume "x"
    std::string tok = nextToken(s, i);
    EXPECT_TRUE(tok.empty());
}

// ---------------------------------------------------------------------------
// 25. nextTokenInSTAR — STAR file token parsing
// ---------------------------------------------------------------------------
TEST(StringsTest, NextTokenInSTAR_ReturnsFirstToken)
{
    std::string line = "12345.0 hello";
    int i = 0;
    std::string tok;
    bool ok = nextTokenInSTAR(line, i, tok);
    EXPECT_TRUE(ok);
    EXPECT_EQ(tok, "12345.0");
}

TEST(StringsTest, NextTokenInSTAR_ReturnsFalseAtEnd)
{
    std::string line = "token";
    int i = 0;
    std::string tok;
    nextTokenInSTAR(line, i, tok);  // consume "token"
    bool ok = nextTokenInSTAR(line, i, tok);
    EXPECT_FALSE(ok);
}

TEST(StringsTest, NextTokenInSTAR_QuotedString)
{
    // Quoted token should return contents without quotes
    std::string line = "\"hello world\"";
    int i = 0;
    std::string tok;
    bool ok = nextTokenInSTAR(line, i, tok);
    EXPECT_TRUE(ok);
    EXPECT_EQ(tok, "hello world");
}

// ---------------------------------------------------------------------------
// 26. firstWord / nextWord — same as firstToken / nextToken but throw on empty
// ---------------------------------------------------------------------------
TEST(StringsTest, FirstWord_ReturnsFirstWord)
{
    std::string s = "alpha beta";
    char* w = firstWord(s);
    ASSERT_NE(w, nullptr);
    EXPECT_STREQ(w, "alpha");
}

TEST(StringsTest, NextWord_ReturnsSecondWord)
{
    std::string s = "alpha beta";
    firstWord(s);
    char* w = nextWord();
    ASSERT_NE(w, nullptr);
    EXPECT_STREQ(w, "beta");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
