/*
 * Unit tests for src/jaz/util/log.cpp
 *
 * Covers Log static methods, MinimalistLogger, and FancyLogger.
 * All output goes to stdout; tests verify no crash / no throw.
 */

#include <gtest/gtest.h>
#include <sstream>
#include "src/jaz/util/log.h"

// ---------------------------------------------------------------------------
// Log static API — smoke tests (no crash)
// ---------------------------------------------------------------------------

TEST(LogTest, Print_DoesNotCrash)
{
    EXPECT_NO_THROW(Log::print("hello world"));
}

TEST(LogTest, Extend_DoesNotCrash)
{
    EXPECT_NO_THROW(Log::extend("extended line"));
}

TEST(LogTest, Warn_DoesNotCrash)
{
    EXPECT_NO_THROW(Log::warn("a warning"));
}

TEST(LogTest, PrintBinaryChoice_TrueCase)
{
    EXPECT_NO_THROW(Log::printBinaryChoice("condition is: ", true, "true", "false"));
}

TEST(LogTest, PrintBinaryChoice_FalseCase)
{
    EXPECT_NO_THROW(Log::printBinaryChoice("condition is: ", false, "true", "false"));
}

TEST(LogTest, BeginEndSection_DoesNotCrash)
{
    EXPECT_NO_THROW(Log::beginSection("test section"));
    EXPECT_NO_THROW(Log::endSection());
}

TEST(LogTest, NestedSections_DoesNotCrash)
{
    EXPECT_NO_THROW(Log::beginSection("outer"));
    EXPECT_NO_THROW(Log::beginSection("inner"));
    EXPECT_NO_THROW(Log::print("inside nested section"));
    EXPECT_NO_THROW(Log::endSection());
    EXPECT_NO_THROW(Log::endSection());
}

TEST(LogTest, BeginUpdateEndProgress_DoesNotCrash)
{
    EXPECT_NO_THROW(Log::beginProgress("working", 100));
    EXPECT_NO_THROW(Log::updateProgress(0));
    EXPECT_NO_THROW(Log::updateProgress(50));
    EXPECT_NO_THROW(Log::updateProgress(100));
    EXPECT_NO_THROW(Log::endProgress());
}

TEST(LogTest, UpdateProgress_ThrottlingDoesNotCrash)
{
    // Updates that are too close together are throttled — should not crash
    EXPECT_NO_THROW(Log::beginProgress("throttle test", 10000));
    for (int i = 0; i < 10000; i++)
        Log::updateProgress(i);
    EXPECT_NO_THROW(Log::endProgress());
}

// ---------------------------------------------------------------------------
// MinimalistLogger direct tests
// ---------------------------------------------------------------------------

TEST(MinimalistLoggerTest, Print_DoesNotCrash)
{
    MinimalistLogger ml;
    EXPECT_NO_THROW(ml.print("hello"));
}

TEST(MinimalistLoggerTest, Extend_DoesNotCrash)
{
    MinimalistLogger ml;
    EXPECT_NO_THROW(ml.extend("extended"));
}

TEST(MinimalistLoggerTest, Warn_DoesNotCrash)
{
    MinimalistLogger ml;
    EXPECT_NO_THROW(ml.warn("warning"));
}

TEST(MinimalistLoggerTest, BeginEndSection_DoesNotCrash)
{
    MinimalistLogger ml;
    EXPECT_NO_THROW(ml.beginSection("section"));
    EXPECT_NO_THROW(ml.endSection());
}

TEST(MinimalistLoggerTest, BeginUpdateEndProgress_DoesNotCrash)
{
    MinimalistLogger ml;
    EXPECT_NO_THROW(ml.beginProgress("progress", 10));
    EXPECT_NO_THROW(ml.updateProgress(5));
    EXPECT_NO_THROW(ml.endProgress());
}

// ---------------------------------------------------------------------------
// FancyLogger direct tests
// ---------------------------------------------------------------------------

TEST(FancyLoggerTest, Print_DoesNotCrash)
{
    FancyLogger fl;
    EXPECT_NO_THROW(fl.print("hello"));
}

TEST(FancyLoggerTest, Extend_DoesNotCrash)
{
    FancyLogger fl;
    EXPECT_NO_THROW(fl.extend("extended"));
}

TEST(FancyLoggerTest, Warn_DoesNotCrash)
{
    FancyLogger fl;
    EXPECT_NO_THROW(fl.warn("warning"));
}

TEST(FancyLoggerTest, BeginEndSection_DoesNotCrash)
{
    FancyLogger fl;
    EXPECT_NO_THROW(fl.beginSection("section A"));
    EXPECT_NO_THROW(fl.endSection());
}

TEST(FancyLoggerTest, NestedSections_DoesNotCrash)
{
    FancyLogger fl;
    EXPECT_NO_THROW(fl.beginSection("outer"));
    EXPECT_NO_THROW(fl.beginSection("inner"));
    EXPECT_NO_THROW(fl.print("deep"));
    EXPECT_NO_THROW(fl.endSection());
    EXPECT_NO_THROW(fl.endSection());
}

TEST(FancyLoggerTest, BeginUpdateEndProgress_DoesNotCrash)
{
    FancyLogger fl;
    EXPECT_NO_THROW(fl.beginProgress("task", 10));
    EXPECT_NO_THROW(fl.updateProgress(5));
    EXPECT_NO_THROW(fl.updateProgress(10));
    EXPECT_NO_THROW(fl.endProgress());
}

TEST(FancyLoggerTest, SetColours_DoesNotCrash)
{
    FancyLogger fl;
    EXPECT_NO_THROW(fl.setColours("\033[31m", "\033[34m"));
    EXPECT_NO_THROW(fl.print("coloured"));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
