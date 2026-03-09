/*
 * Unit tests for src/pipeline_control.h / src/pipeline_control.cpp
 *
 * Covers: is_under_pipeline_control(),
 *         pipeline_control_relion_exit() — modes 0/1/2/invalid,
 *         pipeline_control_check_abort_job(),
 *         pipeline_control_delete_exit_files()
 */

#include <gtest/gtest.h>
#include <fstream>
#include <cstdlib>
#include <sys/stat.h>
#include "src/pipeline_control.h"

// Helper: create a file at the given path
static void touchFile(const std::string& path)
{
    std::ofstream f(path);
    f.close();
}

// Helper: check whether a file exists
static bool fileExists(const std::string& path)
{
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

// ---------------------------------------------------------------------------
// Fixture: isolate the global state by saving/restoring it between tests
// ---------------------------------------------------------------------------

class PipelineControlTest : public ::testing::Test
{
protected:
    std::string saved;

    void SetUp() override
    {
        saved = pipeline_control_outputname;
        pipeline_control_outputname = "";
    }

    void TearDown() override
    {
        pipeline_control_outputname = saved;
    }
};

// ---------------------------------------------------------------------------
// is_under_pipeline_control
// ---------------------------------------------------------------------------

TEST_F(PipelineControlTest, NotUnderControl_WhenEmpty)
{
    pipeline_control_outputname = "";
    EXPECT_FALSE(is_under_pipeline_control());
}

TEST_F(PipelineControlTest, UnderControl_WhenNonEmpty)
{
    pipeline_control_outputname = "/tmp/relion_test_pipeline_";
    EXPECT_TRUE(is_under_pipeline_control());
}

// ---------------------------------------------------------------------------
// pipeline_control_relion_exit — not under pipeline control
// ---------------------------------------------------------------------------

TEST_F(PipelineControlTest, Exit_NotUnderControl_ReturnsMode)
{
    pipeline_control_outputname = "";
    EXPECT_EQ(pipeline_control_relion_exit(0), 0);
    EXPECT_EQ(pipeline_control_relion_exit(1), 1);
    EXPECT_EQ(pipeline_control_relion_exit(2), 2);
}

TEST_F(PipelineControlTest, Exit_NotUnderControl_InvalidMode)
{
    pipeline_control_outputname = "";
    // Even invalid mode returns a value when not under control
    int ret = pipeline_control_relion_exit(99);
    EXPECT_EQ(ret, 99);
}

// ---------------------------------------------------------------------------
// pipeline_control_relion_exit — under pipeline control, creates files
// ---------------------------------------------------------------------------

TEST_F(PipelineControlTest, Exit_Success_CreatesSuccessFile)
{
    pipeline_control_outputname = "/tmp/relion_pctest_success_";
    // Clean up beforehand
    std::remove((pipeline_control_outputname + RELION_JOB_EXIT_SUCCESS).c_str());

    int ret = pipeline_control_relion_exit(0);
    EXPECT_EQ(ret, 0);
    EXPECT_TRUE(fileExists(pipeline_control_outputname + RELION_JOB_EXIT_SUCCESS));

    // Clean up
    std::remove((pipeline_control_outputname + RELION_JOB_EXIT_SUCCESS).c_str());
}

TEST_F(PipelineControlTest, Exit_Failure_CreatesFailureFile)
{
    pipeline_control_outputname = "/tmp/relion_pctest_failure_";
    std::remove((pipeline_control_outputname + RELION_JOB_EXIT_FAILURE).c_str());

    int ret = pipeline_control_relion_exit(1);
    EXPECT_EQ(ret, 1);
    EXPECT_TRUE(fileExists(pipeline_control_outputname + RELION_JOB_EXIT_FAILURE));

    std::remove((pipeline_control_outputname + RELION_JOB_EXIT_FAILURE).c_str());
}

TEST_F(PipelineControlTest, Exit_Aborted_CreatesAbortedFile)
{
    pipeline_control_outputname = "/tmp/relion_pctest_aborted_";
    std::remove((pipeline_control_outputname + RELION_JOB_EXIT_ABORTED).c_str());

    int ret = pipeline_control_relion_exit(2);
    EXPECT_EQ(ret, 2);
    EXPECT_TRUE(fileExists(pipeline_control_outputname + RELION_JOB_EXIT_ABORTED));

    std::remove((pipeline_control_outputname + RELION_JOB_EXIT_ABORTED).c_str());
}

TEST_F(PipelineControlTest, Exit_InvalidMode_Returns12)
{
    pipeline_control_outputname = "/tmp/relion_pctest_inv_";
    int ret = pipeline_control_relion_exit(99);
    EXPECT_EQ(ret, 12);
}

// ---------------------------------------------------------------------------
// pipeline_control_check_abort_job
// ---------------------------------------------------------------------------

TEST_F(PipelineControlTest, CheckAbort_NotUnderControl_ReturnsFalse)
{
    pipeline_control_outputname = "";
    EXPECT_FALSE(pipeline_control_check_abort_job());
}

TEST_F(PipelineControlTest, CheckAbort_UnderControl_NoAbortFile_ReturnsFalse)
{
    pipeline_control_outputname = "/tmp/relion_pctest_abort_";
    std::string abort_file = pipeline_control_outputname + RELION_JOB_ABORT_NOW;
    std::remove(abort_file.c_str()); // ensure it doesn't exist
    EXPECT_FALSE(pipeline_control_check_abort_job());
}

TEST_F(PipelineControlTest, CheckAbort_UnderControl_AbortFilePresent_ReturnsTrue)
{
    pipeline_control_outputname = "/tmp/relion_pctest_abortnow_";
    std::string abort_file = pipeline_control_outputname + RELION_JOB_ABORT_NOW;
    touchFile(abort_file);

    EXPECT_TRUE(pipeline_control_check_abort_job());

    std::remove(abort_file.c_str());
}

// ---------------------------------------------------------------------------
// pipeline_control_delete_exit_files
// ---------------------------------------------------------------------------

TEST_F(PipelineControlTest, DeleteExitFiles_RemovesAllThreeFiles)
{
    pipeline_control_outputname = "/tmp/relion_pctest_delfiles_";
    std::string s = pipeline_control_outputname + RELION_JOB_EXIT_SUCCESS;
    std::string f = pipeline_control_outputname + RELION_JOB_EXIT_FAILURE;
    std::string a = pipeline_control_outputname + RELION_JOB_EXIT_ABORTED;

    touchFile(s);
    touchFile(f);
    touchFile(a);

    ASSERT_TRUE(fileExists(s));
    ASSERT_TRUE(fileExists(f));
    ASSERT_TRUE(fileExists(a));

    pipeline_control_delete_exit_files();

    EXPECT_FALSE(fileExists(s));
    EXPECT_FALSE(fileExists(f));
    EXPECT_FALSE(fileExists(a));
}

TEST_F(PipelineControlTest, DeleteExitFiles_NoCrashWhenFilesAbsent)
{
    pipeline_control_outputname = "/tmp/relion_pctest_nofiles_";
    // Ensure no files exist
    std::remove((pipeline_control_outputname + RELION_JOB_EXIT_SUCCESS).c_str());
    std::remove((pipeline_control_outputname + RELION_JOB_EXIT_FAILURE).c_str());
    std::remove((pipeline_control_outputname + RELION_JOB_EXIT_ABORTED).c_str());

    // Should not crash
    pipeline_control_delete_exit_files();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
