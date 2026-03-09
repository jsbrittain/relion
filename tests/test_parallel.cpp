/*
 * Unit tests for src/parallel.h / src/parallel.cpp
 *
 * Covers: divide_equally, divide_equally_which_group,
 *         ThreadTaskDistributor (serial single-thread usage),
 *         Mutex basic lock/unlock
 */

#include <gtest/gtest.h>
#include "src/parallel.h"

// ---------------------------------------------------------------------------
// divide_equally
// ---------------------------------------------------------------------------

TEST(ParallelTest, DivideEqually_ExactDivision)
{
    // 10 tasks, 2 workers: each gets 5
    long int first, last;
    long int n = divide_equally(10, 2, 0, first, last);
    EXPECT_EQ(first, 0);
    EXPECT_EQ(last,  4);
    EXPECT_EQ(n,     5);

    n = divide_equally(10, 2, 1, first, last);
    EXPECT_EQ(first, 5);
    EXPECT_EQ(last,  9);
    EXPECT_EQ(n,     5);
}

TEST(ParallelTest, DivideEqually_UnexactDivision)
{
    // 7 tasks, 3 workers: workers 0 and 1 get 3, worker 2 gets 1
    long int first, last, n;

    n = divide_equally(7, 3, 0, first, last);
    EXPECT_EQ(n, last - first + 1);
    long int n0 = n;

    n = divide_equally(7, 3, 1, first, last);
    EXPECT_EQ(n, last - first + 1);
    long int n1 = n;

    n = divide_equally(7, 3, 2, first, last);
    EXPECT_EQ(n, last - first + 1);
    long int n2 = n;

    EXPECT_EQ(n0 + n1 + n2, 7);
    // First two workers should have more or equal tasks compared to the last
    EXPECT_GE(n0, n2);
    EXPECT_GE(n1, n2);
}

TEST(ParallelTest, DivideEqually_RangesAreContiguous)
{
    // Ensure ranges cover [0, N-1] without gaps or overlap
    long int N = 13, size = 4;
    long int prev_last = -1;
    for (int rank = 0; rank < size; rank++)
    {
        long int first, last;
        long int n = divide_equally(N, size, rank, first, last);
        if (n > 0)
        {
            EXPECT_EQ(first, prev_last + 1);
            prev_last = last;
        }
    }
    EXPECT_EQ(prev_last, N - 1);
}

TEST(ParallelTest, DivideEqually_SingleWorker)
{
    long int first, last;
    long int n = divide_equally(100, 1, 0, first, last);
    EXPECT_EQ(first, 0);
    EXPECT_EQ(last,  99);
    EXPECT_EQ(n,     100);
}

// ---------------------------------------------------------------------------
// divide_equally_which_group
// ---------------------------------------------------------------------------

TEST(ParallelTest, WhichGroup_ExactDivision)
{
    // 10 tasks, 2 workers: tasks 0-4 → group 0, tasks 5-9 → group 1
    EXPECT_EQ(divide_equally_which_group(10, 2, 0), 0);
    EXPECT_EQ(divide_equally_which_group(10, 2, 4), 0);
    EXPECT_EQ(divide_equally_which_group(10, 2, 5), 1);
    EXPECT_EQ(divide_equally_which_group(10, 2, 9), 1);
}

TEST(ParallelTest, WhichGroup_ConsistentWithDivideEqually)
{
    long int N = 13, size = 4;
    for (long int task = 0; task < N; task++)
    {
        int group = divide_equally_which_group(N, size, task);
        long int first, last;
        divide_equally(N, size, group, first, last);
        EXPECT_GE(task, first);
        EXPECT_LE(task, last);
    }
}

TEST(ParallelTest, WhichGroup_OutOfRange_ReturnsMinusOne)
{
    // myself >= N: not in any group's range → fallthrough return -1
    EXPECT_EQ(divide_equally_which_group(10, 2, 10), -1);
    EXPECT_EQ(divide_equally_which_group(10, 2, 99), -1);
    // Negative index: also not in any group's [0, N-1] range
    EXPECT_EQ(divide_equally_which_group(10, 2, -1), -1);
}

// ---------------------------------------------------------------------------
// ThreadTaskDistributor – serial single-thread usage
// ---------------------------------------------------------------------------

TEST(ParallelTest, ThreadTaskDistributor_SingleBlock)
{
    ThreadTaskDistributor td(10, 10);
    size_t first, last;
    bool got = td.getTasks(first, last);
    EXPECT_TRUE(got);
    EXPECT_EQ(first, (size_t)0);
    EXPECT_EQ(last,  (size_t)9);

    got = td.getTasks(first, last);
    EXPECT_FALSE(got);
}

TEST(ParallelTest, ThreadTaskDistributor_MultipleBlocks)
{
    ThreadTaskDistributor td(10, 3);
    size_t first, last;
    size_t total = 0;
    int blocks = 0;
    while (td.getTasks(first, last))
    {
        total += last - first + 1;
        blocks++;
    }
    EXPECT_EQ(total, (size_t)10);
    EXPECT_GE(blocks, 3); // at least ceil(10/3) = 4 blocks
}

TEST(ParallelTest, ThreadTaskDistributor_Reset)
{
    ThreadTaskDistributor td(5, 5);
    size_t first, last;
    td.getTasks(first, last);
    EXPECT_FALSE(td.getTasks(first, last)); // exhausted

    td.reset();
    EXPECT_TRUE(td.getTasks(first, last));
    EXPECT_EQ(first, (size_t)0);
    EXPECT_EQ(last,  (size_t)4);
}

TEST(ParallelTest, ThreadTaskDistributor_SetBlockSize)
{
    ThreadTaskDistributor td(12, 4);
    EXPECT_EQ(td.getBlockSize(), 4);
    td.setBlockSize(6);
    EXPECT_EQ(td.getBlockSize(), 6);
}

TEST(ParallelTest, ThreadTaskDistributor_Resize)
{
    ThreadTaskDistributor td(10, 5);
    td.resize(20, 10);
    size_t first, last;
    bool got = td.getTasks(first, last);
    EXPECT_TRUE(got);
    EXPECT_EQ(first, (size_t)0);
    EXPECT_EQ(last,  (size_t)9);
}

TEST(ParallelTest, ThreadTaskDistributor_SetAssignedTasks)
{
    ThreadTaskDistributor td(10, 3);
    // Skip first 6 tasks by setting assignedTasks = 6
    td.setAssignedTasks(6);
    size_t first, last;
    bool got = td.getTasks(first, last);
    EXPECT_TRUE(got);
    EXPECT_EQ(first, (size_t)6);
}

TEST(ParallelTest, ThreadTaskDistributor_SetAssignedTasks_OutOfRange_ReturnsFalse)
{
    // tasks >= numberOfTasks triggers the early return false
    ThreadTaskDistributor td(10, 3);
    EXPECT_FALSE(td.setAssignedTasks(10)); // exactly equal to numberOfTasks
    EXPECT_FALSE(td.setAssignedTasks(99)); // well beyond range
    // In-range values still succeed
    EXPECT_TRUE(td.setAssignedTasks(0));
    EXPECT_TRUE(td.setAssignedTasks(9));
}

TEST(ParallelTest, ParallelTaskDistributor_Resize_InvalidArgs_Throws)
{
    // bSize == 0
    EXPECT_THROW(ThreadTaskDistributor(10, 0),  RelionError);
    // nTasks == 0
    EXPECT_THROW(ThreadTaskDistributor(0, 1),   RelionError);
    // bSize > nTasks
    EXPECT_THROW(ThreadTaskDistributor(5, 10),  RelionError);
    // Valid call must not throw
    EXPECT_NO_THROW(ThreadTaskDistributor(10, 5));
}

// ---------------------------------------------------------------------------
// Mutex – basic lock/unlock (single-threaded correctness)
// ---------------------------------------------------------------------------

TEST(ParallelTest, Mutex_LockUnlock)
{
    Mutex m;
    // Should not deadlock in a single-threaded context
    m.lock();
    m.unlock();
    // Second lock after unlock
    m.lock();
    m.unlock();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
