/*
 * GoogleTest smoke tests for the MlOptimiserMpi::expectation() refactoring.
 *
 * Build and run (requires >=2 MPI ranks):
 *   cmake -DRELION_UNIT_TESTS=ON ...
 *   make test_expectation_mpi
 *   mpirun -n 2 ./tests/test_expectation_mpi
 *
 * What is tested:
 *   1. setupAccelerators()          – is a no-op when no GPU/SYCL/ALTCPU backend
 *                                     is compiled/enabled; accDataBundles stays empty.
 *   2. runLeaderExpectationLoop() / – with zero particles the leader immediately
 *      runFollowerExpectationLoop()   signals "done" to every follower; both sides
 *                                     complete without deadlock.
 *
 * MPI lifecycle:
 *   MpiNode::MpiNode() calls MPI_Init internally, so we construct one shared node
 *   in main() and reuse it across tests by setting opt.node to that pointer.
 *   TearDown() sets opt.node = nullptr before ~MlOptimiserMpi() runs, preventing
 *   a double-free of the shared node.  The shared node (and MPI_Finalize) is
 *   released when main() returns.
 */

#include <gtest/gtest.h>
#include "src/ml_optimiser_mpi.h"

// ---------------------------------------------------------------------------
// Global MPI state – initialised once in main() via MpiNode.
// ---------------------------------------------------------------------------
static MpiNode* g_node = nullptr;

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------
class ExpectationRefactorMpiTest : public ::testing::Test
{
protected:
    MlOptimiserMpi opt;

    void SetUp() override
    {
        // Share the process-wide node; destructor must NOT delete it.
        opt.node = g_node;

        // Minimal settings so the extracted methods run without crashing:
        //   no GPU / SYCL / ALTCPU backend, no particles, no verbosity.
        opt.do_gpu                         = false;
        opt.do_sycl                        = false;
        opt.do_cpu                         = false;
        opt.nr_pool                        = 1;
        opt.subset_size                    = 0;   // 0 → use mydata.numberOfParticles()
        opt.verb                           = 0;
        opt.do_split_random_halves         = false;
        opt.halt_all_followers_except_this = 0;
    }

    void TearDown() override
    {
        // Prevent ~MlOptimiserMpi() from deleting the shared node.
        opt.node = nullptr;
    }
};

// ---------------------------------------------------------------------------
// Test 1: setupAccelerators() leaves accDataBundles empty when no backend
//         is active (do_gpu / do_sycl / do_cpu are all false).
// ---------------------------------------------------------------------------
TEST_F(ExpectationRefactorMpiTest, SetupAcceleratorsNoOp)
{
    ASSERT_TRUE(opt.accDataBundles.empty())
        << "Precondition: accDataBundles must be empty before the call";

    opt.setupAccelerators();

    EXPECT_TRUE(opt.accDataBundles.empty())
        << "setupAccelerators() must not populate accDataBundles "
           "when do_gpu=do_sycl=do_cpu=false";
}

// ---------------------------------------------------------------------------
// Test 2: zero-particle dispatch handshake completes without deadlock.
//
// With 0 particles in mydata the leader computes my_nr_particles_done (0)
// >= nr_particles_todo (0) for every follower request, so it immediately
// sends JOB_NIMG=0 ("done").  Each follower receives the done signal and
// breaks out of its loop.  Both paths must reach the barrier below without
// hanging.
// ---------------------------------------------------------------------------
TEST_F(ExpectationRefactorMpiTest, ZeroParticleDispatch)
{
    if (g_node->size < 2)
        GTEST_SKIP() << "ZeroParticleDispatch requires at least 2 MPI ranks";

    MultidimArray<long int> job_buf(6);

    if (g_node->isLeader())
        opt.runLeaderExpectationLoop(job_buf, /*my_nr_particles=*/0);
    else
        opt.runFollowerExpectationLoop(job_buf);

    // All ranks must reach this barrier; a hang means the handshake is broken.
    MPI_Barrier(MPI_COMM_WORLD);

    SUCCEED();  // reaching here without deadlock is the meaningful assertion
}

// ---------------------------------------------------------------------------
// MPI-aware main: let MpiNode call MPI_Init; finalise via its destructor.
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // MpiNode::MpiNode() calls MPI_Init internally.
    g_node = new MpiNode(argc, argv);

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    // Deleting g_node calls MPI_Finalize() via ~MpiNode().
    delete g_node;
    g_node = nullptr;

    return result;
}
