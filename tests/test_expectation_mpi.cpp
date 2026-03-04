/*
 * GoogleTest smoke tests for the MlOptimiserMpi::expectation() refactoring.
 *
 * Build and run (requires >=2 MPI ranks):
 *   cmake -DRELION_UNIT_TESTS=ON ...
 *   make test_expectation_mpi
 *   mpirun -n 2 ./tests/test_expectation_mpi
 *   mpirun -n 3 ./tests/test_expectation_mpi   # enables additional tests
 *
 * What is tested:
 *   1. setupAccelerators()                    – no-op when no GPU/SYCL/ALTCPU backend.
 *   2. runLeaderExpectationLoop() /
 *      runFollowerExpectationLoop()           – zero-particle handshake completes.
 *   3. setupAccelerators()                    – accOptimisers stays empty (CPU build).
 *   4. setupAccelerators()                    – leader has null backend; followers non-null.
 *   5. makeAccBackend factory                 – non-null fallback for all-false flags.
 *   6. PlainCpuBackend methods                – createBundles/createOptimisers/teardown
 *                                               are all no-ops.
 *   7. broadcastRandomSeed()                  – all ranks agree on a non-negative seed.
 *   8. broadcastNrParticlesPerGroup(1)        – leader's array reaches odd-rank follower.
 *   9. broadcastNrParticlesPerGroup(2)        – safe no-op when no even-rank follower.
 *  10. combineAllWeightedSumsImpl()           – ring-reduce yields correct sum (≥3 ranks).
 *  11. combineWeightedSumsTwoHalvesImpl()     – cross-half sum is correct (≥3 ranks).
 *  12. broadcastSplitHalfReconstructionWith3Ranks – completes without deadlock (≥3 ranks).
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
#include "src/acc/acc_backend_factory.h"

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
// Test 3: setupAccelerators() leaves accOptimisers empty when no backend
//         is active (do_gpu / do_sycl / do_cpu are all false).
// ---------------------------------------------------------------------------
TEST_F(ExpectationRefactorMpiTest, SetupAcceleratorsAccOptimisersEmpty)
{
    ASSERT_TRUE(opt.accOptimisers.empty())
        << "Precondition: accOptimisers must be empty before the call";

    opt.setupAccelerators();

    EXPECT_TRUE(opt.accOptimisers.empty())
        << "setupAccelerators() must not populate accOptimisers "
           "when do_gpu=do_sycl=do_cpu=false";
}

// ---------------------------------------------------------------------------
// Test 4: Backend ownership after setupAccelerators():
//   - Leader (rank 0) must not create a backend; accBackend stays null.
//   - Each follower (rank > 0) must own one; accBackend is non-null.
// ---------------------------------------------------------------------------
TEST_F(ExpectationRefactorMpiTest, SetupAcceleratorsBackendOwnership)
{
    if (g_node->size < 2)
        GTEST_SKIP() << "SetupAcceleratorsBackendOwnership requires at least 2 MPI ranks";

    opt.setupAccelerators();

    if (g_node->isLeader())
        EXPECT_EQ(opt.accBackend, nullptr)
            << "Leader must not create a backend (setupAccelerators skips it)";
    else
        EXPECT_NE(opt.accBackend, nullptr)
            << "Follower must own a backend after setupAccelerators()";

    MPI_Barrier(MPI_COMM_WORLD);
}

// ---------------------------------------------------------------------------
// Test 5: makeAccBackend(false, false, false) returns a non-null PlainCpuBackend.
//         This exercises the factory fallback path; no MPI needed.
// ---------------------------------------------------------------------------
TEST(AccBackendFactoryTest, PlainCpuFallbackIsNonNull)
{
    std::unique_ptr<AccBackend> backend =
        makeAccBackend(/*do_gpu=*/false, /*do_sycl=*/false, /*do_cpu=*/false);

    EXPECT_NE(backend.get(), nullptr)
        << "makeAccBackend must return a non-null PlainCpuBackend "
           "when all flags are false";
}

// ---------------------------------------------------------------------------
// Test 6: PlainCpuBackend::createBundles, createOptimisers, and teardown are
//         all no-ops: accDataBundles and accOptimisers must stay empty.
// ---------------------------------------------------------------------------
TEST_F(ExpectationRefactorMpiTest, PlainCpuBackendMethodsAreNoOps)
{
    PlainCpuBackend backend;

    backend.createBundles(opt);
    EXPECT_TRUE(opt.accDataBundles.empty())
        << "PlainCpuBackend::createBundles must not push any bundles";

    backend.createOptimisers(opt);
    EXPECT_TRUE(opt.accOptimisers.empty())
        << "PlainCpuBackend::createOptimisers must not push any optimisers";

    backend.teardown(opt);
    EXPECT_TRUE(opt.accDataBundles.empty())
        << "PlainCpuBackend::teardown must leave accDataBundles empty";
    EXPECT_TRUE(opt.accOptimisers.empty())
        << "PlainCpuBackend::teardown must leave accOptimisers empty";
}

// ---------------------------------------------------------------------------
// Test 7: broadcastRandomSeed() delivers a consistent, non-negative seed to
//         every MPI rank.
// ---------------------------------------------------------------------------
TEST_F(ExpectationRefactorMpiTest, BroadcastRandomSeedSetsConsistentSeed)
{
    opt.random_seed = -1;

    opt.broadcastRandomSeed();

    // Every rank must now hold a non-negative value.
    EXPECT_GE(opt.random_seed, 0);

    // All ranks must agree on the same value.  Use Allreduce min/max as a
    // cross-rank consistency check without needing an extra Bcast.
    int seed_min = 0, seed_max = 0;
    MPI_Allreduce(&opt.random_seed, &seed_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&opt.random_seed, &seed_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_EQ(seed_min, seed_max)
        << "All MPI ranks must agree on the same random seed";
}

// ---------------------------------------------------------------------------
// Test 8: broadcastNrParticlesPerGroup(1) delivers the leader's array to all
//         odd-ranked followers (rank 1, 3, 5, …).
// ---------------------------------------------------------------------------
TEST_F(ExpectationRefactorMpiTest, BroadcastNrParticlesPerGroupHalfset1)
{
    const std::vector<long int> ref = {42L, 7L, 13L};

    if (g_node->isLeader())
        opt.mymodel.nr_particles_per_group = ref;
    else
        opt.mymodel.nr_particles_per_group.assign(ref.size(), 0L);

    opt.broadcastNrParticlesPerGroup(1);

    // Every odd-ranked follower must now hold the leader's values.
    if (g_node->rank % 2 == 1)
        EXPECT_EQ(opt.mymodel.nr_particles_per_group, ref)
            << "Odd follower (rank " << g_node->rank << ") must receive the leader's array";
}

// ---------------------------------------------------------------------------
// Test 9: broadcastNrParticlesPerGroup(2) is a silent no-op when no even-rank
//         follower exists (i.e. exactly 2 MPI ranks: leader + one odd follower).
// ---------------------------------------------------------------------------
TEST_F(ExpectationRefactorMpiTest, BroadcastNrParticlesPerGroupHalfset2NoopWith2Ranks)
{
    if (g_node->size != 2)
        GTEST_SKIP() << "This test only applies to exactly 2 MPI ranks";

    const std::vector<long int> initial = {1L, 2L, 3L};
    opt.mymodel.nr_particles_per_group = initial;

    opt.broadcastNrParticlesPerGroup(2);  // no rank-2 follower → silent no-op

    EXPECT_EQ(opt.mymodel.nr_particles_per_group, initial)
        << "Array must be unchanged when there is no even-rank follower";
}

// ---------------------------------------------------------------------------
// Helper: set up a minimal wsum_model that survives pack()/unpack() round-trips
// without full model initialisation.
//
// BPref[0] is pre-allocated with initialiseDataAndWeight(0) so that pack() and
// unpack() agree on the same buffer size on every rank.  With padding_factor=0
// and current_size=0, initialiseDataAndWeight sets pad_size=3 and allocates
// data/weight arrays of 18 complex / 18 real elements respectively.
//
// Packed layout (packed_size == 62):
//   [0] LL  [1] ave_Pmax  [2] sigma2_offset  [3] avg_norm_correction
//   [4] sigma2_rot  [5] sigma2_tilt  [6] sigma2_psi  [7] pdf_class[0]
//   [8..61] BPref[0].data (18 complex * 2 + 18 real = 54 elements)
// ---------------------------------------------------------------------------
static void setupMinimalWsumModel(MlOptimiserMpi &opt, RFLOAT ll_value)
{
    auto &w = opt.wsum_model;
    w.nr_classes        = 1;
    w.nr_bodies         = 0;   // nr_classes * nr_bodies = 0: BPref loops skipped
    w.nr_groups         = 0;
    w.nr_optics_groups  = 0;
    w.nr_directions     = 0;
    w.ref_dim           = 3;   // avoids the ref_dim==2 priors path
    w.pdf_class.assign(1, 0.0);
    w.pdf_direction.resize(1); // one empty inner array → 0 bytes packed
    w.BPref.resize(1);         // BPref[0] is accessed by getPackSize() even when empty
    w.BPref[0].ref_dim        = 3;
    w.BPref[0].padding_factor = 0.0;
    w.BPref[0].initialiseDataAndWeight(0); // sets pad_size=3, allocs 18 complex+18 real
    w.BPref[0].data.initZeros();
    w.BPref[0].weight.initZeros();
    w.LL = ll_value;
}

// ---------------------------------------------------------------------------
// Test 10: combineAllWeightedSumsImpl() with 3 ranks.
//
// Leader (rank 0) participates only in the final barrier.  Follower 1 starts
// with LL=10, follower 2 with LL=20.  After the ring-reduce both must hold
// the global sum (30).
//
// Requires ≥ 3 ranks: with 2 ranks there is only 1 follower per subset so
// the calling code's guard "(size-1)/nr_halfsets > 1" is false and the impl
// is never entered.
// ---------------------------------------------------------------------------
TEST_F(ExpectationRefactorMpiTest, CombineAllWeightedSumsImplWith3Ranks)
{
    if (g_node->size < 3)
        GTEST_SKIP() << "CombineAllWeightedSumsImpl requires at least 3 MPI ranks";

    const RFLOAT ll1 = 10.0, ll2 = 20.0;
    if (!g_node->isLeader())
        setupMinimalWsumModel(opt, g_node->rank == 1 ? ll1 : ll2);

    opt.do_split_random_halves = false;  // nr_halfsets = 1 → all followers one subset

    opt.combineAllWeightedSumsImpl();

    if (!g_node->isLeader())
        EXPECT_DOUBLE_EQ(opt.wsum_model.LL, ll1 + ll2)
            << "Every follower must hold the global LL sum after ring-reduce";
}

// ---------------------------------------------------------------------------
// Test 11: combineWeightedSumsTwoHalvesImpl() with 3 ranks.
//
// Follower 1 (rank 1, halfset 1) starts with LL=10; follower 2 (rank 2,
// halfset 2) starts with LL=20.  After combining and broadcasting, every
// follower must hold the sum (30).
//
// Requires ≥ 3 ranks so that both rank 1 and rank 2 exist.
// ---------------------------------------------------------------------------
TEST_F(ExpectationRefactorMpiTest, CombineWeightedSumsTwoHalvesImplWith3Ranks)
{
    if (g_node->size < 3)
        GTEST_SKIP() << "CombineWeightedSumsTwoHalvesImpl requires at least 3 MPI ranks";

    const RFLOAT ll1 = 10.0, ll2 = 20.0;
    if (!g_node->isLeader())
        setupMinimalWsumModel(opt, g_node->rank == 1 ? ll1 : ll2);

    opt.combineWeightedSumsTwoHalvesImpl();

    if (!g_node->isLeader())
        EXPECT_DOUBLE_EQ(opt.wsum_model.LL, ll1 + ll2)
            << "Every follower must hold the cross-half LL sum after combining";
}

// ---------------------------------------------------------------------------
// Test 12: broadcastSplitHalfReconstruction(0) with 3 ranks.
//
// With only one follower per subset there are no cross-subset receivers, so
// the function is a send/recv-free no-op.  The meaningful assertion is that
// all ranks reach the post-call barrier without deadlocking.
// ---------------------------------------------------------------------------
TEST_F(ExpectationRefactorMpiTest, BroadcastSplitHalfReconstructionWith3Ranks)
{
    if (g_node->size < 3)
        GTEST_SKIP() << "BroadcastSplitHalfReconstruction requires at least 3 MPI ranks";

    opt.do_grad = false;  // suppress Igrad1/Igrad2 accesses

    opt.broadcastSplitHalfReconstruction(/*ith_recons=*/0);

    // All ranks reaching this barrier without hanging is the real assertion.
    MPI_Barrier(MPI_COMM_WORLD);
    SUCCEED();
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
