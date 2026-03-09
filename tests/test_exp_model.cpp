/*
 * Unit tests for src/exp_model.h / src/exp_model.cpp
 *
 * Covers: ExpImage copy/assign, ExpParticle copy/numberOfImages,
 *         ExpGroup copy/assign, Experiment constructor/clear,
 *         numberOfGroups, numberOfParticles, addGroup,
 *         getGroupId, getRandomSubset, getOpticsGroup,
 *         numberOfImagesInParticle, divideParticlesInRandomHalves
 */

#include <gtest/gtest.h>
#include "src/exp_model.h"

// ---------------------------------------------------------------------------
// ExpImage
// ---------------------------------------------------------------------------

TEST(ExpImageTest, DefaultConstructor)
{
    ExpImage img;
    // Default constructed — members uninitialized but object must be constructable
    (void)img; // no crash
}

TEST(ExpImageTest, CopyConstructor)
{
    ExpImage src;
    src.particle_id = 7;
    src.defU = 1.5f;
    src.defV = 2.5f;
    src.dose = 3.0f;

    ExpImage dst(src);
    EXPECT_EQ(dst.particle_id, 7);
    EXPECT_FLOAT_EQ(dst.defU, 1.5f);
    EXPECT_FLOAT_EQ(dst.defV, 2.5f);
    EXPECT_FLOAT_EQ(dst.dose, 3.0f);
}

TEST(ExpImageTest, AssignmentOperator)
{
    ExpImage src;
    src.particle_id  = 3;
    src.phase_shift  = 0.5f;
    src.scale        = 1.0f;
    src.bfactor      = 0.0f;

    ExpImage dst;
    dst = src;
    EXPECT_EQ(dst.particle_id, 3);
    EXPECT_FLOAT_EQ(dst.phase_shift, 0.5f);
}

// ---------------------------------------------------------------------------
// ExpParticle
// ---------------------------------------------------------------------------

TEST(ExpParticleTest, DefaultConstructor)
{
    ExpParticle p;
    (void)p; // no crash
}

TEST(ExpParticleTest, NumberOfImages_Empty)
{
    ExpParticle p;
    EXPECT_EQ(p.numberOfImages(), 0);
}

TEST(ExpParticleTest, NumberOfImages_AfterPush)
{
    ExpParticle p;
    ExpImage img;
    p.images.push_back(img);
    p.images.push_back(img);
    EXPECT_EQ(p.numberOfImages(), 2);
}

TEST(ExpParticleTest, CopyConstructor)
{
    ExpParticle src;
    src.id             = 5;
    src.name           = "particle_5";
    src.group_id       = 2;
    src.optics_group   = 0;
    src.random_subset  = 1;
    src.tomogram_id    = 0;
    src.optics_group_id = 3;
    ExpImage img;
    img.particle_id = 5;
    src.images.push_back(img);

    ExpParticle dst(src);
    EXPECT_EQ(dst.id,           5);
    EXPECT_EQ(dst.name,         "particle_5");
    EXPECT_EQ(dst.group_id,     2);
    EXPECT_EQ(dst.random_subset,1);
    EXPECT_EQ(dst.numberOfImages(), 1);
    EXPECT_EQ(dst.images[0].particle_id, 5);
}

TEST(ExpParticleTest, AssignmentOperator)
{
    ExpParticle src;
    src.id = 99;
    src.optics_group = 1;

    ExpParticle dst;
    dst = src;
    EXPECT_EQ(dst.id, 99);
    EXPECT_EQ(dst.optics_group, 1);
}

// ---------------------------------------------------------------------------
// ExpGroup
// ---------------------------------------------------------------------------

TEST(ExpGroupTest, DefaultConstructor)
{
    ExpGroup g;
    (void)g; // no crash
}

TEST(ExpGroupTest, CopyConstructor)
{
    ExpGroup src;
    src.id            = 3;
    src.optics_group  = 0;
    src.name          = "grp3";

    ExpGroup dst(src);
    EXPECT_EQ(dst.id,   3);
    EXPECT_EQ(dst.name, "grp3");
}

TEST(ExpGroupTest, AssignmentOperator)
{
    ExpGroup src;
    src.id   = 10;
    src.name = "grpA";

    ExpGroup dst;
    dst = src;
    EXPECT_EQ(dst.id,   10);
    EXPECT_EQ(dst.name, "grpA");
}

// ---------------------------------------------------------------------------
// Experiment — constructor / clear
// ---------------------------------------------------------------------------

TEST(ExperimentTest, DefaultConstructor_EmptyState)
{
    Experiment exp;
    EXPECT_EQ(exp.numberOfParticles(), 0);
    EXPECT_EQ(exp.numberOfGroups(),    0);
    EXPECT_EQ(exp.nr_particles_subset1, 0);
    EXPECT_EQ(exp.nr_particles_subset2, 0);
    EXPECT_EQ(exp.nr_bodies, 1);
    EXPECT_FALSE(exp.is_tomo);
    EXPECT_FALSE(exp.is_3D);
}

TEST(ExperimentTest, Clear_ResetsState)
{
    Experiment exp;
    // Directly push a particle and a group
    ExpParticle p; p.id = 0; exp.particles.push_back(p);
    ExpGroup g;    g.id = 0; exp.groups.push_back(g);
    ASSERT_EQ(exp.numberOfParticles(), 1);
    ASSERT_EQ(exp.numberOfGroups(),    1);

    exp.clear();
    EXPECT_EQ(exp.numberOfParticles(), 0);
    EXPECT_EQ(exp.numberOfGroups(),    0);
}

// ---------------------------------------------------------------------------
// numberOfParticles
// ---------------------------------------------------------------------------

TEST(ExperimentTest, NumberOfParticles_AllSubsets)
{
    Experiment exp;
    ExpParticle p1, p2, p3;
    p1.id = 0; p2.id = 1; p3.id = 2;
    exp.particles.push_back(p1);
    exp.particles.push_back(p2);
    exp.particles.push_back(p3);
    EXPECT_EQ(exp.numberOfParticles(0), 3); // all
}

TEST(ExperimentTest, NumberOfParticles_BySubset)
{
    Experiment exp;
    exp.nr_particles_subset1 = 4;
    exp.nr_particles_subset2 = 3;
    EXPECT_EQ(exp.numberOfParticles(1), 4);
    EXPECT_EQ(exp.numberOfParticles(2), 3);
}

// ---------------------------------------------------------------------------
// addGroup
// ---------------------------------------------------------------------------

TEST(ExperimentTest, AddGroup_IncreasesGroupCount)
{
    Experiment exp;
    // We can call addGroup directly — it doesn't need obsModel
    exp.groups.clear();
    ExpGroup g0; g0.id = 0; g0.name = "mic1"; g0.optics_group = 0;
    ExpGroup g1; g1.id = 1; g1.name = "mic2"; g1.optics_group = 0;
    exp.groups.push_back(g0);
    exp.groups.push_back(g1);
    EXPECT_EQ(exp.numberOfGroups(), 2);
}

// ---------------------------------------------------------------------------
// Direct particle push — getGroupId / getRandomSubset / getOpticsGroup
// ---------------------------------------------------------------------------

TEST(ExperimentTest, GetGroupId_DirectPush)
{
    Experiment exp;
    ExpParticle p;
    p.id           = 0;
    p.group_id     = 5;
    p.random_subset = 2;
    p.optics_group  = 1;
    exp.particles.push_back(p);

    EXPECT_EQ(exp.getGroupId(0),     5);
    EXPECT_EQ(exp.getRandomSubset(0), 2);
    EXPECT_EQ(exp.getOpticsGroup(0),  1);
}

// ---------------------------------------------------------------------------
// numberOfImagesInParticle
// ---------------------------------------------------------------------------

TEST(ExperimentTest, NumberOfImagesInParticle)
{
    Experiment exp;
    ExpParticle p;
    p.id = 0;
    ExpImage img;
    p.images.push_back(img);
    p.images.push_back(img);
    exp.particles.push_back(p);
    EXPECT_EQ(exp.numberOfImagesInParticle(0), 2);
}

// ---------------------------------------------------------------------------
// divideParticlesInRandomHalves — pre-assigned subsets
// ---------------------------------------------------------------------------

TEST(ExperimentTest, DivideParticles_PreAssigned_CountsCorrectly)
{
    // If random_subset is pre-set to 1 or 2 (not all zero), the function
    // just counts them without re-randomising.
    Experiment exp;
    for (int i = 0; i < 4; i++)
    {
        ExpParticle p;
        p.id           = i;
        p.random_subset = (i < 2) ? 1 : 2;
        p.optics_group  = 0;
        p.group_id      = 0;
        exp.particles.push_back(p);
        exp.sorted_idx.push_back(i);
    }
    // Also need MDimg rows for setValue in counting loop
    for (int i = 0; i < 4; i++)
        exp.MDimg.addObject();

    exp.divideParticlesInRandomHalves(42, false);

    EXPECT_EQ(exp.nr_particles_subset1, 2);
    EXPECT_EQ(exp.nr_particles_subset2, 2);
}

// ---------------------------------------------------------------------------
// compareOpticsGroupsParticles (private struct) — tested indirectly via
// randomiseParticlesOrder(), which stable_sorts sorted_idx by optics_group.
// The struct constructor is invoked as the comparator object; operator() is
// called by std::stable_sort to order pairs of particle indices.
// ---------------------------------------------------------------------------

TEST(ExperimentTest, RandomiseParticlesOrder_SortsByOpticsGroup)
{
    // Create 4 particles: optics groups 1, 0, 1, 0 (deliberately out of order)
    Experiment exp;
    for (int i = 0; i < 4; i++)
    {
        ExpParticle p;
        p.id            = i;
        p.random_subset = 0; // not pre-assigned
        p.group_id      = 0;
        p.optics_group  = (i % 2 == 0) ? 1 : 0; // optics groups: 1,0,1,0
        exp.particles.push_back(p);
        exp.sorted_idx.push_back(i);
    }
    // Add MDimg rows
    for (int i = 0; i < 4; i++)
        exp.MDimg.addObject();

    // divideParticlesInRandomHalves assigns random subsets then counts
    exp.divideParticlesInRandomHalves(42, false);

    // randomiseParticlesOrder with do_split_random_halves=false:
    //   shuffles sorted_idx, then stable_sorts the entire range by optics_group
    //   — exercises compareOpticsGroupsParticles constructor and operator().
    exp.randomiseParticlesOrder(42, false);

    // After the sort the sorted_idx must be non-decreasing in optics_group.
    for (int k = 1; k < (int)exp.sorted_idx.size(); k++)
    {
        int prev_og = exp.particles[exp.sorted_idx[k-1]].optics_group;
        int curr_og = exp.particles[exp.sorted_idx[k  ]].optics_group;
        EXPECT_LE(prev_og, curr_og)
            << "sorted_idx[" << k-1 << "]=" << exp.sorted_idx[k-1]
            << " (og=" << prev_og << ") > sorted_idx[" << k << "]="
            << exp.sorted_idx[k] << " (og=" << curr_og << ")";
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
