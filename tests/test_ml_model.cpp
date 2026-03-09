/*
 * Unit tests for src/ml_model.h / src/ml_model.cpp (MlModel and MlWsumModel)
 *
 * Covers: constructor zeroed state, clear(), initialise() vector sizing,
 *         getResolution, getResolutionAngstrom, getPixelFromResolution,
 *         initialisePdfDirection, initialiseHelicalParametersLists,
 *         assignment operator copy
 */

#include <gtest/gtest.h>
#include "src/ml_model.h"

static const double TOL = 1e-9;

// ---------------------------------------------------------------------------
// MlModel constructor
// ---------------------------------------------------------------------------

TEST(MlModelTest, DefaultConstructor_ZeroedScalars)
{
    MlModel m;
    EXPECT_EQ(m.nr_classes,       0);
    EXPECT_EQ(m.nr_bodies,        0);
    EXPECT_EQ(m.nr_groups,        0);
    EXPECT_EQ(m.nr_optics_groups, 0);
    EXPECT_EQ(m.ori_size,         0);
    EXPECT_NEAR(m.pixel_size,     0.0, TOL);
    EXPECT_NEAR(m.LL,             0.0, TOL);
    EXPECT_FALSE(m.do_grad);
}

TEST(MlModelTest, DefaultConstructor_EmptyVectors)
{
    MlModel m;
    EXPECT_TRUE(m.pdf_class.empty());
    EXPECT_TRUE(m.scale_correction.empty());
    EXPECT_TRUE(m.tau2_class.empty());
    EXPECT_TRUE(m.acc_rot.empty());
    EXPECT_TRUE(m.acc_trans.empty());
    EXPECT_TRUE(m.helical_twist.empty());
    EXPECT_TRUE(m.helical_rise.empty());
}

// ---------------------------------------------------------------------------
// clear()
// ---------------------------------------------------------------------------

TEST(MlModelTest, Clear_ResetsAllVectors)
{
    MlModel m;
    m.nr_classes = 2;
    m.nr_bodies  = 1;
    m.nr_groups  = 3;
    m.nr_optics_groups = 2;
    m.ori_size   = 64;
    m.padding_factor = 2;
    m.interpolator = 0;
    m.r_min_nn = 10;
    m.data_dim = 2;
    m.initialise();

    m.clear();

    EXPECT_TRUE(m.pdf_class.empty());
    EXPECT_TRUE(m.scale_correction.empty());
    EXPECT_TRUE(m.tau2_class.empty());
    EXPECT_EQ(m.nr_classes, 0);
}

// ---------------------------------------------------------------------------
// initialise() — vector sizes
// ---------------------------------------------------------------------------

TEST(MlModelTest, Initialise_SizesForClasses)
{
    MlModel m;
    m.nr_classes       = 3;
    m.nr_bodies        = 1;
    m.nr_groups        = 2;
    m.nr_optics_groups = 1;
    m.ori_size         = 32;
    m.padding_factor   = 2;
    m.interpolator     = 0;
    m.r_min_nn         = 5;
    m.data_dim         = 2;
    m.initialise();

    EXPECT_EQ((int)m.pdf_class.size(),       3);
    EXPECT_EQ((int)m.scale_correction.size(),2);
    EXPECT_EQ((int)m.tau2_class.size(),      3); // nr_classes * nr_bodies
    EXPECT_EQ((int)m.acc_rot.size(),         3);
    EXPECT_EQ((int)m.acc_trans.size(),       3);
    EXPECT_EQ((int)m.helical_twist.size(),   3);
    EXPECT_EQ((int)m.helical_rise.size(),    3);
}

TEST(MlModelTest, Initialise_PdfClassSumsToOne)
{
    MlModel m;
    m.nr_classes       = 4;
    m.nr_bodies        = 1;
    m.nr_groups        = 1;
    m.nr_optics_groups = 1;
    m.ori_size         = 32;
    m.padding_factor   = 2;
    m.interpolator     = 0;
    m.r_min_nn         = 5;
    m.data_dim         = 2;
    m.initialise();

    double sum = 0.;
    for (auto v : m.pdf_class) sum += v;
    EXPECT_NEAR(sum, 1.0, 1e-6);
}

TEST(MlModelTest, Initialise_ScaleCorrectionDefaultOne)
{
    MlModel m;
    m.nr_classes       = 1;
    m.nr_bodies        = 1;
    m.nr_groups        = 3;
    m.nr_optics_groups = 1;
    m.ori_size         = 32;
    m.padding_factor   = 2;
    m.interpolator     = 0;
    m.r_min_nn         = 5;
    m.data_dim         = 2;
    m.initialise();

    for (auto v : m.scale_correction)
        EXPECT_NEAR(v, 1.0, TOL);
}

TEST(MlModelTest, Initialise_BfactorCorrectionDefaultZero)
{
    MlModel m;
    m.nr_classes       = 1;
    m.nr_bodies        = 1;
    m.nr_groups        = 2;
    m.nr_optics_groups = 1;
    m.ori_size         = 32;
    m.padding_factor   = 2;
    m.interpolator     = 0;
    m.r_min_nn         = 5;
    m.data_dim         = 2;
    m.initialise();

    for (auto v : m.bfactor_correction)
        EXPECT_NEAR(v, 0.0, TOL);
}

// ---------------------------------------------------------------------------
// getResolution / getResolutionAngstrom / getPixelFromResolution
// ---------------------------------------------------------------------------

TEST(MlModelTest, GetResolution_ZeroPixel)
{
    MlModel m;
    m.pixel_size = 1.35;
    m.ori_size   = 100;
    // ipix=0 → 0 / (1.35 * 100) = 0
    EXPECT_NEAR(m.getResolution(0), 0.0, TOL);
}

TEST(MlModelTest, GetResolution_Nonzero)
{
    MlModel m;
    m.pixel_size = 1.0;
    m.ori_size   = 100;
    // ipix=10 → 10 / (1.0 * 100) = 0.1
    EXPECT_NEAR(m.getResolution(10), 0.1, TOL);
}

TEST(MlModelTest, GetResolutionAngstrom_ZeroPixel_Returns999)
{
    MlModel m;
    m.pixel_size = 1.35;
    m.ori_size   = 100;
    EXPECT_NEAR(m.getResolutionAngstrom(0), 999.0, TOL);
}

TEST(MlModelTest, GetResolutionAngstrom_Nonzero)
{
    MlModel m;
    m.pixel_size = 1.0;
    m.ori_size   = 100;
    // ipix=10 → (1.0 * 100) / 10 = 10.0 Å
    EXPECT_NEAR(m.getResolutionAngstrom(10), 10.0, TOL);
}

TEST(MlModelTest, GetResolution_GetResolutionAngstrom_Inverse)
{
    MlModel m;
    m.pixel_size = 1.35;
    m.ori_size   = 200;
    int ipix = 20;
    RFLOAT resol = m.getResolution(ipix);
    RFLOAT angst = m.getResolutionAngstrom(ipix);
    EXPECT_NEAR(resol * angst, 1.0, 1e-6);
}

TEST(MlModelTest, GetPixelFromResolution_RoundTrip)
{
    MlModel m;
    m.pixel_size = 1.35;
    m.ori_size   = 200;
    int ipix_orig = 15;
    RFLOAT resol = m.getResolution(ipix_orig);
    int ipix_back = m.getPixelFromResolution(resol);
    EXPECT_EQ(ipix_back, ipix_orig);
}

// ---------------------------------------------------------------------------
// initialisePdfDirection
// ---------------------------------------------------------------------------

TEST(MlModelTest, InitialisePdfDirection_EvenDistribution)
{
    // Use nr_classes=1 so that each class's pdf sums to 1.0
    // (each element = 1/(nr_classes*newsize), so per-class sum = 1/nr_classes)
    MlModel m;
    m.nr_classes       = 1;
    m.nr_bodies        = 1;
    m.nr_groups        = 1;
    m.nr_optics_groups = 1;
    m.ori_size         = 32;
    m.padding_factor   = 2;
    m.interpolator     = 0;
    m.r_min_nn         = 5;
    m.data_dim         = 2;
    m.initialise();

    m.initialisePdfDirection(100);

    EXPECT_EQ(m.nr_directions, 100);
    ASSERT_EQ((long long int)m.pdf_direction[0].nzyxdim, 100LL);
    double sum = 0.;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(m.pdf_direction[0])
        sum += DIRECT_MULTIDIM_ELEM(m.pdf_direction[0], n);
    EXPECT_NEAR(sum, 1.0, 1e-6);
}

// ---------------------------------------------------------------------------
// initialiseHelicalParametersLists
// ---------------------------------------------------------------------------

TEST(MlModelTest, InitialiseHelicalParametersLists_SetsValues)
{
    MlModel m;
    m.nr_classes       = 3;
    m.nr_bodies        = 1;
    m.nr_groups        = 1;
    m.nr_optics_groups = 1;
    m.ori_size         = 32;
    m.padding_factor   = 2;
    m.interpolator     = 0;
    m.r_min_nn         = 5;
    m.data_dim         = 2;
    m.initialise();

    m.initialiseHelicalParametersLists(-1.0, 4.75);

    for (int c = 0; c < m.nr_classes; c++)
    {
        EXPECT_NEAR(m.helical_twist[c], -1.0,  TOL);
        EXPECT_NEAR(m.helical_rise[c],   4.75, TOL);
    }
}

// ---------------------------------------------------------------------------
// Assignment operator
// ---------------------------------------------------------------------------

TEST(MlModelTest, AssignmentOperator_CopiesScalars)
{
    MlModel src;
    src.nr_classes       = 2;
    src.nr_bodies        = 1;
    src.nr_groups        = 3;
    src.nr_optics_groups = 1;
    src.ori_size         = 64;
    src.pixel_size       = 1.35;
    src.LL               = 42.0;
    src.padding_factor   = 2;
    src.interpolator     = 0;
    src.r_min_nn         = 5;
    src.data_dim         = 2;
    src.initialise();

    MlModel dst;
    dst = src;

    EXPECT_EQ(dst.nr_classes,  2);
    EXPECT_EQ(dst.nr_groups,   3);
    EXPECT_EQ(dst.ori_size,    64);
    EXPECT_NEAR(dst.pixel_size, 1.35, TOL);
    EXPECT_NEAR(dst.LL,         42.0, TOL);
    EXPECT_EQ((int)dst.pdf_class.size(), 2);
}

// ---------------------------------------------------------------------------
// MlWsumModel basic construction
// ---------------------------------------------------------------------------

TEST(MlWsumModelTest, DefaultConstructor_EmptyVectors)
{
    MlWsumModel ws;
    EXPECT_TRUE(ws.BPref.empty());
    EXPECT_TRUE(ws.sumw_group.empty());
    EXPECT_EQ(ws.nr_classes, 0);
}

TEST(MlWsumModelTest, Clear_ResetsAll)
{
    MlWsumModel ws;
    ws.sumw_group.push_back(1.0);
    ws.clear();
    EXPECT_TRUE(ws.sumw_group.empty());
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
