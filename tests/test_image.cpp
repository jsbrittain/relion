/*
 * Unit tests for src/image.h (Image<T> class)
 *
 * Tests in-memory operations only — no file I/O.
 *
 * Covers:
 *   - Default constructor: empty data array
 *   - Size constructor: Image(Xdim, Ydim)
 *   - getDimensions: returns correct X/Y/Z/N sizes
 *   - getSize: total element count
 *   - data array: readable and writable
 *   - setStatisticsInHeader / MDMainHeader: avg, stddev, min, max
 *   - setSamplingRateInHeader / samplingRateX / samplingRateY
 *   - clear(): resets to empty state
 */

#include <gtest/gtest.h>
#include <cmath>
#include "src/image.h"
#include "src/multidim_array.h"
#include "src/metadata_label.h"

// ------------------------------------------------------- constructors --

TEST(ImageTest, DefaultConstructor_DataIsEmpty)
{
    Image<RFLOAT> img;
    EXPECT_EQ(img.getSize(), (size_t)0);
}

TEST(ImageTest, SizeConstructor_2D)
{
    Image<RFLOAT> img(64, 64);
    int Xdim, Ydim, Zdim;
    long int Ndim;
    img.getDimensions(Xdim, Ydim, Zdim, Ndim);

    EXPECT_EQ(Xdim, 64);
    EXPECT_EQ(Ydim, 64);
    EXPECT_EQ(Zdim, 1);
    EXPECT_EQ(Ndim, 1L);
}

TEST(ImageTest, SizeConstructor_3D)
{
    Image<RFLOAT> img(10, 10, 5);
    int Xdim, Ydim, Zdim;
    long int Ndim;
    img.getDimensions(Xdim, Ydim, Zdim, Ndim);

    EXPECT_EQ(Xdim, 10);
    EXPECT_EQ(Ydim, 10);
    EXPECT_EQ(Zdim, 5);
    EXPECT_EQ(Ndim, 1L);
}

TEST(ImageTest, SizeConstructor_GetSize)
{
    Image<RFLOAT> img(8, 8);
    // Total elements: 8*8*1*1 = 64
    EXPECT_EQ(img.getSize(), (size_t)64);
}

// --------------------------------------------------------- data access --

TEST(ImageTest, DataFilled_CorrectValues)
{
    Image<RFLOAT> img(4, 4);
    // Fill data array directly
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(img.data)
        DIRECT_MULTIDIM_ELEM(img.data, n) = static_cast<RFLOAT>(n);

    // Verify the first and last elements
    EXPECT_NEAR(DIRECT_MULTIDIM_ELEM(img.data, 0), 0.0, 1e-9);
    EXPECT_NEAR(DIRECT_MULTIDIM_ELEM(img.data, 15), 15.0, 1e-9);
}

// ------------------------------------------- setStatisticsInHeader --

TEST(ImageTest, SetStatistics_ConstantVolume)
{
    // Constant image: avg=5, stddev=0, min=max=5
    Image<RFLOAT> img(4, 4);
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(img.data)
        DIRECT_MULTIDIM_ELEM(img.data, n) = 5.0;

    img.setStatisticsInHeader();

    RFLOAT avg = 0, stddev = 0, minval = 0, maxval = 0;
    img.MDMainHeader.getValue(EMDL_IMAGE_STATS_AVG,    avg);
    img.MDMainHeader.getValue(EMDL_IMAGE_STATS_STDDEV, stddev);
    img.MDMainHeader.getValue(EMDL_IMAGE_STATS_MIN,    minval);
    img.MDMainHeader.getValue(EMDL_IMAGE_STATS_MAX,    maxval);

    EXPECT_NEAR(avg,    5.0, 1e-6);
    EXPECT_NEAR(stddev, 0.0, 1e-6);
    EXPECT_NEAR(minval, 5.0, 1e-6);
    EXPECT_NEAR(maxval, 5.0, 1e-6);
}

TEST(ImageTest, SetStatistics_MinMaxCorrect)
{
    // 4-element image: values 1, 2, 3, 4 → avg=2.5, min=1, max=4
    Image<RFLOAT> img(2, 2);
    DIRECT_MULTIDIM_ELEM(img.data, 0) = 1.0;
    DIRECT_MULTIDIM_ELEM(img.data, 1) = 2.0;
    DIRECT_MULTIDIM_ELEM(img.data, 2) = 3.0;
    DIRECT_MULTIDIM_ELEM(img.data, 3) = 4.0;

    img.setStatisticsInHeader();

    RFLOAT avg = 0, minval = 0, maxval = 0;
    img.MDMainHeader.getValue(EMDL_IMAGE_STATS_AVG, avg);
    img.MDMainHeader.getValue(EMDL_IMAGE_STATS_MIN, minval);
    img.MDMainHeader.getValue(EMDL_IMAGE_STATS_MAX, maxval);

    EXPECT_NEAR(avg,    2.5, 1e-6);
    EXPECT_NEAR(minval, 1.0, 1e-6);
    EXPECT_NEAR(maxval, 4.0, 1e-6);
}

TEST(ImageTest, SetStatistics_StddevNonzero)
{
    // Values 0 and 1 alternating: stddev > 0
    Image<RFLOAT> img(4, 4);
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(img.data)
        DIRECT_MULTIDIM_ELEM(img.data, n) = (n % 2 == 0) ? 0.0 : 1.0;

    img.setStatisticsInHeader();

    RFLOAT stddev = 0;
    img.MDMainHeader.getValue(EMDL_IMAGE_STATS_STDDEV, stddev);
    EXPECT_GT(stddev, 0.0);
}

// ----------------------------------------- setSamplingRateInHeader --

TEST(ImageTest, DefaultSamplingRate_IsOne)
{
    // Before setSamplingRateInHeader, default is 1.0
    Image<RFLOAT> img(4, 4);
    EXPECT_NEAR(img.samplingRateX(), 1.0, 1e-6);
    EXPECT_NEAR(img.samplingRateY(), 1.0, 1e-6);
}

TEST(ImageTest, SetSamplingRate_IsoIsotropic)
{
    Image<RFLOAT> img(4, 4);
    img.setSamplingRateInHeader(2.5);

    EXPECT_NEAR(img.samplingRateX(), 2.5, 1e-6);
    EXPECT_NEAR(img.samplingRateY(), 2.5, 1e-6);  // copies X when Y not specified
}

TEST(ImageTest, SetSamplingRate_Anisotropic)
{
    Image<RFLOAT> img(4, 4);
    img.setSamplingRateInHeader(1.0, 2.0);

    EXPECT_NEAR(img.samplingRateX(), 1.0, 1e-6);
    EXPECT_NEAR(img.samplingRateY(), 2.0, 1e-6);
}

// ------------------------------------------------------------ clear --

TEST(ImageTest, Clear_DataBecomesEmpty)
{
    Image<RFLOAT> img(16, 16);
    ASSERT_EQ(img.getSize(), (size_t)256);

    img.clear();
    EXPECT_EQ(img.getSize(), (size_t)0);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
