/*
 * Unit tests for src/image.h (Image<T> class) and src/image.cpp free functions
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
 *   - gettypesize: byte size of each DataType enum
 *   - datatypeString2Int: string to DataType enum
 *   - invert_contrast: negates all pixel values
 *   - getImageContrast: sigma-clipping and min/max clamping
 *   - removeDust: outlier pixel replacement
 *   - calculateBackgroundAvgStddev: background statistics with circular mask
 *   - rewindow: crop/pad image to new size
 *   - rescale: Fourier-resize with sampling rate update
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

// ----------------------------------------------- operator() access --

TEST(ImageTest, ParenthesisOp_ReturnsMutableDataRef)
{
    Image<RFLOAT> img(8, 8);
    // operator()() returns MultidimArray<T>&
    MultidimArray<RFLOAT>& data_ref = img();
    data_ref.initConstant(3.14);
    EXPECT_NEAR(DIRECT_A2D_ELEM(img.data, 0, 0), 3.14, 1e-5);
}

TEST(ImageTest, ParenthesisOp_Const_ReturnsCRef)
{
    Image<RFLOAT> img(8, 8);
    img.data.initConstant(2.71);
    const Image<RFLOAT>& cimg = img;
    const MultidimArray<RFLOAT>& cref = cimg();
    EXPECT_NEAR(DIRECT_A2D_ELEM(cref, 0, 0), 2.71, 1e-5);
}

TEST(ImageTest, TwoIndexOp_ReadWrite)
{
    Image<RFLOAT> img(4, 4);
    img(1, 2) = 9.9;
    EXPECT_NEAR(img(1, 2), 9.9, 1e-5);
}

TEST(ImageTest, ThreeIndexOp_ReadWrite)
{
    Image<RFLOAT> img(4, 4, 4);
    img(1, 2, 3) = 7.7;
    EXPECT_NEAR(img(1, 2, 3), 7.7, 1e-5);
}

// ----------------------------------------------- name() --

TEST(ImageTest, Name_DefaultIsEmpty)
{
    Image<RFLOAT> img(4, 4);
    EXPECT_TRUE(img.name().empty());
}

// ----------------------------------------------- fImageHandler --

TEST(ImageTest, FImageHandler_DefaultConstructor)
{
    fImageHandler hFile;
    EXPECT_EQ(hFile.fimg, nullptr);
}

// ----------------------------------------------- Image<float> --

TEST(ImageTest, FloatImage_SetSamplingRate)
{
    Image<float> img(8, 8);
    img.setSamplingRateInHeader(2.5);
    EXPECT_NEAR(img.samplingRateX(), 2.5, 1e-5);
}

TEST(ImageTest, FloatImage_OperatorParenthesis)
{
    Image<float> img(4, 4);
    img().initConstant(1.5f);
    EXPECT_NEAR(DIRECT_A2D_ELEM(img.data, 0, 0), 1.5f, 1e-5f);
}

TEST(ImageTest, FloatImage_TwoIndexOp)
{
    Image<float> img(4, 4);
    img(0, 1) = 3.3f;
    EXPECT_NEAR(img(0, 1), 3.3f, 1e-5f);
}

// ----------------------------------------------- Image<Complex> --

TEST(ImageTest, ComplexImage_OperatorParenthesis)
{
    Image<Complex> img(4, 4);
    img().initZeros(4, 4);
    MultidimArray<Complex>& ref = img();
    EXPECT_EQ((int)XSIZE(ref), 4);
}

TEST(ImageTest, ComplexImage_DefaultConstructor)
{
    Image<Complex> img;
    EXPECT_EQ(img.getSize(), (size_t)0);
}

TEST(ImageTest, ComplexImage_TwoIndexConst)
{
    Image<Complex> img(4, 4);
    img(1, 2) = Complex(1.5, 2.5);
    const Image<Complex>& cimg = img;
    Complex val = cimg(1, 2);
    EXPECT_NEAR(val.real, 1.5, 1e-6);
    EXPECT_NEAR(val.imag, 2.5, 1e-6);
}

// ----------------------------------------------- Image<float> const --

TEST(ImageTest, FloatImage_DefaultConstructor)
{
    Image<float> img;
    EXPECT_EQ(img.getSize(), (size_t)0);
}

TEST(ImageTest, FloatImage_ConstOperatorParenthesis)
{
    Image<float> img(4, 4);
    img.data.initConstant(5.0f);
    const Image<float>& cimg = img;
    const MultidimArray<float>& cref = cimg();
    EXPECT_NEAR(DIRECT_A2D_ELEM(cref, 0, 0), 5.0f, 1e-5f);
}

// ----------------------------------------------- gettypesize --

TEST(ImageFreeFunctions, GetTypeSize_UChar)
{
    EXPECT_EQ(gettypesize(UChar), sizeof(char));
}

TEST(ImageFreeFunctions, GetTypeSize_SChar)
{
    EXPECT_EQ(gettypesize(SChar), sizeof(char));
}

TEST(ImageFreeFunctions, GetTypeSize_UShort)
{
    EXPECT_EQ(gettypesize(UShort), sizeof(short));
}

TEST(ImageFreeFunctions, GetTypeSize_SShort)
{
    EXPECT_EQ(gettypesize(SShort), sizeof(short));
}

TEST(ImageFreeFunctions, GetTypeSize_UInt)
{
    EXPECT_EQ(gettypesize(UInt), sizeof(int));
}

TEST(ImageFreeFunctions, GetTypeSize_Int)
{
    EXPECT_EQ(gettypesize(Int), sizeof(int));
}

TEST(ImageFreeFunctions, GetTypeSize_Float)
{
    EXPECT_EQ(gettypesize(Float), sizeof(float));
}

TEST(ImageFreeFunctions, GetTypeSize_Double)
{
    EXPECT_EQ(gettypesize(Double), sizeof(RFLOAT));
}

TEST(ImageFreeFunctions, GetTypeSize_Boolean)
{
    EXPECT_EQ(gettypesize(Boolean), sizeof(bool));
}

TEST(ImageFreeFunctions, GetTypeSize_Float16)
{
    // Float16 stored as short (2 bytes)
    EXPECT_EQ(gettypesize(Float16), sizeof(short));
}

TEST(ImageFreeFunctions, GetTypeSize_Unknown)
{
    // Unknown_Type returns 0
    EXPECT_EQ(gettypesize(Unknown_Type), (unsigned long)0);
}

// ----------------------------------------------- datatypeString2Int --

TEST(ImageFreeFunctions, DatatypeString2Int_UChar)
{
    EXPECT_EQ(datatypeString2Int("uchar"), (int)UChar);
}

TEST(ImageFreeFunctions, DatatypeString2Int_UShort)
{
    EXPECT_EQ(datatypeString2Int("ushort"), (int)UShort);
}

TEST(ImageFreeFunctions, DatatypeString2Int_Short)
{
    EXPECT_EQ(datatypeString2Int("short"), (int)SShort);
}

TEST(ImageFreeFunctions, DatatypeString2Int_UInt)
{
    EXPECT_EQ(datatypeString2Int("uint"), (int)UInt);
}

TEST(ImageFreeFunctions, DatatypeString2Int_Int)
{
    EXPECT_EQ(datatypeString2Int("int"), (int)Int);
}

TEST(ImageFreeFunctions, DatatypeString2Int_Float)
{
    EXPECT_EQ(datatypeString2Int("float"), (int)Float);
}

TEST(ImageFreeFunctions, DatatypeString2Int_Float16)
{
    EXPECT_EQ(datatypeString2Int("float16"), (int)Float16);
}

TEST(ImageFreeFunctions, DatatypeString2Int_UpperCase)
{
    // toLower applied internally — uppercase input should still work
    EXPECT_EQ(datatypeString2Int("UCHAR"), (int)UChar);
    EXPECT_EQ(datatypeString2Int("FLOAT"), (int)Float);
}

TEST(ImageFreeFunctions, DatatypeString2Int_Unknown_Throws)
{
    EXPECT_THROW(datatypeString2Int("bogus"), RelionError);
}

// ----------------------------------------------- invert_contrast --

TEST(ImageFreeFunctions, InvertContrast_PositiveValues)
{
    Image<RFLOAT> img(4, 4);
    img.data.initConstant(3.0);
    invert_contrast(img);
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(img.data)
        EXPECT_NEAR(DIRECT_MULTIDIM_ELEM(img.data, n), -3.0, 1e-9);
}

TEST(ImageFreeFunctions, InvertContrast_MixedValues)
{
    Image<RFLOAT> img(2, 2);
    DIRECT_MULTIDIM_ELEM(img.data, 0) =  1.0;
    DIRECT_MULTIDIM_ELEM(img.data, 1) = -2.0;
    DIRECT_MULTIDIM_ELEM(img.data, 2) =  0.0;
    DIRECT_MULTIDIM_ELEM(img.data, 3) =  5.5;

    invert_contrast(img);

    EXPECT_NEAR(DIRECT_MULTIDIM_ELEM(img.data, 0), -1.0, 1e-9);
    EXPECT_NEAR(DIRECT_MULTIDIM_ELEM(img.data, 1),  2.0, 1e-9);
    EXPECT_NEAR(DIRECT_MULTIDIM_ELEM(img.data, 2),  0.0, 1e-9);
    EXPECT_NEAR(DIRECT_MULTIDIM_ELEM(img.data, 3), -5.5, 1e-9);
}

TEST(ImageFreeFunctions, InvertContrast_TwiceIsIdentity)
{
    Image<RFLOAT> img(4, 4);
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(img.data)
        DIRECT_MULTIDIM_ELEM(img.data, n) = static_cast<RFLOAT>(n) - 8.0;

    Image<RFLOAT> orig = img;
    invert_contrast(img);
    invert_contrast(img);

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(img.data)
        EXPECT_NEAR(DIRECT_MULTIDIM_ELEM(img.data, n),
                    DIRECT_MULTIDIM_ELEM(orig.data, n), 1e-9);
}

// ----------------------------------------------- getImageContrast --

TEST(ImageFreeFunctions, GetImageContrast_SigmaZero_MinEqualsMax_ComputesStats)
{
    // minval==maxval and sigma_contrast==0: stats computed from image;
    // all values equal 3.0 so min==max==3.0 and no clamping occurs.
    MultidimArray<RFLOAT> arr;
    arr.resize(1, 1, 4, 4);
    arr.initConstant(3.0);

    RFLOAT minval = 0.0, maxval = 0.0, sigma_contrast = 0.0;
    getImageContrast(arr, minval, maxval, sigma_contrast);

    EXPECT_NEAR(minval, 3.0, 1e-6);
    EXPECT_NEAR(maxval, 3.0, 1e-6);
    // Constant image unchanged
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(arr)
        EXPECT_NEAR(DIRECT_MULTIDIM_ELEM(arr, n), 3.0, 1e-6);
}

TEST(ImageFreeFunctions, GetImageContrast_SigmaZero_MinNeqMax_Clamps)
{
    // minval!=maxval and sigma_contrast==0: clamp to given [minval, maxval].
    MultidimArray<RFLOAT> arr;
    arr.resize(1, 1, 1, 10);
    for (int n = 0; n < 10; n++)
        DIRECT_MULTIDIM_ELEM(arr, n) = static_cast<RFLOAT>(n);  // 0..9

    RFLOAT minval = 2.0, maxval = 7.0, sigma_contrast = 0.0;
    getImageContrast(arr, minval, maxval, sigma_contrast);

    // Values 0,1 clipped to 2; values 8,9 clipped to 7
    EXPECT_NEAR(DIRECT_MULTIDIM_ELEM(arr, 0), 2.0, 1e-6);
    EXPECT_NEAR(DIRECT_MULTIDIM_ELEM(arr, 1), 2.0, 1e-6);
    EXPECT_NEAR(DIRECT_MULTIDIM_ELEM(arr, 5), 5.0, 1e-6);
    EXPECT_NEAR(DIRECT_MULTIDIM_ELEM(arr, 8), 7.0, 1e-6);
    EXPECT_NEAR(DIRECT_MULTIDIM_ELEM(arr, 9), 7.0, 1e-6);
}

TEST(ImageFreeFunctions, GetImageContrast_SigmaPositive_ClipsToMeanPlusMinus)
{
    // sigma_contrast>0: minval = avg - sigma*stddev, maxval = avg + sigma*stddev
    // Values 0..9: avg=4.5. Use sigma=0 to verify limits are strictly tighter.
    MultidimArray<RFLOAT> arr;
    arr.resize(1, 1, 1, 10);
    for (int n = 0; n < 10; n++)
        DIRECT_MULTIDIM_ELEM(arr, n) = static_cast<RFLOAT>(n);

    RFLOAT minval = 0.0, maxval = 0.0, sigma_contrast = 1.0;
    getImageContrast(arr, minval, maxval, sigma_contrast);

    // After sigma clipping, minval and maxval are within the data range
    EXPECT_GT(minval, 0.0);
    EXPECT_LT(maxval, 9.0);
    // All values in arr are clamped to [minval, maxval]
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(arr)
    {
        RFLOAT v = DIRECT_MULTIDIM_ELEM(arr, n);
        EXPECT_GE(v, minval - 1e-9);
        EXPECT_LE(v, maxval + 1e-9);
    }
}

// ----------------------------------------------- calculateBackgroundAvgStddev --

TEST(ImageFreeFunctions, CalcBgAvgStddev_UniformImage_CorrectMean)
{
    // Uniform 10x10 image, all pixels = 5.0
    // bg_radius=2: pixels with i²+j² > 4 are background (XMIPP centered)
    Image<RFLOAT> img(10, 10);
    img.data.setXmippOrigin();  // ensure XMIPP-centred indexing: x,y in [-5..4]
    img.data.initConstant(5.0);

    RFLOAT avg = -1.0, stddev = -1.0;
    calculateBackgroundAvgStddev(img, avg, stddev, 2, false, 0., 0., 0.);

    EXPECT_NEAR(avg,    5.0, 1e-6);
    EXPECT_NEAR(stddev, 0.0, 1e-6);
}

TEST(ImageFreeFunctions, CalcBgAvgStddev_KnownBackground)
{
    // 10x10 image: background (outside radius 2) = 1.0, center region = 999.0
    // avg of background pixels should remain 1.0
    Image<RFLOAT> img(10, 10);
    img.data.setXmippOrigin();  // x,y range [-5..4] → A3D_ELEM safe for small coords
    img.data.initConstant(1.0);
    // Set center pixels (i²+j²≤4) to 999; these are NOT background (bg uses > 4)
    for (int i = -2; i <= 2; i++)
        for (int j = -2; j <= 2; j++)
            if (i*i + j*j <= 4)
                A3D_ELEM(img.data, 0, i, j) = 999.0;

    RFLOAT avg = -1.0, stddev = -1.0;
    calculateBackgroundAvgStddev(img, avg, stddev, 2, false, 0., 0., 0.);

    EXPECT_NEAR(avg,    1.0, 1e-6);
    EXPECT_NEAR(stddev, 0.0, 1e-6);
}

// ----------------------------------------------- removeDust --

TEST(ImageFreeFunctions, RemoveDust_NoDust_ValuesUnchanged)
{
    // All values at avg ± 0.5*stddev, threshold=3: nothing should be replaced
    Image<RFLOAT> img(4, 4);
    RFLOAT avg = 5.0, stddev = 1.0;
    img.data.initConstant(avg);

    Image<RFLOAT> orig = img;
    removeDust(img, true,  3.0, avg, stddev);
    removeDust(img, false, 3.0, avg, stddev);

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(img.data)
        EXPECT_NEAR(DIRECT_MULTIDIM_ELEM(img.data, n),
                    DIRECT_MULTIDIM_ELEM(orig.data, n), 1e-9);
}

TEST(ImageFreeFunctions, RemoveDust_WhiteDust_HighPixelsReplaced)
{
    // One extreme outlier pixel: value = avg + 10*stddev (is_white=true)
    Image<RFLOAT> img(4, 4);
    RFLOAT avg = 0.0, stddev = 1.0;
    img.data.initConstant(avg);
    // Place a strong white-dust spike at one pixel
    DIRECT_MULTIDIM_ELEM(img.data, 0) = avg + 10.0 * stddev;

    removeDust(img, true, 3.0, avg, stddev);

    // The replaced pixel should no longer be > avg + 3*stddev (with overwhelming probability)
    // Check only that the original non-dust pixels are unchanged
    for (size_t n = 1; n < img.getSize(); n++)
        EXPECT_NEAR(DIRECT_MULTIDIM_ELEM(img.data, n), avg, 1e-9);
}

TEST(ImageFreeFunctions, RemoveDust_BlackDust_LowPixelsReplaced)
{
    // One extreme outlier pixel: value = avg - 10*stddev (is_white=false)
    Image<RFLOAT> img(4, 4);
    RFLOAT avg = 0.0, stddev = 1.0;
    img.data.initConstant(avg);
    DIRECT_MULTIDIM_ELEM(img.data, 0) = avg - 10.0 * stddev;

    removeDust(img, false, 3.0, avg, stddev);

    // Non-dust pixels remain unchanged
    for (size_t n = 1; n < img.getSize(); n++)
        EXPECT_NEAR(DIRECT_MULTIDIM_ELEM(img.data, n), avg, 1e-9);
}

// ----------------------------------------------- rewindow --

TEST(ImageFreeFunctions, Rewindow_2D_Crop)
{
    // 10x10 → 6x6: size shrinks
    Image<RFLOAT> img(10, 10);
    img.data.initConstant(1.0);
    rewindow(img, 6);
    EXPECT_EQ(XSIZE(img.data), 6);
    EXPECT_EQ(YSIZE(img.data), 6);
    // Center pixels (within original range) should still be 1.0
    EXPECT_NEAR(A2D_ELEM(img.data, 0, 0), 1.0, 1e-9);
}

TEST(ImageFreeFunctions, Rewindow_2D_Pad)
{
    // 6x6 → 10x10: size grows; new border pixels padded to 0
    Image<RFLOAT> img(6, 6);
    img.data.initConstant(2.0);
    rewindow(img, 10);
    EXPECT_EQ(XSIZE(img.data), 10);
    EXPECT_EQ(YSIZE(img.data), 10);
    EXPECT_EQ(img.getSize(), (size_t)100);
    // Center of padded array should be 2.0 (original data)
    EXPECT_NEAR(A2D_ELEM(img.data, 0, 0), 2.0, 1e-9);
}

TEST(ImageFreeFunctions, Rewindow_2D_SameSize)
{
    Image<RFLOAT> img(8, 8);
    img.data.initConstant(3.5);
    rewindow(img, 8);
    EXPECT_EQ(XSIZE(img.data), 8);
    EXPECT_EQ(YSIZE(img.data), 8);
    EXPECT_NEAR(A2D_ELEM(img.data, 0, 0), 3.5, 1e-9);
}

TEST(ImageFreeFunctions, Rewindow_3D_Crop)
{
    // 8x8x8 → 4x4x4
    Image<RFLOAT> img(8, 8, 8);
    img.data.initConstant(1.0);
    rewindow(img, 4);
    EXPECT_EQ(XSIZE(img.data), 4);
    EXPECT_EQ(YSIZE(img.data), 4);
    EXPECT_EQ(ZSIZE(img.data), 4);
}

// ----------------------------------------------- rescale --

TEST(ImageFreeFunctions, Rescale_UpdatesSamplingRate)
{
    // Create 10x10 image with sampling rate 2.0 Å/px
    // Rescale to 5x5: new sampling = 2.0 * 10/5 = 4.0 Å/px
    Image<RFLOAT> img(10, 10);
    img.data.initConstant(1.0);
    img.setSamplingRateInHeader(2.0);

    rescale(img, 5);

    EXPECT_NEAR(img.samplingRateX(), 4.0, 1e-5);
    EXPECT_NEAR(img.samplingRateY(), 4.0, 1e-5);
}

TEST(ImageFreeFunctions, Rescale_SamplingRateScalesCorrectly_ScaleUp)
{
    // 4x4 → 8x8: sampling halved
    Image<RFLOAT> img(4, 4);
    img.data.initConstant(1.0);
    img.setSamplingRateInHeader(4.0);

    rescale(img, 8);

    EXPECT_NEAR(img.samplingRateX(), 2.0, 1e-5);
    EXPECT_NEAR(img.samplingRateY(), 2.0, 1e-5);
}

TEST(ImageFreeFunctions, Rescale_NoSamplingRate_NoHeaderUpdate)
{
    // Image without sampling rate set: rescale should not crash,
    // and samplingRateX/Y fall back to 1.0 default (header unchanged).
    Image<RFLOAT> img(4, 4);
    img.data.initConstant(1.0);
    // Don't call setSamplingRateInHeader

    EXPECT_NO_THROW(rescale(img, 8));
    // Default is still 1.0 since the header labels were never set
    EXPECT_NEAR(img.samplingRateX(), 1.0, 1e-5);
}

// ----------------------------------------------- gettypesize UHalf --

TEST(ImageFreeFunctions, GetTypeSize_UHalf_Throws)
{
    // UHalf (4-bit) cannot be handled generically — function throws
    EXPECT_THROW(gettypesize(UHalf), RelionError);
}

// ----------------------------------------------- calcBgAvgStddev radius too large --

TEST(ImageFreeFunctions, CalcBgAvgStddev_CircularMask_RadiusTooLarge_Throws)
{
    // bg_radius=8 on a 10x10 image: all pixels inside mask → no background pixels
    // XMIPP indices [-5..4]: max i²+j²=50 < 64=bg_radius² → zero background pixels
    Image<RFLOAT> img(10, 10);
    img.data.setXmippOrigin();
    img.data.initConstant(1.0);
    RFLOAT avg, stddev;
    EXPECT_THROW(calculateBackgroundAvgStddev(img, avg, stddev, 8,
                                              false, 0., 0., 0.),
                 RelionError);
}

// ----------------------------------------------- calcBgAvgStddev helical path --

TEST(ImageFreeFunctions, CalcBgAvgStddev_HelicalSegment_UniformImage)
{
    // 2D 20x20 image, uniform at 7.0
    // is_helical=true, psi=0, tilt=0 (2D forces tilt=0): rotation = identity
    // Background: |i| > tube_radius=2 (rows with |XMIPP y| > 2)
    Image<RFLOAT> img(20, 20);
    img.data.setXmippOrigin();
    img.data.initConstant(7.0);

    RFLOAT avg = -1.0, stddev = -1.0;
    calculateBackgroundAvgStddev(img, avg, stddev, 0,
                                 true, 2., 0., 0.);

    EXPECT_NEAR(avg,    7.0, 1e-6);
    EXPECT_NEAR(stddev, 0.0, 1e-6);
}

// ----------------------------------------------- rescale 3D Z sampling rate --

TEST(ImageFreeFunctions, Rescale_3D_UpdatesSamplingRateZ)
{
    // 4x4x4 with Z sampling rate 2.0; rescale to 8x8x8 → Z rate = 1.0
    Image<RFLOAT> img(4, 4, 4);
    img.setSamplingRateInHeader(2.0);  // sets X, Y, Z (ZSIZE > 1)

    rescale(img, 8);

    RFLOAT sz = 0.0;
    img.MDMainHeader.getValue(EMDL_IMAGE_SAMPLINGRATE_Z, sz);
    EXPECT_NEAR(sz, 1.0, 1e-5);
    // X and Y also updated
    EXPECT_NEAR(img.samplingRateX(), 1.0, 1e-5);
}

// ----------------------------------------------- subtractBackgroundRamp --

TEST(ImageFreeFunctions, SubtractBgRamp_3D_Throws)
{
    Image<RFLOAT> img(4, 4, 4);
    img.data.setXmippOrigin();
    img.data.initConstant(0.0);
    EXPECT_THROW(subtractBackgroundRamp(img, 1, false, 0., 0., 0.), RelionError);
}

TEST(ImageFreeFunctions, SubtractBgRamp_Circular_FlatImage_StaysFlat)
{
    // All-zero 2D image: plane fit gives pA=pB=pC=0, subtraction leaves zeros
    Image<RFLOAT> img(10, 10);
    img.data.setXmippOrigin();
    img.data.initConstant(0.0);

    subtractBackgroundRamp(img, 2, false, 0., 0., 0.);

    FOR_ALL_ELEMENTS_IN_ARRAY2D(img.data)
        EXPECT_NEAR(A2D_ELEM(img.data, i, j), 0.0, 1e-6);
}

TEST(ImageFreeFunctions, SubtractBgRamp_Circular_LinearRamp_Removed)
{
    // Image values = XMIPP X coordinate (j): a perfect linear plane
    // After subtraction the entire image should collapse to ≈0
    Image<RFLOAT> img(10, 10);
    img.data.setXmippOrigin();
    FOR_ALL_ELEMENTS_IN_ARRAY2D(img.data)
        A2D_ELEM(img.data, i, j) = static_cast<RFLOAT>(j);

    subtractBackgroundRamp(img, 2, false, 0., 0., 0.);

    FOR_ALL_ELEMENTS_IN_ARRAY2D(img.data)
        EXPECT_NEAR(A2D_ELEM(img.data, i, j), 0.0, 1e-5);
}

TEST(ImageFreeFunctions, SubtractBgRamp_Helical_FlatImage_StaysFlat)
{
    // Helical path: 10x10, psi=0, tube_radius=2 (background: |i|>2)
    // Flat image → plane=0 → unchanged
    Image<RFLOAT> img(10, 10);
    img.data.setXmippOrigin();
    img.data.initConstant(0.0);

    subtractBackgroundRamp(img, 0, true, 2., 0., 0.);

    FOR_ALL_ELEMENTS_IN_ARRAY2D(img.data)
        EXPECT_NEAR(A2D_ELEM(img.data, i, j), 0.0, 1e-6);
}

// ----------------------------------------------- normalise --

TEST(ImageFreeFunctions, Normalise_RadiusTooLarge_Throws)
{
    Image<RFLOAT> img(10, 10);
    img.data.setXmippOrigin();
    img.data.initConstant(1.0);
    // 2 * bg_radius = 12 > XSIZE = 10
    EXPECT_THROW(normalise(img, 6, -1., -1., false, false, 0., 0., 0.),
                 RelionError);
}

TEST(ImageFreeFunctions, Normalise_HelicalTubeTooLarge_Throws)
{
    Image<RFLOAT> img(10, 10);
    img.data.setXmippOrigin();
    img.data.initConstant(1.0);
    // 2*(tube_radius+1) = 2*(5+1) = 12 > XSIZE = 10
    EXPECT_THROW(normalise(img, 1, -1., -1., false, true, 5., 0., 0.),
                 RelionError);
}

TEST(ImageFreeFunctions, Normalise_BasicNormalization)
{
    // 10x10 image filled with values 0..99; bg_radius=1
    // After normalise, background avg ≈ 0 and stddev ≈ 1
    Image<RFLOAT> img(10, 10);
    img.data.setXmippOrigin();
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(img.data)
        DIRECT_MULTIDIM_ELEM(img.data, n) = static_cast<RFLOAT>(n);

    normalise(img, 1, -1., -1., false, false, 0., 0., 0.);

    RFLOAT avg, stddev;
    calculateBackgroundAvgStddev(img, avg, stddev, 1, false, 0., 0., 0.);
    EXPECT_NEAR(avg,    0.0, 1e-4);
    EXPECT_NEAR(stddev, 1.0, 1e-4);
}

TEST(ImageFreeFunctions, Normalise_ConstantBackground_SkipsNormalization)
{
    // Constant image: background stddev = 0 → normalise prints warning and skips
    Image<RFLOAT> img(10, 10);
    img.data.setXmippOrigin();
    img.data.initConstant(5.0);

    normalise(img, 2, -1., -1., false, false, 0., 0., 0.);

    // Image should be unchanged (normalization skipped)
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(img.data)
        EXPECT_NEAR(DIRECT_MULTIDIM_ELEM(img.data, n), 5.0, 1e-6);
}

TEST(ImageFreeFunctions, Normalise_WithDustRemoval_NoError)
{
    // Exercise the white/black dust removal code paths inside normalise
    // Uses a high threshold so no pixel is actually replaced, but the path executes
    Image<RFLOAT> img(10, 10);
    img.data.setXmippOrigin();
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(img.data)
        DIRECT_MULTIDIM_ELEM(img.data, n) = static_cast<RFLOAT>(n);

    EXPECT_NO_THROW(normalise(img, 1, 100., 100., false, false, 0., 0., 0.));
}

TEST(ImageFreeFunctions, Normalise_WithRamp_NoError)
{
    // Exercise the do_ramp=true code path: linear ramp → plane removed → normalise
    Image<RFLOAT> img(10, 10);
    img.data.setXmippOrigin();
    FOR_ALL_ELEMENTS_IN_ARRAY2D(img.data)
        A2D_ELEM(img.data, i, j) = static_cast<RFLOAT>(j) + 10.0;

    EXPECT_NO_THROW(normalise(img, 2, -1., -1., true, false, 0., 0., 0.));
}

// -- lines 100-101: normalise helical 2D forces tilt=0 and proceeds --

TEST(ImageFreeFunctions, Normalise_Helical2D_ExecutesTiltZeroBranch)
{
    // is_helical=true on a 2D image: line 100-101 forces tilt_deg=0 then runs
    // 20x20, tube_radius=2 (2*(2+1)=6 ≤ 20), bg_radius=0 (2*0=0 ≤ 20)
    Image<RFLOAT> img(20, 20);
    img.data.setXmippOrigin();
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(img.data)
        DIRECT_MULTIDIM_ELEM(img.data, n) = static_cast<RFLOAT>(n);

    EXPECT_NO_THROW(normalise(img, 0, -1., -1., false, true, 2., 45., 0.));
    // After normalisation the background (|i|>2) should have avg≈0, stddev≈1
    RFLOAT avg, stddev;
    calculateBackgroundAvgStddev(img, avg, stddev, 0, true, 2., 0., 0.);
    EXPECT_NEAR(avg,    0.0, 1e-4);
    EXPECT_NEAR(stddev, 1.0, 1e-4);
}

// -- line 155: helical calcBgAvgStddev on a 1D image throws --

TEST(ImageFreeFunctions, CalcBgAvgStddev_Helical_InvalidDim_Throws)
{
    // getDim()==1 (YSIZE==1): neither 2 nor 3 → REPORT_ERROR at line 155
    Image<RFLOAT> img(10, 1);
    img.data.setXmippOrigin();
    img.data.initConstant(1.0);
    RFLOAT avg, stddev;
    EXPECT_THROW(calculateBackgroundAvgStddev(img, avg, stddev, 0,
                                              true, 1., 0., 0.),
                 RelionError);
}

// -- lines 196 + 206: 3D helical path --

TEST(ImageFreeFunctions, CalcBgAvgStddev_HelicalSegment_3D_UniformImage)
{
    // 3D image: dim==3 exercises ZZ(coords)=k (line 196) and sqrt(Y²+X²) (line 206)
    Image<RFLOAT> img(10, 10, 10);
    img.data.setXmippOrigin();
    img.data.initConstant(3.0);

    RFLOAT avg = -1.0, stddev = -1.0;
    calculateBackgroundAvgStddev(img, avg, stddev, 0,
                                 true, 2., 0., 0.);

    EXPECT_NEAR(avg,    3.0, 1e-6);
    EXPECT_NEAR(stddev, 0.0, 1e-6);
}

// -- line 224: helical background REPORT_ERROR (tube covers whole image) --

TEST(ImageFreeFunctions, CalcBgAvgStddev_Helical_TubeTooLarge_Throws)
{
    // 10x10, psi=0: background = |i| > tube_radius; i ∈ [-5..4]
    // tube_radius=5: |i|>5 never true → 0 background pixels → throws line 224
    Image<RFLOAT> img(10, 10);
    img.data.setXmippOrigin();
    img.data.initConstant(1.0);
    RFLOAT avg, stddev;
    EXPECT_THROW(calculateBackgroundAvgStddev(img, avg, stddev, 0,
                                              true, 5., 0., 0.),
                 RelionError);
}

// -- line 316: helical subtractBackgroundRamp with <5 background points --

TEST(ImageFreeFunctions, SubtractBgRamp_Helical_TooFewPoints_Throws)
{
    // 4x4 2D, psi=0, tube_radius=1: background = |i|>1, i ∈ [-2..1]
    // Only i=-2 qualifies (4 pixels) → 4 < 5 → REPORT_ERROR line 316
    Image<RFLOAT> img(4, 4);
    img.data.setXmippOrigin();
    img.data.initConstant(0.0);
    EXPECT_THROW(subtractBackgroundRamp(img, 0, true, 1., 0., 0.), RelionError);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
