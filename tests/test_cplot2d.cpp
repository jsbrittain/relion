/*
 * Unit tests for src/CPlot2D.h
 *
 * Covers: CDataPoint constructors and accessors,
 *         CDataSet data management and statistics,
 *         CPlot2D accessors (no file I/O)
 */

#include <gtest/gtest.h>
#include "src/CPlot2D.h"

// ---------------------------------------------------------------------------
// CDataPoint
// ---------------------------------------------------------------------------

TEST(CDataPointTest, DefaultConstructor_ZeroValues)
{
    CDataPoint p;
    double x, y;
    p.GetValues(&x, &y);
    EXPECT_NEAR(x, 0.0, 1e-10);
    EXPECT_NEAR(y, 0.0, 1e-10);
}

TEST(CDataPointTest, InitConstructor)
{
    CDataPoint p(3.0, 7.5);
    EXPECT_NEAR(p.GetX(), 3.0, 1e-10);
    EXPECT_NEAR(p.GetY(), 7.5, 1e-10);
}

TEST(CDataPointTest, SetValues)
{
    CDataPoint p;
    p.SetValues(1.5, -2.5);
    double x, y;
    p.GetValues(&x, &y);
    EXPECT_NEAR(x,  1.5, 1e-10);
    EXPECT_NEAR(y, -2.5, 1e-10);
}

// ---------------------------------------------------------------------------
// CDataSet
// ---------------------------------------------------------------------------

TEST(CDataSetTest, DefaultConstructor_EmptySet)
{
    CDataSet ds;
    EXPECT_EQ(ds.GetNumberOfDataPointsInSet(), 0);
}

TEST(CDataSetTest, AddDataPoint_IncreasesCount)
{
    CDataSet ds;
    ds.AddDataPoint(CDataPoint(1.0, 2.0));
    ds.AddDataPoint(CDataPoint(3.0, 4.0));
    EXPECT_EQ(ds.GetNumberOfDataPointsInSet(), 2);
}

TEST(CDataSetTest, GetDataPoint_CorrectValues)
{
    CDataSet ds;
    ds.AddDataPoint(CDataPoint(5.0, 9.0));
    CDataPoint p = ds.GetDataPoint(0);
    EXPECT_NEAR(p.GetX(), 5.0, 1e-10);
    EXPECT_NEAR(p.GetY(), 9.0, 1e-10);
}

TEST(CDataSetTest, SetDataPoint_OverwritesEntry)
{
    CDataSet ds;
    ds.AddDataPoint(CDataPoint(1.0, 2.0));
    ds.SetDataPoint(0, CDataPoint(10.0, 20.0));
    EXPECT_NEAR(ds.GetDataPoint(0).GetX(), 10.0, 1e-10);
    EXPECT_NEAR(ds.GetDataPoint(0).GetY(), 20.0, 1e-10);
}

TEST(CDataSetTest, GetXMinMax)
{
    CDataSet ds;
    ds.AddDataPoint(CDataPoint(1.0, 0.0));
    ds.AddDataPoint(CDataPoint(5.0, 0.0));
    ds.AddDataPoint(CDataPoint(3.0, 0.0));
    EXPECT_NEAR(ds.GetXMinValue(), 1.0, 1e-10);
    EXPECT_NEAR(ds.GetXMaxValue(), 5.0, 1e-10);
}

TEST(CDataSetTest, GetYMinMax)
{
    CDataSet ds;
    ds.AddDataPoint(CDataPoint(0.0, -2.0));
    ds.AddDataPoint(CDataPoint(0.0,  4.0));
    EXPECT_NEAR(ds.GetYMinValue(), -2.0, 1e-10);
    EXPECT_NEAR(ds.GetYMaxValue(),  4.0, 1e-10);
}

TEST(CDataSetTest, GetXExtent)
{
    CDataSet ds;
    ds.AddDataPoint(CDataPoint(1.0, 0.0));
    ds.AddDataPoint(CDataPoint(4.0, 0.0));
    EXPECT_NEAR(ds.GetXExtent(), 3.0, 1e-10);
}

TEST(CDataSetTest, GetYExtent)
{
    CDataSet ds;
    ds.AddDataPoint(CDataPoint(0.0, 10.0));
    ds.AddDataPoint(CDataPoint(0.0,  2.0));
    EXPECT_NEAR(ds.GetYExtent(), 8.0, 1e-10);
}

TEST(CDataSetTest, LineWidthAccessor)
{
    CDataSet ds;
    ds.SetLineWidth(3.5);
    EXPECT_NEAR(ds.GetLineWidth(), 3.5, 1e-10);
}

TEST(CDataSetTest, ColorAccessor)
{
    CDataSet ds;
    ds.SetDatasetColor(0.1, 0.5, 0.9);
    double r, g, b;
    ds.GetDatasetColor(&r, &g, &b);
    EXPECT_NEAR(r, 0.1, 1e-10);
    EXPECT_NEAR(g, 0.5, 1e-10);
    EXPECT_NEAR(b, 0.9, 1e-10);
}

TEST(CDataSetTest, MarkerAccessors)
{
    CDataSet ds;
    ds.SetMarkerSymbol("x");
    EXPECT_EQ(ds.GetMarkerSymbol(), "x");
    ds.SetMarkerSize(12.0);
    EXPECT_NEAR(ds.GetMarkerSize(), 12.0, 1e-10);
}

TEST(CDataSetTest, DrawFlagAccessors)
{
    CDataSet ds;
    ds.SetDrawLine(false);
    EXPECT_FALSE(ds.GetDrawLine());
    ds.SetDrawMarker(false);
    EXPECT_FALSE(ds.GetDrawMarker());
    ds.SetDrawMarkerFilled(false);
    EXPECT_FALSE(ds.GetDrawMarkerFilled());
}

TEST(CDataSetTest, DashedLineAccessors)
{
    CDataSet ds;
    ds.SetDashedLine(true);
    EXPECT_TRUE(ds.GetDashedLine());
    ds.SetDashedLinePattern("dot");
    EXPECT_EQ(ds.GetDashedLinePattern(), "dot");
}

TEST(CDataSetTest, DatasetTitleAccessor)
{
    CDataSet ds;
    ds.SetDatasetTitle("my series");
    EXPECT_EQ(ds.GetDatasetTitle(), "my series");
}

TEST(CDataSetTest, DatasetLegendFontAccessor)
{
    CDataSet ds;
    ds.SetDatasetLegendFont("Helvetica");
    EXPECT_EQ(ds.GetDatasetLegendFont(), "Helvetica");
}

// ---------------------------------------------------------------------------
// CPlot2D
// ---------------------------------------------------------------------------

TEST(CPlot2DTest, DefaultConstructor_DefaultDimensions)
{
    CPlot2D plot;
    EXPECT_GT(plot.GetXTotalSize(), 0.0);
    EXPECT_GT(plot.GetYTotalSize(), 0.0);
}

TEST(CPlot2DTest, SetGetXTotalSize)
{
    CPlot2D plot;
    plot.SetXTotalSize(800.0);
    EXPECT_NEAR(plot.GetXTotalSize(), 800.0, 1e-10);
}

TEST(CPlot2DTest, SetGetYTotalSize)
{
    CPlot2D plot;
    plot.SetYTotalSize(600.0);
    EXPECT_NEAR(plot.GetYTotalSize(), 600.0, 1e-10);
}

TEST(CPlot2DTest, SetGetAxisSizes)
{
    CPlot2D plot;
    plot.SetXAxisSize(500.0);
    plot.SetYAxisSize(400.0);
    EXPECT_NEAR(plot.GetXAxisSize(), 500.0, 1e-10);
    EXPECT_NEAR(plot.GetYAxisSize(), 400.0, 1e-10);
}

TEST(CPlot2DTest, SetGetFrameSizes)
{
    CPlot2D plot;
    plot.SetBottomFrameSize(50.0);
    plot.SetRightFrameSize(20.0);
    plot.SetTopFrameSize(30.0);
    plot.SetLeftFrameSize(60.0);
    EXPECT_NEAR(plot.GetBottomFrameSize(), 50.0, 1e-10);
    EXPECT_NEAR(plot.GetRightFrameSize(),  20.0, 1e-10);
    EXPECT_NEAR(plot.GetTopFrameSize(),    30.0, 1e-10);
    EXPECT_NEAR(plot.GetLeftFrameSize(),   60.0, 1e-10);
}

TEST(CPlot2DTest, SetGetLineWidths)
{
    CPlot2D plot;
    plot.SetFrameLineWidth(2.5);
    plot.SetGridLineWidth(0.5);
    EXPECT_NEAR(plot.GetFrameLineWidth(), 2.5, 1e-10);
    EXPECT_NEAR(plot.GetGridLineWidth(),  0.5, 1e-10);
}

TEST(CPlot2DTest, SetGetFrameColor)
{
    CPlot2D plot;
    plot.SetFrameColor(0.2, 0.4, 0.6);
    double r, g, b;
    plot.GetFrameColor(&r, &g, &b);
    EXPECT_NEAR(r, 0.2, 1e-10);
    EXPECT_NEAR(g, 0.4, 1e-10);
    EXPECT_NEAR(b, 0.6, 1e-10);
}

TEST(CPlot2DTest, SetGetGridColor)
{
    CPlot2D plot;
    plot.SetGridColor(0.8, 0.1, 0.3);
    double r, g, b;
    plot.GetGridColor(&r, &g, &b);
    EXPECT_NEAR(r, 0.8, 1e-10);
    EXPECT_NEAR(g, 0.1, 1e-10);
    EXPECT_NEAR(b, 0.3, 1e-10);
}

TEST(CPlot2DTest, SetGetTickAccessors)
{
    CPlot2D plot;
    plot.SetXAxisNumberOfTicks(10);
    plot.SetYAxisNumberOfTicks(8);
    EXPECT_EQ(plot.GetXAxisNumberOfTicks(), 10);
    EXPECT_EQ(plot.GetYAxisNumberOfTicks(),  8);
}

TEST(CPlot2DTest, SetGetGridFlags)
{
    CPlot2D plot;
    plot.SetDrawXAxisGridLines(true);
    plot.SetDrawYAxisGridLines(false);
    plot.SetDrawGridLinesDashed(true);
    EXPECT_TRUE (plot.GetDrawXAxisGridLines());
    EXPECT_FALSE(plot.GetDrawYAxisGridLines());
    EXPECT_TRUE (plot.GetDrawGridLinesDashed());
}

TEST(CPlot2DTest, SetGetTickMarkFlags)
{
    CPlot2D plot;
    plot.SetDrawXAxisTickMarks(false);
    plot.SetDrawYAxisTickMarks(true);
    EXPECT_FALSE(plot.GetDrawXAxisTickMarks());
    EXPECT_TRUE (plot.GetDrawYAxisTickMarks());
}

TEST(CPlot2DTest, SetGetAxisTitles)
{
    CPlot2D plot;
    plot.SetXAxisTitle("X label");
    plot.SetYAxisTitle("Y label");
    EXPECT_EQ(plot.GetXAxisTitle(), "X label");
    EXPECT_EQ(plot.GetYAxisTitle(), "Y label");
}

TEST(CPlot2DTest, SetGetAxisFonts)
{
    CPlot2D plot;
    plot.SetXAxisLabelFont("Times");
    plot.SetYAxisLabelFont("Helvetica");
    EXPECT_EQ(plot.GetXAxisLabelFont(), "Times");
    EXPECT_EQ(plot.GetYAxisLabelFont(), "Helvetica");
}

TEST(CPlot2DTest, SetGetTitleFonts)
{
    CPlot2D plot;
    plot.SetXAxisTitleFont("Arial");
    plot.SetYAxisTitleFont("Courier");
    EXPECT_EQ(plot.GetXAxisTitleFont(), "Arial");
    EXPECT_EQ(plot.GetYAxisTitleFont(), "Courier");
}

TEST(CPlot2DTest, SetGetTitleFontSizes)
{
    CPlot2D plot;
    plot.SetXAxisTitleFontSize(14.0);
    plot.SetYAxisTitleFontSize(12.0);
    EXPECT_NEAR(plot.GetXAxisTitleFontSize(), 14.0, 1e-10);
    EXPECT_NEAR(plot.GetYAxisTitleFontSize(), 12.0, 1e-10);
}

TEST(CPlot2DTest, SetGetAxisTitleColors)
{
    CPlot2D plot;
    plot.SetXAxisTitleColor(0.1, 0.2, 0.3);
    double r, g, b;
    plot.GetXAxisTitleColor(&r, &g, &b);
    EXPECT_NEAR(r, 0.1, 1e-10);
    EXPECT_NEAR(g, 0.2, 1e-10);
    EXPECT_NEAR(b, 0.3, 1e-10);
}

TEST(CPlot2DTest, SetGetLegendFlag)
{
    CPlot2D plot;
    plot.SetDrawLegend(false);
    EXPECT_FALSE(plot.GetDrawLegend());
    plot.SetDrawLegend(true);
    EXPECT_TRUE(plot.GetDrawLegend());
}

TEST(CPlot2DTest, SetGetFlipY)
{
    CPlot2D plot;
    plot.SetFlipY(true);
    EXPECT_TRUE(plot.GetFlipY());
    plot.SetFlipY(false);
    EXPECT_FALSE(plot.GetFlipY());
}

TEST(CPlot2DTest, AddDataSet_CDataSet)
{
    CPlot2D plot;
    CDataSet ds;
    ds.AddDataPoint(CDataPoint(1.0, 2.0));
    ds.AddDataPoint(CDataPoint(3.0, 4.0));
    plot.AddDataSet(ds);
    // No crash is the main test here
}

TEST(CPlot2DTest, AddDataSet_TwoVectors)
{
    CPlot2D plot;
    std::vector<RFLOAT> xs = {1.0, 2.0, 3.0};
    std::vector<RFLOAT> ys = {4.0, 5.0, 6.0};
    plot.AddDataSet(xs, ys);
    // No crash
}

TEST(CPlot2DTest, AddDataSet_SingleYVector)
{
    CPlot2D plot;
    std::vector<RFLOAT> ys = {1.0, 2.0, 3.0, 4.0};
    plot.AddDataSet(ys);
    // No crash
}

TEST(CPlot2DTest, SetTitle_IsCallable)
{
    CPlot2D plot;
    // SetTitle has no corresponding getter; verify it does not crash.
    plot.SetTitle("My Plot Title");
    plot.SetTitle("");            // empty string
    plot.SetTitle("Second Title"); // overwrite
    // No crash is the assertion
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
