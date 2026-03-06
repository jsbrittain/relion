/*
 * GoogleTest unit tests for MetaDataTable (src/metadata_table.h/.cpp).
 *
 * Build and run:
 *   cmake -DBUILD_TESTS=ON ..
 *   make test_metadata_table
 *   ./build/bin/test_metadata_table
 *
 * No MPI required; pure unit tests with no GPU dependency.
 *
 * What is tested:
 *   1.  Default constructor     – isEmpty, numberOfObjects == 0
 *   2.  addObject               – numberOfObjects increments
 *   3.  setValue/getValue       – RFLOAT, int, string, bool round-trips
 *   4.  containsLabel           – true after setValue, false for missing label
 *   5.  addLabel                – label present before any row is added
 *   6.  deactivateLabel         – label absent after deactivation
 *   7.  setName/getName         – stored and retrieved correctly
 *   8.  setComment/getComment   – stored and retrieved correctly
 *   9.  copy constructor        – deep copy (mutation doesn't affect original)
 *  10.  assignment operator     – deep copy (mutation doesn't affect original)
 *  11.  clear()                 – table becomes empty
 *  12.  removeObject            – size decrements by one
 *  13.  append                  – rows and labels merged
 *  14.  FOR_ALL_OBJECTS macro   – iterates all rows exactly once
 *  15.  compareLabels           – equal tables match; different label sets differ
 *  16.  sort (numeric)          – rows reordered by label value
 *  17.  STAR write/read round-trip – values survive serialisation
 *  18.  getActiveLabels         – returns exactly the active set
 */

#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <sstream>
#include <cstdio>
#include "src/metadata_table.h"
#include "src/metadata_label.h"

// ---------------------------------------------------------------------------
// Helper: create a table with a few rows of CTF metadata
// ---------------------------------------------------------------------------
static MetaDataTable makeCTFTable(int nrows = 3)
{
    MetaDataTable mdt;
    for (int i = 0; i < nrows; i++)
    {
        mdt.addObject();
        mdt.setValue(EMDL_CTF_DEFOCUSU,  (RFLOAT)(10000.0 + i * 1000.0));
        mdt.setValue(EMDL_CTF_DEFOCUSV,  (RFLOAT)(10000.0 + i * 1000.0));
        mdt.setValue(EMDL_IMAGE_NAME,     std::string("particle_") + std::to_string(i));
    }
    return mdt;
}

// ---------------------------------------------------------------------------
// 1. Default constructor
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, DefaultConstructorIsEmpty)
{
    MetaDataTable mdt;
    EXPECT_TRUE(mdt.isEmpty());
    EXPECT_EQ(mdt.numberOfObjects(), (size_t)0);
}

// ---------------------------------------------------------------------------
// 2. addObject
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, AddObjectIncrementsCount)
{
    MetaDataTable mdt;
    mdt.addObject();
    EXPECT_EQ(mdt.numberOfObjects(), (size_t)1);
    mdt.addObject();
    EXPECT_EQ(mdt.numberOfObjects(), (size_t)2);
}

TEST(MetaDataTableTest, AddObjectNotEmpty)
{
    MetaDataTable mdt;
    mdt.addObject();
    EXPECT_FALSE(mdt.isEmpty());
}

// ---------------------------------------------------------------------------
// 3. setValue / getValue round-trips
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, SetGetRfloat)
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)15000.0);
    RFLOAT val = 0;
    EXPECT_TRUE(mdt.getValue(EMDL_CTF_DEFOCUSU, val, 0));
    EXPECT_NEAR(val, 15000.0, 1e-6);
}

TEST(MetaDataTableTest, SetGetInt)
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.setValue(EMDL_MLMODEL_NR_CLASSES, 8);
    int val = 0;
    EXPECT_TRUE(mdt.getValue(EMDL_MLMODEL_NR_CLASSES, val, 0));
    EXPECT_EQ(val, 8);
}

TEST(MetaDataTableTest, SetGetString)
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.setValue(EMDL_IMAGE_NAME, std::string("my_particle.mrcs"));
    std::string val;
    EXPECT_TRUE(mdt.getValue(EMDL_IMAGE_NAME, val, 0));
    EXPECT_EQ(val, "my_particle.mrcs");
}

TEST(MetaDataTableTest, SetGetBool)
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.setValue(EMDL_OPTIMISER_DO_ZERO_MASK, true);
    bool val = false;
    EXPECT_TRUE(mdt.getValue(EMDL_OPTIMISER_DO_ZERO_MASK, val, 0));
    EXPECT_TRUE(val);
}

TEST(MetaDataTableTest, GetValueMissingLabelReturnsFalse)
{
    MetaDataTable mdt;
    mdt.addObject();
    RFLOAT val = 0;
    EXPECT_FALSE(mdt.getValue(EMDL_CTF_DEFOCUSU, val, 0));
}

// Multiple rows: explicit objectID selects the right row
TEST(MetaDataTableTest, SetGetMultipleRowsByObjectID)
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)1000.0);
    mdt.addObject();
    mdt.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)2000.0);

    RFLOAT v0 = 0, v1 = 0;
    mdt.getValue(EMDL_CTF_DEFOCUSU, v0, 0);
    mdt.getValue(EMDL_CTF_DEFOCUSU, v1, 1);
    EXPECT_NEAR(v0, 1000.0, 1e-6);
    EXPECT_NEAR(v1, 2000.0, 1e-6);
}

// ---------------------------------------------------------------------------
// 4. containsLabel
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, ContainsLabelTrueAfterSetValue)
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)15000.0);
    EXPECT_TRUE(mdt.containsLabel(EMDL_CTF_DEFOCUSU));
}

TEST(MetaDataTableTest, ContainsLabelFalseForUnusedLabel)
{
    MetaDataTable mdt;
    mdt.addObject();
    EXPECT_FALSE(mdt.containsLabel(EMDL_CTF_DEFOCUSU));
}

// ---------------------------------------------------------------------------
// 5. addLabel
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, AddLabelMakesContainsLabelTrue)
{
    MetaDataTable mdt;
    mdt.addLabel(EMDL_CTF_VOLTAGE);
    EXPECT_TRUE(mdt.containsLabel(EMDL_CTF_VOLTAGE));
}

// ---------------------------------------------------------------------------
// 6. deactivateLabel
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, DeactivateLabelRemovesFromActive)
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)15000.0);
    ASSERT_TRUE(mdt.containsLabel(EMDL_CTF_DEFOCUSU));
    mdt.deactivateLabel(EMDL_CTF_DEFOCUSU);
    EXPECT_FALSE(mdt.containsLabel(EMDL_CTF_DEFOCUSU));
}

// ---------------------------------------------------------------------------
// 7. setName / getName
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, SetGetName)
{
    MetaDataTable mdt;
    mdt.setName("particles");
    EXPECT_EQ(mdt.getName(), "particles");
}

TEST(MetaDataTableTest, DefaultNameIsEmpty)
{
    MetaDataTable mdt;
    EXPECT_EQ(mdt.getName(), "");
}

// ---------------------------------------------------------------------------
// 8. setComment / getComment
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, SetGetComment)
{
    MetaDataTable mdt;
    mdt.setComment("a test comment");
    EXPECT_EQ(mdt.getComment(), "a test comment");
    EXPECT_TRUE(mdt.containsComment());
}

TEST(MetaDataTableTest, DefaultCommentNotContained)
{
    MetaDataTable mdt;
    EXPECT_FALSE(mdt.containsComment());
}

// ---------------------------------------------------------------------------
// 9. Copy constructor — deep copy
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, CopyConstructorDeepCopy)
{
    MetaDataTable orig;
    orig.addObject();
    orig.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)5000.0);

    MetaDataTable copy(orig);
    // mutate copy; original must be unchanged
    copy.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)9999.0, 0);

    RFLOAT orig_val = 0;
    orig.getValue(EMDL_CTF_DEFOCUSU, orig_val, 0);
    EXPECT_NEAR(orig_val, 5000.0, 1e-6);
}

TEST(MetaDataTableTest, CopyConstructorSameNumberOfObjects)
{
    MetaDataTable orig = makeCTFTable(4);
    MetaDataTable copy(orig);
    EXPECT_EQ(copy.numberOfObjects(), orig.numberOfObjects());
}

// ---------------------------------------------------------------------------
// 10. Assignment operator — deep copy
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, AssignmentDeepCopy)
{
    MetaDataTable orig;
    orig.addObject();
    orig.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)7000.0);

    MetaDataTable assigned;
    assigned = orig;
    assigned.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)1.0, 0);

    RFLOAT orig_val = 0;
    orig.getValue(EMDL_CTF_DEFOCUSU, orig_val, 0);
    EXPECT_NEAR(orig_val, 7000.0, 1e-6);
}

// ---------------------------------------------------------------------------
// 11. clear()
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, ClearMakesEmpty)
{
    MetaDataTable mdt = makeCTFTable(3);
    ASSERT_EQ(mdt.numberOfObjects(), (size_t)3);
    mdt.clear();
    EXPECT_TRUE(mdt.isEmpty());
    EXPECT_EQ(mdt.numberOfObjects(), (size_t)0);
}

// ---------------------------------------------------------------------------
// 12. removeObject
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, RemoveObjectDecrementsCount)
{
    MetaDataTable mdt = makeCTFTable(3);
    mdt.removeObject(0);
    EXPECT_EQ(mdt.numberOfObjects(), (size_t)2);
}

TEST(MetaDataTableTest, RemoveObjectCorrectRowRemains)
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)1000.0);
    mdt.addObject();
    mdt.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)2000.0);
    mdt.addObject();
    mdt.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)3000.0);

    mdt.removeObject(1);  // remove row with 2000
    ASSERT_EQ(mdt.numberOfObjects(), (size_t)2);

    RFLOAT v0 = 0, v1 = 0;
    mdt.getValue(EMDL_CTF_DEFOCUSU, v0, 0);
    mdt.getValue(EMDL_CTF_DEFOCUSU, v1, 1);
    EXPECT_NEAR(v0, 1000.0, 1e-6);
    EXPECT_NEAR(v1, 3000.0, 1e-6);
}

// ---------------------------------------------------------------------------
// 13. append
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, AppendCombinesRows)
{
    MetaDataTable a = makeCTFTable(2);
    MetaDataTable b = makeCTFTable(3);
    a.append(b);
    EXPECT_EQ(a.numberOfObjects(), (size_t)5);
}

TEST(MetaDataTableTest, AppendPreservesValues)
{
    MetaDataTable a;
    a.addObject();
    a.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)100.0);

    MetaDataTable b;
    b.addObject();
    b.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)200.0);

    a.append(b);
    RFLOAT v1 = 0;
    a.getValue(EMDL_CTF_DEFOCUSU, v1, 1);
    EXPECT_NEAR(v1, 200.0, 1e-6);
}

// ---------------------------------------------------------------------------
// 14. FOR_ALL_OBJECTS_IN_METADATA_TABLE macro
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, ForAllObjectsIteratesAll)
{
    MetaDataTable mdt = makeCTFTable(5);
    int count = 0;
    FOR_ALL_OBJECTS_IN_METADATA_TABLE(mdt)
    {
        count++;
    }
    EXPECT_EQ(count, 5);
}

TEST(MetaDataTableTest, ForAllObjectsReadsValues)
{
    MetaDataTable mdt;
    for (int i = 0; i < 4; i++)
    {
        mdt.addObject();
        mdt.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)(i * 500.0));
    }

    RFLOAT sum = 0;
    FOR_ALL_OBJECTS_IN_METADATA_TABLE(mdt)
    {
        RFLOAT v = 0;
        mdt.getValue(EMDL_CTF_DEFOCUSU, v);
        sum += v;
    }
    // 0+500+1000+1500 = 3000
    EXPECT_NEAR(sum, 3000.0, 1e-6);
}

// ---------------------------------------------------------------------------
// 15. compareLabels
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, CompareLabelsEqualTables)
{
    MetaDataTable a = makeCTFTable(2);
    MetaDataTable b = makeCTFTable(3);
    EXPECT_TRUE(MetaDataTable::compareLabels(a, b));
}

TEST(MetaDataTableTest, CompareLabelsDifferentSets)
{
    MetaDataTable a;
    a.addObject();
    a.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)1.0);

    MetaDataTable b;
    b.addObject();
    b.setValue(EMDL_CTF_VOLTAGE, (RFLOAT)300.0);

    EXPECT_FALSE(MetaDataTable::compareLabels(a, b));
}

// ---------------------------------------------------------------------------
// 16. sort (numeric)
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, SortAscending)
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)3000.0);
    mdt.addObject();
    mdt.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)1000.0);
    mdt.addObject();
    mdt.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)2000.0);

    mdt.sort(EMDL_CTF_DEFOCUSU);

    RFLOAT v0 = 0, v1 = 0, v2 = 0;
    mdt.getValue(EMDL_CTF_DEFOCUSU, v0, 0);
    mdt.getValue(EMDL_CTF_DEFOCUSU, v1, 1);
    mdt.getValue(EMDL_CTF_DEFOCUSU, v2, 2);
    EXPECT_LT(v0, v1);
    EXPECT_LT(v1, v2);
}

TEST(MetaDataTableTest, SortDescending)
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)1000.0);
    mdt.addObject();
    mdt.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)3000.0);
    mdt.addObject();
    mdt.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)2000.0);

    mdt.sort(EMDL_CTF_DEFOCUSU, /*do_reverse=*/true);

    RFLOAT v0 = 0, v2 = 0;
    mdt.getValue(EMDL_CTF_DEFOCUSU, v0, 0);
    mdt.getValue(EMDL_CTF_DEFOCUSU, v2, 2);
    EXPECT_GT(v0, v2);
}

// ---------------------------------------------------------------------------
// 17. STAR write / read round-trip
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, StarWriteReadRoundTrip)
{
    // Build a table
    MetaDataTable orig;
    orig.setName("particles");
    orig.addObject();
    orig.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)12000.0);
    orig.setValue(EMDL_CTF_DEFOCUSV, (RFLOAT)11500.0);
    orig.setValue(EMDL_IMAGE_NAME,   std::string("part_001.mrcs"));
    orig.addObject();
    orig.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)14000.0);
    orig.setValue(EMDL_CTF_DEFOCUSV, (RFLOAT)13500.0);
    orig.setValue(EMDL_IMAGE_NAME,   std::string("part_002.mrcs"));

    // Write to temp file
    char tmpname[] = "/tmp/test_mdt_XXXXXX";
    int fd = mkstemp(tmpname);
    ASSERT_NE(fd, -1);
    close(fd);
    orig.write(std::string(tmpname));

    // Read back
    MetaDataTable read_back;
    std::ifstream ifs(tmpname);
    ASSERT_TRUE(ifs.is_open());
    read_back.readStar(ifs, "particles");
    ifs.close();
    std::remove(tmpname);

    EXPECT_EQ(read_back.numberOfObjects(), (size_t)2);

    RFLOAT def_u0 = 0, def_u1 = 0;
    read_back.getValue(EMDL_CTF_DEFOCUSU, def_u0, 0);
    read_back.getValue(EMDL_CTF_DEFOCUSU, def_u1, 1);
    EXPECT_NEAR(def_u0, 12000.0, 1e-3);
    EXPECT_NEAR(def_u1, 14000.0, 1e-3);

    std::string name;
    read_back.getValue(EMDL_IMAGE_NAME, name, 0);
    EXPECT_EQ(name, "part_001.mrcs");
}

// ---------------------------------------------------------------------------
// 18. getActiveLabels
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, GetActiveLabelsContainsSetLabels)
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)1.0);
    mdt.setValue(EMDL_CTF_DEFOCUSV, (RFLOAT)1.0);

    std::vector<EMDLabel> active = mdt.getActiveLabels();
    bool found_u = false, found_v = false;
    for (EMDLabel lbl : active)
    {
        if (lbl == EMDL_CTF_DEFOCUSU) found_u = true;
        if (lbl == EMDL_CTF_DEFOCUSV) found_v = true;
    }
    EXPECT_TRUE(found_u);
    EXPECT_TRUE(found_v);
}

TEST(MetaDataTableTest, GetActiveLabelsExcludesDeactivated)
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)1.0);
    mdt.setValue(EMDL_CTF_DEFOCUSV, (RFLOAT)1.0);
    mdt.deactivateLabel(EMDL_CTF_DEFOCUSU);

    std::vector<EMDLabel> active = mdt.getActiveLabels();
    for (EMDLabel lbl : active)
        EXPECT_NE(lbl, EMDL_CTF_DEFOCUSU);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
