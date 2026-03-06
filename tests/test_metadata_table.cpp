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
// 19. Convenience typed accessors: getInt, getRfloat, getBool, getString
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, GetInt_ReturnsIntValue)
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.setValue(EMDL_MLMODEL_NR_CLASSES, 4);
    EXPECT_EQ(mdt.getInt(EMDL_MLMODEL_NR_CLASSES, 0), 4);
}

TEST(MetaDataTableTest, GetRfloat_ReturnsRfloatValue)
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)17500.0);
    EXPECT_NEAR(mdt.getRfloat(EMDL_CTF_DEFOCUSU, 0), 17500.0, 1e-3);
}

TEST(MetaDataTableTest, GetBool_ReturnsBoolValue)
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.setValue(EMDL_OPTIMISER_DO_ZERO_MASK, true);
    EXPECT_TRUE(mdt.getBool(EMDL_OPTIMISER_DO_ZERO_MASK, 0));
}

TEST(MetaDataTableTest, GetString_ReturnsStringValue)
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.setValue(EMDL_IMAGE_NAME, std::string("test.mrcs"));
    EXPECT_EQ(mdt.getString(EMDL_IMAGE_NAME, 0), "test.mrcs");
}

// ---------------------------------------------------------------------------
// 20. getValueToString — returns string representation of a stored value
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, GetValueToString_RfloatAsString)
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)12345.0);
    std::string str;
    bool ok = mdt.getValueToString(EMDL_CTF_DEFOCUSU, str, 0);
    EXPECT_TRUE(ok);
    // String must represent the value 12345
    EXPECT_NE(str.find("12345"), std::string::npos);
}

TEST(MetaDataTableTest, GetValueToString_StringLabel)
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.setValue(EMDL_IMAGE_NAME, std::string("particle.mrcs"));
    std::string str;
    bool ok = mdt.getValueToString(EMDL_IMAGE_NAME, str, 0);
    EXPECT_TRUE(ok);
    EXPECT_NE(str.find("particle.mrcs"), std::string::npos);
}

// ---------------------------------------------------------------------------
// 21. setIsList / isAList
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, Default_IsNotAList)
{
    MetaDataTable mdt;
    EXPECT_FALSE(mdt.isAList());
}

TEST(MetaDataTableTest, SetIsList_True)
{
    MetaDataTable mdt;
    mdt.setIsList(true);
    EXPECT_TRUE(mdt.isAList());
}

TEST(MetaDataTableTest, SetIsList_False)
{
    MetaDataTable mdt;
    mdt.setIsList(true);
    mdt.setIsList(false);
    EXPECT_FALSE(mdt.isAList());
}

// ---------------------------------------------------------------------------
// 22. getVersion / setVersion / getCurrentVersion
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, GetVersionDefault)
{
    MetaDataTable mdt;
    // Default version should equal getCurrentVersion()
    EXPECT_EQ(mdt.getVersion(), MetaDataTable::getCurrentVersion());
}

TEST(MetaDataTableTest, SetGetVersion)
{
    MetaDataTable mdt;
    mdt.setVersion(42);
    EXPECT_EQ(mdt.getVersion(), 42);
}

TEST(MetaDataTableTest, GetCurrentVersion_IsPositive)
{
    EXPECT_GT(MetaDataTable::getCurrentVersion(), 0);
}

// ---------------------------------------------------------------------------
// 23. randomiseOrder — shuffles rows, count unchanged
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, RandomiseOrder_PreservesCount)
{
    MetaDataTable mdt = makeCTFTable(10);
    mdt.randomiseOrder();
    EXPECT_EQ(mdt.numberOfObjects(), (size_t)10);
}

TEST(MetaDataTableTest, RandomiseOrder_PreservesLabels)
{
    MetaDataTable mdt = makeCTFTable(5);
    mdt.randomiseOrder();
    // All rows must still have EMDL_CTF_DEFOCUSU
    EXPECT_TRUE(mdt.containsLabel(EMDL_CTF_DEFOCUSU));
}

TEST(MetaDataTableTest, RandomiseOrder_PreservesValues)
{
    // Build table with unique values; after shuffle, sum must be same.
    MetaDataTable mdt;
    RFLOAT expected_sum = 0;
    for (int i = 1; i <= 5; i++)
    {
        mdt.addObject();
        mdt.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)(i * 1000.0));
        expected_sum += i * 1000.0;
    }
    mdt.randomiseOrder();

    RFLOAT actual_sum = 0;
    FOR_ALL_OBJECTS_IN_METADATA_TABLE(mdt)
    {
        RFLOAT v = 0;
        mdt.getValue(EMDL_CTF_DEFOCUSU, v);
        actual_sum += v;
    }
    EXPECT_NEAR(actual_sum, expected_sum, 1e-3);
}

// ---------------------------------------------------------------------------
// 24. newSort — string-based sort (by IMAGE_NAME)
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, NewSort_StringAscending)
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.setValue(EMDL_IMAGE_NAME, std::string("ccc.mrcs"));
    mdt.addObject();
    mdt.setValue(EMDL_IMAGE_NAME, std::string("aaa.mrcs"));
    mdt.addObject();
    mdt.setValue(EMDL_IMAGE_NAME, std::string("bbb.mrcs"));

    mdt.newSort(EMDL_IMAGE_NAME);

    std::string s0 = mdt.getString(EMDL_IMAGE_NAME, 0);
    std::string s1 = mdt.getString(EMDL_IMAGE_NAME, 1);
    std::string s2 = mdt.getString(EMDL_IMAGE_NAME, 2);
    EXPECT_LT(s0, s1);
    EXPECT_LT(s1, s2);
}

TEST(MetaDataTableTest, NewSort_StringDescending)
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.setValue(EMDL_IMAGE_NAME, std::string("aaa.mrcs"));
    mdt.addObject();
    mdt.setValue(EMDL_IMAGE_NAME, std::string("bbb.mrcs"));

    mdt.newSort(EMDL_IMAGE_NAME, /*do_reverse=*/true);

    std::string s0 = mdt.getString(EMDL_IMAGE_NAME, 0);
    std::string s1 = mdt.getString(EMDL_IMAGE_NAME, 1);
    EXPECT_GT(s0, s1);
}

// ---------------------------------------------------------------------------
// 25. addMissingLabels — adds labels from another table that are absent
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, AddMissingLabels_AddsAbsentLabel)
{
    MetaDataTable src;
    src.addObject();
    src.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)1.0);
    src.setValue(EMDL_CTF_VOLTAGE, (RFLOAT)300.0);

    MetaDataTable dst;
    dst.addObject();
    dst.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)2.0);
    // EMDL_CTF_VOLTAGE is absent from dst

    dst.addMissingLabels(&src);

    EXPECT_TRUE(dst.containsLabel(EMDL_CTF_VOLTAGE));
}

TEST(MetaDataTableTest, AddMissingLabels_ExistingLabelUnchanged)
{
    MetaDataTable src;
    src.addObject();
    src.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)999.0);

    MetaDataTable dst;
    dst.addObject();
    dst.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)1.0);

    dst.addMissingLabels(&src);

    // EMDL_CTF_DEFOCUSU was already present; value must be unchanged
    RFLOAT v = 0;
    dst.getValue(EMDL_CTF_DEFOCUSU, v, 0);
    EXPECT_NEAR(v, 1.0, 1e-6);
}

// ---------------------------------------------------------------------------
// 26. getIntMinusOne — returns stored int value minus one
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, GetIntMinusOne_ReturnsIntMinus1)
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.setValue(EMDL_MLMODEL_NR_CLASSES, 5);
    EXPECT_EQ(mdt.getIntMinusOne(EMDL_MLMODEL_NR_CLASSES, 0), 4);
}

TEST(MetaDataTableTest, GetIntMinusOne_ZeroValue)
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.setValue(EMDL_MLMODEL_NR_CLASSES, 1);
    EXPECT_EQ(mdt.getIntMinusOne(EMDL_MLMODEL_NR_CLASSES, 0), 0);
}

// ---------------------------------------------------------------------------
// 27. getDouble — same as getRfloat but cast to double
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, GetDouble_ReturnsDoubleValue)
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)20000.0);
    EXPECT_NEAR(mdt.getDouble(EMDL_CTF_DEFOCUSU, 0), 20000.0, 1e-3);
}

// ---------------------------------------------------------------------------
// 28. getAngleInRad — converts stored degree value to radians
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, GetAngleInRad_ConvertsDegToRad)
{
    MetaDataTable mdt;
    mdt.addObject();
    // Store 180 degrees; expect π radians back
    mdt.setValue(EMDL_CTF_DEFOCUS_ANGLE, (RFLOAT)180.0);
    EXPECT_NEAR(mdt.getAngleInRad(EMDL_CTF_DEFOCUS_ANGLE, 0), M_PI, 1e-6);
}

TEST(MetaDataTableTest, GetAngleInRad_ZeroDegrees)
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.setValue(EMDL_CTF_DEFOCUS_ANGLE, (RFLOAT)0.0);
    EXPECT_NEAR(mdt.getAngleInRad(EMDL_CTF_DEFOCUS_ANGLE, 0), 0.0, 1e-10);
}

// ---------------------------------------------------------------------------
// 29. labelExists — alias for containsLabel
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, LabelExists_TrueAfterSetValue)
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.setValue(EMDL_CTF_VOLTAGE, (RFLOAT)300.0);
    EXPECT_TRUE(mdt.labelExists(EMDL_CTF_VOLTAGE));
}

TEST(MetaDataTableTest, LabelExists_FalseForMissingLabel)
{
    MetaDataTable mdt;
    mdt.addObject();
    EXPECT_FALSE(mdt.labelExists(EMDL_CTF_VOLTAGE));
}

// ---------------------------------------------------------------------------
// 30. setValueFromString — parses string to typed value
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, SetValueFromString_Rfloat)
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.addLabel(EMDL_CTF_DEFOCUSU);
    EXPECT_TRUE(mdt.setValueFromString(EMDL_CTF_DEFOCUSU, "13500.0", 0));
    EXPECT_NEAR(mdt.getRfloat(EMDL_CTF_DEFOCUSU, 0), 13500.0, 1e-2);
}

TEST(MetaDataTableTest, SetValueFromString_String)
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.addLabel(EMDL_IMAGE_NAME);
    EXPECT_TRUE(mdt.setValueFromString(EMDL_IMAGE_NAME, "test_particle.mrcs", 0));
    EXPECT_EQ(mdt.getString(EMDL_IMAGE_NAME, 0), "test_particle.mrcs");
}

TEST(MetaDataTableTest, SetValueFromString_Int)
{
    MetaDataTable mdt;
    mdt.addObject();
    mdt.addLabel(EMDL_MLMODEL_NR_CLASSES);
    EXPECT_TRUE(mdt.setValueFromString(EMDL_MLMODEL_NR_CLASSES, "7", 0));
    EXPECT_EQ(mdt.getInt(EMDL_MLMODEL_NR_CLASSES, 0), 7);
}

// ---------------------------------------------------------------------------
// 31. reserve — does not crash and preserves existing rows
// ---------------------------------------------------------------------------
TEST(MetaDataTableTest, Reserve_DoesNotCrash)
{
    MetaDataTable mdt = makeCTFTable(3);
    EXPECT_NO_THROW(mdt.reserve(100));
    EXPECT_EQ(mdt.numberOfObjects(), (size_t)3);
}

// ---------------------------------------------------------------------------
// goToObject
// ---------------------------------------------------------------------------

TEST(MetaDataTableTest, GoToObject_ReturnsObjectId)
{
    MetaDataTable mdt = makeCTFTable(3);
    long result = mdt.goToObject(1);
    EXPECT_EQ(result, 1L);
}

TEST(MetaDataTableTest, GoToObject_CurrentObjectIdUpdated)
{
    MetaDataTable mdt = makeCTFTable(3);
    mdt.goToObject(2);
    RFLOAT val;
    mdt.getValue(EMDL_CTF_DEFOCUSU, val, -1);  // -1 uses current
    EXPECT_NEAR(val, 12000.0, 1.0);
}

// ---------------------------------------------------------------------------
// printLabels
// ---------------------------------------------------------------------------

TEST(MetaDataTableTest, PrintLabels_DoesNotCrash)
{
    MetaDataTable mdt = makeCTFTable(1);
    std::ostringstream oss;
    EXPECT_NO_THROW(mdt.printLabels(oss));
}

TEST(MetaDataTableTest, PrintLabels_ContainsLabelName)
{
    MetaDataTable mdt = makeCTFTable(1);
    std::ostringstream oss;
    mdt.printLabels(oss);
    EXPECT_NE(oss.str().find("rlnDefocusU"), std::string::npos);
}

// ---------------------------------------------------------------------------
// addObject (no-arg) and addObject(MetaDataContainer*)
// ---------------------------------------------------------------------------

TEST(MetaDataTableTest, AddObjectNoArg_IncreasesCount)
{
    MetaDataTable mdt = makeCTFTable(2);
    size_t n_before = mdt.numberOfObjects();
    mdt.addObject();
    EXPECT_EQ(mdt.numberOfObjects(), n_before + 1);
}

// ---------------------------------------------------------------------------
// subsetMetaDataTable (RFLOAT range)
// ---------------------------------------------------------------------------

TEST(MetaDataTableTest, SubsetByRange_CorrectCount)
{
    MetaDataTable mdt = makeCTFTable(5);  // defocusU: 10000, 11000, 12000, 13000, 14000
    MetaDataTable sub = subsetMetaDataTable(mdt, EMDL_CTF_DEFOCUSU, 11000.0, 13000.0);
    EXPECT_EQ(sub.numberOfObjects(), (size_t)3);  // 11000, 12000, 13000
}

TEST(MetaDataTableTest, SubsetByRange_EmptyResult)
{
    MetaDataTable mdt = makeCTFTable(3);
    MetaDataTable sub = subsetMetaDataTable(mdt, EMDL_CTF_DEFOCUSU, 99000.0, 99999.0);
    EXPECT_EQ(sub.numberOfObjects(), (size_t)0);
}

TEST(MetaDataTableTest, SubsetByRange_AllRows)
{
    MetaDataTable mdt = makeCTFTable(3);
    MetaDataTable sub = subsetMetaDataTable(mdt, EMDL_CTF_DEFOCUSU, 0.0, 99999.0);
    EXPECT_EQ(sub.numberOfObjects(), (size_t)3);
}

// ---------------------------------------------------------------------------
// subsetMetaDataTable (string search)
// ---------------------------------------------------------------------------

TEST(MetaDataTableTest, SubsetByString_Include_CorrectCount)
{
    MetaDataTable mdt = makeCTFTable(4);  // particle_0, particle_1, particle_2, particle_3
    MetaDataTable sub = subsetMetaDataTable(mdt, EMDL_IMAGE_NAME, std::string("particle_1"), false);
    EXPECT_EQ(sub.numberOfObjects(), (size_t)1);
}

TEST(MetaDataTableTest, SubsetByString_Exclude_CorrectCount)
{
    MetaDataTable mdt = makeCTFTable(4);
    MetaDataTable sub = subsetMetaDataTable(mdt, EMDL_IMAGE_NAME, std::string("particle_0"), true);
    EXPECT_EQ(sub.numberOfObjects(), (size_t)3);
}

// ---------------------------------------------------------------------------
// newSort with DOUBLE and INT labels (exercises MdDoubleComparator/MdIntComparator)
// ---------------------------------------------------------------------------

TEST(MetaDataTableTest, NewSort_DoubleDescending)
{
    MetaDataTable mdt;
    for (int i = 0; i < 4; i++)
    {
        mdt.addObject();
        mdt.setValue(EMDL_CTF_DEFOCUSU, (RFLOAT)(i * 1000.0));
    }
    mdt.newSort(EMDL_CTF_DEFOCUSU, /*descend=*/true);
    RFLOAT first;
    mdt.getValue(EMDL_CTF_DEFOCUSU, first, 0);
    EXPECT_NEAR(first, 3000.0, 1.0);
}

TEST(MetaDataTableTest, NewSort_IntAscending)
{
    MetaDataTable mdt;
    for (int i = 3; i >= 0; i--)
    {
        mdt.addObject();
        mdt.setValue(EMDL_MLMODEL_NR_CLASSES, i);
    }
    mdt.newSort(EMDL_MLMODEL_NR_CLASSES, /*descend=*/false);
    int first;
    mdt.getValue(EMDL_MLMODEL_NR_CLASSES, first, 0);
    EXPECT_EQ(first, 0);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
