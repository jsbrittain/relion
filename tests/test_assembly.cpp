/*
 * Unit tests for src/assembly.h / src/assembly.cpp
 *
 * Covers: Atom, Residue, Molecule, Assembly construction,
 *         addAtom, addResidue, insertResidue, insertResidues,
 *         addMolecule, numberOfAtoms/Residues/Molecules,
 *         join, sortResidues, applyTransformation
 */

#include <gtest/gtest.h>
#include "src/assembly.h"
#include "src/matrix1d.h"
#include "src/matrix2d.h"

// ---------------------------------------------------------------------------
// Atom
// ---------------------------------------------------------------------------

TEST(AtomTest, DefaultConstructor)
{
    Atom a;
    EXPECT_TRUE(a.name.empty());
    EXPECT_NEAR(a.occupancy, 0.0, 1e-10);
    EXPECT_NEAR(a.bfactor,   0.0, 1e-10);
}

TEST(AtomTest, NamedConstructor)
{
    Atom a("CA");
    EXPECT_EQ(a.name, "CA");
}

TEST(AtomTest, GetCoordinates)
{
    Atom a("CA");
    a.coords.resize(3);
    XX(a.coords) = 1.0;
    YY(a.coords) = 2.0;
    ZZ(a.coords) = 3.0;
    Matrix1D<RFLOAT> c = a.getCoordinates();
    EXPECT_NEAR(XX(c), 1.0, 1e-10);
    EXPECT_NEAR(YY(c), 2.0, 1e-10);
    EXPECT_NEAR(ZZ(c), 3.0, 1e-10);
}

// ---------------------------------------------------------------------------
// Residue
// ---------------------------------------------------------------------------

TEST(ResidueTest, DefaultConstructor)
{
    Residue r;
    EXPECT_TRUE(r.name.empty());
    EXPECT_EQ(r.numberOfAtoms(), 0);
}

TEST(ResidueTest, NamedConstructor)
{
    Residue r("ALA", 5);
    EXPECT_EQ(r.name, "ALA");
    EXPECT_EQ(r.number, 5);
}

TEST(ResidueTest, AddAtom)
{
    Residue r("GLY", 1);
    r.addAtom("CA", 1.0, 2.0, 3.0, 1.0, 0.0);
    EXPECT_EQ(r.numberOfAtoms(), 1);
    EXPECT_EQ(r.atoms[0].name, "CA");
    EXPECT_NEAR(XX(r.atoms[0].coords), 1.0, 1e-10);
    EXPECT_NEAR(YY(r.atoms[0].coords), 2.0, 1e-10);
    EXPECT_NEAR(ZZ(r.atoms[0].coords), 3.0, 1e-10);
    EXPECT_NEAR(r.atoms[0].occupancy, 1.0, 1e-10);
    EXPECT_NEAR(r.atoms[0].bfactor,   0.0, 1e-10);
}

TEST(ResidueTest, AddMultipleAtoms)
{
    Residue r("ALA", 1);
    r.addAtom("N",  0.0, 0.0, 0.0);
    r.addAtom("CA", 1.0, 0.0, 0.0);
    r.addAtom("C",  2.0, 0.0, 0.0);
    EXPECT_EQ(r.numberOfAtoms(), 3);
}

// ---------------------------------------------------------------------------
// Molecule
// ---------------------------------------------------------------------------

TEST(MoleculeTest, DefaultConstructor)
{
    Molecule mol;
    EXPECT_TRUE(mol.name.empty());
    EXPECT_EQ(mol.numberOfResidues(), 0);
}

TEST(MoleculeTest, NamedConstructor)
{
    Molecule mol("chainA", "A");
    EXPECT_EQ(mol.name, "chainA");
    EXPECT_EQ(mol.alt_name, "A");
}

TEST(MoleculeTest, AddResidue_ByRef)
{
    Molecule mol("mol");
    Residue r("ALA", 1);
    mol.addResidue(r);
    EXPECT_EQ(mol.numberOfResidues(), 1);
    EXPECT_EQ(mol.residues[0].name, "ALA");
}

TEST(MoleculeTest, AddResidue_ByNameNumber)
{
    Molecule mol("mol");
    mol.addResidue("GLY", 1);
    mol.addResidue("ALA", 2);
    EXPECT_EQ(mol.numberOfResidues(), 2);
}

TEST(MoleculeTest, InsertResidue_AtPosition)
{
    Molecule mol("mol");
    mol.addResidue("ALA", 1);
    mol.addResidue("ALA", 3);
    Residue r("GLY", 2);
    mol.insertResidue(r, 1); // insert at position 1 (between index 0 and 1)
    EXPECT_EQ(mol.numberOfResidues(), 3);
    EXPECT_EQ(mol.residues[1].number, 2);
}

TEST(MoleculeTest, InsertResidues_EntireMolecule)
{
    Molecule mol1("molA");
    mol1.addResidue("ALA", 1);
    mol1.addResidue("GLY", 2);
    mol1.addResidue("VAL", 3);

    Molecule mol2("molB");
    // Insert all residues from mol1 into mol2
    mol2.insertResidues(mol1);
    EXPECT_EQ(mol2.numberOfResidues(), 3);
}

TEST(MoleculeTest, InsertResidues_Range)
{
    Molecule mol1("molA");
    mol1.addResidue("ALA", 1);
    mol1.addResidue("GLY", 2);
    mol1.addResidue("VAL", 3);
    mol1.addResidue("PRO", 4);

    Molecule mol2("molB");
    mol2.insertResidues(mol1, 2, 3); // residues numbered 2 and 3
    EXPECT_EQ(mol2.numberOfResidues(), 2);
    EXPECT_EQ(mol2.residues[0].number, 2);
    EXPECT_EQ(mol2.residues[1].number, 3);
}

// ---------------------------------------------------------------------------
// Assembly
// ---------------------------------------------------------------------------

TEST(AssemblyTest, DefaultConstructor)
{
    Assembly asm1;
    EXPECT_TRUE(asm1.name.empty());
    EXPECT_EQ(asm1.numberOfMolecules(), 0);
    EXPECT_EQ(asm1.numberOfAtoms(),     0);
    EXPECT_EQ(asm1.numberOfResidues(),  0);
}

TEST(AssemblyTest, NamedConstructor)
{
    Assembly asm1("myAssembly");
    EXPECT_EQ(asm1.name, "myAssembly");
}

TEST(AssemblyTest, AddMolecule_ByName)
{
    Assembly asm1;
    asm1.addMolecule("molA", "A");
    EXPECT_EQ(asm1.numberOfMolecules(), 1);
    EXPECT_EQ(asm1.molecules[0].name, "molA");
}

TEST(AssemblyTest, AddMolecule_ByRef)
{
    Assembly asm1;
    Molecule mol("chainB");
    asm1.addMolecule(mol);
    EXPECT_EQ(asm1.numberOfMolecules(), 1);
}

TEST(AssemblyTest, NumberOfAtoms)
{
    Assembly asm1;
    asm1.addMolecule("molA", "A");
    asm1.molecules[0].addResidue("ALA", 1);
    asm1.molecules[0].residues[0].addAtom("CA", 1.0, 2.0, 3.0);
    asm1.molecules[0].residues[0].addAtom("N",  0.0, 0.0, 0.0);
    EXPECT_EQ(asm1.numberOfAtoms(), 2);
}

TEST(AssemblyTest, NumberOfResidues)
{
    Assembly asm1;
    asm1.addMolecule("molA", "A");
    asm1.molecules[0].addResidue("ALA", 1);
    asm1.molecules[0].addResidue("GLY", 2);
    EXPECT_EQ(asm1.numberOfResidues(), 2);
}

TEST(AssemblyTest, SortResidues)
{
    Assembly asm1;
    asm1.addMolecule("molA", "A");
    asm1.molecules[0].addResidue("VAL", 3);
    asm1.molecules[0].addResidue("ALA", 1);
    asm1.molecules[0].addResidue("GLY", 2);
    asm1.sortResidues();
    EXPECT_EQ(asm1.molecules[0].residues[0].number, 1);
    EXPECT_EQ(asm1.molecules[0].residues[1].number, 2);
    EXPECT_EQ(asm1.molecules[0].residues[2].number, 3);
}

TEST(AssemblyTest, Join_MergesMolecules)
{
    Assembly asm1("A"), asm2("B");
    asm1.addMolecule("molA", "A");
    asm2.addMolecule("molB", "B");
    asm1.join(asm2);
    EXPECT_EQ(asm1.numberOfMolecules(), 2);
}

TEST(AssemblyTest, CopyConstructor)
{
    Assembly asm1("orig");
    asm1.addMolecule("mol1", "A");
    Assembly asm2(asm1);
    EXPECT_EQ(asm2.name, "orig");
    EXPECT_EQ(asm2.numberOfMolecules(), 1);
}

TEST(AssemblyTest, ApplyTransformation_Identity)
{
    Assembly asm1;
    asm1.addMolecule("molA", "A");
    asm1.molecules[0].addResidue("ALA", 1);
    asm1.molecules[0].residues[0].addAtom("CA", 1.0, 2.0, 3.0);

    Matrix2D<RFLOAT> rot;
    rot.initIdentity(3);

    Matrix1D<RFLOAT> shift;
    shift.initZeros(3);

    asm1.applyTransformation(rot, shift);

    EXPECT_NEAR(XX(asm1.molecules[0].residues[0].atoms[0].coords), 1.0, 1e-9);
    EXPECT_NEAR(YY(asm1.molecules[0].residues[0].atoms[0].coords), 2.0, 1e-9);
    EXPECT_NEAR(ZZ(asm1.molecules[0].residues[0].atoms[0].coords), 3.0, 1e-9);
}

TEST(AssemblyTest, ApplyTransformation_PureShift)
{
    Assembly asm1;
    asm1.addMolecule("molA", "A");
    asm1.molecules[0].addResidue("ALA", 1);
    asm1.molecules[0].residues[0].addAtom("CA", 0.0, 0.0, 0.0);

    Matrix2D<RFLOAT> rot;
    rot.initIdentity(3);

    Matrix1D<RFLOAT> shift;
    shift.initZeros(3);
    XX(shift) = 5.0;
    YY(shift) = -3.0;
    ZZ(shift) = 1.0;

    asm1.applyTransformation(rot, shift);

    EXPECT_NEAR(XX(asm1.molecules[0].residues[0].atoms[0].coords),  5.0, 1e-9);
    EXPECT_NEAR(YY(asm1.molecules[0].residues[0].atoms[0].coords), -3.0, 1e-9);
    EXPECT_NEAR(ZZ(asm1.molecules[0].residues[0].atoms[0].coords),  1.0, 1e-9);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
