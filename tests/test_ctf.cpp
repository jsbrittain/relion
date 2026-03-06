/*
 * GoogleTest unit tests for CTF (src/ctf.h/.cpp).
 *
 * Build and run:
 *   cmake -DBUILD_TESTS=ON ...
 *   make test_ctf
 *   ./build/bin/test_ctf
 *
 * No MPI required; pure CPU unit tests.
 * The only code paths exercised are setValues() + initialise() + getCTF(),
 * getDeltaF(), getLowOrderGamma(), getK(), getAxx/Axy/Ayy(), and
 * CTF::write(MetaDataTable&).  No ObservationModel, no file I/O.
 *
 * What is tested:
 *   1.  Default constructor – object is constructible without crash.
 *   2.  setValues() stores public members correctly.
 *   3.  getK() – vector has 6 elements; K1 K2 K3 > 0; K4=K5=0 for
 *               Bfac=0/phase_shift=0.
 *   4.  Electron wavelength (lambda) is in the expected range for 300 kV.
 *   5.  getCTF(0,0) – value derived from amplitude-contrast K3.
 *   6.  getCTF central symmetry: getCTF(X,Y) == getCTF(-X,-Y).
 *   7.  getCTF XY symmetry for isotropic defocus: getCTF(X,Y) == getCTF(Y,X).
 *   8.  getCTF do_abs=true – all values non-negative.
 *   9.  getCTF do_only_flip_phases – result is ±1.
 *  10.  getCTF do_intact_until_first_peak – returns ±scale at small gamma.
 *  11.  getCTF do_damping=false vs true – differ when Bfac ≠ 0.
 *  12.  Astigmatic defocus: Axx ≠ Ayy when DeltafU ≠ DeltafV.
 *  13.  Isotropic defocus: Axx == Ayy, Axy == 0 when azimuthal_angle=0 and
 *               DeltafU == DeltafV.
 *  14.  getDeltaF at origin returns 0.
 *  15.  getDeltaF isotropic – returns same value in all directions.
 *  16.  getLowOrderGamma at origin == -K5 - K3.
 *  17.  Phase shift changes the CTF value.
 *  18.  Scale factor scales the CTF linearly.
 *  19.  CTF::write() sets EMDL_CTF_DEFOCUSU/V/ANGLE correctly in a
 *               MetaDataTable.
 */

#include <gtest/gtest.h>
#include <cmath>
#include "src/ctf.h"
#include "src/metadata_table.h"
#include "src/metadata_label.h"
#include "src/macros.h"

static constexpr double EPS    = 1e-9;   // exact equality tolerance
static constexpr double NEAR   = 1e-5;   // floating-point near tolerance
static constexpr double FNEAR  = 1e-4;   // looser tolerance for trig results

// Canonical CTF parameters used throughout the suite.
static constexpr double DEF_U   = 15000.0;  // Angstroms
static constexpr double DEF_V   = 15000.0;
static constexpr double DEF_ANG = 0.0;      // degrees
static constexpr double KV      = 300.0;    // kV
static constexpr double CS      = 2.7;      // mm
static constexpr double Q0      = 0.1;
static constexpr double BFAC    = 0.0;
static constexpr double SCALE   = 1.0;
static constexpr double PSHIFT  = 0.0;

static CTF makeCTF(double defU = DEF_U, double defV = DEF_V,
                   double angle = DEF_ANG, double kv = KV,
                   double cs = CS, double q0 = Q0,
                   double bfac = BFAC, double scale = SCALE,
                   double pshift = PSHIFT)
{
    CTF ctf;
    ctf.setValues(defU, defV, angle, kv, cs, q0, bfac, scale, pshift);
    return ctf;
}

// ---------------------------------------------------------------------------
// 1. Default constructor
// ---------------------------------------------------------------------------
TEST(CTFTest, DefaultConstructorDoesNotCrash)
{
    CTF ctf;
    EXPECT_NEAR(ctf.kV, 200.0, EPS);
}

// ---------------------------------------------------------------------------
// 2. setValues stores parameters
// ---------------------------------------------------------------------------
TEST(CTFTest, SetValuesStoresParameters)
{
    CTF ctf = makeCTF();
    EXPECT_NEAR(ctf.DeltafU,         DEF_U,   EPS);
    EXPECT_NEAR(ctf.DeltafV,         DEF_V,   EPS);
    EXPECT_NEAR(ctf.azimuthal_angle, DEF_ANG, EPS);
    EXPECT_NEAR(ctf.kV,              KV,      EPS);
    EXPECT_NEAR(ctf.Cs,              CS,      EPS);
    EXPECT_NEAR(ctf.Q0,              Q0,      EPS);
    EXPECT_NEAR(ctf.Bfac,            BFAC,    EPS);
    EXPECT_NEAR(ctf.scale,           SCALE,   EPS);
}

// ---------------------------------------------------------------------------
// 3. getK() – structure and sign
// ---------------------------------------------------------------------------
TEST(CTFTest, GetKReturns6Elements)
{
    CTF ctf = makeCTF();
    EXPECT_EQ(ctf.getK().size(), (size_t)6);
}

TEST(CTFTest, K1IsPositive)
{
    CTF ctf = makeCTF();
    EXPECT_GT(ctf.getK()[1], 0.0);
}

TEST(CTFTest, K2IsPositiveForPositiveCs)
{
    CTF ctf = makeCTF();
    EXPECT_GT(ctf.getK()[2], 0.0);
}

TEST(CTFTest, K3IsPositiveForPositiveQ0)
{
    CTF ctf = makeCTF();
    EXPECT_GT(ctf.getK()[3], 0.0);
}

TEST(CTFTest, K4IsZeroForZeroBfac)
{
    CTF ctf = makeCTF(DEF_U, DEF_V, DEF_ANG, KV, CS, Q0, 0.0);
    EXPECT_NEAR(ctf.getK()[4], 0.0, EPS);
}

TEST(CTFTest, K5IsZeroForZeroPhaseShift)
{
    CTF ctf = makeCTF();
    EXPECT_NEAR(ctf.getK()[5], 0.0, EPS);
}

TEST(CTFTest, K4IsNegativeForPositiveBfac)
{
    CTF ctf = makeCTF(DEF_U, DEF_V, DEF_ANG, KV, CS, Q0, 100.0);
    EXPECT_LT(ctf.getK()[4], 0.0);  // K4 = -Bfac/4 < 0
}

TEST(CTFTest, K5IsNonZeroForNonZeroPhaseShift)
{
    CTF ctf = makeCTF(DEF_U, DEF_V, DEF_ANG, KV, CS, Q0, 0.0, 1.0, 90.0);
    EXPECT_GT(std::abs(ctf.getK()[5]), 1.0);  // 90° in radians = π/2 ≈ 1.57
}

// ---------------------------------------------------------------------------
// 4. Electron wavelength is physically reasonable for 300 kV
// ---------------------------------------------------------------------------
TEST(CTFTest, LambdaInReasonableRange)
{
    CTF ctf = makeCTF();
    // 300 kV electrons: lambda ≈ 0.01969 Angstroms
    // Allow wide tolerance since we're not replicating the formula here.
    EXPECT_GT(ctf.lambda, 0.010);
    EXPECT_LT(ctf.lambda, 0.030);
}

// ---------------------------------------------------------------------------
// 5. getCTF(0,0) – from amplitude-contrast K3 only
// ---------------------------------------------------------------------------
TEST(CTFTest, GetCTFAtOriginIsFromAmplitudeContrast)
{
    CTF ctf = makeCTF();
    // gamma(0,0) = -K5-K3; with K5=0: gamma = -K3
    // CTF = -sin(-K3) = sin(K3) ≈ Q0 for small Q0
    double val = ctf.getCTF(0.0, 0.0);
    // Must be positive (sin of positive angle)
    EXPECT_GT(val, 0.0);
    // Rough magnitude check: sin(K3) ≈ Q0/(sqrt(1+Q0^2/(1-Q0^2))) ≈ Q0
    EXPECT_NEAR(val, std::sin(ctf.getK()[3]), FNEAR);
}

// ---------------------------------------------------------------------------
// 6. Central symmetry: getCTF(X,Y) == getCTF(-X,-Y)
// ---------------------------------------------------------------------------
TEST(CTFTest, CentralSymmetry)
{
    CTF ctf = makeCTF();
    const double X = 0.005, Y = 0.003;
    EXPECT_NEAR(ctf.getCTF(X, Y), ctf.getCTF(-X, -Y), EPS);
}

// ---------------------------------------------------------------------------
// 7. XY symmetry for isotropic defocus
// ---------------------------------------------------------------------------
TEST(CTFTest, XYSymmetryForIsotropicDefocus)
{
    CTF ctf = makeCTF();  // DeltafU == DeltafV, angle == 0
    const double X = 0.004, Y = 0.007;
    EXPECT_NEAR(ctf.getCTF(X, Y), ctf.getCTF(Y, X), EPS);
}

// ---------------------------------------------------------------------------
// 8. do_abs=true – all values non-negative
// ---------------------------------------------------------------------------
TEST(CTFTest, DoAbsGivesNonNegativeValues)
{
    CTF ctf = makeCTF();
    const double freqs[] = {0.0, 0.002, 0.005, 0.010, 0.015, 0.020};
    for (double f : freqs)
        EXPECT_GE(ctf.getCTF(f, 0.0, /*do_abs=*/true), 0.0);
}

// ---------------------------------------------------------------------------
// 9. do_only_flip_phases – result is ±1 (scaled by scale factor)
// ---------------------------------------------------------------------------
TEST(CTFTest, DoOnlyFlipPhasesReturnsPlusMinusOne)
{
    CTF ctf = makeCTF();
    const double freqs[] = {0.001, 0.005, 0.010, 0.020};
    for (double f : freqs)
    {
        double v = ctf.getCTF(f, 0.0, false, /*do_only_flip_phases=*/true);
        EXPECT_NEAR(std::abs(v), 1.0, FNEAR);
    }
}

// ---------------------------------------------------------------------------
// 10. do_intact_until_first_peak – returns scale at small gamma
// ---------------------------------------------------------------------------
TEST(CTFTest, DoIntactUntilFirstPeakAtOrigin)
{
    CTF ctf = makeCTF();
    // At origin, |gamma| = K3 ≈ 0.1 which is < PI/2
    double v = ctf.getCTF(0.0, 0.0, false, false, /*do_intact_until_first_peak=*/true);
    EXPECT_NEAR(v, SCALE, FNEAR);
}

// ---------------------------------------------------------------------------
// 11. do_damping=false vs true – different when Bfac ≠ 0 at nonzero u
// ---------------------------------------------------------------------------
TEST(CTFTest, DampingMatterWhenBfacNonZero)
{
    CTF ctf = makeCTF(DEF_U, DEF_V, DEF_ANG, KV, CS, Q0, 500.0);
    const double f = 0.010;
    double with_damp    = ctf.getCTF(f, 0.0, false, false, false, /*do_damping=*/true);
    double without_damp = ctf.getCTF(f, 0.0, false, false, false, /*do_damping=*/false);
    // At nonzero spatial frequency and Bfac=500, these should differ
    EXPECT_NE(with_damp, without_damp);
}

TEST(CTFTest, NoDampingBothEquivalentAtOrigin)
{
    // At the origin u2=0 so e^(K4*u2)=1 regardless of Bfac
    CTF ctf = makeCTF(DEF_U, DEF_V, DEF_ANG, KV, CS, Q0, 500.0);
    double with_damp    = ctf.getCTF(0.0, 0.0, false, false, false, true);
    double without_damp = ctf.getCTF(0.0, 0.0, false, false, false, false);
    EXPECT_NEAR(with_damp, without_damp, EPS);
}

// ---------------------------------------------------------------------------
// 12. Astigmatic defocus: Axx ≠ Ayy
// ---------------------------------------------------------------------------
TEST(CTFTest, AstigmaticDefocusGivesAxxNeqAyy)
{
    CTF ctf = makeCTF(15000.0, 12000.0, 0.0);
    EXPECT_NE(ctf.getAxx(), ctf.getAyy());
}

TEST(CTFTest, AstigmaticDefocusAngle45GivesNonZeroAxy)
{
    CTF ctf = makeCTF(15000.0, 12000.0, 45.0);
    EXPECT_NE(ctf.getAxy(), 0.0);
}

// ---------------------------------------------------------------------------
// 13. Isotropic defocus: Axx == Ayy, Axy == 0
// ---------------------------------------------------------------------------
TEST(CTFTest, IsotropicDefocusGivesAxxEqAyy)
{
    CTF ctf = makeCTF();  // DeltafU == DeltafV, angle == 0
    EXPECT_NEAR(ctf.getAxx(), ctf.getAyy(), EPS);
}

TEST(CTFTest, IsotropicZeroAngleGivesZeroAxy)
{
    CTF ctf = makeCTF();
    EXPECT_NEAR(ctf.getAxy(), 0.0, EPS);
}

// ---------------------------------------------------------------------------
// 14. getDeltaF at origin returns 0
// ---------------------------------------------------------------------------
TEST(CTFTest, GetDeltaFAtOriginIsZero)
{
    CTF ctf = makeCTF();
    EXPECT_NEAR(ctf.getDeltaF(0.0, 0.0), 0.0, EPS);
}

// ---------------------------------------------------------------------------
// 15. getDeltaF isotropic – same value in all directions
// ---------------------------------------------------------------------------
TEST(CTFTest, GetDeltaFIsotropicIsSameInAllDirections)
{
    CTF ctf = makeCTF();  // DeltafU == DeltafV
    double f = 0.01;
    EXPECT_NEAR(ctf.getDeltaF(f, 0.0),  ctf.getDeltaF(0.0, f),  FNEAR);
    EXPECT_NEAR(ctf.getDeltaF(f, 0.0),  ctf.getDeltaF(f, f),    FNEAR);
}

// ---------------------------------------------------------------------------
// 16. getLowOrderGamma at origin == -K5 - K3
// ---------------------------------------------------------------------------
TEST(CTFTest, LowOrderGammaAtOriginEqualsMinusK5MinusK3)
{
    CTF ctf = makeCTF();
    const auto K = ctf.getK();
    EXPECT_NEAR(ctf.getLowOrderGamma(0.0, 0.0), -K[5] - K[3], NEAR);
}

// ---------------------------------------------------------------------------
// 17. Phase shift changes the CTF value at origin
// ---------------------------------------------------------------------------
TEST(CTFTest, PhaseShiftChangesCtfAtOrigin)
{
    CTF ctf_no  = makeCTF(DEF_U, DEF_V, DEF_ANG, KV, CS, Q0, 0.0, 1.0,   0.0);
    CTF ctf_90  = makeCTF(DEF_U, DEF_V, DEF_ANG, KV, CS, Q0, 0.0, 1.0,  90.0);
    EXPECT_NE(ctf_no.getCTF(0.0, 0.0), ctf_90.getCTF(0.0, 0.0));
}

// ---------------------------------------------------------------------------
// 18. Scale factor scales the CTF linearly
// ---------------------------------------------------------------------------
TEST(CTFTest, ScaleFactorScalesLinearly)
{
    CTF ctf1 = makeCTF(DEF_U, DEF_V, DEF_ANG, KV, CS, Q0, 0.0, 1.0);
    CTF ctf2 = makeCTF(DEF_U, DEF_V, DEF_ANG, KV, CS, Q0, 0.0, 2.0);
    const double f = 0.005;
    EXPECT_NEAR(ctf2.getCTF(f, 0.0), 2.0 * ctf1.getCTF(f, 0.0), NEAR);
}

// ---------------------------------------------------------------------------
// 19. CTF::write() populates MetaDataTable correctly
// ---------------------------------------------------------------------------
TEST(CTFTest, WritePopulatesMetaDataTable)
{
    CTF ctf = makeCTF(12345.0, 11111.0, 30.0, KV, CS, Q0, 50.0, 0.8, 10.0);
    MetaDataTable mdt;
    mdt.addObject();
    ctf.write(mdt);

    RFLOAT defU, defV, ang, bfac, scale, pshift;
    EXPECT_TRUE(mdt.getValue(EMDL_CTF_DEFOCUSU,    defU,  0));
    EXPECT_TRUE(mdt.getValue(EMDL_CTF_DEFOCUSV,    defV,  0));
    EXPECT_TRUE(mdt.getValue(EMDL_CTF_DEFOCUS_ANGLE, ang, 0));
    EXPECT_TRUE(mdt.getValue(EMDL_CTF_BFACTOR,     bfac,  0));
    EXPECT_TRUE(mdt.getValue(EMDL_CTF_SCALEFACTOR, scale, 0));
    EXPECT_TRUE(mdt.getValue(EMDL_CTF_PHASESHIFT,  pshift,0));

    EXPECT_NEAR(defU,   12345.0, NEAR);
    EXPECT_NEAR(defV,   11111.0, NEAR);
    EXPECT_NEAR(ang,       30.0, NEAR);
    EXPECT_NEAR(bfac,      50.0, NEAR);
    EXPECT_NEAR(scale,      0.8, NEAR);
    EXPECT_NEAR(pshift,    10.0, NEAR);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
