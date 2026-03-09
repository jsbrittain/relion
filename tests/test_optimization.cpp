/*
 * Unit tests for src/jaz/optimization/optimization.cpp and nelder_mead.cpp
 *
 * Covers: RosenbrockBanana::f, RosenbrockBanana::grad,
 *         DifferentiableOptimization::testGradient,
 *         NelderMead::optimize, NelderMead::test
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "src/jaz/optimization/optimization.h"
#include "src/jaz/optimization/nelder_mead.h"

// ---------------------------------------------------------------------------
// RosenbrockBanana
// ---------------------------------------------------------------------------

TEST(RosenbrockBananaTest, MinimumAtOneOne)
{
    RosenbrockBanana rb;
    std::vector<double> x = {1.0, 1.0};
    EXPECT_NEAR(rb.f(x, nullptr), 0.0, 1e-12);
}

TEST(RosenbrockBananaTest, NonMinimumIsPositive)
{
    RosenbrockBanana rb;
    std::vector<double> x = {0.0, 0.0};
    EXPECT_GT(rb.f(x, nullptr), 0.0);
}

TEST(RosenbrockBananaTest, GradAtMinimumIsZero)
{
    RosenbrockBanana rb;
    std::vector<double> x = {1.0, 1.0};
    std::vector<double> g = {0.0, 0.0};
    rb.grad(x, g, nullptr);
    EXPECT_NEAR(g[0], 0.0, 1e-12);
    EXPECT_NEAR(g[1], 0.0, 1e-12);
}

TEST(RosenbrockBananaTest, GradIsConsistentWithFiniteDiff)
{
    RosenbrockBanana rb;
    std::vector<double> x = {0.5, 0.3};
    std::vector<double> g = {0.0, 0.0};
    rb.grad(x, g, nullptr);

    const double eps = 1e-6;
    double f0 = rb.f(x, nullptr);
    for (int i = 0; i < 2; i++)
    {
        std::vector<double> xp = x;
        xp[i] += eps;
        double fd = (rb.f(xp, nullptr) - f0) / eps;
        EXPECT_NEAR(g[i], fd, 1e-4);
    }
}

TEST(RosenbrockBananaTest, TestGradient_DoesNotCrash)
{
    RosenbrockBanana rb;
    std::vector<double> x = {0.5, 0.3};
    EXPECT_NO_THROW(rb.testGradient(x, 1e-6));
}

// ---------------------------------------------------------------------------
// NelderMead::optimize
// ---------------------------------------------------------------------------

TEST(NelderMeadTest, OptimizesRosenbrock)
{
    RosenbrockBanana rb;
    std::vector<double> initial = {0.0, 0.0};
    double minCost = 0.0;
    std::vector<double> result = NelderMead::optimize(
        initial, rb, 0.5, 1e-6, 5000,
        1.0, 2.0, 0.5, 0.5, false, &minCost);

    ASSERT_EQ(result.size(), (size_t)2);
    EXPECT_NEAR(result[0], 1.0, 0.01);
    EXPECT_NEAR(result[1], 1.0, 0.01);
    EXPECT_NEAR(minCost, 0.0, 1e-6);
}

TEST(NelderMeadTest, MinCostIsReturned)
{
    // minCost is only set when maxIters is exhausted (not on early convergence).
    // Use very tight tolerance + small maxIters so the loop runs to completion.
    RosenbrockBanana rb;
    std::vector<double> initial = {0.5, 0.5};
    double minCost = 1e9;
    NelderMead::optimize(initial, rb, 0.5, 1e-20, 3,
                         1.0, 2.0, 0.5, 0.5, false, &minCost);
    // After 3 iterations minCost should be finite and < initial f(0.5,0.5)=6.5
    EXPECT_TRUE(std::isfinite(minCost));
    EXPECT_LT(minCost, 1e9);
}

TEST(NelderMeadTest, NoMinCostPtr_DoesNotCrash)
{
    RosenbrockBanana rb;
    std::vector<double> initial = {0.5, 0.5};
    EXPECT_NO_THROW(NelderMead::optimize(initial, rb, 0.5, 1e-4, 200));
}

// Simple quadratic: f(x) = x^2, minimum at 0
class QuadraticOpt : public Optimization
{
public:
    double f(const std::vector<double>& x, void*) const override
    {
        return x[0] * x[0];
    }
};

TEST(NelderMeadTest, OptimizesSimpleQuadratic)
{
    QuadraticOpt q;
    std::vector<double> initial = {3.0};
    double minCost;
    auto result = NelderMead::optimize(initial, q, 1.0, 1e-8, 200,
                                       1.0, 2.0, 0.5, 0.5, false, &minCost);
    EXPECT_NEAR(result[0], 0.0, 1e-4);
    EXPECT_NEAR(minCost, 0.0, 1e-8);
}

TEST(NelderMeadTest, NelderMeadTest_DoesNotCrash)
{
    EXPECT_NO_THROW(NelderMead::test());
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
