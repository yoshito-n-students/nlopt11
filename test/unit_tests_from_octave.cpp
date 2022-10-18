#include <cmath>
#include <vector>

#include <nlopt11.hpp>

#include <gtest/gtest.h>

// https://octave.org/doc/v4.2.1/Nonlinear-Programming.html#Nonlinear-Programming
TEST(FromOctave, QP5) {
  // form the problem
  nlopt::opt11 opt(/* algorithm = */ nlopt::LN_COBYLA, /* n_params = */ 5);
  opt.set_max_objective([](const int n, const double *const x) {
    EXPECT_EQ(n, 5);
    const double y0 = x[0] * x[1] * x[2] * x[3] * x[4],
                 y1 = x[0] * x[0] * x[0] + x[1] * x[1] * x[1] + 1.;
    return -std::exp(y0) + 0.5 * y1 * y1;
  });
  opt.add_equality_mconstraint(
      [](const int m, double *const result, const int n, const double *const x) {
        EXPECT_EQ(m, 2);
        EXPECT_EQ(n, 5);
        const double y0 = x[0] * x[0], y1 = x[1] * x[1];
        result[0] = y0 + y1 + x[2] * x[2] + x[3] * x[3] + x[4] * x[4] - 10.;
        result[1] = y0 * x[0] + y1 * x[1] + 1.;
      },
      /* tolerance = */ {1e-8, 1e-8});
  opt.add_equality_constraint(
      [](const int n, const double *const x) {
        EXPECT_EQ(n, 5);
        return x[1] * x[2] - 5. * x[3] * x[4];
      },
      1e-8);
  opt.set_xtol_rel(1e-5);

  // solve the problem
  std::vector<double> x = {-1.8, 1.7, 1.9, -0.8, -0.8};
  double minf;
  const nlopt::result result = opt.optimize(x, minf);

  // verify the solution
  EXPECT_EQ(result, nlopt::XTOL_REACHED);
  EXPECT_NEAR(x[0], -1.71714, 1e-4);
  EXPECT_NEAR(x[1], 1.59571, 1e-4);
  EXPECT_NEAR(x[2], 1.82725, 1e-4);
  EXPECT_NEAR(x[3], -0.76364, 1e-4);
  EXPECT_NEAR(x[4], -0.76364, 1e-4);
}