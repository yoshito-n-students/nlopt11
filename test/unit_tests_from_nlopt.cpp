#include <cmath>
#include <limits>
#include <vector>

#include <nlopt11.hpp>

#include <gtest/gtest.h>

// https://nlopt.readthedocs.io/en/latest/NLopt_Tutorial/#example-in-cc
TEST(FromNLopt, Tutorial) {
  // local lambda functions
  // - a function to be an objective
  static const auto f = [](const std::vector<double> &x, std::vector<double> &grad) {
    const double val = std::sqrt(x[1]);
    if (!grad.empty()) {
      grad[0] = 0.0;
      grad[1] = 0.5 / val;
    }
    return val;
  };
  // - another function to be a constraint
  static const auto g = [](const std::vector<double> &x, std::vector<double> &grad, const double a,
                           const double b) {
    const double y = a * x[0] + b;
    const double yy = y * y;
    if (!grad.empty()) {
      grad[0] = 3. * a * yy;
      grad[1] = -1.0;
    }
    return yy * y - x[1];
  };

  // form the problem
  nlopt::opt11 opt(nlopt::LD_MMA, 2);
  // - objective function
  opt.set_min_objective(f);
  // - bounded constraints
  opt.set_lower_bounds({-std::numeric_limits<double>::infinity(), 0.});
  // - nonlinear constraints
  opt.add_inequality_constraint(
      [](const std::vector<double> &x, std::vector<double> &grad) { return g(x, grad, 2., 0.); },
      /* tolerance = */ 1e-8);
  opt.add_inequality_constraint(
      [](const std::vector<double> &x, std::vector<double> &grad) { return g(x, grad, -1., 1.); },
      1e-8);
  // - stopping criteria
  opt.set_xtol_rel(1e-4);

  // solve the problem
  std::vector<double> x = {1.234, 5.678};
  double minf;
  const nlopt::result result = opt.optimize(x, minf);

  // verify the solution
  EXPECT_EQ(result, nlopt::XTOL_REACHED);
  EXPECT_NEAR(x[0], 0.333334, 1e-4);
  EXPECT_NEAR(x[1], 0.296296, 1e-4);
}