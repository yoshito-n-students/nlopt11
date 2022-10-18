#include <cmath>
#include <vector>

#include <nlopt11.hpp>

#include <gtest/gtest.h>

// https://www.mathworks.com/help/optim/ug/solve-constrained-nonlinear-optimization-problem-based.html
TEST(FromMatlab, QP2) {
  // form the problem
  nlopt::opt11 opt(/* algorithm = */ nlopt::LN_COBYLA, /* n_params = */ 2);
  opt.set_min_objective([](const std::vector<double> &x) {
    return std::exp(x[0]) *
           (4. * x[0] * x[0] + 2. * x[1] * x[1] + 4. * x[0] * x[1] + 2. * x[1] - 1.);
  });
  opt.add_inequality_constraint(
      [](const std::vector<double> &x) {
        const double y0 = x[0] + 2., y1 = x[1] - 2.;
        return x[0] * x[1] / 2. + y0 * y0 + y1 * y1 / 2. - 2.;
      },
      /* tolerance = */ 1e-8);
  opt.set_lower_bounds({-5.5, -0.5});
  opt.set_upper_bounds({0., 7.});
  opt.set_xtol_rel(1e-4);

  // solve the problem
  const std::vector<double> x = opt.optimize(/* initial_x = */ {-1., 1.});

  // verify the answer (the problem has 2 local minimums)
  if (x[0] < -3.) {
    EXPECT_NEAR(x[0], -5.2813, 1e-3);
    EXPECT_NEAR(x[1], 4.6815, 1e-3);
  } else {
    EXPECT_NEAR(x[0], -0.8210, 1e-3);
    EXPECT_NEAR(x[1], 0.6696, 1e-3);
  }
}

// https://www.mathworks.com/help/optim/ug/quadratic-programming-bound-constraints-problem-based.html
TEST(FromMatlab, QP400) {
  // form the problem
  nlopt::opt11 opt(/* algorithm = */ nlopt::LD_VAR2, /* n_params = */ 400);
  // - objective
  opt.set_min_objective([](const std::vector<double> &x, std::vector<double> &grad) {
    // gradients
    if (!grad.empty()) {
      grad[0] = 2. * (2. * x[0] - x[1] - 1.);
      for (std::size_t i = 1; i < 399; ++i) {
        grad[i] = 2. * (2. * x[i] - x[i - 1] - x[i + 1]);
      }
      grad[399] = 2. * (2. * x[399] - x[398] - 1.);
    }
    // objective func
    double sum0 = 0.;
    for (const double &xi : x) {
      sum0 += xi * xi;
    }
    double sum1 = 0.;
    for (std::size_t i = 0; i < 399; ++i) {
      sum1 += x[i] * x[i + 1];
    }
    return 2. * (sum0 - sum1 - x[0] - x[399]);
  });
  // - bounds
  std::vector<double> lb(400, 0.);
  lb[399] = -std::numeric_limits<double>::infinity();
  opt.set_lower_bounds(lb);
  std::vector<double> ub(400, 0.9);
  ub[399] = std::numeric_limits<double>::infinity();
  opt.set_upper_bounds(ub);
  // - stopping criteria
  opt.set_xtol_rel(1e-5);

  // solve the problem
  std::vector<double> x(400, 0.5);
  double minf;
  const nlopt::result result = opt.optimize(x, minf);

  // verify the solution
  EXPECT_EQ(result, nlopt::XTOL_REACHED);
  EXPECT_LE(minf, -1.9849);
}

// https://www.mathworks.com/help/optim/ug/solve-nonlinear-optimization-problem-based.html
TEST(FromMatlab, Rosenbrock) {
  // form the problem
  nlopt::opt11 opt(/* algorithm = */ nlopt::LN_COBYLA, /* n_params = */ 2);
  opt.set_min_objective([](const int n, const double *const x) {
    EXPECT_EQ(n, 2);
    const double y0 = x[1] - x[0] * x[0], y1 = 1. - x[0];
    return 100. * y0 * y0 + y1 * y1;
  });
  opt.add_inequality_constraint(
      [](const int n, const double *const x) {
        EXPECT_EQ(n, 2);
        return x[0] * x[0] + x[1] * x[1] - 1.;
      },
      /* tolerance = */ 1e-8);
  opt.set_xtol_rel(1e-4);

  // solve the problem
  const std::vector<double> x = opt.optimize({0., 0.});

  // verify the solution
  EXPECT_NEAR(x[0], 0.7864, 1e-3);
  EXPECT_NEAR(x[1], 0.6177, 1e-3);
}