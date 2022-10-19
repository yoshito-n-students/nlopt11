#include <nlopt11.hpp>

#include <gtest/gtest.h>

// All problems are from the book "Shikkari Manabu Suuri Saitekika" by Umetani

TEST(FromUmetaniBook, 3_9a) {
  // form the problem
  nlopt::opt11 opt(/* algorithm = */ nlopt::LD_SLSQP, /* n_params = */ 2);
  opt.set_min_objective([](const std::vector<double> &x, std::vector<double> &grad) {
    // gradients
    if (!grad.empty()) {
      grad[0] = 2. * x[0];
      grad[1] = -2. * x[1];
    }
    // objective func
    return x[0] * x[0] - x[1] * x[1];
  });
  opt.add_equality_constraint(
      [](const std::vector<double> &x, std::vector<double> &grad) {
        // gradients
        if (!grad.empty()) {
          // grad[i] = d(constraint)/d(x[i])
          grad[0] = 2. * x[0];
          grad[1] = 8. * x[1];
        }
        // constraint
        return x[0] * x[0] + 4. * x[1] * x[1] - 1.;
      },
      /* tolerance = */ 1e-8);
  opt.set_xtol_rel(1e-4);

  // solve the problem
  std::vector<double> x(2, std::sqrt(0.2));
  double minf;
  const nlopt::result result = opt.optimize(x, minf);

  // TODO: verify the solution
  EXPECT_EQ(result, nlopt::XTOL_REACHED);
}

TEST(FromUmetaniBook, 3_9b) {
  // form the problem
  nlopt::opt11 opt(/* algorithm = */ nlopt::LD_SLSQP, /* n_params = */ 2);
  opt.set_min_objective([](const int n, const double *const x, double *const grad) {
    EXPECT_EQ(n, 2);
    // gradients
    if (grad) {
      grad[0] = 8. * x[0] - 4. * x[1];
      grad[1] = -4. * x[0] + 6. * x[1] - 8.;
    }
    // objective func
    return 4. * x[0] * x[0] - 4. * x[0] * x[1] + 3. * x[1] * x[1] - 8. * x[1];
  });
  opt.add_inequality_constraint(
      [](const int n, const double *const x, double *const grad) {
        EXPECT_EQ(n, 2);
        // gradients
        if (grad) {
          // grad[i] = d(constraint)/d(x[i])
          grad[0] = 1.;
          grad[1] = 1.;
        }
        // constraint
        return x[0] + x[1] - 4.;
      },
      /* tolerance = */ 1e-8);
  opt.set_xtol_rel(1e-4);

  // solve the problem
  std::vector<double> x = {2., 2.};
  double minf;
  const nlopt::result result = opt.optimize(x, minf);

  // TODO: verify the solution
  EXPECT_EQ(result, nlopt::XTOL_REACHED);
}

TEST(FromUmetaniBook, QP2) {
  // form the problem
  nlopt::opt11 opt(/* algorithm = */ nlopt::LD_MMA, /* n_prams = */ 2);
  opt.set_min_objective([](const int n, const double *const x, double *const grad) {
    EXPECT_EQ(n, 2);
    const double y0 = x[0] - 1., y1 = x[1] - 2.5;
    // gradients
    if (grad) {
      grad[0] = 2. * y0;
      grad[1] = 2. * y1;
    }
    // objective func
    return y0 * y0 + y1 * y1;
  });
  opt.add_inequality_mconstraint(
      [](const int m, double *const result, const int n, const double *const x,
         double *const grad) {
        EXPECT_EQ(m, 5);
        EXPECT_EQ(n, 2);
        // gradients
        if (grad) {
          // grad[i * n + j] = d(constraint[i])/d(x[j])
          grad[0 * n + 0] = -1.;
          grad[0 * n + 1] = 2.;
          grad[1 * n + 0] = 1.;
          grad[1 * n + 1] = 2.;
          grad[2 * n + 0] = 1.;
          grad[2 * n + 1] = -2.;
          grad[3 * n + 0] = -1.;
          grad[3 * n + 1] = 0.;
          grad[4 * n + 0] = 0.;
          grad[4 * n + 1] = -1.;
        }
        // constraints
        result[0] = -x[0] + 2. * x[1] - 2.;
        result[1] = x[0] + 2. * x[1] - 6.;
        result[2] = x[0] - 2. * x[1] - 2.;
        result[3] = -x[0];
        result[4] = -x[1];
      },
      /* tolerance = */ {1e-8, 1e-8, 1e-8, 1e-8, 1e-8});
  opt.set_xtol_rel(1e-5);

  // solve the problem
  const std::vector<double> x = opt.optimize({2., 0.});

  // TODO: verify the solution
}