#include <cstdio>
#include <limits>
#include <vector>

#include <nlopt11.hpp>

int main(int argc, char *argv[]) {
  // number of optimized params
  const unsigned int n = 400;

  // Form the problem (expecting a gradient-based algorithm).
  // It is also found in online document of Matlab's optimization toolbox
  // https://www.mathworks.com/help/optim/ug/quadratic-programming-bound-constraints-problem-based.html
  nlopt::opt11 opt(/* algorithm = */ nlopt::LD_VAR2, n);
  opt.set_min_objective([n](const std::vector<double> &x, std::vector<double> &grad) -> double {
    // gradients
    if (grad.size() == n) {
      grad[0] = 2. * (2. * x[0] - x[1] - 1.);
      for (std::size_t i = 1; i < n - 1; ++i) {
        grad[i] = 2. * (2. * x[i] - x[i - 1] - x[i + 1]);
      }
      grad[n - 1] = 2. * (2. * x[n - 1] - x[n - 2] - 1.);
    }
    // objective func
    double sum0 = 0.;
    for (const double &xi : x) {
      sum0 += xi * xi;
    }
    double sum1 = 0.;
    for (std::size_t i = 0; i < n - 1; ++i) {
      sum1 += x[i] * x[i + 1];
    }
    return 2. * (sum0 - sum1 - x[0] - x[n - 1]);
  });

  std::vector<double> lb(n, 0.);
  lb[n - 1] = -std::numeric_limits<double>::infinity();
  opt.set_lower_bounds(lb);

  std::vector<double> ub(n, 0.9);
  ub[n - 1] = std::numeric_limits<double>::infinity();
  opt.set_upper_bounds(ub);

  opt.set_xtol_rel(1e-5);

  // Solve it!
  try {
    std::vector<double> x(n, 0.5);
    double minf;
    const nlopt::result result = opt.optimize(x, minf);
    std::printf("found minimum at f = %f\n", minf);
  } catch (const std::exception &e) {
    std::printf("nlopt failed: %s\n", e.what());
  }

  return 0;
}