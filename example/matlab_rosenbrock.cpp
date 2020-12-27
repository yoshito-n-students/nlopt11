#include <cstdio>
#include <vector>

#include <nlopt11.hpp>

int main(int argc, char *argv[]) {
  // Form the Rosenbrock problem (expecting a derivative-free algorithm).
  // It is also found in online document of Matlab's optimization toolbox
  // https://www.mathworks.com/help/optim/ug/solve-nonlinear-optimization-problem-based.html
  nlopt::opt11 opt(/* algorithm = */ nlopt::LN_COBYLA, /* n_params = */ 2);
  opt.set_min_objective([](const std::vector<double> &x) -> double {
    const double y0 = x[1] - x[0] * x[0], y1 = 1. - x[0];
    return 100. * y0 * y0 + y1 * y1;
  });
  opt.add_inequality_constraint(
      [](const std::vector<double> &x) -> double { return x[0] * x[0] + x[1] * x[1] - 1.; },
      /* tolerance = */ 1e-8);
  opt.set_xtol_rel(1e-4);

  // Solve it!
  try {
    const std::vector<double> x = opt.optimize({0.,0.});
    std::printf("found minimum at f(%f, %f) = %f\n", x[0], x[1], opt.last_optimum_value());
  } catch (const std::exception &e) {
    std::printf("nlopt failed: %s\n", e.what());
  }

  return 0;
}