#include <cmath>
#include <cstdio>
#include <vector>

#include <nlopt11.hpp>

int main(int argc, char *argv[]) {
  // Form the problem (expecting a derivative-free algorithm).
  // It is also found in Sec. "Nonlinear Programming" of Octave's online document
  // https://octave.org/doc/v4.2.1/Nonlinear-Programming.html#Nonlinear-Programming
  nlopt::opt11 opt(/* algorithm = */ nlopt::LN_COBYLA, /* n_params = */ 5);
  opt.set_max_objective([](const std::vector<double> &x) -> double {
    // objective func
    const double y0 = x[0] * x[1] * x[2] * x[3] * x[4],
                 y1 = x[0] * x[0] * x[0] + x[1] * x[1] * x[1] + 1.;
    return -std::exp(y0) + 0.5 * y1 * y1;
  });
  opt.add_equality_mconstraint(
      [](std::vector<double> &result, const std::vector<double> &x) -> void {
        const double y0 = x[0] * x[0], y1 = x[1] * x[1];
        result[0] = y0 + y1 + x[2] * x[2] + x[3] * x[3] + x[4] * x[4] - 10.;
        result[1] = y0 * x[0] + y1 * x[1] + 1.;
      },
      /* tolerance = */ {1e-8, 1e-8});
  opt.add_equality_constraint(
      [](const std::vector<double> &x) -> double { return x[1] * x[2] - 5. * x[3] * x[4]; }, 1e-8);
  opt.set_xtol_rel(1e-5);

  // Solve it!
  try {
    std::vector<double> x = {-1.8, 1.7, 1.9, -0.8, -0.8};
    double minf;
    nlopt::result result = opt.optimize(x, minf);
    std::printf("found minimum at f(%f, %f, %f, %f, %f) = %f\n", x[0], x[1], x[2], x[3], x[4],
                minf);
  } catch (const std::exception &e) {
    std::printf("nlopt failed: %s\n", e.what());
  }

  return 0;
}