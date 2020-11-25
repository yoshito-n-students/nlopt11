#include <cstdio>
#include <vector>

#include <nlopt11.hpp>

int main(int argc, char *argv[]) {
  // Form the problem (expecting a gradient-based algorithm).
  // It is also found in a book "Shikkari Manabu Suuri Saitekika" by Umetani
  nlopt::opt11 opt(/* algorithm = */ nlopt::LD_SLSQP, /* n_params = */ 2);
  opt.set_min_objective([](const std::vector<double> &x, std::vector<double> &grad) -> double {
    // gradients
    if (!grad.empty()) {
      grad[0] = 2. * x[0];
      grad[1] = -2. * x[1];
    }
    // objective func
    return x[0] * x[0] - x[1] * x[1];
  });
  opt.add_equality_constraint(
      [](const std::vector<double> &x, std::vector<double> &grad) -> double {
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

  // Solve it!
  try {
    std::vector<double> x(2, std::sqrt(0.2));
    double minf;
    nlopt::result result = opt.optimize(x, minf);
    std::printf("found minimum at f(%f, %f) = %f\n", x[0], x[1], minf);
  } catch (std::exception &e) {
    std::printf("nlopt failed: %s\n", e.what());
  }

  return 0;
}