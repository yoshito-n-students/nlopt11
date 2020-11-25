#include <cstdio>
#include <vector>

#include <nlopt11.hpp>

int main(int argc, char *argv[]) {
  // Form the problem (expecting a gradient-based algorithm).
  // It is also found in a book "Shikkari Manabu Suuri Saitekika" by Umetani
  nlopt::opt11 opt(/* algorithm = */ nlopt::LD_MMA, /* n_params = */ 2);
  opt.set_min_objective([](const std::vector<double> &x, std::vector<double> &grad) -> double {
    const double y0 = x[0] - 1., y1 = x[1] - 2.5;
    // gradients
    if (!grad.empty()) {
      grad[0] = 2. * y0;
      grad[1] = 2. * y1;
    }
    // objective func
    return y0 * y0 + y1 * y1;
  });
  opt.add_inequality_mconstraint(
      [](std::vector<double> &result, const std::vector<double> &x,
         std::vector<std::vector<double>> &grad) -> void {
        // gradients
        if (!grad.empty()) {
          // grad[i][j] = d(constraint[i])/d(x[j])
          grad[0][0] = -1.;
          grad[0][1] = 2.;
          grad[1][0] = 1.;
          grad[1][1] = 2.;
          grad[2][0] = 1.;
          grad[2][1] = -2.;
          grad[3][0] = -1.;
          grad[3][1] = 0.;
          grad[4][0] = 0.;
          grad[4][1] = -1.;
        }
        // constraints
        result[0] = -x[0] + 2. * x[1] - 2.;
        result[1] = x[0] + 2. * x[1] - 6.;
        result[2] = x[0] - 2. * x[1] - 2.;
        result[3] = -x[0];
        result[4] = -x[1];
      },
      /* tolerance = */ std::vector<double>(5, 1e-8));
  opt.set_xtol_rel(1e-5);

  // Solve it!
  try {
    std::vector<double> x = {2., 0.};
    double minf;
    nlopt::result result = opt.optimize(x, minf);
    std::printf("found minimum at f(%f, %f) = %f\n", x[0], x[1], minf);
  } catch (std::exception &e) {
    std::printf("nlopt failed: %s\n", e.what());
  }

  return 0;
}