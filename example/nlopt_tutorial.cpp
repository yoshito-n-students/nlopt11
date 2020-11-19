#include <cstdio>
#include <functional> // for std::bind()
#include <limits>
#include <vector>

#include <nlopt11.hpp>

// The objective function
double f(const std::vector<double> &x, std::vector<double> &grad) {
  if (!grad.empty()) {
    grad[0] = 0.0;
    grad[1] = 0.5 / sqrt(x[1]);
  }
  return sqrt(x[1]);
}

// A constraint with coefficients
double g(const std::vector<double> &x, std::vector<double> &grad, const double a, const double b) {
  const double y = a * x[0] + b;
  const double yy = y * y;
  if (!grad.empty()) {
    grad[0] = 3. * a * yy;
    grad[1] = -1.0;
  }
  return yy * y - x[1];
}

int main(int argc, char *argv[]) {
  namespace sp = std::placeholders;

  nlopt::opt11 opt(nlopt::LN_COBYLA, 2);
  // The objective function
  opt.set_min_objective(f);
  // Bounded constraints
  opt.set_lower_bounds({-std::numeric_limits<double>::infinity(), 0.});
  // Nonlinear constraints
  opt.add_inequality_constraint(std::bind(g, sp::_1, sp::_2, 2., 0.), 1e-8);
  opt.add_inequality_constraint(std::bind(g, sp::_1, sp::_2, -1., 1.), 1e-8);
  opt.set_xtol_rel(1e-4);

  try {
    std::vector<double> x = {1.234, 5.678};
    double minf;
    nlopt::result result = opt.optimize(x, minf);
    std::printf("found minimum at f(%f, %f) = %f\n", x[0], x[1], minf);
  } catch (const std::exception &e) {
    std::printf("nlopt failed: %s\n", e.what());
  }

  return 0;
}