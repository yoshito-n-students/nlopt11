#include <array>
#include <cmath>
#include <cstdio>

#include <nlopt11x.hpp>

nlopt::opt11x<2> form(const nlopt::algorithm a) {
  // Form the problem (expecting a derivative-free algorithm).
  // It is also found in online document of Matlab's optimization toolbox
  // https://www.mathworks.com/help/optim/ug/solve-constrained-nonlinear-optimization-problem-based.html
  nlopt::opt11x</* n_params = */ 2> opt(/* algorithm = */ a);
  opt.set_min_objective([](const std::array<double, 2> &x) -> double {
    return std::exp(x[0]) *
           (4. * x[0] * x[0] + 2. * x[1] * x[1] + 4. * x[0] * x[1] + 2. * x[1] - 1.);
  });
  opt.add_inequality_constraint(
      [](const std::array<double, 2> &x) -> double {
        const double y0 = x[0] + 2., y1 = x[1] - 2.;
        return x[0] * x[1] / 2. + y0 * y0 + y1 * y1 / 2. - 2.;
      },
      /* tolerance = */ 1e-8);
  opt.set_lower_bounds({-5.5, -0.5});
  opt.set_upper_bounds({0., 7.});
  opt.set_xtol_rel(1e-4);
  return opt;
}

void solve(nlopt::opt11x<2> &opt, const std::array<double, 2> &initial_x) {
  std::array<double, 2> x = initial_x;
  double minf;
  nlopt::result result = opt.optimize(x, minf);
  std::printf("  found minimum at f(%f, %f) = %f, started from (%f, %f)\n", x[0], x[1], minf,
              initial_x[0], initial_x[1]);
}

int main(int argc, char *argv[]) {
  nlopt::opt11x<2> opt_l = form(nlopt::LN_COBYLA), opt_g = form(nlopt::GN_ISRES);

  std::printf("by a LOCAL algorithm:\n");
  solve(opt_l, {-1., 1.});
  solve(opt_l, {-3., 3.});
  std::printf("by a GLOBAL algorithm:\n");
  solve(opt_g, {-1., 1.});
  solve(opt_g, {-3., 3.});

  return 0;
}