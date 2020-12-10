#ifndef NLOPT11X_HPP
#define NLOPT11X_HPP

#include <algorithm>
#include <array>
#include <functional>
#include <stdexcept>
#include <vector>

#include <nlopt.hpp>

namespace nlopt {

///////////////////////////////////////////////////////////////
// An enhanced version of nlplot::opt in flavor of c++11.
// nlopt::opt11x requires dimension known at compilation time.
// This may constrain user codes but also eliminates
// copy between c array and std::vector.

template <std::size_t N> class opt11x : public opt {
public:
  /////////////////////////////////////////////////////////////////////////////////////////
  // Function types
  // (namely '[m]{n,d}func_type'. m: vector-valued, n: derivative-free, d: gradient-based)

  typedef std::function<double(const std::array<double, N> & /* x */)> nfunc_type;
  typedef std::function<double(const std::array<double, N> & /* x */,
                               std::array<double, N> * /* grad */)>
      dfunc_type;
  template <std::size_t M>
  using mnfunc_type = std::function<void(std::array<double, M> & /* result */,
                                         const std::array<double, N> & /* x */)>;
  template <std::size_t M>
  using mdfunc_type = std::function<void(std::array<double, M> & /* result */,
                                         const std::array<double, N> & /* x */,
                                         std::array<std::array<double, N>, M> * /* grad */)>;

public:
  ////////////////
  // Constructors

  opt11x(const algorithm a) : opt::opt(a, N) {}

  opt11x(const opt &f) : opt::opt(f) {
    if (opt::get_dimension() != N) {
      throw std::invalid_argument("Dimension mismatch");
    }
  }

  opt11x &operator=(const opt &f) {
    if (f.get_dimension() != N) {
      throw std::invalid_argument("Dimension mismatch");
    }
    opt::operator=(f);
    return *this;
  }

  //////////////////////
  // Objective function

  using opt::set_min_objective;

  void set_min_objective(const nfunc_type &nf) {
    opt::set_min_objective(dispatch_nfunc, new nfunc_type(nf), free_<nfunc_type>,
                           copy_<nfunc_type>);
  }

  void set_min_objective(const dfunc_type &df) {
    opt::set_min_objective(dispatch_dfunc, new dfunc_type(df), free_<dfunc_type>,
                           copy_<dfunc_type>);
  }

  using opt::set_max_objective;

  void set_max_objective(const nfunc_type &nf) {
    opt::set_max_objective(dispatch_nfunc, new nfunc_type(nf), free_<nfunc_type>,
                           copy_<nfunc_type>);
  }

  void set_max_objective(const dfunc_type &df) {
    opt::set_max_objective(dispatch_dfunc, new dfunc_type(df), free_<dfunc_type>,
                           copy_<dfunc_type>);
  }

  ////////////////////////////////////
  // Nonlinear inequality constraints

  using opt::add_inequality_constraint;

  void add_inequality_constraint(const nfunc_type &nf, const double tol = 0.) {
    opt::add_inequality_constraint(dispatch_nfunc, new nfunc_type(nf), free_<nfunc_type>,
                                   copy_<nfunc_type>, tol);
  }

  void add_inequality_constraint(const dfunc_type &df, const double tol = 0.) {
    opt::add_inequality_constraint(dispatch_dfunc, new dfunc_type(df), free_<dfunc_type>,
                                   copy_<dfunc_type>, tol);
  }

  using opt::add_inequality_mconstraint;

  template <std::size_t M>
  void add_inequality_mconstraint(const mnfunc_type<M> &mnf, const std::array<double, M> &tol) {
    opt::add_inequality_mconstraint(dispatch_mnfunc<M>, new mnfunc_type<M>(mnf),
                                    free_<mnfunc_type<M>>, copy_<mnfunc_type<M>>,
                                    std::vector<double>(tol.begin(), tol.end()));
  }

  template <std::size_t M>
  void add_inequality_mconstraint(const mdfunc_type<M> &mdf, const std::array<double, M> &tol) {
    opt::add_inequality_mconstraint(dispatch_mdfunc<M>, new mdfunc_type<M>(mdf),
                                    free_<mdfunc_type<M>>, copy_<mdfunc_type<M>>,
                                    std::vector<double>(tol.begin(), tol.end()));
  }

  //////////////////////////////////
  // Nonlinear equality constraints

  using opt::add_equality_constraint;

  void add_equality_constraint(const nfunc_type &nf, const double tol = 0.) {
    opt::add_equality_constraint(dispatch_nfunc, new nfunc_type(nf), free_<nfunc_type>,
                                 copy_<nfunc_type>, tol);
  }

  void add_equality_constraint(const dfunc_type &df, const double tol = 0.) {
    opt::add_equality_constraint(dispatch_dfunc, new dfunc_type(df), free_<dfunc_type>,
                                 copy_<dfunc_type>, tol);
  }

  using opt::add_equality_mconstraint;

  template <std::size_t M>
  void add_equality_mconstraint(const mnfunc_type<M> &mnf, const std::array<double, M> &tol) {
    opt::add_equality_mconstraint(dispatch_mnfunc<M>, new mnfunc_type<M>(mnf),
                                  free_<mnfunc_type<M>>, copy_<mnfunc_type<M>>,
                                  std::vector<double>(tol.begin(), tol.end()));
  }

  template <std::size_t M>
  void add_equality_mconstraint(const mdfunc_type<M> &mdf, const std::array<double, M> &tol) {
    opt::add_equality_mconstraint(dispatch_mdfunc<M>, new mdfunc_type<M>(mdf),
                                  free_<mdfunc_type<M>>, copy_<mdfunc_type<M>>,
                                  std::vector<double>(tol.begin(), tol.end()));
  }

  ////////////////
  // Optimization

  using opt::optimize;

  result optimize(std::array<double, N> &x, double &opt_f) {
    std::vector<double> vx(x.begin(), x.end());
    const result res = opt::optimize(vx, opt_f);
    std::copy(vx.begin(), vx.end(), x.begin());
    return res;
  }

protected:
  ///////////////////////////////////
  // C-style wrappers (internal use)

  static double dispatch_nfunc(unsigned int /* n */, const double *x, double *grad, void *nf) {
    if (grad) {
      throw std::invalid_argument("Non-null gradient for a derivative-free function");
    }
    return (*static_cast<nfunc_type *>(nf))(reinterpret_cast<const std::array<double, N> &>(*x));
  }

  static double dispatch_dfunc(unsigned int /* n */, const double *x, double *grad, void *df) {
    return (*static_cast<dfunc_type *>(df))(reinterpret_cast<const std::array<double, N> &>(*x),
                                            reinterpret_cast<std::array<double, N> *>(grad));
  }

  template <std::size_t M>
  static void dispatch_mnfunc(unsigned int /* m */, double *result, unsigned int /* n */,
                              const double *x, double *grad, void *mnf) {
    if (grad) {
      throw std::invalid_argument("Non-null gradient for a vector-valued derivative-free function");
    }
    (*static_cast<mnfunc_type<M> *>(mnf))(reinterpret_cast<std::array<double, M> &>(*result),
                                          reinterpret_cast<const std::array<double, N> &>(*x));
  }

  template <std::size_t M>
  static void dispatch_mdfunc(unsigned int /* m */, double *result, unsigned int /* n */,
                              const double *x, double *grad, void *mdf) {
    (*static_cast<mdfunc_type<M> *>(mdf))(
        reinterpret_cast<std::array<double, M> &>(*result),
        reinterpret_cast<const std::array<double, N> &>(*x),
        reinterpret_cast<std::array<std::array<double, N>, M> *>(grad));
  }

  template <class Func> static void *free_(void *func) {
    if (func) {
      delete static_cast<Func *>(func);
    }
    return NULL;
  }

  template <class Func> static void *copy_(void *func) {
    if (func) {
      return new Func(*static_cast<Func *>(func));
    } else {
      return NULL;
    }
  }
};

} // namespace nlopt

#endif