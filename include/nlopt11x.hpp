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

  using nfunc_type = std::function<double(const std::array<double, N> & /* x */)>;
  using dfunc_type = std::function<double(const std::array<double, N> & /* x */,
                                          std::array<double, N> * /* grad */)>;
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

  // This method cannot be declared like "add_xxx(const mnfunc_type<M> & mnf, ...)"
  // because M in mnfunc_type<M> is not deduced in most cases
  // (ex. when mnf is given as a function pointer, lambda expression, etc.).
  // So this method first deduces M only with the argument tol
  // and then deduces the type of the argument mf by calling an overloaded impl method.
  template <class F, std::size_t M>
  void add_inequality_mconstraint(const F &mf, const std::array<double, M> &tol) {
    add_inequality_mconstraint_impl<M>(mf, to_vector(tol));
  }

  // This method matches a call of "o.add_xxx(xxx, {0., 0.})"
  // and successfully deduces the template param M.
  // If this method does not exist, the call matches the above std::array version
  // but does not deduce M because initialization by "{0., 0.}" is valid for any M >= 2.
  template <class F, std::size_t M>
  void add_inequality_mconstraint(const F &mf, const double (&tol)[M]) {
    add_inequality_mconstraint_impl<M>(mf, to_vector(tol));
  }

protected:
  template <std::size_t M>
  void add_inequality_mconstraint_impl(const mnfunc_type<M> &mnf, const std::vector<double> &tol) {
    opt::add_inequality_mconstraint(dispatch_mnfunc<M>, new mnfunc_type<M>(mnf),
                                    free_<mnfunc_type<M>>, copy_<mnfunc_type<M>>, tol);
  }

  template <std::size_t M>
  void add_inequality_mconstraint_impl(const mdfunc_type<M> &mdf, const std::vector<double> &tol) {
    opt::add_inequality_mconstraint(dispatch_mdfunc<M>, new mdfunc_type<M>(mdf),
                                    free_<mdfunc_type<M>>, copy_<mdfunc_type<M>>, tol);
  }

public:
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

  template <class F, std::size_t M>
  void add_equality_mconstraint(const F &mf, const double (&tol)[M]) {
    add_equality_mconstraint_impl<M>(mf, to_vector(tol));
  }

  template <class F, std::size_t M>
  void add_equality_mconstraint(const F &mf, const std::array<double, M> &tol) {
    add_equality_mconstraint_impl<M>(mf, to_vector(tol));
  }

protected:
  template <std::size_t M>
  void add_equality_mconstraint_impl(const mnfunc_type<M> &mnf, const std::vector<double> &tol) {
    opt::add_equality_mconstraint(dispatch_mnfunc<M>, new mnfunc_type<M>(mnf),
                                  free_<mnfunc_type<M>>, copy_<mnfunc_type<M>>, tol);
  }

  template <std::size_t M>
  void add_equality_mconstraint_impl(const mdfunc_type<M> &mdf, const std::vector<double> &tol) {
    opt::add_equality_mconstraint(dispatch_mdfunc<M>, new mdfunc_type<M>(mdf),
                                  free_<mdfunc_type<M>>, copy_<mdfunc_type<M>>, tol);
  }

public:
  /////////////////////
  // Bound constraints

#define NLOPT11X_GETSET_ARRAY(name)                                                                \
  using opt::get_##name;                                                                           \
                                                                                                   \
  void get_##name(std::array<double, N> &val) const { to_array(opt::get_##name(), val); }          \
                                                                                                   \
  std::array<double, N> get_##name() const { return to_array<N>(opt::get_##name()); }              \
                                                                                                   \
  using opt::set_##name;                                                                           \
                                                                                                   \
  void set_##name(const std::array<double, N> &val) { opt::set_##name(to_vector(val)); }           \
                                                                                                   \
  /* This method matches a call of "o.set_xxx({0., 0.})".                    */                    \
  /* If this method does not exist, the call does not compile                */                    \
  /* because matching both std::array and std::vector versions of set_xxx(). */                    \
  void set_##name(const double(&val)[N]) { opt::set_##name(to_vector(val)); }

  NLOPT11X_GETSET_ARRAY(lower_bounds);
  NLOPT11X_GETSET_ARRAY(upper_bounds);

  /////////////////////
  // Stopping criteria

  NLOPT11X_GETSET_ARRAY(xtol_abs);

  /////////////////////////////////
  // Algorithm-specific parameters

  NLOPT11X_GETSET_ARRAY(initial_step);

#undef NLOPT11X_GETSET_ARRAY

  void get_initial_step(const std::array<double, N> &x, std::array<double, N> &dx) const {
    std::vector<double> vdx = to_vector(dx);
    opt::get_initial_step(to_vector(x), vdx);
    to_array(vdx, dx);
  }

  using opt::get_initial_step_;

  std::array<double, N> get_initial_step_(const std::array<double, N> &x) const {
    return to_array<N>(opt::get_initial_step_(to_vector(x)));
  }

  using opt::set_default_initial_step;

  void set_default_initial_step(const std::array<double, N> &x) {
    opt::set_default_initial_step(to_vector(x));
  }

  ////////////////
  // Optimization

  using opt::optimize;

  result optimize(std::array<double, N> &x, double &opt_f) {
    std::vector<double> vx = to_vector(x);
    const result res = opt::optimize(vx, opt_f);
    to_array(vx, x);
    return res;
  }

  std::array<double, N> optimize(const std::array<double, N> &x) {
    return to_array<N>(opt::optimize(to_vector(x)));
  }

  std::array<double, N> optimize(const double (&x)[N]) {
    return to_array<N>(opt::optimize(to_vector(x)));
  }

protected:
  //////////////////////////////
  // std::array <-> std::vector

  template <std::size_t M>
  static void to_array(const std::vector<double> &v, std::array<double, M> &a) {
    if (v.size() != a.size()) {
      throw std::invalid_argument("Vector size mismatch");
    }
    std::copy(v.begin(), v.end(), a.begin());
  }

  template <std::size_t M> static std::array<double, M> to_array(const std::vector<double> &v) {
    std::array<double, M> a;
    to_array(v, a);
    return a;
  }

  template <std::size_t M> static std::vector<double> to_vector(const std::array<double, M> &a) {
    return std::vector<double>(a.begin(), a.end());
  }

  template <std::size_t M> static std::vector<double> to_vector(const double (&a)[M]) {
    return std::vector<double>(a, a + M);
  }

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