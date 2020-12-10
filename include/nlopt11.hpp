#ifndef NLOPT11_HPP
#define NLOPT11_HPP

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <vector>

#include <nlopt.hpp>

namespace nlopt {

/////////////////////////////////////////////////////////
// An enhanced version of nlplot::opt in flavor of c++11

class opt11 : public opt {
public:
  /////////////////////////////////////////////////////////////////////////////////////////
  // Function types
  // (namely '[m]{n,d}func_type'. m: vector-valued, n: derivative-free, d: gradient-based)

  typedef std::function<double(const std::vector<double> & /* x */)> nfunc_type;
  typedef std::function<double(const std::vector<double> & /* x */,
                               std::vector<double> & /* grad */)>
      dfunc_type;
  typedef std::function<void(std::vector<double> & /* result */,
                             const std::vector<double> & /* x */)>
      mnfunc_type;
  typedef std::function<void(std::vector<double> & /* result */,
                             const std::vector<double> & /* x */,
                             std::vector<std::vector<double>> & /* grad */)>
      mdfunc_type;

public:
  ///////////////////////////////////////////
  // Constructors (inherits from the parent)

  using opt::opt;

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

  void add_inequality_mconstraint(const mnfunc_type &mnf, const std::vector<double> &tol) {
    opt::add_inequality_mconstraint(dispatch_mnfunc, new mnfunc_type(mnf), free_<mnfunc_type>,
                                    copy_<mnfunc_type>, tol);
  }

  void add_inequality_mconstraint(const mdfunc_type &mdf, const std::vector<double> &tol) {
    opt::add_inequality_mconstraint(dispatch_mdfunc, new mdfunc_type(mdf), free_<mdfunc_type>,
                                    copy_<mdfunc_type>, tol);
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

  void add_equality_mconstraint(const mnfunc_type &mnf, const std::vector<double> &tol) {
    opt::add_equality_mconstraint(dispatch_mnfunc, new mnfunc_type(mnf), free_<mnfunc_type>,
                                  copy_<mnfunc_type>, tol);
  }

  void add_equality_mconstraint(const mdfunc_type &mdf, const std::vector<double> &tol) {
    opt::add_equality_mconstraint(dispatch_mdfunc, new mdfunc_type(mdf), free_<mdfunc_type>,
                                  copy_<mdfunc_type>, tol);
  }

protected:
  /////////////////////////
  // C <-> C++ conversions

  static std::vector<double> carray2vector(const double *const arr, const unsigned int n) {
    return arr ? std::vector<double>(arr, arr + n) : std::vector<double>();
  }

  static void vector2carray(const std::vector<double> &vec, double *const arr) {
    if (arr) {
      std::copy(vec.begin(), vec.end(), arr);
    }
  }

  static std::vector<std::vector<double>> carray2vector2d(const double *arr, const unsigned int m,
                                                          const unsigned int n) {
    if (arr) {
      std::vector<std::vector<double>> vec2d(m);
      for (std::vector<double> &vec : vec2d) {
        vec = carray2vector(arr, n);
        arr += n;
      }
      return vec2d;
    } else {
      return std::vector<std::vector<double>>();
    }
  }

  static void vector2d2carray(const std::vector<std::vector<double>> &vec2d, double *arr) {
    if (arr) {
      for (const std::vector<double> &vec : vec2d) {
        vector2carray(vec, arr);
        arr += vec.size();
      }
    }
  }

  ///////////////////////////////////
  // C-style wrappers (internal use)

  static double dispatch_nfunc(unsigned int n, const double *x_c, double *grad_c, void *nf) {
    if (grad_c) {
      throw std::invalid_argument("Non-null gradient for a derivative-free function");
    }
    // c -> cpp
    const std::vector<double> x = carray2vector(x_c, n);
    // Dispatch
    return (*static_cast<nfunc_type *>(nf))(x);
  }

  static double dispatch_dfunc(unsigned int n, const double *x_c, double *grad_c, void *df) {
    // c -> cpp
    const std::vector<double> x = carray2vector(x_c, n);
    std::vector<double> grad = carray2vector(grad_c, n);
    // Dispatch
    const double result = (*static_cast<dfunc_type *>(df))(x, grad);
    // cpp -> c
    vector2carray(grad, grad_c);
    return result;
  }

  static void dispatch_mnfunc(unsigned int m, double *result_c, unsigned int n, const double *x_c,
                              double *grad_c, void *mnf) {
    if (grad_c) {
      throw std::invalid_argument("Non-null gradient for a vector-valued derivative-free function");
    }
    // c -> cpp
    std::vector<double> result = carray2vector(result_c, m);
    const std::vector<double> x = carray2vector(x_c, n);
    // Dispatch
    (*static_cast<mnfunc_type *>(mnf))(result, x);
    // cpp -> c
    vector2carray(result, result_c);
  }

  static void dispatch_mdfunc(unsigned int m, double *result_c, unsigned int n, const double *x_c,
                              double *grad_c, void *mdf) {
    // c -> cpp
    std::vector<double> result = carray2vector(result_c, m);
    const std::vector<double> x = carray2vector(x_c, n);
    std::vector<std::vector<double>> grad = carray2vector2d(grad_c, m, n);
    // Dispatch
    (*static_cast<mdfunc_type *>(mdf))(result, x, grad);
    // cpp -> c
    vector2carray(result, result_c);
    vector2d2carray(grad, grad_c);
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