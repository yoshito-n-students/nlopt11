#ifndef NLOPT11_HPP
#define NLOPT11_HPP

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <vector>

#include <nlopt.hpp>

namespace nlopt {

//////////////////////////////////////////////////////////
// An enhanced version of nlplot::opt in flavor of c++11.
// nlopt::opt11 supports dimension unknown until runtime.

class opt11 : public opt {
public:
  ////////////////////////////////////////////////////
  // Function types (namely '[m]{n,d}{p,v}func_type')
  //   - m: vector-valued
  //   - n: derivative-free, or d: gradient-based
  //   - p: pointer, or v: std::vector

  using npfunc_type = std::function<double(const int /* n */, const double *const /* x */)>;
  using nvfunc_type = std::function<double(const std::vector<double> & /* x */)>;
  using dpfunc_type =
      std::function<double(const int /*n*/, const double *const /* x */, double *const /* grad */)>;
  using dvfunc_type =
      std::function<double(const std::vector<double> & /* x */, std::vector<double> & /* grad */)>;
  using mnpfunc_type = std::function<void(const int /* m */, double *const /* result */,
                                          const int /* n */, const double *const /* x */)>;
  using mnvfunc_type =
      std::function<void(std::vector<double> & /* result */, const std::vector<double> & /* x */)>;
  using mdpfunc_type =
      std::function<void(const int /* m */, double *const /* result */, const int /* n */,
                         const double *const /* x */, double *const /* grad */)>;
  using mdvfunc_type =
      std::function<void(std::vector<double> & /* result */, const std::vector<double> & /* x */,
                         std::vector<std::vector<double>> & /* grad */)>;

public:
  /////////////////////////////////////////////////////
  // Constructors (basically inherits from the parent)

  using opt::opt;

  opt11 &operator=(const opt &f) {
    opt::operator=(f);
    return *this;
  }

  //////////////////////
  // Objective function

#define NLOPT11_DEFINE_FUNC_OBJ(name, prefix)                                                      \
  void name(const prefix##unc_type &prefix) {                                                      \
    opt::name(dispatch_##prefix##unc, new prefix##unc_type(prefix), free_<prefix##unc_type>,       \
              copy_<prefix##unc_type>);                                                            \
  }

  using opt::set_min_objective;
  NLOPT11_DEFINE_FUNC_OBJ(set_min_objective, npf)
  NLOPT11_DEFINE_FUNC_OBJ(set_min_objective, nvf)
  NLOPT11_DEFINE_FUNC_OBJ(set_min_objective, dpf)
  NLOPT11_DEFINE_FUNC_OBJ(set_min_objective, dvf)

  using opt::set_max_objective;
  NLOPT11_DEFINE_FUNC_OBJ(set_max_objective, npf)
  NLOPT11_DEFINE_FUNC_OBJ(set_max_objective, nvf)
  NLOPT11_DEFINE_FUNC_OBJ(set_max_objective, dpf)
  NLOPT11_DEFINE_FUNC_OBJ(set_max_objective, dvf)

  ////////////////////////////////////
  // Nonlinear inequality constraints

#define NLOPT11_DEFINE_FUNC_CONS(name, prefix)                                                     \
  void name(const prefix##unc_type &prefix, const double tol = 0.) {                               \
    opt::name(dispatch_##prefix##unc, new prefix##unc_type(prefix), free_<prefix##unc_type>,       \
              copy_<prefix##unc_type>, tol);                                                       \
  }

  using opt::add_inequality_constraint;
  NLOPT11_DEFINE_FUNC_CONS(add_inequality_constraint, npf)
  NLOPT11_DEFINE_FUNC_CONS(add_inequality_constraint, nvf)
  NLOPT11_DEFINE_FUNC_CONS(add_inequality_constraint, dpf)
  NLOPT11_DEFINE_FUNC_CONS(add_inequality_constraint, dvf)

#define NLOPT11_DEFINE_FUNC_MCONS(name, prefix)                                                    \
  void name(const prefix##unc_type &prefix, const std::vector<double> &tol) {                      \
    opt::name(dispatch_##prefix##unc, new prefix##unc_type(prefix), free_<prefix##unc_type>,       \
              copy_<prefix##unc_type>, tol);                                                       \
  }

  using opt::add_inequality_mconstraint;
  NLOPT11_DEFINE_FUNC_MCONS(add_inequality_mconstraint, mnpf)
  NLOPT11_DEFINE_FUNC_MCONS(add_inequality_mconstraint, mnvf)
  NLOPT11_DEFINE_FUNC_MCONS(add_inequality_mconstraint, mdpf)
  NLOPT11_DEFINE_FUNC_MCONS(add_inequality_mconstraint, mdvf)

  //////////////////////////////////
  // Nonlinear equality constraints

  using opt::add_equality_constraint;
  NLOPT11_DEFINE_FUNC_CONS(add_equality_constraint, npf)
  NLOPT11_DEFINE_FUNC_CONS(add_equality_constraint, nvf)
  NLOPT11_DEFINE_FUNC_CONS(add_equality_constraint, dpf)
  NLOPT11_DEFINE_FUNC_CONS(add_equality_constraint, dvf)

  using opt::add_equality_mconstraint;
  NLOPT11_DEFINE_FUNC_MCONS(add_equality_mconstraint, mnpf)
  NLOPT11_DEFINE_FUNC_MCONS(add_equality_mconstraint, mnvf)
  NLOPT11_DEFINE_FUNC_MCONS(add_equality_mconstraint, mdpf)
  NLOPT11_DEFINE_FUNC_MCONS(add_equality_mconstraint, mdvf)

protected:
  /////////////////////////
  // C <-> C++ conversions

  static std::vector<double> to_vector(const double *const arr, const unsigned int n) {
    return arr ? std::vector<double>(arr, arr + n) : std::vector<double>();
  }

  static void to_carray(const std::vector<double> &vec, double *const arr) {
    if (arr) {
      std::copy(vec.begin(), vec.end(), arr);
    }
  }

  static std::vector<std::vector<double>> to_vector2d(const double *arr, const unsigned int m,
                                                      const unsigned int n) {
    if (arr) {
      std::vector<std::vector<double>> vec2d(m);
      for (std::vector<double> &vec : vec2d) {
        vec = to_vector(arr, n);
        arr += n;
      }
      return vec2d;
    } else {
      return std::vector<std::vector<double>>();
    }
  }

  static void to_carray2d(const std::vector<std::vector<double>> &vec2d, double *arr) {
    if (arr) {
      for (const std::vector<double> &vec : vec2d) {
        to_carray(vec, arr);
        arr += vec.size();
      }
    }
  }

  ///////////////////////////////////
  // C-style wrappers (internal use)

  static double dispatch_npfunc(unsigned int n, const double *x, double *grad, void *npf) {
    if (grad) {
      throw std::invalid_argument("Non-null gradient for a derivative-free function");
    }
    // Dispatch
    return (*static_cast<npfunc_type *>(npf))(n, x);
  }

  static double dispatch_nvfunc(unsigned int n, const double *x_c, double *grad_c, void *nvf) {
    if (grad_c) {
      throw std::invalid_argument("Non-null gradient for a derivative-free function");
    }
    // c -> cpp
    const std::vector<double> x = to_vector(x_c, n);
    // Dispatch
    return (*static_cast<nvfunc_type *>(nvf))(x);
  }

  static double dispatch_dpfunc(unsigned int n, const double *x, double *grad, void *dpf) {
    // Dispatch
    return (*static_cast<dpfunc_type *>(dpf))(n, x, grad);
  }

  static double dispatch_dvfunc(unsigned int n, const double *x_c, double *grad_c, void *dvf) {
    // c -> cpp
    const std::vector<double> x = to_vector(x_c, n);
    std::vector<double> grad = to_vector(grad_c, n);
    // Dispatch
    const double result = (*static_cast<dvfunc_type *>(dvf))(x, grad);
    // cpp -> c
    to_carray(grad, grad_c);
    return result;
  }

  static void dispatch_mnpfunc(unsigned int m, double *result, unsigned int n, const double *x,
                               double *grad, void *mnpf) {
    if (grad) {
      throw std::invalid_argument("Non-null gradient for a vector-valued derivative-free function");
    }
    // Dispatch
    (*static_cast<mnpfunc_type *>(mnpf))(m, result, n, x);
  }

  static void dispatch_mnvfunc(unsigned int m, double *result_c, unsigned int n, const double *x_c,
                               double *grad_c, void *mnvf) {
    if (grad_c) {
      throw std::invalid_argument("Non-null gradient for a vector-valued derivative-free function");
    }
    // c -> cpp
    std::vector<double> result = to_vector(result_c, m);
    const std::vector<double> x = to_vector(x_c, n);
    // Dispatch
    (*static_cast<mnvfunc_type *>(mnvf))(result, x);
    // cpp -> c
    to_carray(result, result_c);
  }

  static void dispatch_mdpfunc(unsigned int m, double *result, unsigned int n, const double *x,
                               double *grad, void *mdpf) {
    // Dispatch
    (*static_cast<mdpfunc_type *>(mdpf))(m, result, n, x, grad);
  }

  static void dispatch_mdvfunc(unsigned int m, double *result_c, unsigned int n, const double *x_c,
                               double *grad_c, void *mdvf) {
    // c -> cpp
    std::vector<double> result = to_vector(result_c, m);
    const std::vector<double> x = to_vector(x_c, n);
    std::vector<std::vector<double>> grad = to_vector2d(grad_c, m, n);
    // Dispatch
    (*static_cast<mdvfunc_type *>(mdvf))(result, x, grad);
    // cpp -> c
    to_carray(result, result_c);
    to_carray2d(grad, grad_c);
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