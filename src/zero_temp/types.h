#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <array>
#include <complex>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace meanfi::zero_temp_native {

constexpr double kPi = 3.141592653589793238462643383279502884;

using Float1D = nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig>;
using Float2D = nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>;
using Float3D = nb::ndarray<nb::numpy, const double, nb::ndim<3>, nb::c_contig>;
using Int1D = nb::ndarray<nb::numpy, const std::int64_t, nb::ndim<1>, nb::c_contig>;
using Int2D = nb::ndarray<nb::numpy, const std::int64_t, nb::ndim<2>, nb::c_contig>;
using Complex2D =
    nb::ndarray<nb::numpy, const std::complex<double>, nb::ndim<2>, nb::c_contig>;
using Complex3D =
    nb::ndarray<nb::numpy, const std::complex<double>, nb::ndim<3>, nb::c_contig>;

template <typename T>
nb::ndarray<nb::numpy, T> make_array(
    std::vector<T> &&data,
    std::initializer_list<size_t> shape
) {
    T *raw = new T[data.size()];
    std::move(data.begin(), data.end(), raw);
    nb::capsule owner(raw, [](void *p) noexcept { delete[] static_cast<T *>(p); });
    return nb::ndarray<nb::numpy, T>(raw, shape, owner);
}

struct ChargeSolveOptions {
    double mu_guess = 0.0;
    double charge_tol = 1e-6;
    double mu_xtol = 1e-6;
    std::int64_t max_mu_iterations = 64;
    std::int64_t max_subdivisions = -1;
    double bulk_theta = 0.5;
};

struct ChargeSolveResult {
    double mu = 0.0;
    double charge = 0.0;
    double charge_error = 0.0;
    double dcharge_dmu = 0.0;
    std::int64_t root_iterations = 0;
    std::int64_t charge_integration_calls = 0;
    std::int64_t evaluator_evals = 0;
    std::int64_t subdivisions = 0;
    std::int64_t n_leaves = 0;
    std::int64_t n_leaf_nodes = 0;
    bool converged = false;
    bool error_estimate_available = true;
};

struct DensityIntegrateOptions {
    double density_atol = 1e-6;
    double density_rtol = 0.0;
    std::int64_t max_subdivisions = -1;
    double bulk_theta = 0.5;
};

struct DensityIntegrateResult {
    std::vector<std::complex<double>> estimate;
    std::vector<double> error_vector;
    double error_scalar = 0.0;
    std::int64_t evaluator_evals = 0;
    std::int64_t subdivisions = 0;
    std::int64_t n_leaves = 0;
    std::int64_t n_leaf_nodes = 0;
    bool converged = false;
    bool error_estimate_available = true;
};

struct RefinementBatch {
    int refinements = 0;
    std::vector<std::int64_t> parent_ids;
    std::vector<std::int64_t> child_offsets{0};
    std::vector<std::int64_t> child_ids;
    std::vector<std::int64_t> parent_vertex_ids;
    std::vector<std::int64_t> child_vertex_ids;
    std::vector<std::int64_t> midpoint_ids;
    std::vector<std::int64_t> bisected_edges;

    nb::tuple as_tuple(size_t ndim) const;
};

struct SimplexRecord {
    std::int64_t simplex_id = -1;
    std::vector<std::int64_t> vertex_ids;
    std::int64_t parent_id = -1;
    std::vector<std::int64_t> children;
    bool active = true;
    size_t level = 0;
    std::array<std::int64_t, 2> split_edge{-1, -1};
    std::int64_t midpoint_vertex_id = -1;
};

}  // namespace meanfi::zero_temp_native
