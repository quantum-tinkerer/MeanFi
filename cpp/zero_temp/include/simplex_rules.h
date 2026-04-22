#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace meanfi::zero_temp::simplex_rules {

double simplex_volume_from_flat(
    const std::vector<double> &points,
    size_t n_vertices,
    size_t ndim
);

std::pair<double, double> simplex_fraction_and_derivative(
    const double *energies,
    double mu,
    size_t dimension,
    const double *weights,
    double tol
);

std::vector<std::vector<double>> occupied_subsimplices_from_flat(
    const double *simplex_points,
    const double *energies,
    size_t n_vertices,
    size_t ndim,
    double mu,
    double tol
);

std::vector<double> occupied_linear_moment_fractions(
    const double *sorted_energies,
    size_t ndim,
    double mu,
    double tol
);

}  // namespace meanfi::zero_temp::simplex_rules
