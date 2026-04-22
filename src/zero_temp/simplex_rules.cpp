#include "simplex_rules.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>

namespace meanfi::zero_temp_native::simplex_rules {

namespace {

double reference_simplex_volume(size_t ndim) {
    double factorial = 1.0;
    for (size_t value = 2; value <= ndim; ++value) {
        factorial *= static_cast<double>(value);
    }
    return 1.0 / factorial;
}

std::vector<double> reference_simplex_points_flat(size_t ndim) {
    std::vector<double> points((ndim + 1) * ndim, 0.0);
    for (size_t vertex = 1; vertex <= ndim; ++vertex) {
        points[vertex * ndim + (vertex - 1)] = 1.0;
    }
    return points;
}

double barycentric_coordinate(const double *point, size_t ndim, size_t vertex) {
    if (vertex == 0) {
        double value = 1.0;
        for (size_t axis = 0; axis < ndim; ++axis) {
            value -= point[axis];
        }
        return value;
    }
    return point[vertex - 1];
}

}  // namespace

double simplex_volume_from_flat(
    const std::vector<double> &points,
    size_t n_vertices,
    size_t ndim
) {
    if (n_vertices != ndim + 1) {
        throw std::runtime_error("simplex_volume_from_flat: expected ndim + 1 vertices");
    }
    if (ndim == 0) {
        return 1.0;
    }

    std::vector<double> matrix(ndim * ndim);
    const double *base = points.data();
    for (size_t row = 0; row < ndim; ++row) {
        for (size_t col = 0; col < ndim; ++col) {
            matrix[row * ndim + col] = points[(row + 1) * ndim + col] - base[col];
        }
    }

    double det = 1.0;
    for (size_t pivot = 0; pivot < ndim; ++pivot) {
        size_t pivot_row = pivot;
        double pivot_abs = std::abs(matrix[pivot * ndim + pivot]);
        for (size_t row = pivot + 1; row < ndim; ++row) {
            const double candidate_abs = std::abs(matrix[row * ndim + pivot]);
            if (candidate_abs > pivot_abs) {
                pivot_abs = candidate_abs;
                pivot_row = row;
            }
        }
        if (pivot_abs <= 0.0) {
            return 0.0;
        }
        if (pivot_row != pivot) {
            for (size_t col = 0; col < ndim; ++col) {
                std::swap(matrix[pivot * ndim + col], matrix[pivot_row * ndim + col]);
            }
            det = -det;
        }
        const double pivot_value = matrix[pivot * ndim + pivot];
        det *= pivot_value;
        for (size_t row = pivot + 1; row < ndim; ++row) {
            const double factor = matrix[row * ndim + pivot] / pivot_value;
            for (size_t col = pivot + 1; col < ndim; ++col) {
                matrix[row * ndim + col] -= factor * matrix[pivot * ndim + col];
            }
        }
    }

    double factorial = 1.0;
    for (size_t value = 2; value <= ndim; ++value) {
        factorial *= static_cast<double>(value);
    }
    return std::abs(det) / factorial;
}

std::pair<double, double> simplex_fraction_and_derivative(
    const double *energies,
    double mu,
    size_t dimension,
    const double *weights,
    double tol
) {
    if (dimension == 1) {
        const double e0 = energies[0];
        const double e1 = energies[1];
        const double denom = e1 - e0;
        const double fraction = std::clamp((mu - e0) / denom, 0.0, 1.0);
        const double derivative = (mu > e0 && mu < e1) ? 1.0 / denom : 0.0;
        return {fraction, derivative};
    }
    if (dimension == 2) {
        const double e0 = energies[0];
        const double e1 = energies[1];
        const double e2 = energies[2];
        if (mu < e1) {
            const double denom = (e1 - e0) * (e2 - e0);
            const double x = mu - e0;
            return {std::clamp((x * x) / denom, 0.0, 1.0), 2.0 * x / denom};
        }
        const double denom = (e2 - e0) * (e2 - e1);
        const double x = e2 - mu;
        return {std::clamp(1.0 - (x * x) / denom, 0.0, 1.0), 2.0 * x / denom};
    }
    if (dimension == 3) {
        const double e0 = energies[0];
        const double e1 = energies[1];
        const double e2 = energies[2];
        const double e3 = energies[3];
        if (mu < e1) {
            const double denom = (e1 - e0) * (e2 - e0) * (e3 - e0);
            const double x = mu - e0;
            return {std::clamp((x * x * x) / denom, 0.0, 1.0), 3.0 * x * x / denom};
        }
        if (mu < e2) {
            const double denom0 = (e1 - e0) * (e2 - e0) * (e3 - e0);
            const double denom1 = (e0 - e1) * (e2 - e1) * (e3 - e1);
            const double x0 = mu - e0;
            const double x1 = mu - e1;
            return {
                std::clamp(
                    (x0 * x0 * x0) / denom0 + (x1 * x1 * x1) / denom1,
                    0.0,
                    1.0
                ),
                3.0 * (x0 * x0 / denom0 + x1 * x1 / denom1),
            };
        }
        const double denom = (e3 - e0) * (e3 - e1) * (e3 - e2);
        const double x = e3 - mu;
        return {std::clamp(1.0 - (x * x * x) / denom, 0.0, 1.0), 3.0 * x * x / denom};
    }

    double fraction = 0.0;
    double derivative = 0.0;
    for (size_t vertex = 0; vertex < dimension + 1; ++vertex) {
        const double delta = std::max(mu - energies[vertex], 0.0);
        fraction += weights[vertex] * std::pow(delta, static_cast<int>(dimension));
        if (dimension == 1) {
            derivative += weights[vertex] * (delta > tol ? 1.0 : 0.0);
        } else if (delta > tol) {
            derivative += static_cast<double>(dimension) *
                          weights[vertex] *
                          std::pow(delta, static_cast<int>(dimension - 1));
        }
    }
    return {std::clamp(fraction, 0.0, 1.0), derivative};
}

std::vector<std::vector<double>> occupied_subsimplices_from_flat(
    const double *simplex_points,
    const double *energies,
    size_t n_vertices,
    size_t ndim,
    double mu,
    double tol
) {
    std::vector<size_t> order(n_vertices);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(
        order.begin(),
        order.end(),
        [&](size_t left, size_t right) { return energies[left] < energies[right]; }
    );

    std::vector<double> ordered_points(n_vertices * ndim);
    std::vector<double> ordered_energies(n_vertices);
    for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
        ordered_energies[vertex] = energies[order[vertex]];
        const double *src = simplex_points + order[vertex] * ndim;
        std::copy_n(src, ndim, ordered_points.data() + vertex * ndim);
    }

    size_t n_inside = 0;
    while (n_inside < n_vertices && ordered_energies[n_inside] <= mu) {
        ++n_inside;
    }
    if (n_inside == 0) {
        return {};
    }
    if (n_inside >= n_vertices) {
        return {ordered_points};
    }

    const size_t n_outside = n_vertices - n_inside;
    std::vector<std::vector<std::vector<double>>> grid(
        n_inside,
        std::vector<std::vector<double>>(n_outside + 1, std::vector<double>(ndim, 0.0))
    );

    for (size_t inside = 0; inside < n_inside; ++inside) {
        std::copy_n(
            ordered_points.data() + inside * ndim,
            ndim,
            grid[inside][0].data()
        );
        const double e_inside = ordered_energies[inside];
        for (size_t offset = 1; offset <= n_outside; ++offset) {
            const size_t outside_index = n_inside + offset - 1;
            const double e_outside = ordered_energies[outside_index];
            double alpha = 0.0;
            if (e_outside > e_inside) {
                alpha = (mu - e_inside) / (e_outside - e_inside);
            }
            alpha = std::clamp(alpha, 0.0, 1.0);
            for (size_t axis = 0; axis < ndim; ++axis) {
                const double inside_value = ordered_points[inside * ndim + axis];
                const double outside_value = ordered_points[outside_index * ndim + axis];
                grid[inside][offset][axis] =
                    (1.0 - alpha) * inside_value + alpha * outside_value;
            }
        }
    }

    std::vector<std::vector<double>> simplices;
    std::vector<double> path;
    path.reserve(n_vertices * ndim);
    path.insert(path.end(), grid[0][0].begin(), grid[0][0].end());

    std::function<void(size_t, size_t)> extend_path = [&](size_t i, size_t j) {
        if (i == n_inside - 1 && j == n_outside) {
            if (simplex_volume_from_flat(path, n_vertices, ndim) > tol) {
                simplices.push_back(path);
            }
            return;
        }
        if (i < n_inside - 1) {
            path.insert(path.end(), grid[i + 1][j].begin(), grid[i + 1][j].end());
            extend_path(i + 1, j);
            path.resize(path.size() - ndim);
        }
        if (j < n_outside) {
            path.insert(path.end(), grid[i][j + 1].begin(), grid[i][j + 1].end());
            extend_path(i, j + 1);
            path.resize(path.size() - ndim);
        }
    };

    extend_path(0, 0);
    return simplices;
}

std::vector<double> occupied_linear_moment_fractions(
    const double *sorted_energies,
    size_t ndim,
    double mu,
    double tol
) {
    const size_t n_vertices = ndim + 1;
    if (ndim == 0) {
        return {sorted_energies[0] <= mu + tol ? 1.0 : 0.0};
    }

    const auto reference_points = reference_simplex_points_flat(ndim);
    const auto pieces = occupied_subsimplices_from_flat(
        reference_points.data(),
        sorted_energies,
        n_vertices,
        ndim,
        mu,
        tol
    );

    std::vector<double> fractions(n_vertices, 0.0);
    if (pieces.empty()) {
        size_t occupied = 0;
        while (occupied < n_vertices && sorted_energies[occupied] <= mu + tol) {
            ++occupied;
        }
        if (occupied == 0) {
            return fractions;
        }
        if (occupied >= n_vertices) {
            std::fill(fractions.begin(), fractions.end(), 1.0 / static_cast<double>(n_vertices));
            return fractions;
        }
        for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
            fractions[vertex] = sorted_energies[vertex] <= mu + tol
                ? 1.0 / static_cast<double>(n_vertices)
                : 0.0;
        }
        return fractions;
    }

    const double ref_volume = reference_simplex_volume(ndim);
    for (const auto &piece : pieces) {
        const double piece_volume = simplex_volume_from_flat(piece, n_vertices, ndim);
        const double scale = piece_volume / (ref_volume * static_cast<double>(n_vertices));
        for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
            double bary_sum = 0.0;
            for (size_t point_index = 0; point_index < n_vertices; ++point_index) {
                const double *point = piece.data() + point_index * ndim;
                bary_sum += barycentric_coordinate(point, ndim, vertex);
            }
            fractions[vertex] += scale * bary_sum;
        }
    }
    return fractions;
}

}  // namespace meanfi::zero_temp_native::simplex_rules
