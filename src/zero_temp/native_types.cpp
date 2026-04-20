#include "native_types.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>

namespace meanfi::zero_temp_native {

nb::ndarray<nb::numpy, std::int64_t> NativeRefinementDescriptor::child_ids_array() const {
    return make_array(std::vector<std::int64_t>(child_ids), {child_ids.size()});
}

nb::ndarray<nb::numpy, std::int64_t> NativeRefinementDescriptor::parent_vertex_ids_array() const {
    return make_array(std::vector<std::int64_t>(parent_vertex_ids), {parent_vertex_ids.size()});
}

nb::ndarray<nb::numpy, std::int64_t> NativeRefinementDescriptor::child_vertex_ids_array() const {
    return make_array(
        std::vector<std::int64_t>(child_vertex_ids),
        {child_ids.size(), n_vertices}
    );
}

nb::ndarray<nb::numpy, std::int64_t> NativeRefinementDescriptor::bisected_edge_array() const {
    return make_array(
        std::vector<std::int64_t>{bisected_edge[0], bisected_edge[1]},
        {2}
    );
}

nb::tuple NativeRefinementBatch::as_tuple(size_t ndim) const {
    return nb::make_tuple(
        refinements,
        make_array(std::vector<std::int64_t>(parent_ids), {parent_ids.size()}),
        make_array(std::vector<std::int64_t>(child_offsets), {child_offsets.size()}),
        make_array(std::vector<std::int64_t>(child_ids), {child_ids.size()}),
        make_array(
            std::vector<std::int64_t>(parent_vertex_ids),
            {parent_ids.size(), ndim + 1}
        ),
        make_array(
            std::vector<std::int64_t>(child_vertex_ids),
            {child_ids.size(), ndim + 1}
        ),
        make_array(std::vector<std::int64_t>(midpoint_ids), {midpoint_ids.size()}),
        make_array(
            std::vector<std::int64_t>(bisected_edges),
            {parent_ids.size(), 2}
        )
    );
}

std::string point_key_from_ptr(const double *point, size_t ndim, double tol) {
    std::vector<double> normalized(ndim);
    for (size_t axis = 0; axis < ndim; ++axis) {
        const double value = point[axis];
        normalized[axis] = std::abs(value) <= tol ? 0.0 : value;
    }
    return std::string(
        reinterpret_cast<const char *>(normalized.data()),
        normalized.size() * sizeof(double)
    );
}

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
            matrix[row * ndim + col] =
                points[(row + 1) * ndim + col] - base[col];
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

}  // namespace meanfi::zero_temp_native
