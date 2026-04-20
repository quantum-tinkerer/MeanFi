#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

using Float1D = nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig>;
using Float2D = nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>;
using Float3D = nb::ndarray<nb::numpy, const double, nb::ndim<3>, nb::c_contig>;
using Int1D = nb::ndarray<nb::numpy, const std::int64_t, nb::ndim<1>, nb::c_contig>;
using Int2D = nb::ndarray<nb::numpy, const std::int64_t, nb::ndim<2>, nb::c_contig>;
using Bool1D = nb::ndarray<nb::numpy, const bool, nb::ndim<1>, nb::c_contig>;
using Complex1D = nb::ndarray<nb::numpy, const std::complex<double>, nb::ndim<1>, nb::c_contig>;
using Complex2D = nb::ndarray<nb::numpy, const std::complex<double>, nb::ndim<2>, nb::c_contig>;
using Complex3D = nb::ndarray<nb::numpy, const std::complex<double>, nb::ndim<3>, nb::c_contig>;

template <typename T>
nb::ndarray<nb::numpy, T> make_array(std::vector<T> &&data, std::initializer_list<size_t> shape) {
    T *raw = new T[data.size()];
    std::move(data.begin(), data.end(), raw);
    nb::capsule owner(raw, [](void *p) noexcept { delete[] static_cast<T *>(p); });
    return nb::ndarray<nb::numpy, T>(raw, shape, owner);
}

std::string point_key_from_ptr(const double *point, size_t ndim, double tol) {
    std::vector<double> normalized(ndim);
    for (size_t axis = 0; axis < ndim; ++axis) {
        double value = point[axis];
        normalized[axis] = std::abs(value) <= tol ? 0.0 : value;
    }
    return std::string(
        reinterpret_cast<const char *>(normalized.data()),
        normalized.size() * sizeof(double)
    );
}

std::vector<std::int64_t> owner_unique_from_runs(
    const std::int64_t *owner_ids,
    size_t count,
    std::vector<std::int64_t> &owner_inverse
) {
    owner_inverse.resize(count);
    std::vector<std::int64_t> owner_unique;
    if (count == 0) {
        return owner_unique;
    }

    std::int64_t current = owner_ids[0];
    std::int64_t run_index = 0;
    owner_unique.push_back(current);
    owner_inverse[0] = 0;
    for (size_t index = 1; index < count; ++index) {
        if (owner_ids[index] != current) {
            current = owner_ids[index];
            ++run_index;
            owner_unique.push_back(current);
        }
        owner_inverse[index] = run_index;
    }
    return owner_unique;
}

double simplex_volume_from_flat(const std::vector<double> &points, size_t n_vertices, size_t ndim) {
    if (n_vertices != ndim + 1) {
        throw std::runtime_error("simplex_volume_from_flat: expected ndim + 1 vertices");
    }
    if (ndim == 0) {
        return 0.0;
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
            double candidate_abs = std::abs(matrix[row * ndim + pivot]);
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

class PointRegistry {
public:
    PointRegistry() = default;

    nb::ndarray<nb::numpy, std::int64_t> register_many(Float2D points, nb::object vertex_ids_obj) {
        const auto n_points = points.shape(0);
        const auto ndim = points.shape(1);
        if (ndim_ == 0) {
            ndim_ = ndim;
        } else if (ndim_ != ndim) {
            throw std::runtime_error("PointRegistry dimension mismatch");
        }

        std::vector<std::int64_t> point_ids(n_points);
        const std::int64_t *vertex_ids = nullptr;
        if (!vertex_ids_obj.is_none()) {
            auto vertex_ids_arr = nb::cast<Int1D>(vertex_ids_obj);
            if (vertex_ids_arr.shape(0) != n_points) {
                throw std::runtime_error("vertex_ids length mismatch");
            }
            vertex_ids = vertex_ids_arr.data();
        }

        const double *point_ptr = points.data();
        for (size_t index = 0; index < n_points; ++index) {
            const double *row = point_ptr + index * ndim;
            std::string key = point_key_from_ptr(row, ndim, tol_);
            auto it = lookup_.find(key);
            std::int64_t vertex_id = vertex_ids ? vertex_ids[index] : -1;
            if (it == lookup_.end()) {
                const std::int64_t point_id = static_cast<std::int64_t>(vertex_ids_.size());
                lookup_.emplace(std::move(key), point_id);
                points_.insert(points_.end(), row, row + ndim);
                vertex_ids_.push_back(vertex_id);
                point_ids[index] = point_id;
            } else {
                const std::int64_t point_id = it->second;
                if (vertex_id >= 0 && vertex_ids_[point_id] < 0) {
                    vertex_ids_[point_id] = vertex_id;
                }
                point_ids[index] = point_id;
            }
        }

        return make_array(std::move(point_ids), {n_points});
    }

    nb::ndarray<nb::numpy, double> points_for_ids(Int1D point_ids) const {
        const auto n_ids = point_ids.shape(0);
        std::vector<double> points(n_ids * ndim_);
        const std::int64_t *ids = point_ids.data();
        for (size_t index = 0; index < n_ids; ++index) {
            const std::int64_t point_id = ids[index];
            if (point_id < 0 || static_cast<size_t>(point_id) >= vertex_ids_.size()) {
                throw std::runtime_error("point_id out of range");
            }
            const size_t offset = static_cast<size_t>(point_id) * ndim_;
            std::copy_n(points_.data() + offset, ndim_, points.data() + index * ndim_);
        }
        return make_array(std::move(points), {n_ids, ndim_});
    }

    nb::ndarray<nb::numpy, std::int64_t> vertex_ids_for_ids(Int1D point_ids) const {
        const auto n_ids = point_ids.shape(0);
        std::vector<std::int64_t> values(n_ids);
        const std::int64_t *ids = point_ids.data();
        for (size_t index = 0; index < n_ids; ++index) {
            const std::int64_t point_id = ids[index];
            if (point_id < 0 || static_cast<size_t>(point_id) >= vertex_ids_.size()) {
                throw std::runtime_error("point_id out of range");
            }
            values[index] = vertex_ids_[point_id];
        }
        return make_array(std::move(values), {n_ids});
    }

    size_t size() const { return vertex_ids_.size(); }

private:
    size_t ndim_ = 0;
    double tol_ = 1e-14;
    std::unordered_map<std::string, std::int64_t> lookup_;
    std::vector<double> points_;
    std::vector<std::int64_t> vertex_ids_;
};

class NativeTightBindingModel {
public:
    NativeTightBindingModel(Int2D keys, Complex3D matrices) {
        ndim_ = keys.shape(1);
        nterms_ = keys.shape(0);
        if (matrices.shape(0) != nterms_) {
            throw std::runtime_error("NativeTightBindingModel: term axis mismatch");
        }
        if (matrices.shape(1) != matrices.shape(2)) {
            throw std::runtime_error("NativeTightBindingModel: matrices must be square");
        }
        ndof_ = matrices.shape(1);
        keys_.assign(keys.data(), keys.data() + nterms_ * ndim_);
        matrices_.assign(matrices.data(), matrices.data() + nterms_ * ndof_ * ndof_);
    }

    size_t ndim() const noexcept { return ndim_; }
    size_t ndof() const noexcept { return ndof_; }
    size_t nterms() const noexcept { return nterms_; }

    nb::ndarray<nb::numpy, std::int64_t> keys_array() const {
        return make_array(std::vector<std::int64_t>(keys_), {nterms_, ndim_});
    }

    nb::ndarray<nb::numpy, std::complex<double>> matrices_array() const {
        return make_array(std::vector<std::complex<double>>(matrices_), {nterms_, ndof_, ndof_});
    }

    nb::ndarray<nb::numpy, std::complex<double>> evaluate_many(Float2D points) const {
        const auto n_points = points.shape(0);
        if (points.shape(1) != ndim_) {
            throw std::runtime_error("NativeTightBindingModel: point dimension mismatch");
        }
        std::vector<std::complex<double>> out(n_points * ndof_ * ndof_);
        const double *point_ptr = points.data();
        for (size_t point_index = 0; point_index < n_points; ++point_index) {
            Eigen::MatrixXcd h = evaluate_point_raw(point_ptr + point_index * ndim_);
            for (size_t row = 0; row < ndof_; ++row) {
                for (size_t col = 0; col < ndof_; ++col) {
                    out[(point_index * ndof_ + row) * ndof_ + col] = h(row, col);
                }
            }
        }
        return make_array(std::move(out), {n_points, ndof_, ndof_});
    }

    nb::ndarray<nb::numpy, std::complex<double>> evaluate_point(Float1D point) const {
        if (point.shape(0) != ndim_) {
            throw std::runtime_error("NativeTightBindingModel: point dimension mismatch");
        }
        Eigen::MatrixXcd h = evaluate_point_raw(point.data());
        std::vector<std::complex<double>> out(ndof_ * ndof_);
        for (size_t row = 0; row < ndof_; ++row) {
            for (size_t col = 0; col < ndof_; ++col) {
                out[row * ndof_ + col] = h(row, col);
            }
        }
        return make_array(std::move(out), {ndof_, ndof_});
    }

    Eigen::MatrixXcd evaluate_point_raw(const double *point) const {
        Eigen::MatrixXcd h = Eigen::MatrixXcd::Zero(ndof_, ndof_);
        for (size_t term = 0; term < nterms_; ++term) {
            double phase_arg = 0.0;
            for (size_t axis = 0; axis < ndim_; ++axis) {
                phase_arg += point[axis] * static_cast<double>(keys_[term * ndim_ + axis]);
            }
            const std::complex<double> phase = std::exp(std::complex<double>(0.0, -phase_arg));
            for (size_t row = 0; row < ndof_; ++row) {
                for (size_t col = 0; col < ndof_; ++col) {
                    h(row, col) += phase * matrices_[(term * ndof_ + row) * ndof_ + col];
                }
            }
        }
        return h;
    }

private:
    size_t ndim_ = 0;
    size_t ndof_ = 0;
    size_t nterms_ = 0;
    std::vector<std::int64_t> keys_;
    std::vector<std::complex<double>> matrices_;
};

class NativeSpectralCache {
public:
    explicit NativeSpectralCache(std::shared_ptr<NativeTightBindingModel> model, double tol = 1e-14)
        : model_(std::move(model)), tol_(tol) {
        if (!model_) {
            throw std::runtime_error("NativeSpectralCache: model must not be null");
        }
    }

    void clear() {
        cache_.clear();
    }

    void invalidate() {
        ++generation_;
        clear();
    }

    std::uint64_t generation() const noexcept { return generation_; }
    size_t ndim() const noexcept { return model_->ndim(); }
    size_t ndof() const noexcept { return model_->ndof(); }
    size_t size() const noexcept { return cache_.size(); }
    std::uint64_t n_kernel_evals() const noexcept { return n_kernel_evals_; }

    nb::tuple evaluate_many(Float2D points) {
        return evaluate_many_impl(points, false);
    }

    nb::tuple get_many(Float2D points) {
        return evaluate_many_impl(points, true);
    }

    nb::ndarray<nb::numpy, double> get_many_values(Float2D points) {
        auto result = evaluate_many_impl(points, true);
        return nb::cast<nb::ndarray<nb::numpy, double>>(result[0]);
    }

private:
    friend class NativeChargeEvaluator;
    friend class NativeDensityEvaluator;

    struct CacheEntry {
        std::vector<double> eigenvalues;
        std::vector<std::complex<double>> eigenvectors;
    };

    nb::tuple evaluate_many_impl(Float2D points, bool use_cache) {
        const auto n_points = points.shape(0);
        if (points.shape(1) != model_->ndim()) {
            throw std::runtime_error("NativeSpectralCache: point dimension mismatch");
        }
        const size_t ndof = model_->ndof();
        std::vector<double> values(n_points * ndof);
        std::vector<std::complex<double>> vectors(n_points * ndof * ndof);
        const double *point_ptr = points.data();

        for (size_t point_index = 0; point_index < n_points; ++point_index) {
            const double *row = point_ptr + point_index * model_->ndim();
            const std::string key = point_key_from_ptr(row, model_->ndim(), tol_);
            CacheEntry *entry = nullptr;
            if (use_cache) {
                auto it = cache_.find(key);
                if (it != cache_.end()) {
                    entry = &it->second;
                }
            }
            if (entry == nullptr) {
                CacheEntry fresh = diagonalize_point(row);
                if (use_cache) {
                    auto [it, _inserted] = cache_.emplace(key, std::move(fresh));
                    entry = &it->second;
                } else {
                    temp_entry_ = std::move(fresh);
                    entry = &temp_entry_;
                }
            }
            std::copy_n(entry->eigenvalues.data(), ndof, values.data() + point_index * ndof);
            std::copy_n(entry->eigenvectors.data(), ndof * ndof, vectors.data() + point_index * ndof * ndof);
        }

        return nb::make_tuple(
            make_array(std::move(values), {n_points, ndof}),
            make_array(std::move(vectors), {n_points, ndof, ndof})
        );
    }

    CacheEntry diagonalize_point(const double *point) {
        Eigen::MatrixXcd h = model_->evaluate_point_raw(point);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(h);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("NativeSpectralCache: eigensolve failed");
        }

        ++n_kernel_evals_;
        const size_t ndof = model_->ndof();
        CacheEntry entry;
        entry.eigenvalues.resize(ndof);
        entry.eigenvectors.resize(ndof * ndof);

        const auto evals = solver.eigenvalues();
        const auto evecs = solver.eigenvectors();
        for (size_t i = 0; i < ndof; ++i) {
            entry.eigenvalues[i] = evals(static_cast<Eigen::Index>(i));
        }
        for (size_t row = 0; row < ndof; ++row) {
            for (size_t col = 0; col < ndof; ++col) {
                entry.eigenvectors[row * ndof + col] = evecs(row, col);
            }
        }
        return entry;
    }

    std::shared_ptr<NativeTightBindingModel> model_;
    double tol_ = 1e-14;
    std::uint64_t generation_ = 0;
    std::uint64_t n_kernel_evals_ = 0;
    std::unordered_map<std::string, CacheEntry> cache_;
    CacheEntry temp_entry_;
};

struct NativeRefinementDescriptor {
    std::int64_t parent_id = -1;
    std::vector<std::int64_t> child_ids;
    std::vector<std::int64_t> parent_vertex_ids;
    std::vector<std::int64_t> child_vertex_ids;
    std::int64_t new_midpoint_vertex_id = -1;
    std::array<std::int64_t, 2> bisected_edge{ -1, -1 };
    size_t n_vertices = 0;

    nb::ndarray<nb::numpy, std::int64_t> child_ids_array() const {
        return make_array(std::vector<std::int64_t>(child_ids), {child_ids.size()});
    }

    nb::ndarray<nb::numpy, std::int64_t> parent_vertex_ids_array() const {
        return make_array(std::vector<std::int64_t>(parent_vertex_ids), {parent_vertex_ids.size()});
    }

    nb::ndarray<nb::numpy, std::int64_t> child_vertex_ids_array() const {
        return make_array(
            std::vector<std::int64_t>(child_vertex_ids),
            {child_ids.size(), n_vertices}
        );
    }

    nb::ndarray<nb::numpy, std::int64_t> bisected_edge_array() const {
        return make_array(
            std::vector<std::int64_t>{bisected_edge[0], bisected_edge[1]},
            {2}
        );
    }
};

class NativeChargeEvaluator;
class NativeDensityEvaluator;

struct NativeSimplexRecord {
    std::int64_t simplex_id = -1;
    std::vector<std::int64_t> vertex_ids;
    std::int64_t parent_id = -1;
    std::vector<std::int64_t> children;
    bool active = true;
    size_t level = 0;
    std::array<std::int64_t, 2> split_edge{ -1, -1 };
    std::int64_t midpoint_vertex_id = -1;
};

class NativeGeometry {
public:
    static std::shared_ptr<NativeGeometry> root(
        size_t ndim,
        std::int64_t root_subcells_per_axis = 2,
        double tol = 1e-14
    ) {
        auto geometry = std::make_shared<NativeGeometry>(ndim, root_subcells_per_axis, tol);
        geometry->build_root();
        return geometry;
    }

    NativeGeometry(size_t ndim, std::int64_t root_subcells_per_axis = 2, double tol = 1e-14)
        : ndim_(ndim), root_subcells_per_axis_(root_subcells_per_axis), tol_(tol) {}

    size_t ndim() const noexcept { return ndim_; }
    std::int64_t root_subcells_per_axis() const noexcept { return root_subcells_per_axis_; }
    size_t n_vertices() const noexcept { return vertex_count_; }
    size_t n_simplices() const noexcept { return simplex_records_.size(); }
    size_t n_active() const noexcept { return active_simplex_ids_.size(); }

    nb::ndarray<nb::numpy, double> vertices_array() const {
        return make_array(std::vector<double>(vertices_), {n_vertices(), ndim_});
    }

    nb::ndarray<nb::numpy, std::int64_t> active_simplex_ids() const {
        return make_array(std::vector<std::int64_t>(active_simplex_ids_), {active_simplex_ids_.size()});
    }

    nb::ndarray<nb::numpy, std::int64_t> simplex_vertex_ids(std::int64_t simplex_id) const {
        const auto &record = simplex_record(simplex_id);
        return make_array(std::vector<std::int64_t>(record.vertex_ids), {record.vertex_ids.size()});
    }

    nb::ndarray<nb::numpy, double> simplex_points(std::int64_t simplex_id) const {
        std::vector<double> points = simplex_points_flat(simplex_id);
        return make_array(std::move(points), {ndim_ + 1, ndim_});
    }

    double simplex_volume(std::int64_t simplex_id) {
        ensure_volume_cache_size();
        if (simplex_volume_cache_[simplex_id] >= 0.0) {
            return simplex_volume_cache_[simplex_id];
        }
        double volume = simplex_volume_impl(simplex_id);
        simplex_volume_cache_[simplex_id] = volume;
        return volume;
    }

    nb::ndarray<nb::numpy, std::int64_t> ensure_children(std::int64_t simplex_id) {
        const auto &children = ensure_children_impl(simplex_id);
        return make_array(std::vector<std::int64_t>(children), {children.size()});
    }

    nb::ndarray<nb::numpy, std::int64_t> descendant_leaves(std::int64_t simplex_id, std::int64_t levels) {
        std::vector<std::int64_t> leaves;
        descendant_leaves_impl(simplex_id, levels, leaves);
        return make_array(std::move(leaves), {leaves.size()});
    }

    nb::tuple refine(Int1D marked_ids_arr) {
        std::unordered_set<std::int64_t> marked(
            marked_ids_arr.data(),
            marked_ids_arr.data() + marked_ids_arr.shape(0)
        );
        if (marked.empty()) {
            return nb::make_tuple(
                0,
                make_array(std::vector<std::int64_t>{}, {0}),
                make_array(std::vector<std::int64_t>{0}, {1}),
                make_array(std::vector<std::int64_t>{}, {0}),
                make_array(std::vector<std::int64_t>{}, {0, ndim_ + 1}),
                make_array(std::vector<std::int64_t>{}, {0, ndim_ + 1}),
                make_array(std::vector<std::int64_t>{}, {0}),
                make_array(std::vector<std::int64_t>{}, {0, 2})
            );
        }

        std::vector<std::int64_t> refined_active;
        refined_active.reserve(active_simplex_ids_.size() * 2);
        int refinements = 0;
        std::vector<std::int64_t> parent_ids;
        std::vector<std::int64_t> child_offsets{0};
        std::vector<std::int64_t> child_ids_flat;
        std::vector<std::int64_t> parent_vertex_ids_flat;
        std::vector<std::int64_t> child_vertex_ids_flat;
        std::vector<std::int64_t> midpoint_ids;
        std::vector<std::int64_t> bisected_edges_flat;

        for (std::int64_t simplex_id : active_simplex_ids_) {
            if (marked.find(simplex_id) == marked.end()) {
                refined_active.push_back(simplex_id);
                continue;
            }

            ++refinements;
            simplex_records_[simplex_id].active = false;
            const auto &children = ensure_children_impl(simplex_id);
            const auto &parent = simplex_records_[simplex_id];
            for (std::int64_t child_id : children) {
                simplex_records_[child_id].active = true;
                refined_active.push_back(child_id);
            }

            parent_ids.push_back(simplex_id);
            child_ids_flat.insert(child_ids_flat.end(), children.begin(), children.end());
            child_offsets.push_back(static_cast<std::int64_t>(child_ids_flat.size()));
            parent_vertex_ids_flat.insert(
                parent_vertex_ids_flat.end(),
                parent.vertex_ids.begin(),
                parent.vertex_ids.end()
            );
            for (std::int64_t child_id : children) {
                const auto &child_vertices = simplex_records_[child_id].vertex_ids;
                child_vertex_ids_flat.insert(
                    child_vertex_ids_flat.end(),
                    child_vertices.begin(),
                    child_vertices.end()
                );
            }
            midpoint_ids.push_back(parent.midpoint_vertex_id);
            bisected_edges_flat.push_back(parent.split_edge[0]);
            bisected_edges_flat.push_back(parent.split_edge[1]);
        }

        active_simplex_ids_ = std::move(refined_active);
        return nb::make_tuple(
            refinements,
            make_array(std::move(parent_ids), {static_cast<size_t>(refinements)}),
            make_array(std::move(child_offsets), {static_cast<size_t>(refinements) + 1}),
            make_array(std::move(child_ids_flat), {static_cast<size_t>(2 * refinements)}),
            make_array(std::move(parent_vertex_ids_flat), {static_cast<size_t>(refinements), ndim_ + 1}),
            make_array(std::move(child_vertex_ids_flat), {static_cast<size_t>(2 * refinements), ndim_ + 1}),
            make_array(std::move(midpoint_ids), {static_cast<size_t>(refinements)}),
            make_array(std::move(bisected_edges_flat), {static_cast<size_t>(refinements), 2})
        );
    }

    const std::vector<std::int64_t> &active_simplex_ids_vector() const noexcept {
        return active_simplex_ids_;
    }

    const std::vector<std::int64_t> &simplex_vertex_ids_vector(std::int64_t simplex_id) const {
        return simplex_record(simplex_id).vertex_ids;
    }

private:
    friend class NativeChargeEvaluator;
    friend class NativeDensityEvaluator;

    void build_root() {
        const double step = 1.0 / static_cast<double>(root_subcells_per_axis_);
        const size_t n_offsets = ndim_ == 0 ? 1 : static_cast<size_t>(std::pow(root_subcells_per_axis_, ndim_));

        for (size_t offset_index = 0; offset_index < n_offsets; ++offset_index) {
            std::vector<double> base(ndim_, 0.0);
            size_t remainder = offset_index;
            for (size_t axis = 0; axis < ndim_; ++axis) {
                const size_t digit = remainder % static_cast<size_t>(root_subcells_per_axis_);
                remainder /= static_cast<size_t>(root_subcells_per_axis_);
                base[axis] = step * static_cast<double>(digit);
            }

            std::vector<int> perm(ndim_);
            std::iota(perm.begin(), perm.end(), 0);

            if (ndim_ == 0) {
                const std::int64_t vertex_id = get_or_add_vertex(base);
                active_simplex_ids_.push_back(add_simplex({vertex_id}));
                continue;
            }

            do {
                std::vector<double> point = base;
                std::vector<std::int64_t> simplex;
                simplex.reserve(ndim_ + 1);
                simplex.push_back(get_or_add_vertex(point));
                for (int axis : perm) {
                    point[static_cast<size_t>(axis)] += step;
                    simplex.push_back(get_or_add_vertex(point));
                }
                active_simplex_ids_.push_back(add_simplex(simplex));
            } while (std::next_permutation(perm.begin(), perm.end()));
        }
    }

    std::int64_t get_or_add_vertex(const std::vector<double> &point) {
        const std::string key = point_key_from_ptr(point.data(), ndim_, tol_);
        auto it = vertex_lookup_.find(key);
        if (it != vertex_lookup_.end()) {
            return it->second;
        }
        const std::int64_t vertex_id = static_cast<std::int64_t>(vertex_count_);
        vertices_.insert(vertices_.end(), point.begin(), point.end());
        vertex_lookup_.emplace(key, vertex_id);
        ++vertex_count_;
        return vertex_id;
    }

    std::int64_t add_simplex(
        const std::vector<std::int64_t> &vertex_ids,
        std::int64_t parent_id = -1,
        size_t level = 0,
        bool active = true
    ) {
        const std::int64_t simplex_id = static_cast<std::int64_t>(simplex_records_.size());
        NativeSimplexRecord record;
        record.simplex_id = simplex_id;
        record.vertex_ids = vertex_ids;
        record.parent_id = parent_id;
        record.level = level;
        record.active = active;
        simplex_records_.push_back(std::move(record));
        simplex_volume_cache_.push_back(-1.0);
        return simplex_id;
    }

    const NativeSimplexRecord &simplex_record(std::int64_t simplex_id) const {
        if (simplex_id < 0 || static_cast<size_t>(simplex_id) >= simplex_records_.size()) {
            throw std::runtime_error("NativeGeometry: simplex_id out of range");
        }
        return simplex_records_[static_cast<size_t>(simplex_id)];
    }

    std::vector<double> simplex_points_flat(std::int64_t simplex_id) const {
        const auto &record = simplex_record(simplex_id);
        std::vector<double> points(record.vertex_ids.size() * ndim_);
        for (size_t vertex = 0; vertex < record.vertex_ids.size(); ++vertex) {
            const std::int64_t vertex_id = record.vertex_ids[vertex];
            const size_t src_offset = static_cast<size_t>(vertex_id) * ndim_;
            std::copy_n(
                vertices_.data() + src_offset,
                ndim_,
                points.data() + vertex * ndim_
            );
        }
        return points;
    }

    void ensure_volume_cache_size() {
        if (simplex_volume_cache_.size() < simplex_records_.size()) {
            simplex_volume_cache_.resize(simplex_records_.size(), -1.0);
        }
    }

    double simplex_volume_impl(std::int64_t simplex_id) const {
        if (ndim_ == 0) {
            return 1.0;
        }
        const std::vector<double> points = simplex_points_flat(simplex_id);
        return simplex_volume_from_flat(points, ndim_ + 1, ndim_);
    }

    std::pair<std::int64_t, std::int64_t> longest_edge(std::int64_t simplex_id) const {
        if (ndim_ == 0) {
            throw std::runtime_error("NativeGeometry: zero-dimensional simplices cannot be bisected");
        }
        const std::vector<double> points = simplex_points_flat(simplex_id);
        std::pair<std::int64_t, std::int64_t> best{0, 1};
        double best_length = -1.0;
        for (size_t i = 0; i < ndim_ + 1; ++i) {
            for (size_t j = i + 1; j < ndim_ + 1; ++j) {
                double length = 0.0;
                for (size_t axis = 0; axis < ndim_; ++axis) {
                    const double delta = points[i * ndim_ + axis] - points[j * ndim_ + axis];
                    length += delta * delta;
                }
                if (length > best_length) {
                    best_length = length;
                    best = {static_cast<std::int64_t>(i), static_cast<std::int64_t>(j)};
                }
            }
        }
        return best;
    }

    const std::vector<std::int64_t> &ensure_children_impl(std::int64_t simplex_id) {
        auto &record = simplex_records_[static_cast<size_t>(simplex_id)];
        if (!record.children.empty()) {
            return record.children;
        }

        const size_t parent_level = record.level;
        const std::vector<std::int64_t> parent_vertex_ids = record.vertex_ids;
        const auto [edge_i, edge_j] = longest_edge(simplex_id);
        std::vector<double> points = simplex_points_flat(simplex_id);
        std::vector<double> midpoint(ndim_, 0.0);
        for (size_t axis = 0; axis < ndim_; ++axis) {
            midpoint[axis] =
                0.5 * (points[static_cast<size_t>(edge_i) * ndim_ + axis] +
                       points[static_cast<size_t>(edge_j) * ndim_ + axis]);
        }
        const std::int64_t midpoint_id = get_or_add_vertex(midpoint);

        std::vector<std::int64_t> child_a = parent_vertex_ids;
        std::vector<std::int64_t> child_b = parent_vertex_ids;
        child_a[static_cast<size_t>(edge_i)] = midpoint_id;
        child_b[static_cast<size_t>(edge_j)] = midpoint_id;
        const std::int64_t child_a_id = add_simplex(child_a, simplex_id, parent_level + 1, false);
        const std::int64_t child_b_id = add_simplex(child_b, simplex_id, parent_level + 1, false);

        auto &updated_record = simplex_records_[static_cast<size_t>(simplex_id)];
        updated_record.children = {child_a_id, child_b_id};
        updated_record.midpoint_vertex_id = midpoint_id;
        updated_record.split_edge = {edge_i, edge_j};
        return updated_record.children;
    }

    void descendant_leaves_impl(
        std::int64_t simplex_id,
        std::int64_t levels,
        std::vector<std::int64_t> &out
    ) {
        if (levels <= 0) {
            out.push_back(simplex_id);
            return;
        }
        const auto &children = ensure_children_impl(simplex_id);
        if (levels == 1) {
            out.insert(out.end(), children.begin(), children.end());
            return;
        }
        for (std::int64_t child_id : children) {
            descendant_leaves_impl(child_id, levels - 1, out);
        }
    }

    size_t ndim_ = 0;
    std::int64_t root_subcells_per_axis_ = 2;
    double tol_ = 1e-14;
    size_t vertex_count_ = 0;
    std::vector<double> vertices_;
    std::unordered_map<std::string, std::int64_t> vertex_lookup_;
    std::vector<NativeSimplexRecord> simplex_records_;
    std::vector<double> simplex_volume_cache_;
    std::vector<std::int64_t> active_simplex_ids_;
};

class NativeFrontier {
public:
    static NativeFrontier from_geometry(NativeGeometry &geometry) {
        return NativeFrontier(geometry);
    }

    explicit NativeFrontier(NativeGeometry &geometry)
        : geometry_(&geometry) {
        sync_from_geometry();
    }

    void sync_from_geometry() {
        active_simplex_ids_ = geometry_->active_simplex_ids_vector();
    }

    size_t n_active() const noexcept { return active_simplex_ids_.size(); }
    std::uint64_t generation() const noexcept { return generation_; }

    nb::ndarray<nb::numpy, std::int64_t> active_simplex_ids() const {
        return make_array(std::vector<std::int64_t>(active_simplex_ids_), {active_simplex_ids_.size()});
    }

    nb::ndarray<nb::numpy, std::int64_t> vertex_ids() const {
        const size_t n_vertices = geometry_->ndim() + 1;
        std::vector<std::int64_t> out(active_simplex_ids_.size() * n_vertices);
        for (size_t simplex = 0; simplex < active_simplex_ids_.size(); ++simplex) {
            const auto &vertex_ids = geometry_->simplex_vertex_ids_vector(active_simplex_ids_[simplex]);
            std::copy_n(vertex_ids.data(), n_vertices, out.data() + simplex * n_vertices);
        }
        return make_array(std::move(out), {active_simplex_ids_.size(), n_vertices});
    }

    nb::ndarray<nb::numpy, double> volumes() {
        std::vector<double> out(active_simplex_ids_.size());
        for (size_t simplex = 0; simplex < active_simplex_ids_.size(); ++simplex) {
            out[simplex] = geometry_->simplex_volume(active_simplex_ids_[simplex]);
        }
        return make_array(std::move(out), {active_simplex_ids_.size()});
    }

    void apply_refinement(Int1D parent_ids, Int1D child_offsets, Int1D child_ids) {
        std::unordered_map<std::int64_t, std::vector<std::int64_t>> replacements;
        const auto n_parents = parent_ids.shape(0);
        const auto *parent_ptr = parent_ids.data();
        const auto *offset_ptr = child_offsets.data();
        const auto *child_ptr = child_ids.data();
        for (size_t index = 0; index < n_parents; ++index) {
            const std::int64_t start = offset_ptr[index];
            const std::int64_t stop = offset_ptr[index + 1];
            replacements.emplace(
                parent_ptr[index],
                std::vector<std::int64_t>(child_ptr + start, child_ptr + stop)
            );
        }

        if (replacements.empty()) {
            return;
        }

        std::vector<std::int64_t> updated;
        updated.reserve(active_simplex_ids_.size() + replacements.size());
        for (std::int64_t simplex_id : active_simplex_ids_) {
            auto it = replacements.find(simplex_id);
            if (it == replacements.end()) {
                updated.push_back(simplex_id);
                continue;
            }
            updated.insert(updated.end(), it->second.begin(), it->second.end());
        }
        active_simplex_ids_ = std::move(updated);
        ++generation_;
    }

private:
    friend class NativeChargeEvaluator;
    friend class NativeDensityEvaluator;

    NativeGeometry *geometry_ = nullptr;
    std::vector<std::int64_t> active_simplex_ids_;
    std::uint64_t generation_ = 0;
};

class NativeChargeEvaluator {
public:
    NativeChargeEvaluator(
        NativeGeometry &geometry,
        NativeSpectralCache &spectral_cache,
        std::int64_t refine_levels = 0,
        double tol = 1e-14
    )
        : geometry_(&geometry),
          spectral_cache_(&spectral_cache),
          refine_levels_(std::max<std::int64_t>(0, refine_levels)),
          tol_(tol) {}

    nb::tuple evaluate(NativeFrontier &frontier, double mu) {
        const auto result = evaluate_impl(frontier, mu);
        return nb::make_tuple(
            result.total_charge,
            result.derivative_exact ? result.total_derivative : std::numeric_limits<double>::quiet_NaN(),
            result.derivative_exact,
            make_array(std::vector<std::int64_t>(result.owner_ids), {result.owner_ids.size()}),
            make_array(std::vector<double>(result.owner_charges), {result.owner_charges.size()})
        );
    }

    double simplex_charge(std::int64_t simplex_id, double mu, std::int64_t levels = -1) {
        const auto &group = prepared_group(simplex_id, levels < 0 ? refine_levels_ : levels);
        double total_charge = 0.0;
        double total_derivative = 0.0;
        bool derivative_exact = true;
        for (const auto &cell : group.cells) {
            double charge = 0.0;
            double derivative = 0.0;
            bool exact = true;
            evaluate_cell(cell, mu, charge, derivative, exact);
            total_charge += charge;
            total_derivative += derivative;
            derivative_exact = derivative_exact && exact;
        }
        return total_charge;
    }

private:
    struct PreparedChargeCell {
        std::vector<double> points_flat;
        std::vector<double> vertex_energies;
        std::vector<double> sorted_energies;
        std::vector<double> simplex_weights;
        std::vector<double> band_min;
        std::vector<double> band_max;
        std::vector<double> flat_energy;
        std::vector<std::uint8_t> distinct_mask;
        std::vector<std::uint8_t> flat_mask;
        double volume = 0.0;
    };

    struct PreparedChargeGroup {
        std::int64_t owner_id = -1;
        std::vector<std::int64_t> leaf_ids;
        std::vector<PreparedChargeCell> cells;
    };

    struct ChargeEvalResult {
        double total_charge = 0.0;
        double total_derivative = 0.0;
        bool derivative_exact = true;
        std::vector<std::int64_t> owner_ids;
        std::vector<double> owner_charges;
    };

    struct GroupKey {
        std::int64_t simplex_id = -1;
        std::int64_t levels = 0;

        bool operator==(const GroupKey &other) const noexcept {
            return simplex_id == other.simplex_id && levels == other.levels;
        }
    };

    struct GroupKeyHash {
        size_t operator()(const GroupKey &key) const noexcept {
            return std::hash<std::int64_t>{}(key.simplex_id) ^ (std::hash<std::int64_t>{}(key.levels) << 1);
        }
    };

    ChargeEvalResult evaluate_impl(NativeFrontier &frontier, double mu) {
        ChargeEvalResult result;
        result.owner_ids = frontier.active_simplex_ids_;
        result.owner_charges.resize(result.owner_ids.size(), 0.0);
        for (size_t owner_index = 0; owner_index < result.owner_ids.size(); ++owner_index) {
            const auto simplex_id = result.owner_ids[owner_index];
            const auto &group = prepared_group(simplex_id, refine_levels_);
            double owner_charge = 0.0;
            double owner_derivative = 0.0;
            bool owner_exact = true;
            for (const auto &cell : group.cells) {
                double charge = 0.0;
                double derivative = 0.0;
                bool exact = true;
                evaluate_cell(cell, mu, charge, derivative, exact);
                owner_charge += charge;
                owner_derivative += derivative;
                owner_exact = owner_exact && exact;
            }
            result.owner_charges[owner_index] = owner_charge;
            result.total_charge += owner_charge;
            result.total_derivative += owner_derivative;
            result.derivative_exact = result.derivative_exact && owner_exact;
        }
        return result;
    }

    const PreparedChargeGroup &prepared_group(std::int64_t simplex_id, std::int64_t levels) {
        const GroupKey key{simplex_id, levels};
        auto it = prepared_group_cache_.find(key);
        if (it != prepared_group_cache_.end()) {
            return it->second;
        }

        PreparedChargeGroup group;
        group.owner_id = simplex_id;
        if (levels <= 0) {
            group.leaf_ids.push_back(simplex_id);
        } else {
            geometry_->descendant_leaves_impl(simplex_id, levels, group.leaf_ids);
        }
        group.cells.reserve(group.leaf_ids.size());
        for (const auto leaf_id : group.leaf_ids) {
            group.cells.push_back(prepared_cell(leaf_id));
        }
        auto [inserted, _ok] = prepared_group_cache_.emplace(key, std::move(group));
        return inserted->second;
    }

    PreparedChargeCell prepared_cell(std::int64_t simplex_id) {
        auto it = prepared_cell_cache_.find(simplex_id);
        if (it != prepared_cell_cache_.end()) {
            return it->second;
        }

        PreparedChargeCell cell;
        const auto &record = geometry_->simplex_records_[static_cast<size_t>(simplex_id)];
        const size_t n_vertices = record.vertex_ids.size();
        const size_t ndof = spectral_cache_->model_->ndof();
        const size_t dimension = geometry_->ndim_;

        cell.points_flat = geometry_->simplex_points_flat(simplex_id);
        cell.vertex_energies.resize(n_vertices * ndof);
        cell.sorted_energies.resize(ndof * n_vertices);
        cell.simplex_weights.assign(ndof * n_vertices, 0.0);
        cell.band_min.resize(ndof);
        cell.band_max.resize(ndof);
        cell.flat_energy.resize(ndof);
        cell.distinct_mask.resize(ndof, 1);
        cell.flat_mask.resize(ndof, 0);
        cell.volume = geometry_->simplex_volume(simplex_id);

        for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
            const auto vertex_id = record.vertex_ids[vertex];
            const auto &values = vertex_eigenvalues(vertex_id);
            std::copy_n(values.data(), ndof, cell.vertex_energies.data() + vertex * ndof);
        }

        std::vector<double> scratch(n_vertices);
        for (size_t band = 0; band < ndof; ++band) {
            for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
                scratch[vertex] = cell.vertex_energies[vertex * ndof + band];
            }
            std::sort(scratch.begin(), scratch.end());
            cell.band_min[band] = scratch.front();
            cell.band_max[band] = scratch.back();
            cell.flat_energy[band] = cell.vertex_energies[band];
            cell.flat_mask[band] = (scratch.back() - scratch.front()) <= tol_ ? 1 : 0;
            bool distinct = true;
            for (size_t vertex = 1; vertex < n_vertices; ++vertex) {
                if (scratch[vertex] - scratch[vertex - 1] <= tol_) {
                    distinct = false;
                }
                cell.sorted_energies[band * n_vertices + vertex - 1] = scratch[vertex - 1];
            }
            cell.sorted_energies[band * n_vertices + n_vertices - 1] = scratch[n_vertices - 1];
            cell.distinct_mask[band] = distinct ? 1 : 0;
            if (dimension > 3 && distinct) {
                for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
                    double denom = 1.0;
                    for (size_t other = 0; other < n_vertices; ++other) {
                        if (other == vertex) {
                            continue;
                        }
                        denom *= scratch[vertex] - scratch[other];
                    }
                    cell.simplex_weights[band * n_vertices + vertex] = 1.0 / denom;
                }
            }
        }

        auto [inserted, _ok] = prepared_cell_cache_.emplace(simplex_id, std::move(cell));
        return inserted->second;
    }

    const std::vector<double> &vertex_eigenvalues(std::int64_t vertex_id) {
        auto it = vertex_values_cache_.find(vertex_id);
        if (it != vertex_values_cache_.end()) {
            return it->second;
        }

        const size_t ndim = geometry_->ndim_;
        std::vector<double> k_point(ndim);
        const size_t base = static_cast<size_t>(vertex_id) * ndim;
        for (size_t axis = 0; axis < ndim; ++axis) {
            k_point[axis] = 2.0 * kPi * geometry_->vertices_[base + axis] - kPi;
        }
        const std::string key = point_key_from_ptr(k_point.data(), ndim, spectral_cache_->tol_);
        auto cache_it = spectral_cache_->cache_.find(key);
        if (cache_it == spectral_cache_->cache_.end()) {
            auto [inserted, _ok] = spectral_cache_->cache_.emplace(key, spectral_cache_->diagonalize_point(k_point.data()));
            cache_it = inserted;
        }
        auto [inserted, _ok] = vertex_values_cache_.emplace(vertex_id, cache_it->second.eigenvalues);
        return inserted->second;
    }

    static std::pair<double, double> simplex_fraction_and_derivative(
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
                    std::clamp((x0 * x0 * x0) / denom0 + (x1 * x1 * x1) / denom1, 0.0, 1.0),
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
                derivative += static_cast<double>(dimension) * weights[vertex] * std::pow(delta, static_cast<int>(dimension - 1));
            }
        }
        return {std::clamp(fraction, 0.0, 1.0), derivative};
    }

    void evaluate_cell(
        const PreparedChargeCell &cell,
        double mu,
        double &charge,
        double &derivative,
        bool &derivative_exact
    ) const {
        const size_t n_vertices = geometry_->ndim_ + 1;
        const size_t ndof = spectral_cache_->model_->ndof();
        double cell_min = std::numeric_limits<double>::infinity();
        double cell_max = -std::numeric_limits<double>::infinity();
        for (size_t band = 0; band < ndof; ++band) {
            cell_min = std::min(cell_min, cell.band_min[band]);
            cell_max = std::max(cell_max, cell.band_max[band]);
        }
        if (cell_max <= mu) {
            charge += cell.volume * static_cast<double>(ndof);
            return;
        }
        if (cell_min > mu) {
            return;
        }

        for (size_t band = 0; band < ndof; ++band) {
            const bool full = cell.band_max[band] <= mu;
            const bool empty = cell.band_min[band] > mu;
            const bool flat_half =
                cell.flat_mask[band] &&
                !full &&
                !empty &&
                std::abs(cell.flat_energy[band] - mu) <= tol_;
            if (full) {
                charge += cell.volume;
                continue;
            }
            if (empty) {
                continue;
            }
            if (flat_half) {
                charge += 0.5 * cell.volume;
                continue;
            }
            if (cell.distinct_mask[band]) {
                const double *energies = cell.sorted_energies.data() + band * n_vertices;
                const double *weights =
                    geometry_->ndim_ > 3 ? cell.simplex_weights.data() + band * n_vertices : nullptr;
                const auto [fraction, dfraction] = simplex_fraction_and_derivative(
                    energies,
                    mu,
                    geometry_->ndim_,
                    weights,
                    tol_
                );
                charge += cell.volume * fraction;
                derivative += cell.volume * dfraction;
                continue;
            }

            derivative_exact = false;
            std::vector<double> band_energies(n_vertices);
            for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
                band_energies[vertex] = cell.vertex_energies[vertex * ndof + band];
            }
            auto pieces = occupied_subsimplices_from_flat(
                cell.points_flat.data(),
                band_energies.data(),
                n_vertices,
                geometry_->ndim_,
                mu,
                tol_
            );
            for (const auto &piece : pieces) {
                charge += simplex_volume_from_flat(piece, n_vertices, geometry_->ndim_);
            }
        }
    }

    NativeGeometry *geometry_ = nullptr;
    NativeSpectralCache *spectral_cache_ = nullptr;
    std::int64_t refine_levels_ = 0;
    double tol_ = 1e-14;
    std::unordered_map<std::int64_t, std::vector<double>> vertex_values_cache_;
    std::unordered_map<std::int64_t, PreparedChargeCell> prepared_cell_cache_;
    std::unordered_map<GroupKey, PreparedChargeGroup, GroupKeyHash> prepared_group_cache_;
};

class NativeDensityEvaluator {
public:
    NativeDensityEvaluator(
        NativeGeometry &geometry,
        NativeSpectralCache &spectral_cache,
        Float2D keys,
        double tol = 1e-14
    )
        : geometry_(&geometry),
          spectral_cache_(&spectral_cache),
          tol_(tol),
          n_keys_(keys.shape(0)),
          ndim_(keys.shape(1)),
          ndof_(spectral_cache.ndof()),
          ncomp_(spectral_cache.ndof() * spectral_cache.ndof() * keys.shape(0)) {
        keys_.assign(keys.data(), keys.data() + n_keys_ * ndim_);
    }

    void clear() {
        value_cache_.clear();
        current_mu_ = std::numeric_limits<double>::quiet_NaN();
    }

    nb::tuple evaluate_many(Int1D simplex_ids, double mu) {
        ensure_mu(mu);
        const auto count = simplex_ids.shape(0);
        const auto *ids = simplex_ids.data();
        std::vector<std::complex<double>> estimates(count * ncomp_);
        std::vector<double> error_vectors(count * ncomp_);
        std::vector<double> error_scalars(count);
        std::int64_t total_evaluator_evals = 0;

        for (size_t index = 0; index < count; ++index) {
            const auto &cached = cached_value(ids[index]);
            std::copy_n(cached.estimate.data(), ncomp_, estimates.data() + index * ncomp_);
            std::copy_n(cached.error_vector.data(), ncomp_, error_vectors.data() + index * ncomp_);
            error_scalars[index] = cached.error_scalar;
            total_evaluator_evals += cached.evaluator_evals;
        }

        return nb::make_tuple(
            make_array(std::move(estimates), {count, ncomp_}),
            make_array(std::move(error_vectors), {count, ncomp_}),
            make_array(std::move(error_scalars), {count}),
            total_evaluator_evals
        );
    }

private:
    struct CachedDensityValue {
        std::vector<std::complex<double>> estimate;
        std::vector<double> error_vector;
        double error_scalar = 0.0;
        std::int64_t evaluator_evals = 0;
    };

    struct PointSpectrum {
        std::vector<double> eigenvalues;
        std::vector<std::complex<double>> eigenvectors;
    };

    void ensure_mu(double mu) {
        if (std::isnan(current_mu_) || std::abs(mu - current_mu_) > tol_) {
            current_mu_ = mu;
            value_cache_.clear();
        }
    }

    const CachedDensityValue &cached_value(std::int64_t simplex_id) {
        auto it = value_cache_.find(simplex_id);
        if (it != value_cache_.end()) {
            return it->second;
        }
        auto [inserted, _ok] = value_cache_.emplace(simplex_id, evaluate_simplex(simplex_id));
        return inserted->second;
    }

    const std::vector<double> &vertex_eigenvalues(std::int64_t vertex_id) {
        ensure_vertex_value_capacity(vertex_id);
        if (vertex_value_ready_[vertex_id]) {
            return vertex_values_[vertex_id];
        }
        const auto &entry = vertex_entry(vertex_id);
        vertex_values_[vertex_id] = entry.eigenvalues;
        vertex_value_ready_[vertex_id] = 1;
        return vertex_values_[vertex_id];
    }

    const std::vector<std::complex<double>> &vertex_tables(std::int64_t vertex_id) {
        ensure_vertex_table_capacity(vertex_id);
        if (vertex_table_ready_[vertex_id]) {
            return vertex_tables_[vertex_id];
        }
        std::vector<double> reduced_point(ndim_);
        const size_t offset = static_cast<size_t>(vertex_id) * ndim_;
        std::copy_n(geometry_->vertices_.data() + offset, ndim_, reduced_point.data());
        const auto &entry = vertex_entry(vertex_id);
        vertex_tables_[vertex_id] = density_tables_for_point(
            reduced_point.data(),
            entry.eigenvectors.data(),
            ndof_,
            nullptr,
            ndof_
        );
        vertex_table_ready_[vertex_id] = 1;
        return vertex_tables_[vertex_id];
    }

    void ensure_vertex_value_capacity(std::int64_t vertex_id) {
        const size_t needed = static_cast<size_t>(vertex_id) + 1;
        if (vertex_values_.size() >= needed) {
            return;
        }
        vertex_values_.resize(needed);
        vertex_value_ready_.resize(needed, 0);
    }

    void ensure_vertex_table_capacity(std::int64_t vertex_id) {
        const size_t needed = static_cast<size_t>(vertex_id) + 1;
        if (vertex_tables_.size() >= needed) {
            return;
        }
        vertex_tables_.resize(needed);
        vertex_table_ready_.resize(needed, 0);
    }

    const NativeSpectralCache::CacheEntry &vertex_entry(std::int64_t vertex_id) {
        std::vector<double> k_point(ndim_);
        const size_t base = static_cast<size_t>(vertex_id) * ndim_;
        for (size_t axis = 0; axis < ndim_; ++axis) {
            k_point[axis] = 2.0 * kPi * geometry_->vertices_[base + axis] - kPi;
        }
        const std::string key = point_key_from_ptr(k_point.data(), ndim_, spectral_cache_->tol_);
        auto it = spectral_cache_->cache_.find(key);
        if (it == spectral_cache_->cache_.end()) {
            auto [inserted, _ok] = spectral_cache_->cache_.emplace(key, spectral_cache_->diagonalize_point(k_point.data()));
            it = inserted;
        }
        return it->second;
    }

    PointSpectrum uncached_point_spectrum(const double *reduced_point) {
        std::vector<double> k_point(ndim_);
        for (size_t axis = 0; axis < ndim_; ++axis) {
            k_point[axis] = 2.0 * kPi * reduced_point[axis] - kPi;
        }
        const auto entry = spectral_cache_->diagonalize_point(k_point.data());
        return PointSpectrum{entry.eigenvalues, entry.eigenvectors};
    }

    std::vector<std::complex<double>> density_tables_for_point(
        const double *reduced_point,
        const std::complex<double> *eigenvectors,
        size_t n_all_bands,
        const std::int64_t *selected_bands,
        size_t n_selected_bands
    ) const {
        std::vector<std::complex<double>> out(n_selected_bands * ncomp_);
        std::vector<std::complex<double>> phases(n_keys_);
        for (size_t key_index = 0; key_index < n_keys_; ++key_index) {
            double phase_arg = 0.0;
            for (size_t axis = 0; axis < ndim_; ++axis) {
                phase_arg +=
                    (2.0 * kPi * reduced_point[axis] - kPi) *
                    keys_[key_index * ndim_ + axis];
            }
            phases[key_index] = std::exp(std::complex<double>(0.0, phase_arg));
        }

        for (size_t band_index = 0; band_index < n_selected_bands; ++band_index) {
            const size_t band = selected_bands ? static_cast<size_t>(selected_bands[band_index]) : band_index;
            for (size_t i = 0; i < ndof_; ++i) {
                const std::complex<double> ui = eigenvectors[i * n_all_bands + band];
                for (size_t j = 0; j < ndof_; ++j) {
                    const std::complex<double> projector =
                        ui * std::conj(eigenvectors[j * n_all_bands + band]);
                    const size_t base =
                        ((band_index * ndof_ + i) * ndof_ + j) * n_keys_;
                    for (size_t key_index = 0; key_index < n_keys_; ++key_index) {
                        out[base + key_index] = projector * phases[key_index];
                    }
                }
            }
        }
        return out;
    }

    CachedDensityValue evaluate_simplex(std::int64_t simplex_id) {
        CachedDensityValue result;
        result.estimate.assign(ncomp_, std::complex<double>(0.0, 0.0));
        result.error_vector.assign(ncomp_, 0.0);

        const auto &record = geometry_->simplex_records_[static_cast<size_t>(simplex_id)];
        const size_t n_vertices = record.vertex_ids.size();
        const double volume = geometry_->simplex_volume(simplex_id);
        const std::vector<double> points_flat = geometry_->simplex_points_flat(simplex_id);
        std::vector<double> centroid(ndim_, 0.0);
        for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
            for (size_t axis = 0; axis < ndim_; ++axis) {
                centroid[axis] += points_flat[vertex * ndim_ + axis];
            }
        }
        for (size_t axis = 0; axis < ndim_; ++axis) {
            centroid[axis] /= static_cast<double>(n_vertices);
        }

        std::vector<double> vertex_energies(n_vertices * ndof_);
        for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
            const auto vertex_id = record.vertex_ids[vertex];
            const auto &values = vertex_eigenvalues(vertex_id);
            std::copy_n(values.data(), ndof_, vertex_energies.data() + vertex * ndof_);
        }

        const auto centroid_spectrum = uncached_point_spectrum(centroid.data());
        const auto centroid_tables_all = density_tables_for_point(
            centroid.data(),
            centroid_spectrum.eigenvectors.data(),
            ndof_,
            nullptr,
            ndof_
        );

        auto occupation = [&](double energy) noexcept -> double {
            if (std::abs(energy - current_mu_) <= tol_) {
                return 0.5;
            }
            return energy < current_mu_ ? 1.0 : 0.0;
        };

        std::vector<std::complex<double>> estimate_low(ncomp_, std::complex<double>(0.0, 0.0));
        std::vector<std::complex<double>> estimate_high(ncomp_, std::complex<double>(0.0, 0.0));
        std::int64_t evaluator_evals = 0;
        std::vector<double> band_energies(n_vertices);

        for (size_t band = 0; band < ndof_; ++band) {
            double band_min = std::numeric_limits<double>::infinity();
            double band_max = -std::numeric_limits<double>::infinity();
            bool half_mask = true;
            bool vertex_matches_centroid = true;
            const double centroid_energy = centroid_spectrum.eigenvalues[band];
            const double centroid_occ = occupation(centroid_energy);

            for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
                const double energy = vertex_energies[vertex * ndof_ + band];
                band_energies[vertex] = energy;
                band_min = std::min(band_min, energy);
                band_max = std::max(band_max, energy);
                if (std::abs(energy - current_mu_) > tol_) {
                    half_mask = false;
                }
                if (occupation(energy) != centroid_occ) {
                    vertex_matches_centroid = false;
                }
            }

            const bool full_mask = (band_max <= current_mu_) && vertex_matches_centroid && !half_mask;
            const bool empty_mask = (band_min > current_mu_) && vertex_matches_centroid && !half_mask;
            if (half_mask || full_mask) {
                const double weight = half_mask ? 0.5 : 1.0;
                for (size_t comp = 0; comp < ncomp_; ++comp) {
                    estimate_low[comp] += volume * weight * centroid_tables_all[band * ncomp_ + comp];
                    std::complex<double> avg(0.0, 0.0);
                    for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
                        const auto &tables = vertex_tables(record.vertex_ids[vertex]);
                        avg += tables[band * ncomp_ + comp];
                    }
                    estimate_high[comp] += volume * weight * avg / static_cast<double>(n_vertices);
                }
                evaluator_evals += static_cast<std::int64_t>(n_vertices);
                continue;
            }
            if (empty_mask) {
                continue;
            }

            auto pieces = occupied_subsimplices_from_flat(
                points_flat.data(),
                band_energies.data(),
                n_vertices,
                ndim_,
                current_mu_,
                tol_
            );
            if (pieces.empty()) {
                for (size_t comp = 0; comp < ncomp_; ++comp) {
                    estimate_low[comp] += volume * centroid_occ * centroid_tables_all[band * ncomp_ + comp];
                    std::complex<double> avg(0.0, 0.0);
                    for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
                        const auto &tables = vertex_tables(record.vertex_ids[vertex]);
                        avg += occupation(vertex_energies[vertex * ndof_ + band]) * tables[band * ncomp_ + comp];
                    }
                    estimate_high[comp] += volume * avg / static_cast<double>(n_vertices);
                }
                evaluator_evals += static_cast<std::int64_t>(n_vertices);
                continue;
            }

            evaluator_evals += static_cast<std::int64_t>(pieces.size());
            for (const auto &piece : pieces) {
                const double piece_volume = simplex_volume_from_flat(piece, n_vertices, ndim_);
                std::vector<double> piece_centroid(ndim_, 0.0);
                for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
                    for (size_t axis = 0; axis < ndim_; ++axis) {
                        piece_centroid[axis] += piece[vertex * ndim_ + axis];
                    }
                }
                for (size_t axis = 0; axis < ndim_; ++axis) {
                    piece_centroid[axis] /= static_cast<double>(n_vertices);
                }
                const auto piece_centroid_spectrum = uncached_point_spectrum(piece_centroid.data());
                const std::int64_t band_id = static_cast<std::int64_t>(band);
                const auto piece_centroid_table = density_tables_for_point(
                    piece_centroid.data(),
                    piece_centroid_spectrum.eigenvectors.data(),
                    ndof_,
                    &band_id,
                    1
                );
                std::vector<std::complex<double>> piece_vertex_tables_flat(n_vertices * ncomp_);
                for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
                    const double *piece_vertex = piece.data() + vertex * ndim_;
                    const auto piece_vertex_spectrum = uncached_point_spectrum(piece_vertex);
                    const auto piece_vertex_table = density_tables_for_point(
                        piece_vertex,
                        piece_vertex_spectrum.eigenvectors.data(),
                        ndof_,
                        &band_id,
                        1
                    );
                    std::copy_n(
                        piece_vertex_table.data(),
                        ncomp_,
                        piece_vertex_tables_flat.data() + vertex * ncomp_
                    );
                }
                for (size_t comp = 0; comp < ncomp_; ++comp) {
                    estimate_low[comp] += piece_volume * piece_centroid_table[comp];
                    std::complex<double> avg(0.0, 0.0);
                    for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
                        avg += piece_vertex_tables_flat[vertex * ncomp_ + comp];
                    }
                    estimate_high[comp] += piece_volume * avg / static_cast<double>(n_vertices);
                }
                evaluator_evals += static_cast<std::int64_t>(n_vertices);
            }
        }

        double error_norm_sq = 0.0;
        for (size_t comp = 0; comp < ncomp_; ++comp) {
            const std::complex<double> diff = estimate_high[comp] - estimate_low[comp];
            result.estimate[comp] = estimate_high[comp];
            result.error_vector[comp] = std::abs(diff);
            error_norm_sq += result.error_vector[comp] * result.error_vector[comp];
        }
        result.error_scalar = std::sqrt(error_norm_sq);
        result.evaluator_evals = evaluator_evals;
        return result;
    }

    NativeGeometry *geometry_ = nullptr;
    NativeSpectralCache *spectral_cache_ = nullptr;
    double tol_ = 1e-14;
    size_t n_keys_ = 0;
    size_t ndim_ = 0;
    size_t ndof_ = 0;
    size_t ncomp_ = 0;
    std::vector<double> keys_;
    double current_mu_ = std::numeric_limits<double>::quiet_NaN();
    std::vector<std::vector<double>> vertex_values_;
    std::vector<std::uint8_t> vertex_value_ready_;
    std::vector<std::vector<std::complex<double>>> vertex_tables_;
    std::vector<std::uint8_t> vertex_table_ready_;
    std::unordered_map<std::int64_t, CachedDensityValue> value_cache_;
};

nb::bytes point_key_bytes(Float1D point, double tol) {
    std::string key = point_key_from_ptr(point.data(), point.shape(0), tol);
    return nb::bytes(key.data(), key.size());
}

nb::tuple prepare_charge_batch_metadata(Float3D vertex_energies, Float1D volumes, Int1D owner_ids, double tol) {
    const auto n_cells = vertex_energies.shape(0);
    const auto n_vertices = vertex_energies.shape(1);
    const auto n_bands = vertex_energies.shape(2);
    const auto dimension = n_vertices - 1;

    if (volumes.shape(0) != n_cells || owner_ids.shape(0) != n_cells) {
        throw std::runtime_error("prepare_charge_batch_metadata: incompatible leading dimensions");
    }

    const double *values = vertex_energies.data();
    std::vector<double> sorted_energies(n_cells * n_bands * n_vertices);
    std::vector<double> simplex_weights(n_cells * n_bands * n_vertices, 0.0);
    std::vector<std::uint8_t> distinct_mask(n_cells * n_bands, 1);
    std::vector<double> band_min(n_cells * n_bands);
    std::vector<double> band_max(n_cells * n_bands);
    std::vector<double> flat_energy(n_cells * n_bands);
    std::vector<std::uint8_t> flat_mask(n_cells * n_bands, 0);
    std::vector<double> cell_min(n_cells);
    std::vector<double> cell_max(n_cells);
    std::vector<double> scratch(n_vertices);

    for (size_t cell = 0; cell < n_cells; ++cell) {
        double local_min = std::numeric_limits<double>::infinity();
        double local_max = -std::numeric_limits<double>::infinity();
        for (size_t band = 0; band < n_bands; ++band) {
            const size_t band_offset = cell * n_vertices * n_bands + band;
            for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
                scratch[vertex] = values[cell * n_vertices * n_bands + vertex * n_bands + band];
            }
            std::sort(scratch.begin(), scratch.end());
            double bmin = scratch.front();
            double bmax = scratch.back();
            band_min[cell * n_bands + band] = bmin;
            band_max[cell * n_bands + band] = bmax;
            flat_energy[cell * n_bands + band] = values[band_offset];
            flat_mask[cell * n_bands + band] = (bmax - bmin) <= tol ? 1 : 0;
            local_min = std::min(local_min, bmin);
            local_max = std::max(local_max, bmax);

            bool distinct = true;
            for (size_t vertex = 1; vertex < n_vertices; ++vertex) {
                if (scratch[vertex] - scratch[vertex - 1] <= tol) {
                    distinct = false;
                }
                sorted_energies[(cell * n_bands + band) * n_vertices + vertex - 1] = scratch[vertex - 1];
            }
            sorted_energies[(cell * n_bands + band) * n_vertices + n_vertices - 1] = scratch[n_vertices - 1];
            distinct_mask[cell * n_bands + band] = distinct ? 1 : 0;

            if (dimension > 3 && distinct) {
                for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
                    double denom = 1.0;
                    for (size_t other = 0; other < n_vertices; ++other) {
                        if (other == vertex) {
                            continue;
                        }
                        denom *= scratch[vertex] - scratch[other];
                    }
                    simplex_weights[(cell * n_bands + band) * n_vertices + vertex] = 1.0 / denom;
                }
            }
        }
        cell_min[cell] = local_min;
        cell_max[cell] = local_max;
    }

    std::vector<std::int64_t> owner_inverse;
    std::vector<std::int64_t> owner_unique = owner_unique_from_runs(owner_ids.data(), n_cells, owner_inverse);

    return nb::make_tuple(
        make_array(std::move(sorted_energies), {n_cells, n_bands, n_vertices}),
        make_array(std::move(simplex_weights), {n_cells, n_bands, n_vertices}),
        make_array(std::move(distinct_mask), {n_cells, n_bands}),
        make_array(std::move(band_min), {n_cells, n_bands}),
        make_array(std::move(band_max), {n_cells, n_bands}),
        make_array(std::move(flat_energy), {n_cells, n_bands}),
        make_array(std::move(flat_mask), {n_cells, n_bands}),
        make_array(std::move(cell_min), {n_cells}),
        make_array(std::move(cell_max), {n_cells}),
        make_array(std::move(owner_unique), {owner_unique.size()}),
        make_array(std::move(owner_inverse), {n_cells})
    );
}

nb::tuple unique_first_indices_int64(Int1D values) {
    const auto count = values.shape(0);
    const std::int64_t *ptr = values.data();

    std::unordered_map<std::int64_t, std::int64_t> seen;
    seen.reserve(count);

    std::vector<std::int64_t> unique_values;
    std::vector<std::int64_t> first_indices;
    unique_values.reserve(count);
    first_indices.reserve(count);

    for (size_t index = 0; index < count; ++index) {
        const std::int64_t value = ptr[index];
        auto [it, inserted] = seen.emplace(value, static_cast<std::int64_t>(unique_values.size()));
        if (inserted) {
            unique_values.push_back(value);
            first_indices.push_back(static_cast<std::int64_t>(index));
        }
    }

    return nb::make_tuple(
        make_array(std::move(unique_values), {first_indices.size()}),
        make_array(std::move(first_indices), {first_indices.size()})
    );
}

nb::tuple plan_vertex_prefetch(Int1D vertex_ids, Bool1D ready_mask) {
    const auto count = vertex_ids.shape(0);
    const auto ready_size = ready_mask.shape(0);
    const std::int64_t *ids = vertex_ids.data();
    const bool *ready = ready_mask.data();

    std::unordered_map<std::int64_t, std::int64_t> seen;
    seen.reserve(count);

    std::vector<std::int64_t> unique_ids;
    std::vector<std::int64_t> first_indices;
    std::vector<std::int64_t> missing_ids;
    std::vector<std::int64_t> missing_first_indices;
    unique_ids.reserve(count);
    first_indices.reserve(count);

    for (size_t index = 0; index < count; ++index) {
        const std::int64_t vertex_id = ids[index];
        auto [it, inserted] = seen.emplace(vertex_id, static_cast<std::int64_t>(unique_ids.size()));
        if (!inserted) {
            continue;
        }
        unique_ids.push_back(vertex_id);
        first_indices.push_back(static_cast<std::int64_t>(index));
        if (vertex_id < 0 || static_cast<size_t>(vertex_id) >= ready_size || !ready[vertex_id]) {
            missing_ids.push_back(vertex_id);
            missing_first_indices.push_back(static_cast<std::int64_t>(index));
        }
    }

    return nb::make_tuple(
        make_array(std::move(unique_ids), {first_indices.size()}),
        make_array(std::move(first_indices), {first_indices.size()}),
        make_array(std::move(missing_ids), {missing_first_indices.size()}),
        make_array(std::move(missing_first_indices), {missing_first_indices.size()})
    );
}

nb::ndarray<nb::numpy, double> gather_vertex_values(Float2D values, Int1D vertex_ids) {
    const auto n_rows = vertex_ids.shape(0);
    const auto ndof = values.shape(1);
    const double *src = values.data();
    const std::int64_t *ids = vertex_ids.data();

    std::vector<double> out(n_rows * ndof);
    for (size_t row = 0; row < n_rows; ++row) {
        const std::int64_t vertex_id = ids[row];
        const double *src_row = src + static_cast<size_t>(vertex_id) * ndof;
        std::copy_n(src_row, ndof, out.data() + row * ndof);
    }
    return make_array(std::move(out), {n_rows, ndof});
}

nb::ndarray<nb::numpy, std::complex<double>> gather_vertex_tables(
    Complex3D tables,
    Int1D vertex_ids,
    nb::object bands_obj
) {
    const auto n_rows = vertex_ids.shape(0);
    const auto ndof = tables.shape(1);
    const auto ncomp = tables.shape(2);
    const std::complex<double> *src = tables.data();
    const std::int64_t *ids = vertex_ids.data();

    if (bands_obj.is_none()) {
        std::vector<std::complex<double>> out(n_rows * ndof * ncomp);
        for (size_t row = 0; row < n_rows; ++row) {
            const size_t vertex_id = static_cast<size_t>(ids[row]);
            const std::complex<double> *src_row = src + vertex_id * ndof * ncomp;
            std::copy_n(src_row, ndof * ncomp, out.data() + row * ndof * ncomp);
        }
        return make_array(std::move(out), {n_rows, ndof, ncomp});
    }

    auto bands = nb::cast<Int1D>(bands_obj);
    const auto nbands = bands.shape(0);
    const std::int64_t *band_ids = bands.data();
    std::vector<std::complex<double>> out(n_rows * nbands * ncomp);
    for (size_t row = 0; row < n_rows; ++row) {
        const size_t vertex_id = static_cast<size_t>(ids[row]);
        for (size_t band_index = 0; band_index < nbands; ++band_index) {
            const size_t band = static_cast<size_t>(band_ids[band_index]);
            const std::complex<double> *src_band = src + (vertex_id * ndof + band) * ncomp;
            std::copy_n(src_band, ncomp, out.data() + (row * nbands + band_index) * ncomp);
        }
    }
    return make_array(std::move(out), {n_rows, nbands, ncomp});
}

nb::tuple prepare_density_band_metadata(
    Float3D vertex_energies,
    Float2D centroid_energies,
    double mu,
    double tol
) {
    const auto n_simplex = vertex_energies.shape(0);
    const auto n_vertices = vertex_energies.shape(1);
    const auto n_bands = vertex_energies.shape(2);

    if (centroid_energies.shape(0) != n_simplex || centroid_energies.shape(1) != n_bands) {
        throw std::runtime_error("prepare_density_band_metadata: incompatible shapes");
    }

    const double *vertex_ptr = vertex_energies.data();
    const double *centroid_ptr = centroid_energies.data();

    std::vector<std::int64_t> whole_counts(n_simplex, 0);
    std::vector<std::int64_t> candidate_counts(n_simplex, 0);
    std::vector<std::int64_t> whole_bands;
    std::vector<double> whole_weights;
    std::vector<std::int64_t> candidate_bands;
    whole_bands.reserve(n_simplex * n_bands);
    whole_weights.reserve(n_simplex * n_bands);
    candidate_bands.reserve(n_simplex * n_bands);

    auto occupation = [mu, tol](double energy) noexcept -> double {
        if (std::abs(energy - mu) <= tol) {
            return 0.5;
        }
        return energy < mu ? 1.0 : 0.0;
    };

    for (size_t simplex = 0; simplex < n_simplex; ++simplex) {
        for (size_t band = 0; band < n_bands; ++band) {
            double band_min = std::numeric_limits<double>::infinity();
            double band_max = -std::numeric_limits<double>::infinity();
            bool half_mask = true;
            bool vertex_matches_centroid = true;
            const double centroid_occ = occupation(centroid_ptr[simplex * n_bands + band]);

            for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
                const double energy = vertex_ptr[(simplex * n_vertices + vertex) * n_bands + band];
                band_min = std::min(band_min, energy);
                band_max = std::max(band_max, energy);
                if (std::abs(energy - mu) > tol) {
                    half_mask = false;
                }
                if (occupation(energy) != centroid_occ) {
                    vertex_matches_centroid = false;
                }
            }

            const bool full_mask = (band_max <= mu) && vertex_matches_centroid && !half_mask;
            const bool empty_mask = (band_min > mu) && vertex_matches_centroid && !half_mask;
            if (half_mask || full_mask) {
                whole_bands.push_back(static_cast<std::int64_t>(band));
                whole_weights.push_back(half_mask ? 0.5 : 1.0);
                ++whole_counts[simplex];
                continue;
            }
            if (empty_mask) {
                continue;
            }

            candidate_bands.push_back(static_cast<std::int64_t>(band));
            ++candidate_counts[simplex];
        }
    }

    return nb::make_tuple(
        make_array(std::move(whole_counts), {n_simplex}),
        make_array(std::move(whole_bands), {whole_bands.size()}),
        make_array(std::move(whole_weights), {whole_weights.size()}),
        make_array(std::move(candidate_counts), {n_simplex}),
        make_array(std::move(candidate_bands), {candidate_bands.size()})
    );
}

nb::tuple prepare_density_piece_metadata(
    Float3D simplex_points,
    Float3D vertex_energies,
    Int1D candidate_offsets,
    Int1D candidate_bands,
    double mu,
    double tol
) {
    const auto n_simplex = simplex_points.shape(0);
    const auto n_vertices = simplex_points.shape(1);
    const auto ndim = simplex_points.shape(2);
    const auto n_bands = vertex_energies.shape(2);

    if (vertex_energies.shape(0) != n_simplex || vertex_energies.shape(1) != n_vertices) {
        throw std::runtime_error("prepare_density_piece_metadata: incompatible simplex/value shapes");
    }
    if (candidate_offsets.shape(0) != n_simplex + 1) {
        throw std::runtime_error("prepare_density_piece_metadata: bad candidate_offsets length");
    }

    const double *points_ptr = simplex_points.data();
    const double *energy_ptr = vertex_energies.data();
    const std::int64_t *candidate_offset_ptr = candidate_offsets.data();
    const std::int64_t *candidate_band_ptr = candidate_bands.data();

    std::vector<std::int64_t> step_counts(n_simplex, 0);
    std::vector<std::int64_t> step_bands;
    std::vector<std::int64_t> piece_counts(n_simplex, 0);
    std::vector<std::int64_t> piece_bands;
    std::vector<double> piece_volumes;
    std::vector<double> piece_centroid_points;
    std::vector<std::int64_t> piece_vertex_offsets{0};
    std::vector<double> piece_vertex_points;
    std::vector<double> energy_scratch(n_vertices);

    for (size_t simplex = 0; simplex < n_simplex; ++simplex) {
        const std::int64_t start = candidate_offset_ptr[simplex];
        const std::int64_t stop = candidate_offset_ptr[simplex + 1];
        const double *simplex_point_ptr = points_ptr + simplex * n_vertices * ndim;

        for (std::int64_t flat_index = start; flat_index < stop; ++flat_index) {
            const std::int64_t band = candidate_band_ptr[flat_index];
            for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
                energy_scratch[vertex] =
                    energy_ptr[(simplex * n_vertices + vertex) * n_bands + static_cast<size_t>(band)];
            }
            auto pieces = occupied_subsimplices_from_flat(
                simplex_point_ptr,
                energy_scratch.data(),
                n_vertices,
                ndim,
                mu,
                tol
            );
            if (pieces.empty()) {
                step_bands.push_back(band);
                ++step_counts[simplex];
                continue;
            }

            for (const auto &piece : pieces) {
                piece_bands.push_back(band);
                piece_volumes.push_back(simplex_volume_from_flat(piece, n_vertices, ndim));
                std::vector<double> centroid(ndim, 0.0);
                for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
                    for (size_t axis = 0; axis < ndim; ++axis) {
                        centroid[axis] += piece[vertex * ndim + axis];
                    }
                }
                for (size_t axis = 0; axis < ndim; ++axis) {
                    centroid[axis] /= static_cast<double>(n_vertices);
                }
                piece_centroid_points.insert(
                    piece_centroid_points.end(),
                    centroid.begin(),
                    centroid.end()
                );
                piece_vertex_points.insert(piece_vertex_points.end(), piece.begin(), piece.end());
                piece_vertex_offsets.push_back(
                    piece_vertex_offsets.back() + static_cast<std::int64_t>(n_vertices)
                );
                ++piece_counts[simplex];
            }
        }
    }

    const size_t n_pieces = piece_volumes.size();
    const size_t total_piece_vertices = piece_vertex_offsets.back();
    const size_t n_piece_offsets = piece_vertex_offsets.size();

    return nb::make_tuple(
        make_array(std::move(step_counts), {n_simplex}),
        make_array(std::move(step_bands), {step_bands.size()}),
        make_array(std::move(piece_counts), {n_simplex}),
        make_array(std::move(piece_bands), {piece_bands.size()}),
        make_array(std::move(piece_volumes), {n_pieces}),
        make_array(std::move(piece_centroid_points), {n_pieces, ndim}),
        make_array(std::move(piece_vertex_offsets), {n_piece_offsets}),
        make_array(std::move(piece_vertex_points), {total_piece_vertices, ndim})
    );
}

nb::tuple prepare_density_cells_metadata(
    Float3D simplex_points,
    Float3D vertex_energies,
    Float2D centroid_energies,
    double mu,
    double tol
) {
    const auto n_simplex = simplex_points.shape(0);
    const auto n_vertices = simplex_points.shape(1);
    const auto ndim = simplex_points.shape(2);
    const auto n_bands = vertex_energies.shape(2);

    if (vertex_energies.shape(0) != n_simplex || vertex_energies.shape(1) != n_vertices) {
        throw std::runtime_error("prepare_density_cells_metadata: incompatible simplex/value shapes");
    }
    if (centroid_energies.shape(0) != n_simplex || centroid_energies.shape(1) != n_bands) {
        throw std::runtime_error("prepare_density_cells_metadata: incompatible centroid/value shapes");
    }

    const double *points_ptr = simplex_points.data();
    const double *vertex_ptr = vertex_energies.data();
    const double *centroid_ptr = centroid_energies.data();

    std::vector<std::int64_t> whole_counts(n_simplex, 0);
    std::vector<std::int64_t> step_counts(n_simplex, 0);
    std::vector<std::int64_t> piece_counts(n_simplex, 0);
    std::vector<std::int64_t> whole_bands;
    std::vector<double> whole_weights;
    std::vector<std::int64_t> step_bands;
    std::vector<std::int64_t> piece_bands;
    std::vector<double> piece_volumes;
    std::vector<double> piece_centroid_points;
    std::vector<std::int64_t> piece_vertex_offsets{0};
    std::vector<double> piece_vertex_points;
    std::vector<double> energy_scratch(n_vertices);

    auto occupation = [mu, tol](double energy) noexcept -> double {
        if (std::abs(energy - mu) <= tol) {
            return 0.5;
        }
        return energy < mu ? 1.0 : 0.0;
    };

    for (size_t simplex = 0; simplex < n_simplex; ++simplex) {
        const double *simplex_point_ptr = points_ptr + simplex * n_vertices * ndim;
        for (size_t band = 0; band < n_bands; ++band) {
            double band_min = std::numeric_limits<double>::infinity();
            double band_max = -std::numeric_limits<double>::infinity();
            bool half_mask = true;
            bool vertex_matches_centroid = true;
            const double centroid_occ = occupation(centroid_ptr[simplex * n_bands + band]);

            for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
                const double energy = vertex_ptr[(simplex * n_vertices + vertex) * n_bands + band];
                energy_scratch[vertex] = energy;
                band_min = std::min(band_min, energy);
                band_max = std::max(band_max, energy);
                if (std::abs(energy - mu) > tol) {
                    half_mask = false;
                }
                if (occupation(energy) != centroid_occ) {
                    vertex_matches_centroid = false;
                }
            }

            const bool full_mask = (band_max <= mu) && vertex_matches_centroid && !half_mask;
            const bool empty_mask = (band_min > mu) && vertex_matches_centroid && !half_mask;
            if (half_mask || full_mask) {
                whole_bands.push_back(static_cast<std::int64_t>(band));
                whole_weights.push_back(half_mask ? 0.5 : 1.0);
                ++whole_counts[simplex];
                continue;
            }
            if (empty_mask) {
                continue;
            }

            auto pieces = occupied_subsimplices_from_flat(
                simplex_point_ptr,
                energy_scratch.data(),
                n_vertices,
                ndim,
                mu,
                tol
            );
            if (pieces.empty()) {
                step_bands.push_back(static_cast<std::int64_t>(band));
                ++step_counts[simplex];
                continue;
            }

            for (const auto &piece : pieces) {
                piece_bands.push_back(static_cast<std::int64_t>(band));
                piece_volumes.push_back(simplex_volume_from_flat(piece, n_vertices, ndim));
                std::vector<double> centroid(ndim, 0.0);
                for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
                    for (size_t axis = 0; axis < ndim; ++axis) {
                        centroid[axis] += piece[vertex * ndim + axis];
                    }
                }
                for (size_t axis = 0; axis < ndim; ++axis) {
                    centroid[axis] /= static_cast<double>(n_vertices);
                }
                piece_centroid_points.insert(piece_centroid_points.end(), centroid.begin(), centroid.end());
                piece_vertex_points.insert(piece_vertex_points.end(), piece.begin(), piece.end());
                piece_vertex_offsets.push_back(piece_vertex_offsets.back() + static_cast<std::int64_t>(n_vertices));
                ++piece_counts[simplex];
            }
        }
    }

    std::vector<std::int64_t> whole_offsets(n_simplex + 1, 0);
    std::vector<std::int64_t> step_offsets(n_simplex + 1, 0);
    std::vector<std::int64_t> piece_offsets(n_simplex + 1, 0);
    for (size_t simplex = 0; simplex < n_simplex; ++simplex) {
        whole_offsets[simplex + 1] = whole_offsets[simplex] + whole_counts[simplex];
        step_offsets[simplex + 1] = step_offsets[simplex] + step_counts[simplex];
        piece_offsets[simplex + 1] = piece_offsets[simplex] + piece_counts[simplex];
    }

    const size_t n_pieces = piece_volumes.size();
    const size_t total_piece_vertices = piece_vertex_offsets.back();

    return nb::make_tuple(
        make_array(std::move(whole_offsets), {n_simplex + 1}),
        make_array(std::move(whole_bands), {whole_bands.size()}),
        make_array(std::move(whole_weights), {whole_weights.size()}),
        make_array(std::move(step_offsets), {n_simplex + 1}),
        make_array(std::move(step_bands), {step_bands.size()}),
        make_array(std::move(piece_offsets), {n_simplex + 1}),
        make_array(std::move(piece_bands), {piece_bands.size()}),
        make_array(std::move(piece_volumes), {n_pieces}),
        make_array(std::move(piece_centroid_points), {n_pieces, ndim}),
        make_array(std::move(piece_vertex_offsets), {piece_vertex_offsets.size()}),
        make_array(std::move(piece_vertex_points), {total_piece_vertices, ndim})
    );
}

nb::ndarray<nb::numpy, std::complex<double>> density_tables_from_eigenvectors(
    Float2D points,
    Complex3D eigenvectors,
    Float2D keys,
    nb::object bands_obj
) {
    const auto n_points = points.shape(0);
    const auto ndim = points.shape(1);
    const auto ndof = eigenvectors.shape(1);
    const auto n_all_bands = eigenvectors.shape(2);
    const auto n_keys = keys.shape(0);
    if (eigenvectors.shape(0) != n_points || keys.shape(1) != ndim) {
        throw std::runtime_error("density_tables_from_eigenvectors: incompatible shapes");
    }

    std::vector<std::int64_t> selected_bands;
    if (bands_obj.is_none()) {
        selected_bands.resize(n_all_bands);
        std::iota(selected_bands.begin(), selected_bands.end(), 0);
    } else {
        auto bands = nb::cast<Int1D>(bands_obj);
        selected_bands.assign(bands.data(), bands.data() + bands.shape(0));
    }

    const auto n_bands = selected_bands.size();
    std::vector<std::complex<double>> out(n_points * n_bands * ndof * ndof * n_keys);
    const double *point_ptr = points.data();
    const double *key_ptr = keys.data();
    const std::complex<double> *vector_ptr = eigenvectors.data();
    std::vector<std::complex<double>> phases(n_keys);

    for (size_t point_index = 0; point_index < n_points; ++point_index) {
        for (size_t key_index = 0; key_index < n_keys; ++key_index) {
            double phase_arg = 0.0;
            for (size_t axis = 0; axis < ndim; ++axis) {
                phase_arg +=
                    (2.0 * kPi * point_ptr[point_index * ndim + axis] - kPi) *
                    key_ptr[key_index * ndim + axis];
            }
            phases[key_index] = std::exp(std::complex<double>(0.0, phase_arg));
        }

        for (size_t band_index = 0; band_index < n_bands; ++band_index) {
            const size_t band = static_cast<size_t>(selected_bands[band_index]);
            for (size_t i = 0; i < ndof; ++i) {
                const std::complex<double> ui = vector_ptr[(point_index * ndof + i) * n_all_bands + band];
                for (size_t j = 0; j < ndof; ++j) {
                    const std::complex<double> projector =
                        ui * std::conj(vector_ptr[(point_index * ndof + j) * n_all_bands + band]);
                    const size_t base =
                        (((point_index * n_bands + band_index) * ndof + i) * ndof + j) * n_keys;
                    for (size_t key_index = 0; key_index < n_keys; ++key_index) {
                        out[base + key_index] = projector * phases[key_index];
                    }
                }
            }
        }
    }

    return make_array(std::move(out), {n_points, n_bands, ndof * ndof * n_keys});
}

nb::tuple accumulate_density_terms(
    Complex2D whole_centroid_tables,
    Complex3D whole_vertex_tables,
    Float1D whole_weights,
    Complex2D step_centroid_tables,
    Complex3D step_vertex_tables,
    Float1D step_centroid_occ,
    Float2D step_vertex_occ,
    double volume,
    Float1D piece_volumes,
    Complex2D piece_centroid_tables,
    Int1D piece_vertex_offsets,
    Complex2D piece_vertex_tables
) {
    const size_t ncomp =
        whole_centroid_tables.shape(1) ? whole_centroid_tables.shape(1) :
        (step_centroid_tables.shape(1) ? step_centroid_tables.shape(1) :
        (piece_centroid_tables.shape(1) ? piece_centroid_tables.shape(1) : 0));

    std::vector<std::complex<double>> low(ncomp, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> high(ncomp, std::complex<double>(0.0, 0.0));

    const size_t nwhole = whole_centroid_tables.shape(0);
    const size_t nwhole_vertices = whole_vertex_tables.shape(0);
    const auto *whole_centroid_ptr = whole_centroid_tables.data();
    const auto *whole_vertex_ptr = whole_vertex_tables.data();
    const auto *whole_weights_ptr = whole_weights.data();
    for (size_t band = 0; band < nwhole; ++band) {
        const double weight = whole_weights_ptr[band] * volume;
        for (size_t comp = 0; comp < ncomp; ++comp) {
            low[comp] += weight * whole_centroid_ptr[band * ncomp + comp];
            std::complex<double> avg(0.0, 0.0);
            for (size_t vertex = 0; vertex < nwhole_vertices; ++vertex) {
                avg += whole_vertex_ptr[(vertex * nwhole + band) * ncomp + comp];
            }
            high[comp] += weight * avg / static_cast<double>(nwhole_vertices);
        }
    }

    const size_t nstep = step_centroid_tables.shape(0);
    const size_t nstep_vertices = step_vertex_tables.shape(0);
    const auto *step_centroid_ptr = step_centroid_tables.data();
    const auto *step_vertex_ptr = step_vertex_tables.data();
    const auto *step_centroid_occ_ptr = step_centroid_occ.data();
    const auto *step_vertex_occ_ptr = step_vertex_occ.data();
    for (size_t band = 0; band < nstep; ++band) {
        const double low_weight = volume * step_centroid_occ_ptr[band];
        for (size_t comp = 0; comp < ncomp; ++comp) {
            low[comp] += low_weight * step_centroid_ptr[band * ncomp + comp];
            std::complex<double> avg(0.0, 0.0);
            for (size_t vertex = 0; vertex < nstep_vertices; ++vertex) {
                avg += step_vertex_occ_ptr[band * nstep_vertices + vertex] *
                       step_vertex_ptr[(vertex * nstep + band) * ncomp + comp];
            }
            high[comp] += volume * avg / static_cast<double>(nstep_vertices);
        }
    }

    const size_t npieces = piece_volumes.shape(0);
    const auto *piece_volumes_ptr = piece_volumes.data();
    const auto *piece_centroid_ptr = piece_centroid_tables.data();
    const auto *piece_vertex_offsets_ptr = piece_vertex_offsets.data();
    const auto *piece_vertex_ptr = piece_vertex_tables.data();
    for (size_t piece = 0; piece < npieces; ++piece) {
        const double piece_volume = piece_volumes_ptr[piece];
        const size_t start = static_cast<size_t>(piece_vertex_offsets_ptr[piece]);
        const size_t stop = static_cast<size_t>(piece_vertex_offsets_ptr[piece + 1]);
        const size_t count = stop - start;
        for (size_t comp = 0; comp < ncomp; ++comp) {
            low[comp] += piece_volume * piece_centroid_ptr[piece * ncomp + comp];
            std::complex<double> avg(0.0, 0.0);
            for (size_t vertex = start; vertex < stop; ++vertex) {
                avg += piece_vertex_ptr[vertex * ncomp + comp];
            }
            high[comp] += piece_volume * avg / static_cast<double>(count);
        }
    }

    return nb::make_tuple(
        make_array(std::move(low), {ncomp}),
        make_array(std::move(high), {ncomp})
    );
}

} // namespace

NB_MODULE(_zero_temp_native, m) {
    m.doc() = "Native helpers for meanfi.zero_temp";

    nb::class_<NativeRefinementDescriptor>(m, "NativeRefinementDescriptor")
        .def_prop_ro("parent_id", [](const NativeRefinementDescriptor &self) { return self.parent_id; })
        .def_prop_ro("child_ids", &NativeRefinementDescriptor::child_ids_array)
        .def_prop_ro("parent_vertex_ids", &NativeRefinementDescriptor::parent_vertex_ids_array)
        .def_prop_ro("child_vertex_ids", &NativeRefinementDescriptor::child_vertex_ids_array)
        .def_prop_ro("new_midpoint_vertex_id", [](const NativeRefinementDescriptor &self) {
            return self.new_midpoint_vertex_id;
        })
        .def_prop_ro("bisected_edge", &NativeRefinementDescriptor::bisected_edge_array);

    nb::class_<NativeGeometry>(m, "NativeGeometry")
        .def_static("root", &NativeGeometry::root, "ndim"_a, "root_subcells_per_axis"_a = 2, "tol"_a = 1e-14)
        .def_prop_ro("ndim", &NativeGeometry::ndim)
        .def_prop_ro("root_subcells_per_axis", &NativeGeometry::root_subcells_per_axis)
        .def_prop_ro("n_vertices", &NativeGeometry::n_vertices)
        .def_prop_ro("n_simplices", &NativeGeometry::n_simplices)
        .def_prop_ro("n_active", &NativeGeometry::n_active)
        .def("vertices_array", &NativeGeometry::vertices_array)
        .def("active_simplex_ids", &NativeGeometry::active_simplex_ids)
        .def("simplex_vertex_ids", &NativeGeometry::simplex_vertex_ids, "simplex_id"_a)
        .def("simplex_points", &NativeGeometry::simplex_points, "simplex_id"_a)
        .def("simplex_volume", &NativeGeometry::simplex_volume, "simplex_id"_a)
        .def("ensure_children", &NativeGeometry::ensure_children, "simplex_id"_a)
        .def("descendant_leaves", &NativeGeometry::descendant_leaves, "simplex_id"_a, "levels"_a)
        .def("refine", &NativeGeometry::refine, "marked_ids"_a);

    nb::class_<NativeFrontier>(m, "NativeFrontier")
        .def_static("from_geometry", &NativeFrontier::from_geometry, "geometry"_a)
        .def("sync_from_geometry", &NativeFrontier::sync_from_geometry)
        .def(
            "apply_refinement",
            &NativeFrontier::apply_refinement,
            "parent_ids"_a,
            "child_offsets"_a,
            "child_ids"_a
        )
        .def_prop_ro("n_active", &NativeFrontier::n_active)
        .def_prop_ro("generation", &NativeFrontier::generation)
        .def("active_simplex_ids", &NativeFrontier::active_simplex_ids)
        .def("vertex_ids", &NativeFrontier::vertex_ids)
        .def("volumes", &NativeFrontier::volumes);

    nb::class_<NativeChargeEvaluator>(m, "NativeChargeEvaluator")
        .def(
            nb::init<NativeGeometry &, NativeSpectralCache &, std::int64_t, double>(),
            "geometry"_a,
            "spectral_cache"_a,
            "refine_levels"_a = 0,
            "tol"_a = 1e-14
        )
        .def("evaluate", &NativeChargeEvaluator::evaluate, "frontier"_a, "mu"_a)
        .def(
            "simplex_charge",
            &NativeChargeEvaluator::simplex_charge,
            "simplex_id"_a,
            "mu"_a,
            "levels"_a = -1
        );

    nb::class_<NativeDensityEvaluator>(m, "NativeDensityEvaluator")
        .def(
            nb::init<NativeGeometry &, NativeSpectralCache &, Float2D, double>(),
            "geometry"_a,
            "spectral_cache"_a,
            "keys"_a,
            "tol"_a = 1e-14
        )
        .def("clear", &NativeDensityEvaluator::clear)
        .def("evaluate_many", &NativeDensityEvaluator::evaluate_many, "simplex_ids"_a, "mu"_a);

    nb::class_<NativeTightBindingModel>(m, "NativeTightBindingModel")
        .def(nb::init<Int2D, Complex3D>(), "keys"_a, "matrices"_a)
        .def_prop_ro("ndim", &NativeTightBindingModel::ndim)
        .def_prop_ro("ndof", &NativeTightBindingModel::ndof)
        .def_prop_ro("nterms", &NativeTightBindingModel::nterms)
        .def("keys_array", &NativeTightBindingModel::keys_array)
        .def("matrices_array", &NativeTightBindingModel::matrices_array)
        .def("evaluate_point", &NativeTightBindingModel::evaluate_point, "point"_a)
        .def("evaluate_many", &NativeTightBindingModel::evaluate_many, "points"_a);

    nb::class_<NativeSpectralCache>(m, "NativeSpectralCache")
        .def(nb::init<std::shared_ptr<NativeTightBindingModel>, double>(), "model"_a, "tol"_a = 1e-14)
        .def("clear", &NativeSpectralCache::clear)
        .def("invalidate", &NativeSpectralCache::invalidate)
        .def_prop_ro("generation", &NativeSpectralCache::generation)
        .def_prop_ro("ndim", &NativeSpectralCache::ndim)
        .def_prop_ro("ndof", &NativeSpectralCache::ndof)
        .def_prop_ro("size", &NativeSpectralCache::size)
        .def_prop_ro("n_kernel_evals", &NativeSpectralCache::n_kernel_evals)
        .def("evaluate_many", &NativeSpectralCache::evaluate_many, "points"_a)
        .def("get_many", &NativeSpectralCache::get_many, "points"_a)
        .def("get_many_values", &NativeSpectralCache::get_many_values, "points"_a);

    nb::class_<PointRegistry>(m, "PointRegistry")
        .def(nb::init<>())
        .def("register_many", &PointRegistry::register_many, "points"_a, "vertex_ids"_a = nb::none())
        .def("points_for_ids", &PointRegistry::points_for_ids)
        .def("vertex_ids_for_ids", &PointRegistry::vertex_ids_for_ids)
        .def_prop_ro("size", &PointRegistry::size);

    m.def("point_key_bytes", &point_key_bytes, "point"_a, "tol"_a);
    m.def(
        "prepare_charge_batch_metadata",
        &prepare_charge_batch_metadata,
        "vertex_energies"_a,
        "volumes"_a,
        "owner_ids"_a,
        "tol"_a
    );
    m.def("unique_first_indices_int64", &unique_first_indices_int64, "values"_a);
    m.def("plan_vertex_prefetch", &plan_vertex_prefetch, "vertex_ids"_a, "ready_mask"_a);
    m.def("gather_vertex_values", &gather_vertex_values, "values"_a, "vertex_ids"_a);
    m.def("gather_vertex_tables", &gather_vertex_tables, "tables"_a, "vertex_ids"_a, "bands"_a = nb::none());
    m.def(
        "prepare_density_band_metadata",
        &prepare_density_band_metadata,
        "vertex_energies"_a,
        "centroid_energies"_a,
        "mu"_a,
        "tol"_a
    );
    m.def(
        "prepare_density_piece_metadata",
        &prepare_density_piece_metadata,
        "simplex_points"_a,
        "vertex_energies"_a,
        "candidate_offsets"_a,
        "candidate_bands"_a,
        "mu"_a,
        "tol"_a
    );
    m.def(
        "prepare_density_cells_metadata",
        &prepare_density_cells_metadata,
        "simplex_points"_a,
        "vertex_energies"_a,
        "centroid_energies"_a,
        "mu"_a,
        "tol"_a
    );
    m.def(
        "accumulate_density_terms",
        &accumulate_density_terms,
        "whole_centroid_tables"_a,
        "whole_vertex_tables"_a,
        "whole_weights"_a,
        "step_centroid_tables"_a,
        "step_vertex_tables"_a,
        "step_centroid_occ"_a,
        "step_vertex_occ"_a,
        "volume"_a,
        "piece_volumes"_a,
        "piece_centroid_tables"_a,
        "piece_vertex_offsets"_a,
        "piece_vertex_tables"_a
    );
    m.def(
        "density_tables_from_eigenvectors",
        &density_tables_from_eigenvectors,
        "points"_a,
        "eigenvectors"_a,
        "keys"_a,
        "bands"_a = nb::none()
    );
}
