#include "vertex_cache.h"

#include "geometry.h"

#include <cmath>
#include <stdexcept>

namespace meanfi::zero_temp {

namespace {

double wrap_reduced_coordinate(double value, double tol) {
    double wrapped = std::fmod(value, 1.0);
    if (wrapped < 0.0) {
        wrapped += 1.0;
    }
    if (std::abs(wrapped) <= tol || std::abs(wrapped - 1.0) <= tol) {
        return 0.0;
    }
    return wrapped;
}

std::string point_key_from_wrapped_reduced_point(const double *point, size_t ndim) {
    std::vector<double> wrapped(ndim);
    for (size_t axis = 0; axis < ndim; ++axis) {
        wrapped[axis] = point[axis];
    }
    return std::string(
        reinterpret_cast<const char *>(wrapped.data()),
        wrapped.size() * sizeof(double)
    );
}

std::string phase_layout_key(const std::vector<double> &keys, size_t n_keys, size_t ndim) {
    std::vector<double> payload;
    payload.reserve(2 + keys.size());
    payload.push_back(static_cast<double>(n_keys));
    payload.push_back(static_cast<double>(ndim));
    payload.insert(payload.end(), keys.begin(), keys.end());
    return std::string(
        reinterpret_cast<const char *>(payload.data()),
        payload.size() * sizeof(double)
    );
}

}  // namespace

VertexCache::VertexCache(std::shared_ptr<TightBindingModel> model, double tol)
    : model_(std::move(model)), tol_(tol) {
    if (!model_) {
        throw std::runtime_error("VertexCache: model must not be null");
    }
}

void VertexCache::clear() {
    cache_.clear();
    geometry_vertex_entries_.clear();
    geometry_vertex_ready_.clear();
    for (auto &layout : phase_layouts_) {
        layout.phases.clear();
        layout.ready.clear();
        layout.cache_size = 0;
    }
}

void VertexCache::invalidate() {
    ++generation_;
    clear();
}

void VertexCache::ensure_vertex_capacity(std::int64_t vertex_id) {
    const size_t needed = static_cast<size_t>(vertex_id) + 1;
    if (geometry_vertex_entries_.size() >= needed) {
        return;
    }
    geometry_vertex_entries_.resize(needed, nullptr);
    geometry_vertex_ready_.resize(needed, 0);
}

VertexCache::Entry VertexCache::diagonalize_wrapped_reduced_point(const double *reduced_point) {
    std::vector<double> k_point(model_->ndim());
    for (size_t axis = 0; axis < model_->ndim(); ++axis) {
        const double wrapped = wrap_reduced_coordinate(reduced_point[axis], tol_);
        k_point[axis] = 2.0 * kPi * wrapped - kPi;
    }

    const Eigen::MatrixXcd h = model_->evaluate_point_raw(k_point.data());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(h);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("VertexCache: eigensolve failed");
    }

    ++n_kernel_evals_;
    const size_t ndof = model_->ndof();
    Entry entry;
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

const VertexCache::Entry &VertexCache::entry_for_vertex(const Geometry &geometry, std::int64_t vertex_id) {
    ensure_vertex_capacity(vertex_id);
    if (geometry_vertex_ready_[static_cast<size_t>(vertex_id)]) {
        return *geometry_vertex_entries_[static_cast<size_t>(vertex_id)];
    }

    const double *reduced_point =
        geometry.vertices_.data() + static_cast<size_t>(vertex_id) * geometry.ndim_;
    std::vector<double> wrapped(geometry.ndim_);
    for (size_t axis = 0; axis < geometry.ndim_; ++axis) {
        wrapped[axis] = wrap_reduced_coordinate(reduced_point[axis], tol_);
    }
    const std::string key = point_key_from_wrapped_reduced_point(wrapped.data(), geometry.ndim_);
    auto it = cache_.find(key);
    if (it == cache_.end()) {
        auto [inserted, _ok] = cache_.emplace(key, diagonalize_wrapped_reduced_point(wrapped.data()));
        it = inserted;
    }

    geometry_vertex_entries_[static_cast<size_t>(vertex_id)] = &it->second;
    geometry_vertex_ready_[static_cast<size_t>(vertex_id)] = 1;
    return it->second;
}

size_t VertexCache::register_phase_layout(
    const std::vector<double> &keys,
    size_t n_keys,
    size_t ndim
) {
    const std::string key = phase_layout_key(keys, n_keys, ndim);
    auto it = phase_layout_lookup_.find(key);
    if (it != phase_layout_lookup_.end()) {
        return it->second;
    }

    PhaseLayout layout;
    layout.ndim = ndim;
    layout.n_keys = n_keys;
    layout.keys = keys;
    const size_t layout_id = phase_layouts_.size();
    phase_layout_lookup_.emplace(key, layout_id);
    phase_layouts_.push_back(std::move(layout));
    return layout_id;
}

void VertexCache::ensure_phase_capacity(PhaseLayout &layout, std::int64_t vertex_id) {
    const size_t needed = static_cast<size_t>(vertex_id) + 1;
    if (layout.phases.size() >= needed) {
        return;
    }
    layout.phases.resize(needed);
    layout.ready.resize(needed, 0);
}

const std::vector<std::complex<double>> &VertexCache::phases_for_vertex(
    const Geometry &geometry,
    std::int64_t vertex_id,
    size_t layout_id
) {
    auto &layout = phase_layouts_.at(layout_id);
    ensure_phase_capacity(layout, vertex_id);
    if (layout.ready[static_cast<size_t>(vertex_id)]) {
        return layout.phases[static_cast<size_t>(vertex_id)];
    }

    const double *reduced_point =
        geometry.vertices_.data() + static_cast<size_t>(vertex_id) * geometry.ndim_;
    auto &phases = layout.phases[static_cast<size_t>(vertex_id)];
    phases.resize(layout.n_keys);
    for (size_t key_index = 0; key_index < layout.n_keys; ++key_index) {
        double phase_arg = 0.0;
        for (size_t axis = 0; axis < layout.ndim; ++axis) {
            phase_arg +=
                (2.0 * kPi * reduced_point[axis] - kPi) *
                layout.keys[key_index * layout.ndim + axis];
        }
        phases[key_index] = std::exp(std::complex<double>(0.0, phase_arg));
    }
    layout.ready[static_cast<size_t>(vertex_id)] = 1;
    ++layout.cache_size;
    return phases;
}

size_t VertexCache::phase_cache_size(size_t layout_id) const noexcept {
    if (layout_id >= phase_layouts_.size()) {
        return 0;
    }
    return phase_layouts_[layout_id].cache_size;
}

}  // namespace meanfi::zero_temp
