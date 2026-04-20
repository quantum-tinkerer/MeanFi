#include "native_spectral_cache.h"

#include <stdexcept>

namespace meanfi::zero_temp_native {

NativeSpectralCache::NativeSpectralCache(
    std::shared_ptr<NativeTightBindingModel> model,
    double tol
)
    : model_(std::move(model)), tol_(tol) {
    if (!model_) {
        throw std::runtime_error("NativeSpectralCache: model must not be null");
    }
}

void NativeSpectralCache::clear() {
    cache_.clear();
}

void NativeSpectralCache::invalidate() {
    ++generation_;
    clear();
}

nb::tuple NativeSpectralCache::evaluate_many(Float2D points) {
    return evaluate_many_impl(points, false);
}

nb::tuple NativeSpectralCache::get_many(Float2D points) {
    return evaluate_many_impl(points, true);
}

nb::ndarray<nb::numpy, double> NativeSpectralCache::get_many_values(Float2D points) {
    auto result = evaluate_many_impl(points, true);
    return nb::cast<nb::ndarray<nb::numpy, double>>(result[0]);
}

const NativeSpectralCache::CacheEntry &NativeSpectralCache::entry_for_k_point(const double *point) {
    const std::string key = point_key_from_ptr(point, model_->ndim(), tol_);
    auto it = cache_.find(key);
    if (it == cache_.end()) {
        auto [inserted, _ok] = cache_.emplace(key, diagonalize_k_point(point));
        it = inserted;
    }
    return it->second;
}

const NativeSpectralCache::CacheEntry &NativeSpectralCache::entry_for_reduced_point(
    const double *reduced_point
) {
    std::vector<double> k_point(model_->ndim());
    for (size_t axis = 0; axis < model_->ndim(); ++axis) {
        k_point[axis] = 2.0 * kPi * reduced_point[axis] - kPi;
    }
    return entry_for_k_point(k_point.data());
}

NativeSpectralCache::CacheEntry NativeSpectralCache::evaluate_reduced_point_uncached(
    const double *reduced_point
) {
    std::vector<double> k_point(model_->ndim());
    for (size_t axis = 0; axis < model_->ndim(); ++axis) {
        k_point[axis] = 2.0 * kPi * reduced_point[axis] - kPi;
    }
    return diagonalize_k_point(k_point.data());
}

nb::tuple NativeSpectralCache::evaluate_many_impl(Float2D points, bool use_cache) {
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
        CacheEntry *entry = nullptr;
        if (use_cache) {
            const std::string key = point_key_from_ptr(row, model_->ndim(), tol_);
            auto it = cache_.find(key);
            if (it != cache_.end()) {
                entry = &it->second;
            } else {
                auto [inserted, _ok] = cache_.emplace(key, diagonalize_k_point(row));
                entry = &inserted->second;
            }
        } else {
            temp_entry_ = diagonalize_k_point(row);
            entry = &temp_entry_;
        }

        std::copy_n(entry->eigenvalues.data(), ndof, values.data() + point_index * ndof);
        std::copy_n(
            entry->eigenvectors.data(),
            ndof * ndof,
            vectors.data() + point_index * ndof * ndof
        );
    }

    return nb::make_tuple(
        make_array(std::move(values), {n_points, ndof}),
        make_array(std::move(vectors), {n_points, ndof, ndof})
    );
}

NativeSpectralCache::CacheEntry NativeSpectralCache::diagonalize_k_point(const double *point) {
    const Eigen::MatrixXcd h = model_->evaluate_point_raw(point);
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

}  // namespace meanfi::zero_temp_native
