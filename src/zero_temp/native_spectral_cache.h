#pragma once

#include "native_tight_binding.h"

#include <memory>
#include <unordered_map>

namespace meanfi::zero_temp_native {

class NativeGeometry;

class NativeSpectralCache {
public:
    struct CacheEntry {
        std::vector<double> eigenvalues;
        std::vector<std::complex<double>> eigenvectors;
    };

    explicit NativeSpectralCache(std::shared_ptr<NativeTightBindingModel> model, double tol = 1e-14);

    void clear();
    void invalidate();

    std::uint64_t generation() const noexcept { return generation_; }
    size_t ndim() const noexcept { return model_->ndim(); }
    size_t ndof() const noexcept { return model_->ndof(); }
    size_t size() const noexcept { return cache_.size(); }
    std::uint64_t n_kernel_evals() const noexcept { return n_kernel_evals_; }
    double spectral_bound() const { return model_->spectral_bound(); }

    nb::tuple evaluate_many(Float2D points);
    nb::tuple get_many(Float2D points);
    nb::ndarray<nb::numpy, double> get_many_values(Float2D points);

    const CacheEntry &entry_for_k_point(const double *point);
    const CacheEntry &entry_for_reduced_point(const double *reduced_point);
    const CacheEntry &entry_for_geometry_vertex(const NativeGeometry &geometry, std::int64_t vertex_id);
    CacheEntry evaluate_reduced_point_uncached(const double *reduced_point);
    std::uint64_t n_reduced_point_lookups() const noexcept { return n_reduced_point_lookups_; }
    std::uint64_t n_geometry_vertex_lookups() const noexcept { return n_geometry_vertex_lookups_; }
    size_t geometry_vertex_cache_size() const noexcept { return geometry_vertex_cache_size_; }

private:
    nb::tuple evaluate_many_impl(Float2D points, bool use_cache);
    CacheEntry diagonalize_k_point(const double *point);

    std::shared_ptr<NativeTightBindingModel> model_;
    double tol_ = 1e-14;
    std::uint64_t generation_ = 0;
    std::uint64_t n_kernel_evals_ = 0;
    std::uint64_t n_reduced_point_lookups_ = 0;
    std::uint64_t n_geometry_vertex_lookups_ = 0;
    size_t geometry_vertex_cache_size_ = 0;
    std::unordered_map<std::string, CacheEntry> cache_;
    std::vector<const CacheEntry *> geometry_vertex_entries_;
    std::vector<std::uint8_t> geometry_vertex_ready_;
    CacheEntry temp_entry_;
};

}  // namespace meanfi::zero_temp_native
