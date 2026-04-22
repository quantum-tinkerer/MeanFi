#pragma once

#include "tight_binding.h"

#include <memory>
#include <unordered_map>

namespace meanfi::zero_temp {

class Geometry;

class VertexCache {
public:
    struct Entry {
        std::vector<double> eigenvalues;
        std::vector<std::complex<double>> eigenvectors;
    };

    explicit VertexCache(std::shared_ptr<TightBindingModel> model, double tol = 1e-14);

    void clear();
    void invalidate();

    std::uint64_t generation() const noexcept { return generation_; }
    size_t ndim() const noexcept { return model_->ndim(); }
    size_t ndof() const noexcept { return model_->ndof(); }
    size_t size() const noexcept { return cache_.size(); }
    std::uint64_t n_kernel_evals() const noexcept { return n_kernel_evals_; }
    double spectral_bound() const { return model_->spectral_bound(); }

    const Entry &entry_for_vertex(const Geometry &geometry, std::int64_t vertex_id);
    size_t register_phase_layout(const std::vector<double> &keys, size_t n_keys, size_t ndim);
    const std::vector<std::complex<double>> &phases_for_vertex(
        const Geometry &geometry,
        std::int64_t vertex_id,
        size_t layout_id
    );
    size_t phase_cache_size(size_t layout_id) const noexcept;

private:
    struct PhaseLayout {
        size_t ndim = 0;
        size_t n_keys = 0;
        std::vector<double> keys;
        std::vector<std::vector<std::complex<double>>> phases;
        std::vector<std::uint8_t> ready;
        size_t cache_size = 0;
    };

    Entry diagonalize_wrapped_reduced_point(const double *reduced_point);
    void ensure_vertex_capacity(std::int64_t vertex_id);
    void ensure_phase_capacity(PhaseLayout &layout, std::int64_t vertex_id);

    std::shared_ptr<TightBindingModel> model_;
    double tol_ = 1e-14;
    std::uint64_t generation_ = 0;
    std::uint64_t n_kernel_evals_ = 0;
    std::unordered_map<std::string, Entry> cache_;
    std::vector<const Entry *> geometry_vertex_entries_;
    std::vector<std::uint8_t> geometry_vertex_ready_;
    std::unordered_map<std::string, size_t> phase_layout_lookup_;
    std::vector<PhaseLayout> phase_layouts_;
};

}  // namespace meanfi::zero_temp
