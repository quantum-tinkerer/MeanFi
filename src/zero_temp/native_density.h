#pragma once

#include "native_frontier.h"
#include "native_simplex_cache.h"
#include "native_spectral_cache.h"

#include <queue>
#include <limits>
#include <unordered_map>

namespace meanfi::zero_temp_native {

class NativeDensityEvaluator {
public:
    NativeDensityEvaluator(
        NativeGeometry &geometry,
        NativeSpectralCache &spectral_cache,
        Float2D keys,
        double tol = 1e-14
    );

    void clear();
    nb::tuple evaluate(NativeFrontier &frontier, double mu, std::int64_t levels = 0);
    nb::tuple evaluate_many(Int1D simplex_ids, double mu);
    size_t phase_cache_size() const noexcept { return phase_cache_size_; }
    size_t cached_simplex_value_count() const noexcept { return value_cache_.size(); }
    std::int64_t leaf_build_count() const noexcept { return leaf_build_count_; }
    DensityIntegrateResult integrate_adaptive(
        NativeFrontier &frontier,
        double mu,
        const DensityIntegrateOptions &options
    );

private:
    struct SimplexDensityEstimate {
        std::vector<std::complex<double>> estimate;
        std::int64_t evaluator_evals = 0;
    };

    struct LeafDensityContribution {
        std::vector<std::complex<double>> coarse_value;
        std::vector<std::complex<double>> preview_value;
        std::vector<double> error_vector;
        double indicator = 0.0;
        std::int64_t evaluator_evals = 0;
    };

    struct DensityEvalResult {
        std::vector<std::complex<double>> total_estimate;
        std::vector<std::int64_t> owner_ids;
        std::vector<std::complex<double>> owner_estimates;
        std::int64_t evaluator_evals = 0;
    };

    struct HeapEntry {
        double error = 0.0;
        std::int64_t simplex_id = -1;

        bool operator<(const HeapEntry &other) const noexcept {
            return error < other.error;
        }
    };

    struct PointSpectrum {
        std::vector<double> eigenvalues;
        std::vector<std::complex<double>> eigenvectors;
    };

    void ensure_mu(double mu);
    DensityEvalResult evaluate_impl(NativeFrontier &frontier, std::int64_t levels);
    const SimplexDensityEstimate &cached_simplex_value(std::int64_t simplex_id);
    const LeafDensityContribution &leaf_contribution(std::int64_t simplex_id);
    const LeafDensityContribution *active_leaf_with_max_error();
    static std::vector<std::int64_t> bulk_mark(
        const std::vector<std::int64_t> &owner_ids,
        const std::vector<double> &indicators,
        double max_indicator,
        double bulk_theta
    );
    const NativeSpectralCache::CacheEntry &vertex_entry(std::int64_t vertex_id);
    const std::vector<std::complex<double>> &vertex_phases(std::int64_t vertex_id);
    void ensure_vertex_phase_capacity(std::int64_t vertex_id);
    std::vector<std::complex<double>> point_phases(const double *reduced_point) const;
    PointSpectrum uncached_point_spectrum(const double *reduced_point);
    void accumulate_density_table_for_band(
        std::vector<std::complex<double>> &out,
        const std::complex<double> *phases,
        const std::complex<double> *eigenvectors,
        size_t n_all_bands,
        size_t band,
        double scale = 1.0
    ) const;
    SimplexDensityEstimate evaluate_simplex(std::int64_t simplex_id);

    NativeGeometry *geometry_ = nullptr;
    NativeSpectralCache *spectral_cache_ = nullptr;
    NativeSimplexCache simplex_cache_;
    double tol_ = 1e-14;
    size_t n_keys_ = 0;
    size_t ndim_ = 0;
    size_t ndof_ = 0;
    size_t ncomp_ = 0;
    std::vector<double> keys_;
    double current_mu_ = std::numeric_limits<double>::quiet_NaN();
    size_t phase_cache_size_ = 0;
    std::int64_t leaf_build_count_ = 0;
    std::vector<std::vector<std::complex<double>>> vertex_phases_;
    std::vector<std::uint8_t> vertex_phase_ready_;
    std::unordered_map<std::int64_t, LeafDensityContribution> leaf_cache_;
    std::priority_queue<HeapEntry> leaf_heap_;
    std::unordered_map<std::int64_t, SimplexDensityEstimate> value_cache_;
};

}  // namespace meanfi::zero_temp_native
