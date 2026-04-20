#pragma once

#include "native_frontier.h"
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
    nb::tuple evaluate_many(Int1D simplex_ids, double mu);
    DensityIntegrateResult integrate_adaptive(
        NativeFrontier &frontier,
        double mu,
        const DensityIntegrateOptions &options
    );

private:
    struct CachedDensityValue {
        std::vector<std::complex<double>> estimate;
        std::vector<double> error_vector;
        double error_scalar = 0.0;
        std::int64_t evaluator_evals = 0;
    };

    struct HeapEntry {
        double error = 0.0;
        std::int64_t simplex_id = -1;
        std::int64_t version = 0;

        bool operator<(const HeapEntry &other) const noexcept {
            return error < other.error;
        }
    };

    struct PointSpectrum {
        std::vector<double> eigenvalues;
        std::vector<std::complex<double>> eigenvectors;
    };

    void ensure_mu(double mu);
    const CachedDensityValue &cached_value(std::int64_t simplex_id);
    std::pair<std::vector<CachedDensityValue>, std::int64_t> evaluate_many_impl(
        const std::vector<std::int64_t> &simplex_ids
    );
    const std::vector<double> &vertex_eigenvalues(std::int64_t vertex_id);
    const std::vector<std::complex<double>> &vertex_tables(std::int64_t vertex_id);
    void ensure_vertex_value_capacity(std::int64_t vertex_id);
    void ensure_vertex_table_capacity(std::int64_t vertex_id);
    const NativeSpectralCache::CacheEntry &vertex_entry(std::int64_t vertex_id);
    PointSpectrum uncached_point_spectrum(const double *reduced_point);
    std::vector<std::complex<double>> density_tables_for_point(
        const double *reduced_point,
        const std::complex<double> *eigenvectors,
        size_t n_all_bands,
        const std::int64_t *selected_bands,
        size_t n_selected_bands
    ) const;
    CachedDensityValue evaluate_simplex(std::int64_t simplex_id);

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

}  // namespace meanfi::zero_temp_native
