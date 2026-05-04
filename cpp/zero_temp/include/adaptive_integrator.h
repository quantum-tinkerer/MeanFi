#pragma once

#include "geometry.h"
#include "simplex_rules.h"
#include "vertex_cache.h"

#include <limits>
#include <queue>

namespace meanfi::zero_temp {

class AdaptiveIntegrator {
public:
    AdaptiveIntegrator(
        Geometry &geometry,
        VertexCache &vertex_cache,
        Float2D keys,
        std::int64_t preview_depth = 1,
        double tol = 1e-14
    );

    void clear();
    std::int64_t preview_depth() const noexcept { return preview_depth_; }

    nb::tuple evaluate_charge(double mu, std::int64_t levels = 0);
    nb::tuple evaluate_density(double mu, std::int64_t levels = 0);
    ChargeSolveResult solve_mu_and_refine(double filling, const ChargeSolveOptions &options);
    DensityIntegrateResult integrate_density(double mu, const DensityIntegrateOptions &options);

    size_t phase_cache_size() const noexcept;
    size_t cached_simplex_value_count() const noexcept { return cached_simplex_value_count_; }
    std::int64_t leaf_build_count() const noexcept { return leaf_build_count_; }

private:
    enum class OccupancyClass : std::uint8_t { Empty, Full, Half, Cut };

    struct BandLayout {
        std::vector<std::uint8_t> order;
        bool strictly_ordered = true;
    };

    struct SimplexLayout {
        std::int64_t simplex_id = -1;
        std::vector<BandLayout> bands;
    };

    struct BandOccupation {
        OccupancyClass classification = OccupancyClass::Empty;
    };

    struct SimplexOccupation {
        std::int64_t simplex_id = -1;
        const SimplexLayout *layout = nullptr;
        std::vector<BandOccupation> bands;
    };

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

    struct ChargeEvalResult {
        double total_charge = 0.0;
        double total_derivative = 0.0;
        bool derivative_exact = true;
        std::vector<std::int64_t> owner_ids;
        std::vector<double> owner_charges;
        std::int64_t evaluator_evals = 0;
    };

    struct RootSolveResult {
        double mu = 0.0;
        double charge = 0.0;
        double derivative = 0.0;
        std::int64_t iterations = 0;
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

    void clear_mu_dependent_caches();
    void ensure_mu(double mu);
    void ensure_layout_capacity(std::int64_t simplex_id);
    void ensure_value_capacity(std::int64_t simplex_id);
    void ensure_leaf_capacity(std::int64_t simplex_id);
    void gather_vertex_entries(
        std::int64_t simplex_id,
        std::vector<const VertexCache::Entry *> &entries
    );
    const SimplexLayout &simplex_layout(
        std::int64_t simplex_id,
        const std::vector<const VertexCache::Entry *> &entries
    );
    SimplexOccupation simplex_occupation(
        std::int64_t simplex_id,
        double mu,
        const SimplexLayout &layout,
        const std::vector<const VertexCache::Entry *> &entries
    );
    void collect_simplex_ids(
        std::int64_t owner_id,
        std::int64_t levels,
        std::vector<std::int64_t> &out
    );

    ChargeEvalResult evaluate_charge_impl(double mu, std::int64_t levels);
    ChargeEvalResult evaluate_charge_counted(
        double mu,
        std::int64_t levels,
        std::int64_t &integration_calls,
        std::int64_t &evaluator_evals
    );
    void evaluate_charge_simplex(
        std::int64_t simplex_id,
        double mu,
        double &charge,
        double &derivative,
        bool &derivative_exact
    );
    RootSolveResult solve_mu_on_preview(
        double filling,
        double mu_guess,
        double lower,
        double upper,
        const ChargeSolveOptions &options,
        std::int64_t &integration_calls,
        std::int64_t &evaluator_evals
    );
    void expand_mu_bracket(
        double filling,
        double &lower,
        double &upper,
        std::int64_t &integration_calls,
        std::int64_t &evaluator_evals
    );

    DensityEvalResult evaluate_density_impl(double mu, std::int64_t levels);
    const SimplexDensityEstimate &cached_simplex_value(std::int64_t simplex_id);
    const LeafDensityContribution &leaf_contribution(std::int64_t simplex_id);
    const LeafDensityContribution *active_leaf_with_max_error();
    static std::vector<std::int64_t> bulk_mark(
        const std::vector<std::int64_t> &owner_ids,
        const std::vector<double> &indicators,
        double max_indicator,
        double bulk_theta
    );
    void accumulate_vertex_band(
        std::vector<std::complex<double>> &out,
        const std::complex<double> *phases,
        const std::complex<double> *eigenvectors,
        size_t n_all_bands,
        size_t band,
        double scale
    ) const;
    SimplexDensityEstimate evaluate_density_simplex(std::int64_t simplex_id);

    Geometry *geometry_ = nullptr;
    VertexCache *vertex_cache_ = nullptr;
    double tol_ = 1e-14;
    size_t n_keys_ = 0;
    size_t ndim_ = 0;
    size_t ndof_ = 0;
    size_t ncomp_ = 0;
    std::vector<double> keys_;
    std::int64_t preview_depth_ = 1;
    size_t phase_layout_id_ = std::numeric_limits<size_t>::max();
    double current_mu_ = std::numeric_limits<double>::quiet_NaN();

    std::vector<SimplexLayout> layout_cache_;
    std::vector<std::uint8_t> layout_ready_;

    std::vector<SimplexDensityEstimate> value_cache_;
    std::vector<std::uint8_t> value_ready_;
    size_t cached_simplex_value_count_ = 0;

    std::vector<LeafDensityContribution> leaf_cache_;
    std::vector<std::uint8_t> leaf_ready_;
    std::priority_queue<HeapEntry> leaf_heap_;
    std::int64_t leaf_build_count_ = 0;
};

}  // namespace meanfi::zero_temp
