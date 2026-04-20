#pragma once

#include "native_frontier.h"
#include "native_spectral_cache.h"

#include <unordered_map>

namespace meanfi::zero_temp_native {

class NativeChargeEvaluator {
public:
    NativeChargeEvaluator(
        NativeGeometry &geometry,
        NativeSpectralCache &spectral_cache,
        double tol = 1e-14
    );

    std::int64_t preview_depth() const noexcept { return 1; }

    nb::tuple evaluate(NativeFrontier &frontier, double mu, std::int64_t levels = 0);
    double simplex_charge(std::int64_t simplex_id, double mu, std::int64_t levels = 0);
    ChargeSolveResult solve_mu_and_refine(
        NativeFrontier &frontier,
        double filling,
        const ChargeSolveOptions &options
    );

private:
    struct PreparedChargeCell {
        std::vector<double> vertex_energies;
        std::vector<double> sorted_energies;
        std::vector<double> simplex_weights;
        std::vector<double> band_min;
        std::vector<double> band_max;
        std::vector<double> flat_energy;
        std::vector<std::uint8_t> distinct_mask;
        std::vector<std::uint8_t> flat_mask;
        std::vector<double> points_flat;
        double volume = 0.0;
    };

    struct PreparedChargeGroup {
        std::int64_t owner_id = -1;
        std::vector<std::int64_t> leaf_ids;
        std::vector<std::int64_t> child_ids;
        std::vector<PreparedChargeCell> cells;
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

    struct GroupKey {
        std::int64_t simplex_id = -1;
        std::int64_t levels = 0;

        bool operator==(const GroupKey &other) const noexcept {
            return simplex_id == other.simplex_id && levels == other.levels;
        }
    };

    struct GroupKeyHash {
        size_t operator()(const GroupKey &key) const noexcept {
            return std::hash<std::int64_t>{}(key.simplex_id) ^
                   (std::hash<std::int64_t>{}(key.levels) << 1);
        }
    };

    ChargeEvalResult evaluate_impl(NativeFrontier &frontier, double mu, std::int64_t levels);
    ChargeEvalResult evaluate_counted(
        NativeFrontier &frontier,
        double mu,
        std::int64_t levels,
        std::int64_t &integration_calls,
        std::int64_t &evaluator_evals
    );
    const PreparedChargeGroup &prepared_group(std::int64_t simplex_id, std::int64_t levels);
    const PreparedChargeCell &prepared_cell(std::int64_t simplex_id);
    const std::vector<double> &vertex_eigenvalues(std::int64_t vertex_id);

    RootSolveResult solve_mu_on_preview(
        NativeFrontier &frontier,
        double filling,
        double mu_guess,
        double lower,
        double upper,
        const ChargeSolveOptions &options,
        std::int64_t &integration_calls,
        std::int64_t &evaluator_evals
    );
    void expand_mu_bracket(
        NativeFrontier &frontier,
        double filling,
        double &lower,
        double &upper,
        std::int64_t &integration_calls,
        std::int64_t &evaluator_evals
    );
    static std::vector<std::int64_t> bulk_mark(
        const std::vector<std::int64_t> &owner_ids,
        const std::vector<double> &indicators,
        double total_indicator,
        double bulk_theta
    );
    static std::pair<double, double> simplex_fraction_and_derivative(
        const double *energies,
        double mu,
        size_t dimension,
        const double *weights,
        double tol
    );
    void evaluate_cell(
        const PreparedChargeCell &cell,
        double mu,
        double &charge,
        double &derivative,
        bool &derivative_exact
    ) const;

    NativeGeometry *geometry_ = nullptr;
    NativeSpectralCache *spectral_cache_ = nullptr;
    double tol_ = 1e-14;
    std::unordered_map<std::int64_t, std::vector<double>> vertex_values_cache_;
    std::unordered_map<std::int64_t, PreparedChargeCell> prepared_cell_cache_;
    std::unordered_map<GroupKey, PreparedChargeGroup, GroupKeyHash> prepared_group_cache_;
};

}  // namespace meanfi::zero_temp_native
