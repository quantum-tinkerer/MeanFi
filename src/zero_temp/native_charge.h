#pragma once

#include "native_frontier.h"
#include "native_simplex_cache.h"
#include "native_spectral_cache.h"

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

    ChargeEvalResult evaluate_impl(NativeFrontier &frontier, double mu, std::int64_t levels);
    ChargeEvalResult evaluate_counted(
        NativeFrontier &frontier,
        double mu,
        std::int64_t levels,
        std::int64_t &integration_calls,
        std::int64_t &evaluator_evals
    );
    void evaluate_simplex(
        std::int64_t simplex_id,
        double mu,
        double &charge,
        double &derivative,
        bool &derivative_exact
    ) const;

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
        double max_indicator,
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
        const NativeSimplexCache::SimplexData &simplex,
        double mu,
        double &charge,
        double &derivative,
        bool &derivative_exact
    ) const;

    NativeGeometry *geometry_ = nullptr;
    NativeSpectralCache *spectral_cache_ = nullptr;
    NativeSimplexCache simplex_cache_;
    double tol_ = 1e-14;
};

}  // namespace meanfi::zero_temp_native
