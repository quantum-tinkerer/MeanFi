#include "native_charge.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace meanfi::zero_temp_native {

NativeChargeEvaluator::NativeChargeEvaluator(
    NativeGeometry &geometry,
    NativeSpectralCache &spectral_cache,
    double tol
)
    : geometry_(&geometry),
      spectral_cache_(&spectral_cache),
      simplex_cache_(geometry),
      tol_(tol) {}

nb::tuple NativeChargeEvaluator::evaluate(NativeFrontier &frontier, double mu, std::int64_t levels) {
    const auto result = evaluate_impl(frontier, mu, levels);
    return nb::make_tuple(
        result.total_charge,
        result.derivative_exact ? result.total_derivative : std::numeric_limits<double>::quiet_NaN(),
        result.derivative_exact,
        make_array(std::vector<std::int64_t>(result.owner_ids), {result.owner_ids.size()}),
        make_array(std::vector<double>(result.owner_charges), {result.owner_charges.size()}),
        result.evaluator_evals
    );
}

double NativeChargeEvaluator::simplex_charge(std::int64_t simplex_id, double mu, std::int64_t levels) {
    double total_charge = 0.0;
    double total_derivative = 0.0;
    bool derivative_exact = true;
    for (const auto leaf_id : simplex_cache_.leaf_ids(simplex_id, levels)) {
        evaluate_simplex(leaf_id, mu, total_charge, total_derivative, derivative_exact);
    }
    (void) total_derivative;
    (void) derivative_exact;
    return total_charge;
}

ChargeSolveResult NativeChargeEvaluator::solve_mu_and_refine(
    NativeFrontier &frontier,
    double filling,
    const ChargeSolveOptions &options
) {
    double bound = spectral_cache_->spectral_bound() + 1.0;
    double lower = -bound;
    double upper = bound;
    std::int64_t integration_calls = 0;
    std::int64_t evaluator_evals = 0;
    std::int64_t refinements = 0;
    std::int64_t remaining = options.max_subdivisions;

    expand_mu_bracket(frontier, filling, lower, upper, integration_calls, evaluator_evals);

    double mu = std::clamp(options.mu_guess, lower, upper);
    if (!(lower < mu && mu < upper)) {
        mu = 0.5 * (lower + upper);
    }

    ChargeSolveResult result;
    result.mu = mu;
    result.converged = false;

    while (true) {
        ChargeSolveOptions root_options = options;
        root_options.charge_tol = std::max(0.25 * options.charge_tol, std::numeric_limits<double>::epsilon());
        const auto root = solve_mu_on_preview(
            frontier,
            filling,
            mu,
            lower,
            upper,
            root_options,
            integration_calls,
            evaluator_evals
        );
        mu = root.mu;
        result.mu = mu;
        result.root_iterations = root.iterations;
        result.dcharge_dmu = root.derivative;

        const auto coarse = evaluate_counted(frontier, mu, 0, integration_calls, evaluator_evals);
        const auto preview = evaluate_counted(frontier, mu, 1, integration_calls, evaluator_evals);
        if (coarse.owner_ids != preview.owner_ids) {
            throw std::runtime_error("NativeChargeEvaluator: coarse/preview owner mismatch");
        }

        std::vector<double> indicators(preview.owner_ids.size(), 0.0);
        double indicator_max = 0.0;
        for (size_t index = 0; index < preview.owner_ids.size(); ++index) {
            indicators[index] = std::abs(preview.owner_charges[index] - coarse.owner_charges[index]);
            indicator_max = std::max(indicator_max, indicators[index]);
        }

        result.charge = preview.total_charge;
        result.charge_error = indicator_max;
        result.error_estimate_available = true;

        if (std::abs(preview.total_charge - filling) <= options.charge_tol &&
            indicator_max <= options.charge_tol) {
            result.converged = true;
            break;
        }

        if (remaining == 0) {
            throw std::runtime_error("Adaptive zero-temperature charge integration did not converge");
        }

        const auto marked = bulk_mark(
            preview.owner_ids,
            indicators,
            indicator_max,
            options.bulk_theta
        );
        if (marked.empty()) {
            break;
        }

        const auto batch = geometry_->refine_marked(marked);
        frontier.apply_refinement(batch);
        refinements += batch.refinements;
        if (remaining > 0) {
            remaining -= batch.refinements;
            if (remaining < 0) {
                throw std::runtime_error("Adaptive zero-temperature charge integration did not converge");
            }
        }
    }

    result.charge_integration_calls = integration_calls;
    result.evaluator_evals = evaluator_evals;
    result.subdivisions = refinements;
    result.n_leaves = static_cast<std::int64_t>(frontier.n_active());
    result.n_leaf_nodes = static_cast<std::int64_t>(frontier.n_leaf_vertices());
    return result;
}

NativeChargeEvaluator::ChargeEvalResult NativeChargeEvaluator::evaluate_impl(
    NativeFrontier &frontier,
    double mu,
    std::int64_t levels
) {
    ChargeEvalResult result;
    result.owner_ids = frontier.active_simplex_ids_;
    result.owner_charges.resize(result.owner_ids.size(), 0.0);

    for (size_t owner_index = 0; owner_index < result.owner_ids.size(); ++owner_index) {
        double owner_charge = 0.0;
        double owner_derivative = 0.0;
        bool owner_exact = true;
        for (const auto leaf_id : simplex_cache_.leaf_ids(result.owner_ids[owner_index], levels)) {
            evaluate_simplex(leaf_id, mu, owner_charge, owner_derivative, owner_exact);
            result.evaluator_evals += static_cast<std::int64_t>(spectral_cache_->ndof());
        }
        result.owner_charges[owner_index] = owner_charge;
        result.total_charge += owner_charge;
        result.total_derivative += owner_derivative;
        result.derivative_exact = result.derivative_exact && owner_exact;
    }
    return result;
}

NativeChargeEvaluator::ChargeEvalResult NativeChargeEvaluator::evaluate_counted(
    NativeFrontier &frontier,
    double mu,
    std::int64_t levels,
    std::int64_t &integration_calls,
    std::int64_t &evaluator_evals
) {
    ++integration_calls;
    auto result = evaluate_impl(frontier, mu, levels);
    evaluator_evals += result.evaluator_evals;
    return result;
}

void NativeChargeEvaluator::evaluate_simplex(
    std::int64_t simplex_id,
    double mu,
    double &charge,
    double &derivative,
    bool &derivative_exact
) const {
    evaluate_cell(simplex_cache_.simplex(simplex_id), mu, charge, derivative, derivative_exact);
}

NativeChargeEvaluator::RootSolveResult NativeChargeEvaluator::solve_mu_on_preview(
    NativeFrontier &frontier,
    double filling,
    double mu_guess,
    double lower,
    double upper,
    const ChargeSolveOptions &options,
    std::int64_t &integration_calls,
    std::int64_t &evaluator_evals
) {
    auto lower_eval = evaluate_counted(frontier, lower, 1, integration_calls, evaluator_evals);
    auto upper_eval = evaluate_counted(frontier, upper, 1, integration_calls, evaluator_evals);
    double lower_charge = lower_eval.total_charge;
    double upper_charge = upper_eval.total_charge;
    double mu = std::clamp(mu_guess, lower, upper);
    if (!(lower < mu && mu < upper)) {
        mu = 0.5 * (lower + upper);
    }

    double last_charge = std::numeric_limits<double>::quiet_NaN();
    double last_derivative = std::numeric_limits<double>::quiet_NaN();
    RootSolveResult result;

    for (std::int64_t iteration = 1; iteration <= options.max_mu_iterations; ++iteration) {
        const auto summary = evaluate_counted(frontier, mu, 1, integration_calls, evaluator_evals);
        last_charge = summary.total_charge;
        const double residual = last_charge - filling;
        if (std::abs(residual) <= options.charge_tol) {
            result.mu = mu;
            result.charge = last_charge;
            result.derivative = last_derivative;
            result.iterations = iteration;
            return result;
        }

        if (residual < 0.0) {
            lower = mu;
            lower_charge = last_charge;
        } else {
            upper = mu;
            upper_charge = last_charge;
        }

        if (upper - lower <= options.mu_xtol) {
            result.mu = 0.5 * (lower + upper);
            result.charge = last_charge;
            result.derivative = last_derivative;
            result.iterations = iteration;
            return result;
        }

        double slope =
            summary.derivative_exact && std::isfinite(summary.total_derivative) && summary.total_derivative > 0.0
                ? summary.total_derivative
                : std::numeric_limits<double>::quiet_NaN();
        if (!std::isfinite(slope) || slope <= 0.0) {
            slope = (upper_charge - lower_charge) / (upper - lower);
        }
        last_derivative = slope;

        double candidate = std::numeric_limits<double>::quiet_NaN();
        if (slope > 0.0 && std::isfinite(slope)) {
            candidate = mu - residual / slope;
        }
        if (!std::isfinite(candidate) || candidate <= lower || candidate >= upper) {
            candidate = 0.5 * (lower + upper);
        }
        mu = candidate;
    }

    const double midpoint = 0.5 * (lower + upper);
    const auto summary = evaluate_counted(frontier, midpoint, 1, integration_calls, evaluator_evals);
    double slope = last_derivative;
    if (!std::isfinite(slope) || slope <= 0.0) {
        slope = (upper_charge - lower_charge) / (upper - lower);
    }
    result.mu = midpoint;
    result.charge = summary.total_charge;
    result.derivative = slope;
    result.iterations = options.max_mu_iterations;
    return result;
}

void NativeChargeEvaluator::expand_mu_bracket(
    NativeFrontier &frontier,
    double filling,
    double &lower,
    double &upper,
    std::int64_t &integration_calls,
    std::int64_t &evaluator_evals
) {
    auto lower_eval = evaluate_counted(frontier, lower, 1, integration_calls, evaluator_evals);
    auto upper_eval = evaluate_counted(frontier, upper, 1, integration_calls, evaluator_evals);
    double lower_charge = lower_eval.total_charge;
    double upper_charge = upper_eval.total_charge;
    while (lower_charge > filling || upper_charge < filling) {
        lower *= 2.0;
        upper *= 2.0;
        lower_eval = evaluate_counted(frontier, lower, 1, integration_calls, evaluator_evals);
        upper_eval = evaluate_counted(frontier, upper, 1, integration_calls, evaluator_evals);
        lower_charge = lower_eval.total_charge;
        upper_charge = upper_eval.total_charge;
    }
}

std::vector<std::int64_t> NativeChargeEvaluator::bulk_mark(
    const std::vector<std::int64_t> &owner_ids,
    const std::vector<double> &indicators,
    double max_indicator,
    double bulk_theta
) {
    if (owner_ids.empty() || max_indicator <= 0.0) {
        return {};
    }
    std::vector<std::int64_t> marked;
    const double threshold = bulk_theta * max_indicator;
    size_t best_index = 0;
    for (size_t index = 0; index < owner_ids.size(); ++index) {
        if (indicators[index] > indicators[best_index]) {
            best_index = index;
        }
        if (indicators[index] >= threshold && indicators[index] > 0.0) {
            marked.push_back(owner_ids[index]);
        }
    }
    if (marked.empty() && indicators[best_index] > 0.0) {
        marked.push_back(owner_ids[best_index]);
    }
    return marked;
}

std::pair<double, double> NativeChargeEvaluator::simplex_fraction_and_derivative(
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
                std::clamp(
                    (x0 * x0 * x0) / denom0 + (x1 * x1 * x1) / denom1,
                    0.0,
                    1.0
                ),
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
            derivative += static_cast<double>(dimension) *
                          weights[vertex] *
                          std::pow(delta, static_cast<int>(dimension - 1));
        }
    }
    return {std::clamp(fraction, 0.0, 1.0), derivative};
}

void NativeChargeEvaluator::evaluate_cell(
    const NativeSimplexCache::SimplexData &simplex,
    double mu,
    double &charge,
    double &derivative,
    bool &derivative_exact
) const {
    const size_t n_vertices = simplex.vertex_ids.size();
    const size_t ndof = spectral_cache_->ndof();
    std::vector<const std::vector<double> *> vertex_values(n_vertices, nullptr);
    for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
        const auto vertex_id = simplex.vertex_ids[vertex];
        const auto &entry = spectral_cache_->entry_for_geometry_vertex(*geometry_, vertex_id);
        vertex_values[vertex] = &entry.eigenvalues;
    }

    std::vector<double> band_energies(n_vertices);
    std::vector<double> sorted_energies(n_vertices);
    std::vector<double> simplex_weights(n_vertices, 0.0);

    for (size_t band = 0; band < ndof; ++band) {
        double band_min = std::numeric_limits<double>::infinity();
        double band_max = -std::numeric_limits<double>::infinity();
        double flat_energy = 0.0;
        bool flat_mask = true;
        for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
            const double energy = (*vertex_values[vertex])[band];
            band_energies[vertex] = energy;
            sorted_energies[vertex] = energy;
            band_min = std::min(band_min, energy);
            band_max = std::max(band_max, energy);
            if (vertex == 0) {
                flat_energy = energy;
            } else if (std::abs(energy - flat_energy) > tol_) {
                flat_mask = false;
            }
        }

        const bool full = band_max <= mu;
        const bool empty = band_min > mu;
        const bool flat_half =
            flat_mask &&
            !full &&
            !empty &&
            std::abs(flat_energy - mu) <= tol_;
        if (full) {
            charge += simplex.volume;
            continue;
        }
        if (empty) {
            continue;
        }
        if (flat_half) {
            charge += 0.5 * simplex.volume;
            continue;
        }

        std::sort(sorted_energies.begin(), sorted_energies.end());
        bool distinct = true;
        for (size_t vertex = 1; vertex < n_vertices; ++vertex) {
            if (sorted_energies[vertex] - sorted_energies[vertex - 1] <= tol_) {
                distinct = false;
                break;
            }
        }

        if (distinct) {
            const double *weights = nullptr;
            if (geometry_->ndim_ > 3) {
                std::fill(simplex_weights.begin(), simplex_weights.end(), 0.0);
                for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
                    double denom = 1.0;
                    for (size_t other = 0; other < n_vertices; ++other) {
                        if (other == vertex) {
                            continue;
                        }
                        denom *= sorted_energies[vertex] - sorted_energies[other];
                    }
                    simplex_weights[vertex] = 1.0 / denom;
                }
                weights = simplex_weights.data();
            }
            const auto [fraction, dfraction] = simplex_fraction_and_derivative(
                sorted_energies.data(),
                mu,
                geometry_->ndim_,
                weights,
                tol_
            );
            charge += simplex.volume * fraction;
            derivative += simplex.volume * dfraction;
            continue;
        }

        derivative_exact = false;
        const auto pieces = occupied_subsimplices_from_flat(
            simplex.points_flat.data(),
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

}  // namespace meanfi::zero_temp_native
