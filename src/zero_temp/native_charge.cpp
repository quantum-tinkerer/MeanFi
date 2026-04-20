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
    : geometry_(&geometry), spectral_cache_(&spectral_cache), tol_(tol) {}

nb::tuple NativeChargeEvaluator::evaluate(NativeFrontier &frontier, double mu, std::int64_t levels) {
    const auto result = evaluate_impl(frontier, mu, levels);
    return nb::make_tuple(
        result.total_charge,
        result.derivative_exact ? result.total_derivative : std::numeric_limits<double>::quiet_NaN(),
        result.derivative_exact,
        make_array(std::vector<std::int64_t>(result.owner_ids), {result.owner_ids.size()}),
        make_array(std::vector<double>(result.owner_charges), {result.owner_charges.size()})
    );
}

double NativeChargeEvaluator::simplex_charge(std::int64_t simplex_id, double mu, std::int64_t levels) {
    const auto &group = prepared_group(simplex_id, levels);
    double total_charge = 0.0;
    double total_derivative = 0.0;
    bool derivative_exact = true;
    for (const auto &cell : group.cells) {
        double charge = 0.0;
        double derivative = 0.0;
        bool exact = true;
        evaluate_cell(cell, mu, charge, derivative, exact);
        total_charge += charge;
        total_derivative += derivative;
        derivative_exact = derivative_exact && exact;
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
        double indicator_sum = 0.0;
        for (size_t index = 0; index < preview.owner_ids.size(); ++index) {
            indicators[index] = std::abs(preview.owner_charges[index] - coarse.owner_charges[index]);
            indicator_sum += indicators[index];
        }

        result.charge = preview.total_charge;
        result.charge_error = indicator_sum;

        if (std::abs(preview.total_charge - filling) <= options.charge_tol &&
            indicator_sum <= options.charge_tol) {
            result.converged = true;
            break;
        }

        if (remaining == 0) {
            throw std::runtime_error("Adaptive zero-temperature charge integration did not converge");
        }

        const auto marked = bulk_mark(preview.owner_ids, indicators, indicator_sum, options.bulk_theta);
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
        const auto simplex_id = result.owner_ids[owner_index];
        const auto &group = prepared_group(simplex_id, levels);
        double owner_charge = 0.0;
        double owner_derivative = 0.0;
        bool owner_exact = true;
        for (const auto &cell : group.cells) {
            double charge = 0.0;
            double derivative = 0.0;
            bool exact = true;
            evaluate_cell(cell, mu, charge, derivative, exact);
            owner_charge += charge;
            owner_derivative += derivative;
            owner_exact = owner_exact && exact;
            result.evaluator_evals += static_cast<std::int64_t>(cell.band_min.size());
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

const NativeChargeEvaluator::PreparedChargeGroup &NativeChargeEvaluator::prepared_group(
    std::int64_t simplex_id,
    std::int64_t levels
) {
    const GroupKey key{simplex_id, levels};
    auto it = prepared_group_cache_.find(key);
    if (it != prepared_group_cache_.end()) {
        return it->second;
    }

    PreparedChargeGroup group;
    group.owner_id = simplex_id;
    if (levels <= 0) {
        group.leaf_ids.push_back(simplex_id);
    } else {
        geometry_->descendant_leaves_impl(simplex_id, levels, group.leaf_ids);
        if (levels == 1) {
            group.child_ids = group.leaf_ids;
        }
    }
    group.cells.reserve(group.leaf_ids.size());
    for (const auto leaf_id : group.leaf_ids) {
        group.cells.push_back(prepared_cell(leaf_id));
    }
    auto [inserted, _ok] = prepared_group_cache_.emplace(key, std::move(group));
    return inserted->second;
}

const NativeChargeEvaluator::PreparedChargeCell &NativeChargeEvaluator::prepared_cell(
    std::int64_t simplex_id
) {
    auto it = prepared_cell_cache_.find(simplex_id);
    if (it != prepared_cell_cache_.end()) {
        return it->second;
    }

    PreparedChargeCell cell;
    const auto &record = geometry_->simplex_records_[static_cast<size_t>(simplex_id)];
    const size_t n_vertices = record.vertex_ids.size();
    const size_t ndof = spectral_cache_->ndof();
    const size_t dimension = geometry_->ndim_;

    cell.points_flat = geometry_->simplex_points_flat(simplex_id);
    cell.vertex_energies.resize(n_vertices * ndof);
    cell.sorted_energies.resize(ndof * n_vertices);
    cell.simplex_weights.assign(ndof * n_vertices, 0.0);
    cell.band_min.resize(ndof);
    cell.band_max.resize(ndof);
    cell.flat_energy.resize(ndof);
    cell.distinct_mask.resize(ndof, 1);
    cell.flat_mask.resize(ndof, 0);
    cell.volume = geometry_->simplex_volume(simplex_id);

    for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
        const auto vertex_id = record.vertex_ids[vertex];
        const auto &values = vertex_eigenvalues(vertex_id);
        std::copy_n(values.data(), ndof, cell.vertex_energies.data() + vertex * ndof);
    }

    std::vector<double> scratch(n_vertices);
    for (size_t band = 0; band < ndof; ++band) {
        for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
            scratch[vertex] = cell.vertex_energies[vertex * ndof + band];
        }
        std::sort(scratch.begin(), scratch.end());
        cell.band_min[band] = scratch.front();
        cell.band_max[band] = scratch.back();
        cell.flat_energy[band] = cell.vertex_energies[band];
        cell.flat_mask[band] = (scratch.back() - scratch.front()) <= tol_ ? 1 : 0;
        bool distinct = true;
        for (size_t vertex = 1; vertex < n_vertices; ++vertex) {
            if (scratch[vertex] - scratch[vertex - 1] <= tol_) {
                distinct = false;
            }
            cell.sorted_energies[band * n_vertices + vertex - 1] = scratch[vertex - 1];
        }
        cell.sorted_energies[band * n_vertices + n_vertices - 1] = scratch[n_vertices - 1];
        cell.distinct_mask[band] = distinct ? 1 : 0;
        if (dimension > 3 && distinct) {
            for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
                double denom = 1.0;
                for (size_t other = 0; other < n_vertices; ++other) {
                    if (other == vertex) {
                        continue;
                    }
                    denom *= scratch[vertex] - scratch[other];
                }
                cell.simplex_weights[band * n_vertices + vertex] = 1.0 / denom;
            }
        }
    }

    auto [inserted, _ok] = prepared_cell_cache_.emplace(simplex_id, std::move(cell));
    return inserted->second;
}

const std::vector<double> &NativeChargeEvaluator::vertex_eigenvalues(std::int64_t vertex_id) {
    auto it = vertex_values_cache_.find(vertex_id);
    if (it != vertex_values_cache_.end()) {
        return it->second;
    }

    const size_t ndim = geometry_->ndim_;
    std::vector<double> reduced_point(ndim);
    const size_t base = static_cast<size_t>(vertex_id) * ndim;
    std::copy_n(geometry_->vertices_.data() + base, ndim, reduced_point.data());
    const auto &entry = spectral_cache_->entry_for_reduced_point(reduced_point.data());
    auto [inserted, _ok] = vertex_values_cache_.emplace(vertex_id, entry.eigenvalues);
    return inserted->second;
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
    double total_indicator,
    double bulk_theta
) {
    if (owner_ids.empty() || total_indicator <= 0.0) {
        return {};
    }
    std::vector<size_t> order(owner_ids.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(
        order.begin(),
        order.end(),
        [&](size_t left, size_t right) { return indicators[left] > indicators[right]; }
    );
    const double target = bulk_theta * total_indicator;
    std::vector<std::int64_t> marked;
    double accumulated = 0.0;
    for (const auto index : order) {
        if (indicators[index] <= 0.0) {
            continue;
        }
        marked.push_back(owner_ids[index]);
        accumulated += indicators[index];
        if (accumulated >= target) {
            break;
        }
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
    const PreparedChargeCell &cell,
    double mu,
    double &charge,
    double &derivative,
    bool &derivative_exact
) const {
    const size_t n_vertices = geometry_->ndim_ + 1;
    const size_t ndof = spectral_cache_->ndof();
    double cell_min = std::numeric_limits<double>::infinity();
    double cell_max = -std::numeric_limits<double>::infinity();
    for (size_t band = 0; band < ndof; ++band) {
        cell_min = std::min(cell_min, cell.band_min[band]);
        cell_max = std::max(cell_max, cell.band_max[band]);
    }
    if (cell_max <= mu) {
        charge += cell.volume * static_cast<double>(ndof);
        return;
    }
    if (cell_min > mu) {
        return;
    }

    for (size_t band = 0; band < ndof; ++band) {
        const bool full = cell.band_max[band] <= mu;
        const bool empty = cell.band_min[band] > mu;
        const bool flat_half =
            cell.flat_mask[band] &&
            !full &&
            !empty &&
            std::abs(cell.flat_energy[band] - mu) <= tol_;
        if (full) {
            charge += cell.volume;
            continue;
        }
        if (empty) {
            continue;
        }
        if (flat_half) {
            charge += 0.5 * cell.volume;
            continue;
        }
        if (cell.distinct_mask[band]) {
            const double *energies = cell.sorted_energies.data() + band * n_vertices;
            const double *weights =
                geometry_->ndim_ > 3 ? cell.simplex_weights.data() + band * n_vertices : nullptr;
            const auto [fraction, dfraction] = simplex_fraction_and_derivative(
                energies,
                mu,
                geometry_->ndim_,
                weights,
                tol_
            );
            charge += cell.volume * fraction;
            derivative += cell.volume * dfraction;
            continue;
        }

        derivative_exact = false;
        std::vector<double> band_energies(n_vertices);
        for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
            band_energies[vertex] = cell.vertex_energies[vertex * ndof + band];
        }
        const auto pieces = occupied_subsimplices_from_flat(
            cell.points_flat.data(),
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
