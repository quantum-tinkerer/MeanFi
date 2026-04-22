#include "adaptive_integrator.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace meanfi::zero_temp_native {

AdaptiveIntegrator::AdaptiveIntegrator(
    Geometry &geometry,
    VertexCache &vertex_cache,
    Float2D keys,
    double tol
)
    : geometry_(&geometry),
      vertex_cache_(&vertex_cache),
      tol_(tol),
      n_keys_(keys.shape(0)),
      ndim_(geometry.ndim()),
      ndof_(vertex_cache.ndof()),
      ncomp_(vertex_cache.ndof() * vertex_cache.ndof() * keys.shape(0)) {
    if (keys.shape(1) != ndim_) {
        throw std::runtime_error("AdaptiveIntegrator: key dimension mismatch");
    }
    keys_.assign(keys.data(), keys.data() + n_keys_ * ndim_);
    if (n_keys_ > 0) {
        phase_layout_id_ = vertex_cache_->register_phase_layout(keys_, n_keys_, ndim_);
    }
}

void AdaptiveIntegrator::clear_mu_dependent_caches() {
    std::fill(value_ready_.begin(), value_ready_.end(), 0);
    std::fill(leaf_ready_.begin(), leaf_ready_.end(), 0);
    cached_simplex_value_count_ = 0;
    leaf_heap_ = std::priority_queue<HeapEntry>();
    leaf_build_count_ = 0;
}

void AdaptiveIntegrator::clear() {
    clear_mu_dependent_caches();
    current_mu_ = std::numeric_limits<double>::quiet_NaN();
}

size_t AdaptiveIntegrator::phase_cache_size() const noexcept {
    if (phase_layout_id_ == std::numeric_limits<size_t>::max()) {
        return 0;
    }
    return vertex_cache_->phase_cache_size(phase_layout_id_);
}

void AdaptiveIntegrator::ensure_mu(double mu) {
    if (std::isnan(current_mu_) || std::abs(mu - current_mu_) > tol_) {
        current_mu_ = mu;
        clear_mu_dependent_caches();
    }
}

void AdaptiveIntegrator::ensure_layout_capacity(std::int64_t simplex_id) {
    const size_t needed = static_cast<size_t>(simplex_id) + 1;
    if (layout_cache_.size() >= needed) {
        return;
    }
    layout_cache_.resize(needed);
    layout_ready_.resize(needed, 0);
}

void AdaptiveIntegrator::ensure_value_capacity(std::int64_t simplex_id) {
    const size_t needed = static_cast<size_t>(simplex_id) + 1;
    if (value_cache_.size() >= needed) {
        return;
    }
    value_cache_.resize(needed);
    value_ready_.resize(needed, 0);
}

void AdaptiveIntegrator::ensure_leaf_capacity(std::int64_t simplex_id) {
    const size_t needed = static_cast<size_t>(simplex_id) + 1;
    if (leaf_cache_.size() >= needed) {
        return;
    }
    leaf_cache_.resize(needed);
    leaf_ready_.resize(needed, 0);
}

void AdaptiveIntegrator::gather_vertex_entries(
    std::int64_t simplex_id,
    std::vector<const VertexCache::Entry *> &entries
) {
    const auto &vertex_ids = geometry_->simplex_vertex_ids_vector(simplex_id);
    entries.resize(vertex_ids.size());
    for (size_t vertex = 0; vertex < vertex_ids.size(); ++vertex) {
        entries[vertex] = &vertex_cache_->entry_for_vertex(*geometry_, vertex_ids[vertex]);
    }
}

const AdaptiveIntegrator::SimplexLayout &AdaptiveIntegrator::simplex_layout(
    std::int64_t simplex_id,
    const std::vector<const VertexCache::Entry *> &entries
) {
    ensure_layout_capacity(simplex_id);
    if (layout_ready_[static_cast<size_t>(simplex_id)]) {
        return layout_cache_[static_cast<size_t>(simplex_id)];
    }

    const size_t n_vertices = geometry_->simplex_vertex_ids_vector(simplex_id).size();
    SimplexLayout layout;
    layout.simplex_id = simplex_id;
    layout.bands.resize(ndof_);

    for (size_t band = 0; band < ndof_; ++band) {
        auto &band_layout = layout.bands[band];
        band_layout.order.resize(n_vertices);
        std::iota(band_layout.order.begin(), band_layout.order.end(), std::uint8_t{0});
        std::stable_sort(
            band_layout.order.begin(),
            band_layout.order.end(),
            [&](std::uint8_t left, std::uint8_t right) {
                return entries[left]->eigenvalues[band] < entries[right]->eigenvalues[band];
            }
        );
        band_layout.strictly_ordered = true;
        for (size_t pos = 1; pos < n_vertices; ++pos) {
            const double curr = entries[band_layout.order[pos]]->eigenvalues[band];
            const double prev = entries[band_layout.order[pos - 1]]->eigenvalues[band];
            if (curr - prev <= tol_) {
                band_layout.strictly_ordered = false;
                break;
            }
        }
    }

    layout_cache_[static_cast<size_t>(simplex_id)] = std::move(layout);
    layout_ready_[static_cast<size_t>(simplex_id)] = 1;
    return layout_cache_[static_cast<size_t>(simplex_id)];
}

AdaptiveIntegrator::SimplexOccupation AdaptiveIntegrator::simplex_occupation(
    std::int64_t simplex_id,
    double mu,
    const SimplexLayout &layout,
    const std::vector<const VertexCache::Entry *> &entries
) {
    SimplexOccupation occupation;
    occupation.simplex_id = simplex_id;
    occupation.layout = &layout;
    occupation.bands.resize(ndof_);

    for (size_t band = 0; band < ndof_; ++band) {
        const auto &order = layout.bands[band].order;
        const double band_min = entries[order.front()]->eigenvalues[band];
        const double band_max = entries[order.back()]->eigenvalues[band];
        bool half_mask = true;
        for (const auto local_vertex : order) {
            if (std::abs(entries[local_vertex]->eigenvalues[band] - mu) > tol_) {
                half_mask = false;
                break;
            }
        }

        auto &band_occ = occupation.bands[band];
        if (band_min > mu + tol_) {
            band_occ.classification = OccupancyClass::Empty;
        } else if (half_mask) {
            band_occ.classification = OccupancyClass::Half;
        } else if (band_max <= mu + tol_) {
            band_occ.classification = OccupancyClass::Full;
        } else {
            band_occ.classification = OccupancyClass::Cut;
        }
    }

    return occupation;
}

void AdaptiveIntegrator::collect_simplex_ids(
    std::int64_t owner_id,
    std::int64_t levels,
    std::vector<std::int64_t> &out
) {
    out.clear();
    geometry_->descendant_leaves_impl(owner_id, levels, out);
}

nb::tuple AdaptiveIntegrator::evaluate_charge(double mu, std::int64_t levels) {
    auto result = evaluate_charge_impl(mu, levels);
    return nb::make_tuple(
        result.total_charge,
        result.derivative_exact ? result.total_derivative : std::numeric_limits<double>::quiet_NaN(),
        result.derivative_exact,
        make_array(std::vector<std::int64_t>(result.owner_ids), {result.owner_ids.size()}),
        make_array(std::vector<double>(result.owner_charges), {result.owner_charges.size()}),
        result.evaluator_evals
    );
}

AdaptiveIntegrator::ChargeEvalResult AdaptiveIntegrator::evaluate_charge_impl(
    double mu,
    std::int64_t levels
) {
    ChargeEvalResult result;
    result.owner_ids = geometry_->active_simplex_ids_vector();
    result.owner_charges.resize(result.owner_ids.size(), 0.0);

    std::vector<std::int64_t> leaf_ids;
    for (size_t owner_index = 0; owner_index < result.owner_ids.size(); ++owner_index) {
        double owner_charge = 0.0;
        double owner_derivative = 0.0;
        bool owner_exact = true;
        collect_simplex_ids(result.owner_ids[owner_index], levels, leaf_ids);
        for (const auto leaf_id : leaf_ids) {
            evaluate_charge_simplex(leaf_id, mu, owner_charge, owner_derivative, owner_exact);
            result.evaluator_evals += static_cast<std::int64_t>(ndof_);
        }
        result.owner_charges[owner_index] = owner_charge;
        result.total_charge += owner_charge;
        result.total_derivative += owner_derivative;
        result.derivative_exact = result.derivative_exact && owner_exact;
    }
    return result;
}

AdaptiveIntegrator::ChargeEvalResult AdaptiveIntegrator::evaluate_charge_counted(
    double mu,
    std::int64_t levels,
    std::int64_t &integration_calls,
    std::int64_t &evaluator_evals
) {
    ++integration_calls;
    auto result = evaluate_charge_impl(mu, levels);
    evaluator_evals += result.evaluator_evals;
    return result;
}

void AdaptiveIntegrator::evaluate_charge_simplex(
    std::int64_t simplex_id,
    double mu,
    double &charge,
    double &derivative,
    bool &derivative_exact
) {
    std::vector<const VertexCache::Entry *> entries;
    gather_vertex_entries(simplex_id, entries);
    const auto &layout = simplex_layout(simplex_id, entries);
    const auto occupation = simplex_occupation(simplex_id, mu, layout, entries);
    const size_t n_vertices = entries.size();
    const double volume = geometry_->simplex_volume(simplex_id);

    std::vector<double> sorted_energies(n_vertices);
    std::vector<double> simplex_weights(n_vertices, 0.0);

    for (size_t band = 0; band < ndof_; ++band) {
        const auto classification = occupation.bands[band].classification;
        if (classification == OccupancyClass::Empty) {
            continue;
        }
        if (classification == OccupancyClass::Full) {
            charge += volume;
            continue;
        }
        if (classification == OccupancyClass::Half) {
            charge += 0.5 * volume;
            continue;
        }

        const auto &order = layout.bands[band].order;
        for (size_t pos = 0; pos < n_vertices; ++pos) {
            sorted_energies[pos] = entries[order[pos]]->eigenvalues[band];
        }

        if (layout.bands[band].strictly_ordered) {
            const double *weights = nullptr;
            if (ndim_ > 3) {
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
            const auto [fraction, dfraction] = simplex_rules::simplex_fraction_and_derivative(
                sorted_energies.data(),
                mu,
                ndim_,
                weights,
                tol_
            );
            charge += volume * fraction;
            derivative += volume * dfraction;
            continue;
        }

        derivative_exact = false;
        const auto fractions = simplex_rules::occupied_linear_moment_fractions(
            sorted_energies.data(),
            ndim_,
            mu,
            tol_
        );
        charge += volume * std::accumulate(fractions.begin(), fractions.end(), 0.0);
    }
}

AdaptiveIntegrator::RootSolveResult AdaptiveIntegrator::solve_mu_on_preview(
    double filling,
    double mu_guess,
    double lower,
    double upper,
    const ChargeSolveOptions &options,
    std::int64_t &integration_calls,
    std::int64_t &evaluator_evals
) {
    auto lower_eval = evaluate_charge_counted(lower, 1, integration_calls, evaluator_evals);
    auto upper_eval = evaluate_charge_counted(upper, 1, integration_calls, evaluator_evals);
    double lower_charge = lower_eval.total_charge;
    double upper_charge = upper_eval.total_charge;
    double mu = std::clamp(mu_guess, lower, upper);
    if (!(lower < mu && mu < upper)) {
        mu = 0.5 * (lower + upper);
    }

    double last_derivative = std::numeric_limits<double>::quiet_NaN();
    RootSolveResult result;

    for (std::int64_t iteration = 1; iteration <= options.max_mu_iterations; ++iteration) {
        const auto summary = evaluate_charge_counted(mu, 1, integration_calls, evaluator_evals);
        const double residual = summary.total_charge - filling;
        if (std::abs(residual) <= options.charge_tol) {
            result.mu = mu;
            result.charge = summary.total_charge;
            result.derivative = last_derivative;
            result.iterations = iteration;
            return result;
        }

        if (residual < 0.0) {
            lower = mu;
            lower_charge = summary.total_charge;
        } else {
            upper = mu;
            upper_charge = summary.total_charge;
        }

        if (upper - lower <= options.mu_xtol) {
            result.mu = 0.5 * (lower + upper);
            result.charge = summary.total_charge;
            result.derivative = last_derivative;
            result.iterations = iteration;
            return result;
        }

        double slope =
            summary.derivative_exact && std::isfinite(summary.total_derivative) &&
                    summary.total_derivative > 0.0
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
    const auto summary = evaluate_charge_counted(midpoint, 1, integration_calls, evaluator_evals);
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

void AdaptiveIntegrator::expand_mu_bracket(
    double filling,
    double &lower,
    double &upper,
    std::int64_t &integration_calls,
    std::int64_t &evaluator_evals
) {
    auto lower_eval = evaluate_charge_counted(lower, 1, integration_calls, evaluator_evals);
    auto upper_eval = evaluate_charge_counted(upper, 1, integration_calls, evaluator_evals);
    double lower_charge = lower_eval.total_charge;
    double upper_charge = upper_eval.total_charge;
    while (lower_charge > filling || upper_charge < filling) {
        lower *= 2.0;
        upper *= 2.0;
        lower_eval = evaluate_charge_counted(lower, 1, integration_calls, evaluator_evals);
        upper_eval = evaluate_charge_counted(upper, 1, integration_calls, evaluator_evals);
        lower_charge = lower_eval.total_charge;
        upper_charge = upper_eval.total_charge;
    }
}

std::vector<std::int64_t> AdaptiveIntegrator::bulk_mark(
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

ChargeSolveResult AdaptiveIntegrator::solve_mu_and_refine(
    double filling,
    const ChargeSolveOptions &options
) {
    double bound = vertex_cache_->spectral_bound() + 1.0;
    double lower = -bound;
    double upper = bound;
    std::int64_t integration_calls = 0;
    std::int64_t evaluator_evals = 0;
    std::int64_t refinements = 0;
    std::int64_t remaining = options.max_subdivisions;

    expand_mu_bracket(filling, lower, upper, integration_calls, evaluator_evals);

    double mu = std::clamp(options.mu_guess, lower, upper);
    if (!(lower < mu && mu < upper)) {
        mu = 0.5 * (lower + upper);
    }

    ChargeSolveResult result;
    result.mu = mu;
    result.converged = false;

    while (true) {
        ChargeSolveOptions root_options = options;
        root_options.charge_tol = std::max(
            0.25 * options.charge_tol,
            std::numeric_limits<double>::epsilon()
        );
        const auto root = solve_mu_on_preview(
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

        const auto coarse = evaluate_charge_counted(mu, 0, integration_calls, evaluator_evals);
        const auto preview = evaluate_charge_counted(mu, 1, integration_calls, evaluator_evals);
        if (coarse.owner_ids != preview.owner_ids) {
            throw std::runtime_error("AdaptiveIntegrator: coarse/preview owner mismatch");
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
    result.n_leaves = static_cast<std::int64_t>(geometry_->n_active());
    result.n_leaf_nodes = static_cast<std::int64_t>(geometry_->n_leaf_vertices());
    return result;
}

nb::tuple AdaptiveIntegrator::evaluate_density(double mu, std::int64_t levels) {
    ensure_mu(mu);
    const auto result = evaluate_density_impl(mu, levels);
    return nb::make_tuple(
        make_array(
            std::move(std::vector<std::complex<double>>(result.total_estimate)),
            {ncomp_}
        ),
        make_array(std::move(std::vector<std::int64_t>(result.owner_ids)), {result.owner_ids.size()}),
        make_array(
            std::move(std::vector<std::complex<double>>(result.owner_estimates)),
            {result.owner_ids.size(), ncomp_}
        ),
        result.evaluator_evals
    );
}

AdaptiveIntegrator::DensityEvalResult AdaptiveIntegrator::evaluate_density_impl(
    double mu,
    std::int64_t levels
) {
    ensure_mu(mu);
    DensityEvalResult result;
    result.total_estimate.assign(ncomp_, std::complex<double>(0.0, 0.0));
    result.owner_ids = geometry_->active_simplex_ids_vector();
    result.owner_estimates.assign(
        result.owner_ids.size() * ncomp_,
        std::complex<double>(0.0, 0.0)
    );

    std::vector<std::int64_t> leaf_ids;
    for (size_t owner_index = 0; owner_index < result.owner_ids.size(); ++owner_index) {
        collect_simplex_ids(result.owner_ids[owner_index], levels, leaf_ids);
        const size_t base = owner_index * ncomp_;
        for (const auto leaf_id : leaf_ids) {
            const auto &value = cached_simplex_value(leaf_id);
            for (size_t comp = 0; comp < ncomp_; ++comp) {
                result.owner_estimates[base + comp] += value.estimate[comp];
                result.total_estimate[comp] += value.estimate[comp];
            }
            result.evaluator_evals += value.evaluator_evals;
        }
    }
    return result;
}

const AdaptiveIntegrator::SimplexDensityEstimate &AdaptiveIntegrator::cached_simplex_value(
    std::int64_t simplex_id
) {
    ensure_value_capacity(simplex_id);
    if (value_ready_[static_cast<size_t>(simplex_id)]) {
        return value_cache_[static_cast<size_t>(simplex_id)];
    }
    value_cache_[static_cast<size_t>(simplex_id)] = evaluate_density_simplex(simplex_id);
    value_ready_[static_cast<size_t>(simplex_id)] = 1;
    ++cached_simplex_value_count_;
    return value_cache_[static_cast<size_t>(simplex_id)];
}

const AdaptiveIntegrator::LeafDensityContribution &AdaptiveIntegrator::leaf_contribution(
    std::int64_t simplex_id
) {
    ensure_leaf_capacity(simplex_id);
    if (leaf_ready_[static_cast<size_t>(simplex_id)]) {
        return leaf_cache_[static_cast<size_t>(simplex_id)];
    }

    LeafDensityContribution leaf;
    const auto &coarse = cached_simplex_value(simplex_id);
    leaf.coarse_value = coarse.estimate;
    leaf.preview_value.assign(ncomp_, std::complex<double>(0.0, 0.0));
    leaf.error_vector.assign(ncomp_, 0.0);
    leaf.evaluator_evals = coarse.evaluator_evals;

    const auto &preview_ids = geometry_->ensure_children_vector(simplex_id);
    for (const auto preview_id : preview_ids) {
        const auto &preview_value = cached_simplex_value(preview_id);
        for (size_t comp = 0; comp < ncomp_; ++comp) {
            leaf.preview_value[comp] += preview_value.estimate[comp];
        }
        leaf.evaluator_evals += preview_value.evaluator_evals;
    }

    for (size_t comp = 0; comp < ncomp_; ++comp) {
        leaf.error_vector[comp] = std::abs(leaf.preview_value[comp] - leaf.coarse_value[comp]);
        leaf.indicator = std::max(leaf.indicator, leaf.error_vector[comp]);
    }

    leaf_cache_[static_cast<size_t>(simplex_id)] = std::move(leaf);
    leaf_ready_[static_cast<size_t>(simplex_id)] = 1;
    ++leaf_build_count_;
    leaf_heap_.push(HeapEntry{leaf_cache_[static_cast<size_t>(simplex_id)].indicator, simplex_id});
    return leaf_cache_[static_cast<size_t>(simplex_id)];
}

const AdaptiveIntegrator::LeafDensityContribution *AdaptiveIntegrator::active_leaf_with_max_error() {
    while (!leaf_heap_.empty()) {
        const auto top = leaf_heap_.top();
        if (!leaf_ready_[static_cast<size_t>(top.simplex_id)]) {
            leaf_heap_.pop();
            continue;
        }
        return &leaf_cache_[static_cast<size_t>(top.simplex_id)];
    }
    return nullptr;
}

void AdaptiveIntegrator::accumulate_vertex_band(
    std::vector<std::complex<double>> &out,
    const std::complex<double> *phases,
    const std::complex<double> *eigenvectors,
    size_t n_all_bands,
    size_t band,
    double scale
) const {
    for (size_t i = 0; i < ndof_; ++i) {
        const std::complex<double> ui = eigenvectors[i * n_all_bands + band];
        for (size_t j = 0; j < ndof_; ++j) {
            const std::complex<double> projector =
                scale * ui * std::conj(eigenvectors[j * n_all_bands + band]);
            const size_t base = (i * ndof_ + j) * n_keys_;
            for (size_t key_index = 0; key_index < n_keys_; ++key_index) {
                out[base + key_index] += projector * phases[key_index];
            }
        }
    }
}

AdaptiveIntegrator::SimplexDensityEstimate AdaptiveIntegrator::evaluate_density_simplex(
    std::int64_t simplex_id
) {
    SimplexDensityEstimate result;
    result.estimate.assign(ncomp_, std::complex<double>(0.0, 0.0));

    std::vector<const VertexCache::Entry *> entries;
    gather_vertex_entries(simplex_id, entries);
    const auto &layout = simplex_layout(simplex_id, entries);
    const auto occupation = simplex_occupation(simplex_id, current_mu_, layout, entries);
    const auto &vertex_ids = geometry_->simplex_vertex_ids_vector(simplex_id);
    const size_t n_vertices = vertex_ids.size();
    const double volume = geometry_->simplex_volume(simplex_id);
    std::vector<double> weights(n_vertices, 0.0);
    std::vector<double> sorted_energies(n_vertices, 0.0);

    for (size_t band = 0; band < ndof_; ++band) {
        std::fill(weights.begin(), weights.end(), 0.0);
        const auto classification = occupation.bands[band].classification;
        if (classification == OccupancyClass::Empty) {
            continue;
        }

        if (classification == OccupancyClass::Full || classification == OccupancyClass::Half) {
            const double band_factor = classification == OccupancyClass::Half ? 0.5 : 1.0;
            const double weight = volume * band_factor / static_cast<double>(n_vertices);
            std::fill(weights.begin(), weights.end(), weight);
        } else {
            const auto &order = layout.bands[band].order;
            for (size_t pos = 0; pos < n_vertices; ++pos) {
                sorted_energies[pos] = entries[order[pos]]->eigenvalues[band];
            }
            const auto fractions = simplex_rules::occupied_linear_moment_fractions(
                sorted_energies.data(),
                ndim_,
                current_mu_,
                tol_
            );
            for (size_t pos = 0; pos < n_vertices; ++pos) {
                const auto local_vertex = static_cast<size_t>(order[pos]);
                weights[local_vertex] += volume * fractions[pos];
            }
        }

        for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
            if (weights[vertex] == 0.0) {
                continue;
            }
            const auto &phases =
                vertex_cache_->phases_for_vertex(*geometry_, vertex_ids[vertex], phase_layout_id_);
            accumulate_vertex_band(
                result.estimate,
                phases.data(),
                entries[vertex]->eigenvectors.data(),
                ndof_,
                band,
                weights[vertex]
            );
        }
        result.evaluator_evals += static_cast<std::int64_t>(n_vertices);
    }

    return result;
}

DensityIntegrateResult AdaptiveIntegrator::integrate_density(
    double mu,
    const DensityIntegrateOptions &options
) {
    ensure_mu(mu);
    std::fill(leaf_ready_.begin(), leaf_ready_.end(), 0);
    leaf_heap_ = std::priority_queue<HeapEntry>();
    leaf_build_count_ = 0;

    DensityIntegrateResult result;
    result.error_estimate_available = true;
    result.converged = false;
    result.estimate.assign(ncomp_, std::complex<double>(0.0, 0.0));

    std::int64_t remaining = options.max_subdivisions;
    for (const auto simplex_id : geometry_->active_simplex_ids_vector()) {
        const auto &leaf = leaf_contribution(simplex_id);
        for (size_t comp = 0; comp < ncomp_; ++comp) {
            result.estimate[comp] += leaf.preview_value[comp];
        }
        result.evaluator_evals += leaf.evaluator_evals;
    }

    while (true) {
        double estimate_max = 0.0;
        for (const auto &value : result.estimate) {
            estimate_max = std::max(estimate_max, std::abs(value));
        }

        const auto *max_leaf = active_leaf_with_max_error();
        const double max_indicator = max_leaf != nullptr ? max_leaf->indicator : 0.0;
        const double tolerance = options.density_atol + options.density_rtol * estimate_max;
        if (max_indicator <= tolerance) {
            result.converged = true;
            break;
        }
        if (remaining == 0) {
            throw std::runtime_error("Adaptive zero-temperature density integration did not converge");
        }

        const auto &active_ids = geometry_->active_simplex_ids_vector();
        std::vector<double> indicators(active_ids.size(), 0.0);
        for (size_t index = 0; index < active_ids.size(); ++index) {
            const auto simplex_id = active_ids[index];
            ensure_leaf_capacity(simplex_id);
            if (leaf_ready_[static_cast<size_t>(simplex_id)]) {
                indicators[index] = leaf_cache_[static_cast<size_t>(simplex_id)].indicator;
            }
        }
        const auto marked = bulk_mark(active_ids, indicators, max_indicator, options.bulk_theta);
        if (marked.empty()) {
            break;
        }

        const auto batch = geometry_->refine_marked(marked);
        result.subdivisions += batch.refinements;
        if (remaining > 0) {
            remaining -= batch.refinements;
            if (remaining < 0) {
                throw std::runtime_error("Adaptive zero-temperature density integration did not converge");
            }
        }

        for (const auto parent_id : batch.parent_ids) {
            ensure_leaf_capacity(parent_id);
            if (!leaf_ready_[static_cast<size_t>(parent_id)]) {
                continue;
            }
            const auto &parent = leaf_cache_[static_cast<size_t>(parent_id)];
            for (size_t comp = 0; comp < ncomp_; ++comp) {
                result.estimate[comp] -= parent.preview_value[comp];
            }
            leaf_ready_[static_cast<size_t>(parent_id)] = 0;
        }
        for (const auto child_id : batch.child_ids) {
            const auto &leaf = leaf_contribution(child_id);
            for (size_t comp = 0; comp < ncomp_; ++comp) {
                result.estimate[comp] += leaf.preview_value[comp];
            }
            result.evaluator_evals += leaf.evaluator_evals;
        }
    }

    result.error_vector.assign(ncomp_, 0.0);
    result.error_scalar = 0.0;
    for (const auto simplex_id : geometry_->active_simplex_ids_vector()) {
        ensure_leaf_capacity(simplex_id);
        if (!leaf_ready_[static_cast<size_t>(simplex_id)]) {
            continue;
        }
        const auto &leaf = leaf_cache_[static_cast<size_t>(simplex_id)];
        result.error_scalar = std::max(result.error_scalar, leaf.indicator);
        for (size_t comp = 0; comp < ncomp_; ++comp) {
            result.error_vector[comp] = std::max(result.error_vector[comp], leaf.error_vector[comp]);
        }
    }

    result.n_leaves = static_cast<std::int64_t>(geometry_->n_active());
    result.n_leaf_nodes = static_cast<std::int64_t>(geometry_->n_leaf_vertices());
    return result;
}

}  // namespace meanfi::zero_temp_native
