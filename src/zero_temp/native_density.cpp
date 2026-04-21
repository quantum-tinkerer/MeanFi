#include "native_density.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace meanfi::zero_temp_native {

NativeDensityEvaluator::NativeDensityEvaluator(
    NativeGeometry &geometry,
    NativeSpectralCache &spectral_cache,
    Float2D keys,
    double tol
)
    : geometry_(&geometry),
      spectral_cache_(&spectral_cache),
      simplex_cache_(geometry),
      tol_(tol),
      n_keys_(keys.shape(0)),
      ndim_(keys.shape(1)),
      ndof_(spectral_cache.ndof()),
      ncomp_(spectral_cache.ndof() * spectral_cache.ndof() * keys.shape(0)) {
    keys_.assign(keys.data(), keys.data() + n_keys_ * ndim_);
}

void NativeDensityEvaluator::clear() {
    value_cache_.clear();
    leaf_cache_.clear();
    leaf_heap_ = std::priority_queue<HeapEntry>();
    leaf_build_count_ = 0;
    current_mu_ = std::numeric_limits<double>::quiet_NaN();
}

nb::tuple NativeDensityEvaluator::evaluate(
    NativeFrontier &frontier,
    double mu,
    std::int64_t levels
) {
    ensure_mu(mu);
    const auto result = evaluate_impl(frontier, levels);
    return nb::make_tuple(
        make_array(std::move(std::vector<std::complex<double>>(result.total_estimate)), {ncomp_}),
        make_array(std::move(std::vector<std::int64_t>(result.owner_ids)), {result.owner_ids.size()}),
        make_array(
            std::move(std::vector<std::complex<double>>(result.owner_estimates)),
            {result.owner_ids.size(), ncomp_}
        ),
        result.evaluator_evals
    );
}

nb::tuple NativeDensityEvaluator::evaluate_many(Int1D simplex_ids, double mu) {
    ensure_mu(mu);
    std::vector<std::int64_t> ids(simplex_ids.data(), simplex_ids.data() + simplex_ids.shape(0));

    const size_t count = ids.size();
    std::vector<std::complex<double>> estimates(count * ncomp_);
    std::vector<double> error_vectors(count * ncomp_, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> error_scalars(count, std::numeric_limits<double>::quiet_NaN());
    std::int64_t evaluator_evals = 0;

    for (size_t index = 0; index < count; ++index) {
        const auto &value = cached_simplex_value(ids[index]);
        std::copy_n(value.estimate.data(), ncomp_, estimates.data() + index * ncomp_);
        evaluator_evals += value.evaluator_evals;
    }

    return nb::make_tuple(
        make_array(std::move(estimates), {count, ncomp_}),
        make_array(std::move(error_vectors), {count, ncomp_}),
        make_array(std::move(error_scalars), {count}),
        evaluator_evals
    );
}

DensityIntegrateResult NativeDensityEvaluator::integrate_adaptive(
    NativeFrontier &frontier,
    double mu,
    const DensityIntegrateOptions &options
) {
    ensure_mu(mu);
    leaf_cache_.clear();
    leaf_heap_ = std::priority_queue<HeapEntry>();
    leaf_build_count_ = 0;

    DensityIntegrateResult result;
    result.error_estimate_available = true;
    result.converged = false;
    result.estimate.assign(ncomp_, std::complex<double>(0.0, 0.0));

    std::int64_t remaining = options.max_subdivisions;
    for (const auto simplex_id : frontier.active_simplex_ids_) {
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

        std::vector<double> indicators(frontier.active_simplex_ids_.size(), 0.0);
        for (size_t index = 0; index < frontier.active_simplex_ids_.size(); ++index) {
            const auto it = leaf_cache_.find(frontier.active_simplex_ids_[index]);
            if (it != leaf_cache_.end()) {
                indicators[index] = it->second.indicator;
            }
        }
        const auto marked = bulk_mark(
            frontier.active_simplex_ids_,
            indicators,
            max_indicator,
            options.bulk_theta
        );
        if (marked.empty()) {
            break;
        }

        const auto batch = geometry_->refine_marked(marked);
        frontier.apply_refinement(batch);
        result.subdivisions += batch.refinements;
        if (remaining > 0) {
            remaining -= batch.refinements;
            if (remaining < 0) {
                throw std::runtime_error("Adaptive zero-temperature density integration did not converge");
            }
        }

        for (const auto parent_id : batch.parent_ids) {
            auto it = leaf_cache_.find(parent_id);
            if (it == leaf_cache_.end()) {
                continue;
            }
            for (size_t comp = 0; comp < ncomp_; ++comp) {
                result.estimate[comp] -= it->second.preview_value[comp];
            }
            leaf_cache_.erase(it);
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
    for (const auto simplex_id : frontier.active_simplex_ids_) {
        const auto it = leaf_cache_.find(simplex_id);
        if (it == leaf_cache_.end()) {
            continue;
        }
        result.error_scalar = std::max(result.error_scalar, it->second.indicator);
        for (size_t comp = 0; comp < ncomp_; ++comp) {
            result.error_vector[comp] = std::max(
                result.error_vector[comp],
                it->second.error_vector[comp]
            );
        }
    }

    result.n_leaves = static_cast<std::int64_t>(frontier.n_active());
    result.n_leaf_nodes = static_cast<std::int64_t>(frontier.n_leaf_vertices());
    return result;
}

void NativeDensityEvaluator::ensure_mu(double mu) {
    if (std::isnan(current_mu_) || std::abs(mu - current_mu_) > tol_) {
        current_mu_ = mu;
        value_cache_.clear();
        leaf_cache_.clear();
        leaf_heap_ = std::priority_queue<HeapEntry>();
        leaf_build_count_ = 0;
    }
}

NativeDensityEvaluator::DensityEvalResult NativeDensityEvaluator::evaluate_impl(
    NativeFrontier &frontier,
    std::int64_t levels
) {
    DensityEvalResult result;
    result.total_estimate.assign(ncomp_, std::complex<double>(0.0, 0.0));
    result.owner_ids = frontier.active_simplex_ids_;
    result.owner_estimates.assign(
        result.owner_ids.size() * ncomp_, std::complex<double>(0.0, 0.0)
    );

    for (size_t owner_index = 0; owner_index < result.owner_ids.size(); ++owner_index) {
        const auto &leaf_ids = simplex_cache_.leaf_ids(result.owner_ids[owner_index], levels);
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

const NativeDensityEvaluator::SimplexDensityEstimate &NativeDensityEvaluator::cached_simplex_value(
    std::int64_t simplex_id
) {
    auto it = value_cache_.find(simplex_id);
    if (it != value_cache_.end()) {
        return it->second;
    }
    auto [inserted, _ok] = value_cache_.emplace(simplex_id, evaluate_simplex(simplex_id));
    return inserted->second;
}

const NativeDensityEvaluator::LeafDensityContribution &NativeDensityEvaluator::leaf_contribution(
    std::int64_t simplex_id
) {
    auto it = leaf_cache_.find(simplex_id);
    if (it != leaf_cache_.end()) {
        return it->second;
    }

    LeafDensityContribution leaf;
    const auto &coarse = cached_simplex_value(simplex_id);
    leaf.coarse_value = coarse.estimate;
    leaf.preview_value.assign(ncomp_, std::complex<double>(0.0, 0.0));
    leaf.error_vector.assign(ncomp_, 0.0);
    leaf.evaluator_evals = coarse.evaluator_evals;

    const auto &preview_ids = simplex_cache_.leaf_ids(simplex_id, 1);
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

    ++leaf_build_count_;
    auto [inserted, _ok] = leaf_cache_.emplace(simplex_id, std::move(leaf));
    leaf_heap_.push(HeapEntry{inserted->second.indicator, simplex_id});
    return inserted->second;
}

const NativeDensityEvaluator::LeafDensityContribution *NativeDensityEvaluator::active_leaf_with_max_error() {
    while (!leaf_heap_.empty()) {
        const auto top = leaf_heap_.top();
        auto it = leaf_cache_.find(top.simplex_id);
        if (it == leaf_cache_.end()) {
            leaf_heap_.pop();
            continue;
        }
        return &it->second;
    }
    return nullptr;
}

std::vector<std::int64_t> NativeDensityEvaluator::bulk_mark(
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

const NativeSpectralCache::CacheEntry &NativeDensityEvaluator::vertex_entry(std::int64_t vertex_id) {
    return spectral_cache_->entry_for_geometry_vertex(*geometry_, vertex_id);
}

const std::vector<std::complex<double>> &NativeDensityEvaluator::vertex_phases(std::int64_t vertex_id) {
    ensure_vertex_phase_capacity(vertex_id);
    if (vertex_phase_ready_[static_cast<size_t>(vertex_id)]) {
        return vertex_phases_[static_cast<size_t>(vertex_id)];
    }
    const double *reduced_point =
        geometry_->vertices_.data() + static_cast<size_t>(vertex_id) * ndim_;
    vertex_phases_[static_cast<size_t>(vertex_id)] = point_phases(reduced_point);
    vertex_phase_ready_[static_cast<size_t>(vertex_id)] = 1;
    ++phase_cache_size_;
    return vertex_phases_[static_cast<size_t>(vertex_id)];
}

void NativeDensityEvaluator::ensure_vertex_phase_capacity(std::int64_t vertex_id) {
    const size_t needed = static_cast<size_t>(vertex_id) + 1;
    if (vertex_phases_.size() >= needed) {
        return;
    }
    vertex_phases_.resize(needed);
    vertex_phase_ready_.resize(needed, 0);
}

std::vector<std::complex<double>> NativeDensityEvaluator::point_phases(const double *reduced_point) const {
    std::vector<std::complex<double>> phases(n_keys_);
    for (size_t key_index = 0; key_index < n_keys_; ++key_index) {
        double phase_arg = 0.0;
        for (size_t axis = 0; axis < ndim_; ++axis) {
            phase_arg +=
                (2.0 * kPi * reduced_point[axis] - kPi) *
                keys_[key_index * ndim_ + axis];
        }
        phases[key_index] = std::exp(std::complex<double>(0.0, phase_arg));
    }
    return phases;
}

NativeDensityEvaluator::PointSpectrum NativeDensityEvaluator::uncached_point_spectrum(
    const double *reduced_point
) {
    const auto entry = spectral_cache_->evaluate_reduced_point_uncached(reduced_point);
    return PointSpectrum{entry.eigenvalues, entry.eigenvectors};
}

void NativeDensityEvaluator::accumulate_density_table_for_band(
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

NativeDensityEvaluator::SimplexDensityEstimate NativeDensityEvaluator::evaluate_simplex(
    std::int64_t simplex_id
) {
    SimplexDensityEstimate result;
    result.estimate.assign(ncomp_, std::complex<double>(0.0, 0.0));

    const auto &simplex = simplex_cache_.simplex(simplex_id);
    const size_t n_vertices = simplex.vertex_ids.size();
    const double volume = simplex.volume;

    std::vector<const NativeSpectralCache::CacheEntry *> vertex_entries(n_vertices, nullptr);
    for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
        vertex_entries[vertex] = &vertex_entry(simplex.vertex_ids[vertex]);
    }

    auto occupation = [&](double energy) noexcept -> double {
        if (std::abs(energy - current_mu_) <= tol_) {
            return 0.5;
        }
        return energy < current_mu_ ? 1.0 : 0.0;
    };

    std::int64_t evaluator_evals = 0;
    std::vector<double> band_energies(n_vertices);

    for (size_t band = 0; band < ndof_; ++band) {
        double band_min = std::numeric_limits<double>::infinity();
        double band_max = -std::numeric_limits<double>::infinity();
        bool half_mask = true;

        for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
            const double energy = vertex_entries[vertex]->eigenvalues[band];
            band_energies[vertex] = energy;
            band_min = std::min(band_min, energy);
            band_max = std::max(band_max, energy);
            if (std::abs(energy - current_mu_) > tol_) {
                half_mask = false;
            }
        }

        if (band_min > current_mu_ + tol_) {
            continue;
        }

        if (half_mask || band_max <= current_mu_ + tol_) {
            const double band_scale = half_mask ? 0.5 : 1.0;
            for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
                const double scale = half_mask
                    ? volume * band_scale / static_cast<double>(n_vertices)
                    : volume * occupation(band_energies[vertex]) / static_cast<double>(n_vertices);
                if (scale == 0.0) {
                    continue;
                }
                const auto &phases = vertex_phases(simplex.vertex_ids[vertex]);
                accumulate_density_table_for_band(
                    result.estimate,
                    phases.data(),
                    vertex_entries[vertex]->eigenvectors.data(),
                    ndof_,
                    band,
                    scale
                );
            }
            evaluator_evals += static_cast<std::int64_t>(n_vertices);
            continue;
        }

        const auto pieces = occupied_subsimplices_from_flat(
            simplex.points_flat.data(),
            band_energies.data(),
            n_vertices,
            ndim_,
            current_mu_,
            tol_
        );
        if (pieces.empty()) {
            for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
                const double weight = occupation(band_energies[vertex]);
                if (weight == 0.0) {
                    continue;
                }
                const auto &phases = vertex_phases(simplex.vertex_ids[vertex]);
                accumulate_density_table_for_band(
                    result.estimate,
                    phases.data(),
                    vertex_entries[vertex]->eigenvectors.data(),
                    ndof_,
                    band,
                    volume * weight / static_cast<double>(n_vertices)
                );
            }
            evaluator_evals += static_cast<std::int64_t>(n_vertices);
            continue;
        }

        for (const auto &piece : pieces) {
            const double piece_volume = simplex_volume_from_flat(piece, n_vertices, ndim_);
            for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
                const double *piece_vertex = piece.data() + vertex * ndim_;
                const auto piece_vertex_spectrum = uncached_point_spectrum(piece_vertex);
                const auto phases = point_phases(piece_vertex);
                accumulate_density_table_for_band(
                    result.estimate,
                    phases.data(),
                    piece_vertex_spectrum.eigenvectors.data(),
                    ndof_,
                    band,
                    piece_volume / static_cast<double>(n_vertices)
                );
            }
            evaluator_evals += static_cast<std::int64_t>(n_vertices);
        }
    }

    result.evaluator_evals = evaluator_evals;
    return result;
}

}  // namespace meanfi::zero_temp_native
