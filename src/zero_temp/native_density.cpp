#include "native_density.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
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
      tol_(tol),
      n_keys_(keys.shape(0)),
      ndim_(keys.shape(1)),
      ndof_(spectral_cache.ndof()),
      ncomp_(spectral_cache.ndof() * spectral_cache.ndof() * keys.shape(0)) {
    keys_.assign(keys.data(), keys.data() + n_keys_ * ndim_);
}

void NativeDensityEvaluator::clear() {
    value_cache_.clear();
    current_mu_ = std::numeric_limits<double>::quiet_NaN();
}

nb::tuple NativeDensityEvaluator::evaluate_many(Int1D simplex_ids, double mu) {
    ensure_mu(mu);
    std::vector<std::int64_t> ids(simplex_ids.data(), simplex_ids.data() + simplex_ids.shape(0));
    auto [values, evaluator_evals] = evaluate_many_impl(ids);

    const size_t count = values.size();
    std::vector<std::complex<double>> estimates(count * ncomp_);
    std::vector<double> error_vectors(count * ncomp_);
    std::vector<double> error_scalars(count);
    for (size_t index = 0; index < count; ++index) {
        std::copy_n(values[index].estimate.data(), ncomp_, estimates.data() + index * ncomp_);
        std::copy_n(
            values[index].error_vector.data(),
            ncomp_,
            error_vectors.data() + index * ncomp_
        );
        error_scalars[index] = values[index].error_scalar;
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
    DensityIntegrateResult result;
    result.estimate.assign(ncomp_, std::complex<double>(0.0, 0.0));
    result.error_vector.assign(ncomp_, 0.0);

    std::unordered_map<std::int64_t, CachedDensityValue> cell_values;
    std::unordered_map<std::int64_t, std::int64_t> error_versions;
    std::priority_queue<HeapEntry> error_heap;

    auto accumulate = [&](const CachedDensityValue &value, double sign) {
        for (size_t comp = 0; comp < ncomp_; ++comp) {
            result.estimate[comp] += sign * value.estimate[comp];
            result.error_vector[comp] += sign * value.error_vector[comp];
        }
        result.error_scalar += sign * value.error_scalar;
    };

    auto store_value = [&](std::int64_t simplex_id, const CachedDensityValue &value) {
        cell_values[simplex_id] = value;
        const auto version = error_versions[simplex_id] + 1;
        error_versions[simplex_id] = version;
        if (value.error_scalar > 0.0) {
            error_heap.push(HeapEntry{value.error_scalar, simplex_id, version});
        }
    };

    auto [initial_values, initial_evals] = evaluate_many_impl(frontier.active_simplex_ids_);
    result.evaluator_evals += initial_evals;
    for (size_t index = 0; index < frontier.active_simplex_ids_.size(); ++index) {
        const auto simplex_id = frontier.active_simplex_ids_[index];
        store_value(simplex_id, initial_values[index]);
        accumulate(initial_values[index], 1.0);
    }

    std::int64_t remaining = options.max_subdivisions;
    while (true) {
        const double tolerance =
            options.density_atol +
            options.density_rtol * std::sqrt(std::real(std::inner_product(
                result.estimate.begin(),
                result.estimate.end(),
                result.estimate.begin(),
                0.0,
                std::plus<>(),
                [](const std::complex<double> &left, const std::complex<double> &right) {
                    return std::real(std::conj(left) * right);
                }
            )));
        if (result.error_scalar <= tolerance) {
            result.converged = true;
            break;
        }
        if (remaining == 0) {
            throw std::runtime_error("Adaptive zero-temperature density integration did not converge");
        }

        const double target = options.bulk_theta * result.error_scalar;
        std::vector<std::int64_t> marked;
        double accumulated = 0.0;
        while (!error_heap.empty() && accumulated < target) {
            const auto entry = error_heap.top();
            error_heap.pop();
            auto cell_it = cell_values.find(entry.simplex_id);
            if (cell_it == cell_values.end()) {
                continue;
            }
            if (error_versions[entry.simplex_id] != entry.version) {
                continue;
            }
            marked.push_back(entry.simplex_id);
            accumulated += entry.error;
        }
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

        std::vector<std::int64_t> new_simplex_ids;
        for (const auto simplex_id : marked) {
            const auto value = cell_values.at(simplex_id);
            accumulate(value, -1.0);
            cell_values.erase(simplex_id);
            error_versions[simplex_id] += 1;
        }
        new_simplex_ids = batch.child_ids;

        auto [child_values, child_evals] = evaluate_many_impl(new_simplex_ids);
        result.evaluator_evals += child_evals;
        for (size_t index = 0; index < new_simplex_ids.size(); ++index) {
            store_value(new_simplex_ids[index], child_values[index]);
            accumulate(child_values[index], 1.0);
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
    }
}

const NativeDensityEvaluator::CachedDensityValue &NativeDensityEvaluator::cached_value(
    std::int64_t simplex_id
) {
    auto it = value_cache_.find(simplex_id);
    if (it != value_cache_.end()) {
        return it->second;
    }
    auto [inserted, _ok] = value_cache_.emplace(simplex_id, evaluate_simplex(simplex_id));
    return inserted->second;
}

std::pair<std::vector<NativeDensityEvaluator::CachedDensityValue>, std::int64_t>
NativeDensityEvaluator::evaluate_many_impl(const std::vector<std::int64_t> &simplex_ids) {
    std::vector<CachedDensityValue> out;
    out.reserve(simplex_ids.size());
    std::int64_t evaluator_evals = 0;
    for (const auto simplex_id : simplex_ids) {
        const auto &cached = cached_value(simplex_id);
        out.push_back(cached);
        evaluator_evals += cached.evaluator_evals;
    }
    return {out, evaluator_evals};
}

const std::vector<double> &NativeDensityEvaluator::vertex_eigenvalues(std::int64_t vertex_id) {
    ensure_vertex_value_capacity(vertex_id);
    if (vertex_value_ready_[static_cast<size_t>(vertex_id)]) {
        return vertex_values_[static_cast<size_t>(vertex_id)];
    }
    const auto &entry = vertex_entry(vertex_id);
    vertex_values_[static_cast<size_t>(vertex_id)] = entry.eigenvalues;
    vertex_value_ready_[static_cast<size_t>(vertex_id)] = 1;
    return vertex_values_[static_cast<size_t>(vertex_id)];
}

const std::vector<std::complex<double>> &NativeDensityEvaluator::vertex_tables(
    std::int64_t vertex_id
) {
    ensure_vertex_table_capacity(vertex_id);
    if (vertex_table_ready_[static_cast<size_t>(vertex_id)]) {
        return vertex_tables_[static_cast<size_t>(vertex_id)];
    }
    std::vector<double> reduced_point(ndim_);
    const size_t offset = static_cast<size_t>(vertex_id) * ndim_;
    std::copy_n(geometry_->vertices_.data() + offset, ndim_, reduced_point.data());
    const auto &entry = vertex_entry(vertex_id);
    vertex_tables_[static_cast<size_t>(vertex_id)] = density_tables_for_point(
        reduced_point.data(),
        entry.eigenvectors.data(),
        ndof_,
        nullptr,
        ndof_
    );
    vertex_table_ready_[static_cast<size_t>(vertex_id)] = 1;
    return vertex_tables_[static_cast<size_t>(vertex_id)];
}

void NativeDensityEvaluator::ensure_vertex_value_capacity(std::int64_t vertex_id) {
    const size_t needed = static_cast<size_t>(vertex_id) + 1;
    if (vertex_values_.size() >= needed) {
        return;
    }
    vertex_values_.resize(needed);
    vertex_value_ready_.resize(needed, 0);
}

void NativeDensityEvaluator::ensure_vertex_table_capacity(std::int64_t vertex_id) {
    const size_t needed = static_cast<size_t>(vertex_id) + 1;
    if (vertex_tables_.size() >= needed) {
        return;
    }
    vertex_tables_.resize(needed);
    vertex_table_ready_.resize(needed, 0);
}

const NativeSpectralCache::CacheEntry &NativeDensityEvaluator::vertex_entry(std::int64_t vertex_id) {
    std::vector<double> reduced_point(ndim_);
    const size_t base = static_cast<size_t>(vertex_id) * ndim_;
    std::copy_n(geometry_->vertices_.data() + base, ndim_, reduced_point.data());
    return spectral_cache_->entry_for_reduced_point(reduced_point.data());
}

NativeDensityEvaluator::PointSpectrum NativeDensityEvaluator::uncached_point_spectrum(
    const double *reduced_point
) {
    const auto entry = spectral_cache_->evaluate_reduced_point_uncached(reduced_point);
    return PointSpectrum{entry.eigenvalues, entry.eigenvectors};
}

std::vector<std::complex<double>> NativeDensityEvaluator::density_tables_for_point(
    const double *reduced_point,
    const std::complex<double> *eigenvectors,
    size_t n_all_bands,
    const std::int64_t *selected_bands,
    size_t n_selected_bands
) const {
    std::vector<std::complex<double>> out(n_selected_bands * ncomp_);
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

    for (size_t band_index = 0; band_index < n_selected_bands; ++band_index) {
        const size_t band = selected_bands ? static_cast<size_t>(selected_bands[band_index]) : band_index;
        for (size_t i = 0; i < ndof_; ++i) {
            const std::complex<double> ui = eigenvectors[i * n_all_bands + band];
            for (size_t j = 0; j < ndof_; ++j) {
                const std::complex<double> projector =
                    ui * std::conj(eigenvectors[j * n_all_bands + band]);
                const size_t base =
                    ((band_index * ndof_ + i) * ndof_ + j) * n_keys_;
                for (size_t key_index = 0; key_index < n_keys_; ++key_index) {
                    out[base + key_index] = projector * phases[key_index];
                }
            }
        }
    }
    return out;
}

NativeDensityEvaluator::CachedDensityValue NativeDensityEvaluator::evaluate_simplex(
    std::int64_t simplex_id
) {
    CachedDensityValue result;
    result.estimate.assign(ncomp_, std::complex<double>(0.0, 0.0));
    result.error_vector.assign(ncomp_, 0.0);

    const auto &record = geometry_->simplex_records_[static_cast<size_t>(simplex_id)];
    const size_t n_vertices = record.vertex_ids.size();
    const double volume = geometry_->simplex_volume(simplex_id);
    const std::vector<double> points_flat = geometry_->simplex_points_flat(simplex_id);
    std::vector<double> centroid(ndim_, 0.0);
    for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
        for (size_t axis = 0; axis < ndim_; ++axis) {
            centroid[axis] += points_flat[vertex * ndim_ + axis];
        }
    }
    for (size_t axis = 0; axis < ndim_; ++axis) {
        centroid[axis] /= static_cast<double>(n_vertices);
    }

    std::vector<double> vertex_energies(n_vertices * ndof_);
    for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
        const auto vertex_id = record.vertex_ids[vertex];
        const auto &values = vertex_eigenvalues(vertex_id);
        std::copy_n(values.data(), ndof_, vertex_energies.data() + vertex * ndof_);
    }

    const auto centroid_spectrum = uncached_point_spectrum(centroid.data());
    const auto centroid_tables_all = density_tables_for_point(
        centroid.data(),
        centroid_spectrum.eigenvectors.data(),
        ndof_,
        nullptr,
        ndof_
    );

    auto occupation = [&](double energy) noexcept -> double {
        if (std::abs(energy - current_mu_) <= tol_) {
            return 0.5;
        }
        return energy < current_mu_ ? 1.0 : 0.0;
    };

    std::vector<std::complex<double>> estimate_low(ncomp_, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> estimate_high(ncomp_, std::complex<double>(0.0, 0.0));
    std::int64_t evaluator_evals = 0;
    std::vector<double> band_energies(n_vertices);

    for (size_t band = 0; band < ndof_; ++band) {
        double band_min = std::numeric_limits<double>::infinity();
        double band_max = -std::numeric_limits<double>::infinity();
        bool half_mask = true;
        bool vertex_matches_centroid = true;
        const double centroid_energy = centroid_spectrum.eigenvalues[band];
        const double centroid_occ = occupation(centroid_energy);

        for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
            const double energy = vertex_energies[vertex * ndof_ + band];
            band_energies[vertex] = energy;
            band_min = std::min(band_min, energy);
            band_max = std::max(band_max, energy);
            if (std::abs(energy - current_mu_) > tol_) {
                half_mask = false;
            }
            if (occupation(energy) != centroid_occ) {
                vertex_matches_centroid = false;
            }
        }

        const bool full_mask = (band_max <= current_mu_) && vertex_matches_centroid && !half_mask;
        const bool empty_mask = (band_min > current_mu_) && vertex_matches_centroid && !half_mask;
        if (half_mask || full_mask) {
            const double weight = half_mask ? 0.5 : 1.0;
            for (size_t comp = 0; comp < ncomp_; ++comp) {
                estimate_low[comp] += volume * weight * centroid_tables_all[band * ncomp_ + comp];
                std::complex<double> avg(0.0, 0.0);
                for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
                    const auto &tables = vertex_tables(record.vertex_ids[vertex]);
                    avg += tables[band * ncomp_ + comp];
                }
                estimate_high[comp] += volume * weight * avg / static_cast<double>(n_vertices);
            }
            evaluator_evals += static_cast<std::int64_t>(n_vertices);
            continue;
        }
        if (empty_mask) {
            continue;
        }

        const auto pieces = occupied_subsimplices_from_flat(
            points_flat.data(),
            band_energies.data(),
            n_vertices,
            ndim_,
            current_mu_,
            tol_
        );
        if (pieces.empty()) {
            for (size_t comp = 0; comp < ncomp_; ++comp) {
                estimate_low[comp] +=
                    volume * centroid_occ * centroid_tables_all[band * ncomp_ + comp];
                std::complex<double> avg(0.0, 0.0);
                for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
                    const auto &tables = vertex_tables(record.vertex_ids[vertex]);
                    avg += occupation(vertex_energies[vertex * ndof_ + band]) *
                           tables[band * ncomp_ + comp];
                }
                estimate_high[comp] += volume * avg / static_cast<double>(n_vertices);
            }
            evaluator_evals += static_cast<std::int64_t>(n_vertices);
            continue;
        }

        evaluator_evals += static_cast<std::int64_t>(pieces.size());
        for (const auto &piece : pieces) {
            const double piece_volume = simplex_volume_from_flat(piece, n_vertices, ndim_);
            std::vector<double> piece_centroid(ndim_, 0.0);
            for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
                for (size_t axis = 0; axis < ndim_; ++axis) {
                    piece_centroid[axis] += piece[vertex * ndim_ + axis];
                }
            }
            for (size_t axis = 0; axis < ndim_; ++axis) {
                piece_centroid[axis] /= static_cast<double>(n_vertices);
            }
            const auto piece_centroid_spectrum = uncached_point_spectrum(piece_centroid.data());
            const std::int64_t band_id = static_cast<std::int64_t>(band);
            const auto piece_centroid_table = density_tables_for_point(
                piece_centroid.data(),
                piece_centroid_spectrum.eigenvectors.data(),
                ndof_,
                &band_id,
                1
            );
            std::vector<std::complex<double>> piece_vertex_tables_flat(n_vertices * ncomp_);
            for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
                const double *piece_vertex = piece.data() + vertex * ndim_;
                const auto piece_vertex_spectrum = uncached_point_spectrum(piece_vertex);
                const auto piece_vertex_table = density_tables_for_point(
                    piece_vertex,
                    piece_vertex_spectrum.eigenvectors.data(),
                    ndof_,
                    &band_id,
                    1
                );
                std::copy_n(
                    piece_vertex_table.data(),
                    ncomp_,
                    piece_vertex_tables_flat.data() + vertex * ncomp_
                );
            }
            for (size_t comp = 0; comp < ncomp_; ++comp) {
                estimate_low[comp] += piece_volume * piece_centroid_table[comp];
                std::complex<double> avg(0.0, 0.0);
                for (size_t vertex = 0; vertex < n_vertices; ++vertex) {
                    avg += piece_vertex_tables_flat[vertex * ncomp_ + comp];
                }
                estimate_high[comp] += piece_volume * avg / static_cast<double>(n_vertices);
            }
            evaluator_evals += static_cast<std::int64_t>(n_vertices);
        }
    }

    double error_norm_sq = 0.0;
    for (size_t comp = 0; comp < ncomp_; ++comp) {
        const std::complex<double> diff = estimate_high[comp] - estimate_low[comp];
        result.estimate[comp] = estimate_high[comp];
        result.error_vector[comp] = std::abs(diff);
        error_norm_sq += result.error_vector[comp] * result.error_vector[comp];
    }
    result.error_scalar = std::sqrt(error_norm_sq);
    result.evaluator_evals = evaluator_evals;
    return result;
}

}  // namespace meanfi::zero_temp_native
