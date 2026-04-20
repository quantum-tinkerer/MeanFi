#include "native_frontier.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace meanfi::zero_temp_native {

NativeFrontier NativeFrontier::from_geometry(NativeGeometry &geometry) {
    return NativeFrontier(geometry);
}

NativeFrontier::NativeFrontier(NativeGeometry &geometry) : geometry_(&geometry) {
    sync_from_geometry();
}

void NativeFrontier::sync_from_geometry() {
    active_simplex_ids_ = geometry_->active_simplex_ids_vector();
}

void NativeFrontier::apply_refinement(const NativeRefinementBatch &batch) {
    if (batch.parent_ids.empty()) {
        return;
    }

    std::unordered_map<std::int64_t, std::pair<std::int64_t, std::int64_t>> replacement_offsets;
    for (size_t index = 0; index < batch.parent_ids.size(); ++index) {
        replacement_offsets.emplace(
            batch.parent_ids[index],
            std::make_pair(batch.child_offsets[index], batch.child_offsets[index + 1])
        );
    }

    std::vector<std::int64_t> updated;
    updated.reserve(active_simplex_ids_.size() + batch.parent_ids.size());
    for (const auto simplex_id : active_simplex_ids_) {
        auto it = replacement_offsets.find(simplex_id);
        if (it == replacement_offsets.end()) {
            updated.push_back(simplex_id);
            continue;
        }
        const auto [start, stop] = it->second;
        updated.insert(
            updated.end(),
            batch.child_ids.begin() + start,
            batch.child_ids.begin() + stop
        );
    }
    active_simplex_ids_ = std::move(updated);
    ++generation_;
}

void NativeFrontier::apply_refinement(Int1D parent_ids, Int1D child_offsets, Int1D child_ids) {
    NativeRefinementBatch batch;
    batch.parent_ids.assign(parent_ids.data(), parent_ids.data() + parent_ids.shape(0));
    batch.child_offsets.assign(
        child_offsets.data(),
        child_offsets.data() + child_offsets.shape(0)
    );
    batch.child_ids.assign(child_ids.data(), child_ids.data() + child_ids.shape(0));
    apply_refinement(batch);
}

size_t NativeFrontier::n_leaf_vertices() const {
    std::unordered_set<std::int64_t> used;
    for (const auto simplex_id : active_simplex_ids_) {
        const auto &vertex_ids = geometry_->simplex_vertex_ids_vector(simplex_id);
        used.insert(vertex_ids.begin(), vertex_ids.end());
    }
    return used.size();
}

nb::ndarray<nb::numpy, std::int64_t> NativeFrontier::active_simplex_ids() const {
    return make_array(
        std::vector<std::int64_t>(active_simplex_ids_),
        {active_simplex_ids_.size()}
    );
}

nb::ndarray<nb::numpy, std::int64_t> NativeFrontier::vertex_ids() const {
    const size_t n_vertices = geometry_->ndim() + 1;
    std::vector<std::int64_t> out(active_simplex_ids_.size() * n_vertices);
    for (size_t simplex = 0; simplex < active_simplex_ids_.size(); ++simplex) {
        const auto &vertex_ids = geometry_->simplex_vertex_ids_vector(active_simplex_ids_[simplex]);
        std::copy_n(vertex_ids.data(), n_vertices, out.data() + simplex * n_vertices);
    }
    return make_array(std::move(out), {active_simplex_ids_.size(), n_vertices});
}

nb::ndarray<nb::numpy, double> NativeFrontier::volumes() {
    std::vector<double> out(active_simplex_ids_.size());
    for (size_t simplex = 0; simplex < active_simplex_ids_.size(); ++simplex) {
        out[simplex] = geometry_->simplex_volume(active_simplex_ids_[simplex]);
    }
    return make_array(std::move(out), {active_simplex_ids_.size()});
}

}  // namespace meanfi::zero_temp_native
