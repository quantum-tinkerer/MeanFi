#include "native_simplex_cache.h"

namespace meanfi::zero_temp_native {

NativeSimplexCache::NativeSimplexCache(NativeGeometry &geometry) : geometry_(&geometry) {}

const NativeSimplexCache::SimplexData &NativeSimplexCache::simplex(std::int64_t simplex_id) const {
    auto it = simplex_cache_.find(simplex_id);
    if (it != simplex_cache_.end()) {
        return it->second;
    }

    SimplexData data;
    data.vertex_ids = geometry_->simplex_vertex_ids_vector(simplex_id);
    data.points_flat = geometry_->simplex_points_flat(simplex_id);
    data.volume = geometry_->simplex_volume(simplex_id);
    auto [inserted, _ok] = simplex_cache_.emplace(simplex_id, std::move(data));
    return inserted->second;
}

const std::vector<std::int64_t> &NativeSimplexCache::leaf_ids(
    std::int64_t simplex_id,
    std::int64_t levels
) const {
    const GroupKey key{simplex_id, levels};
    auto it = group_cache_.find(key);
    if (it != group_cache_.end()) {
        return it->second;
    }

    std::vector<std::int64_t> leaf_ids;
    geometry_->descendant_leaves_impl(simplex_id, levels, leaf_ids);
    auto [inserted, _ok] = group_cache_.emplace(key, std::move(leaf_ids));
    return inserted->second;
}

}  // namespace meanfi::zero_temp_native
