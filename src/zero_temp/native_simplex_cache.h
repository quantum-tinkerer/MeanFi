#pragma once

#include "native_geometry.h"

#include <unordered_map>

namespace meanfi::zero_temp_native {

class NativeSimplexCache {
public:
    explicit NativeSimplexCache(NativeGeometry &geometry);

    struct SimplexData {
        std::vector<std::int64_t> vertex_ids;
        std::vector<double> points_flat;
        double volume = 0.0;
    };

    const SimplexData &simplex(std::int64_t simplex_id) const;
    const std::vector<std::int64_t> &leaf_ids(std::int64_t simplex_id, std::int64_t levels) const;

private:
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

    NativeGeometry *geometry_ = nullptr;
    mutable std::unordered_map<std::int64_t, SimplexData> simplex_cache_;
    mutable std::unordered_map<GroupKey, std::vector<std::int64_t>, GroupKeyHash> group_cache_;
};

}  // namespace meanfi::zero_temp_native
