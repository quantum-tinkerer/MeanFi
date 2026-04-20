#pragma once

#include "native_types.h"

#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace meanfi::zero_temp_native {

class NativeGeometry {
public:
    static std::shared_ptr<NativeGeometry> root(
        size_t ndim,
        std::int64_t root_subcells_per_axis = 2,
        double tol = 1e-14
    );

    NativeGeometry(size_t ndim, std::int64_t root_subcells_per_axis = 2, double tol = 1e-14);

    size_t ndim() const noexcept { return ndim_; }
    std::int64_t root_subcells_per_axis() const noexcept { return root_subcells_per_axis_; }
    size_t n_vertices() const noexcept { return vertex_count_; }
    size_t n_simplices() const noexcept { return simplex_records_.size(); }
    size_t n_active() const noexcept { return active_simplex_ids_.size(); }

    nb::ndarray<nb::numpy, double> vertices_array() const;
    nb::ndarray<nb::numpy, std::int64_t> active_simplex_ids() const;
    nb::ndarray<nb::numpy, std::int64_t> simplex_vertex_ids(std::int64_t simplex_id) const;
    nb::ndarray<nb::numpy, double> simplex_points(std::int64_t simplex_id) const;
    double simplex_volume(std::int64_t simplex_id);
    nb::ndarray<nb::numpy, std::int64_t> ensure_children(std::int64_t simplex_id);
    nb::ndarray<nb::numpy, std::int64_t> descendant_leaves(std::int64_t simplex_id, std::int64_t levels);
    nb::tuple refine(Int1D marked_ids);

    const std::vector<std::int64_t> &active_simplex_ids_vector() const noexcept {
        return active_simplex_ids_;
    }
    const std::vector<std::int64_t> &simplex_vertex_ids_vector(std::int64_t simplex_id) const;
    std::vector<double> simplex_points_flat(std::int64_t simplex_id) const;

    NativeRefinementBatch refine_marked(const std::vector<std::int64_t> &marked_ids);

private:
    friend class NativeChargeEvaluator;
    friend class NativeDensityEvaluator;
    friend class NativeFrontier;

    void build_root();
    std::int64_t get_or_add_vertex(const std::vector<double> &point);
    std::int64_t add_simplex(
        const std::vector<std::int64_t> &vertex_ids,
        std::int64_t parent_id = -1,
        size_t level = 0,
        bool active = true
    );
    const NativeSimplexRecord &simplex_record(std::int64_t simplex_id) const;
    void ensure_volume_cache_size();
    double simplex_volume_impl(std::int64_t simplex_id) const;
    std::pair<std::int64_t, std::int64_t> longest_edge(std::int64_t simplex_id) const;
    const std::vector<std::int64_t> &ensure_children_impl(std::int64_t simplex_id);
    void descendant_leaves_impl(
        std::int64_t simplex_id,
        std::int64_t levels,
        std::vector<std::int64_t> &out
    );

    size_t ndim_ = 0;
    std::int64_t root_subcells_per_axis_ = 2;
    double tol_ = 1e-14;
    size_t vertex_count_ = 0;
    std::vector<double> vertices_;
    std::unordered_map<std::string, std::int64_t> vertex_lookup_;
    std::vector<NativeSimplexRecord> simplex_records_;
    std::vector<double> simplex_volume_cache_;
    std::vector<std::int64_t> active_simplex_ids_;
};

}  // namespace meanfi::zero_temp_native
