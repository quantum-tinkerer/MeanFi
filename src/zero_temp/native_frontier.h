#pragma once

#include "native_geometry.h"

namespace meanfi::zero_temp_native {

class NativeFrontier {
public:
    static NativeFrontier from_geometry(NativeGeometry &geometry);

    explicit NativeFrontier(NativeGeometry &geometry);

    void sync_from_geometry();
    void apply_refinement(const NativeRefinementBatch &batch);
    void apply_refinement(Int1D parent_ids, Int1D child_offsets, Int1D child_ids);

    size_t n_active() const noexcept { return active_simplex_ids_.size(); }
    std::uint64_t generation() const noexcept { return generation_; }
    size_t n_leaf_vertices() const;

    nb::ndarray<nb::numpy, std::int64_t> active_simplex_ids() const;
    nb::ndarray<nb::numpy, std::int64_t> vertex_ids() const;
    nb::ndarray<nb::numpy, double> volumes();

private:
    friend class NativeChargeEvaluator;
    friend class NativeDensityEvaluator;

    NativeGeometry *geometry_ = nullptr;
    std::vector<std::int64_t> active_simplex_ids_;
    std::uint64_t generation_ = 0;
};

}  // namespace meanfi::zero_temp_native
