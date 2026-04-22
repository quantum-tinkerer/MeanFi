#include "types.h"

namespace meanfi::zero_temp {

nb::tuple RefinementBatch::as_tuple(size_t ndim) const {
    return nb::make_tuple(
        refinements,
        make_array(std::vector<std::int64_t>(parent_ids), {parent_ids.size()}),
        make_array(std::vector<std::int64_t>(child_offsets), {child_offsets.size()}),
        make_array(std::vector<std::int64_t>(child_ids), {child_ids.size()}),
        make_array(
            std::vector<std::int64_t>(parent_vertex_ids),
            {parent_ids.size(), ndim + 1}
        ),
        make_array(
            std::vector<std::int64_t>(child_vertex_ids),
            {child_ids.size(), ndim + 1}
        ),
        make_array(std::vector<std::int64_t>(midpoint_ids), {midpoint_ids.size()}),
        make_array(
            std::vector<std::int64_t>(bisected_edges),
            {parent_ids.size(), 2}
        )
    );
}

}  // namespace meanfi::zero_temp
