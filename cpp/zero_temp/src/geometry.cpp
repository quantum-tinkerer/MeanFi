#include "geometry.h"

#include "simplex_rules.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace meanfi::zero_temp {

namespace {

std::string point_key_from_ptr(const double *point, size_t ndim, double tol) {
    std::vector<double> normalized(ndim);
    for (size_t axis = 0; axis < ndim; ++axis) {
        const double value = point[axis];
        normalized[axis] = std::abs(value) <= tol ? 0.0 : value;
    }
    return std::string(
        reinterpret_cast<const char *>(normalized.data()),
        normalized.size() * sizeof(double)
    );
}

}  // namespace

std::shared_ptr<Geometry> Geometry::root(
    size_t ndim,
    std::int64_t root_subcells_per_axis,
    double tol
) {
    auto geometry = std::make_shared<Geometry>(ndim, root_subcells_per_axis, tol);
    geometry->build_root();
    return geometry;
}

Geometry::Geometry(size_t ndim, std::int64_t root_subcells_per_axis, double tol)
    : ndim_(ndim), root_subcells_per_axis_(root_subcells_per_axis), tol_(tol) {}

size_t Geometry::n_leaf_vertices() const {
    std::unordered_set<std::int64_t> used;
    for (const auto simplex_id : active_simplex_ids_) {
        const auto &vertex_ids = simplex_vertex_ids_vector(simplex_id);
        used.insert(vertex_ids.begin(), vertex_ids.end());
    }
    return used.size();
}

nb::ndarray<nb::numpy, double> Geometry::vertices_array() const {
    return make_array(std::vector<double>(vertices_), {n_vertices(), ndim_});
}

nb::ndarray<nb::numpy, std::int64_t> Geometry::active_simplex_ids() const {
    return make_array(
        std::vector<std::int64_t>(active_simplex_ids_),
        {active_simplex_ids_.size()}
    );
}

nb::ndarray<nb::numpy, std::int64_t> Geometry::simplex_vertex_ids(std::int64_t simplex_id) const {
    const auto &record = simplex_record(simplex_id);
    return make_array(
        std::vector<std::int64_t>(record.vertex_ids),
        {record.vertex_ids.size()}
    );
}

nb::ndarray<nb::numpy, double> Geometry::simplex_points(std::int64_t simplex_id) const {
    std::vector<double> points = simplex_points_flat(simplex_id);
    return make_array(std::move(points), {ndim_ + 1, ndim_});
}

double Geometry::simplex_volume(std::int64_t simplex_id) {
    ensure_volume_cache_size();
    if (simplex_volume_cache_[static_cast<size_t>(simplex_id)] >= 0.0) {
        return simplex_volume_cache_[static_cast<size_t>(simplex_id)];
    }
    const double volume = simplex_volume_impl(simplex_id);
    simplex_volume_cache_[static_cast<size_t>(simplex_id)] = volume;
    return volume;
}

nb::ndarray<nb::numpy, std::int64_t> Geometry::ensure_children(std::int64_t simplex_id) {
    const auto &children = ensure_children_impl(simplex_id);
    return make_array(std::vector<std::int64_t>(children), {children.size()});
}

nb::ndarray<nb::numpy, std::int64_t> Geometry::descendant_leaves(
    std::int64_t simplex_id,
    std::int64_t levels
) {
    std::vector<std::int64_t> leaves;
    descendant_leaves_impl(simplex_id, levels, leaves);
    return make_array(std::move(leaves), {leaves.size()});
}

nb::tuple Geometry::refine(Int1D marked_ids) {
    std::vector<std::int64_t> marked(marked_ids.data(), marked_ids.data() + marked_ids.shape(0));
    const auto batch = refine_marked(marked);
    return batch.as_tuple(ndim_);
}

const std::vector<std::int64_t> &Geometry::simplex_vertex_ids_vector(std::int64_t simplex_id) const {
    return simplex_record(simplex_id).vertex_ids;
}

std::vector<double> Geometry::simplex_points_flat(std::int64_t simplex_id) const {
    const auto &record = simplex_record(simplex_id);
    std::vector<double> points(record.vertex_ids.size() * ndim_);
    for (size_t vertex = 0; vertex < record.vertex_ids.size(); ++vertex) {
        const std::int64_t vertex_id = record.vertex_ids[vertex];
        const size_t src_offset = static_cast<size_t>(vertex_id) * ndim_;
        std::copy_n(
            vertices_.data() + src_offset,
            ndim_,
            points.data() + vertex * ndim_
        );
    }
    return points;
}

const std::vector<std::int64_t> &Geometry::ensure_children_vector(std::int64_t simplex_id) {
    return ensure_children_impl(simplex_id);
}

void Geometry::build_root() {
    const double step = 1.0 / static_cast<double>(root_subcells_per_axis_);
    const size_t n_offsets =
        ndim_ == 0 ? 1 : static_cast<size_t>(std::pow(root_subcells_per_axis_, ndim_));

    for (size_t offset_index = 0; offset_index < n_offsets; ++offset_index) {
        std::vector<double> base(ndim_, 0.0);
        size_t remainder = offset_index;
        for (size_t axis = 0; axis < ndim_; ++axis) {
            const size_t digit = remainder % static_cast<size_t>(root_subcells_per_axis_);
            remainder /= static_cast<size_t>(root_subcells_per_axis_);
            base[axis] = step * static_cast<double>(digit);
        }

        std::vector<int> perm(ndim_);
        std::iota(perm.begin(), perm.end(), 0);

        if (ndim_ == 0) {
            const std::int64_t vertex_id = get_or_add_vertex(base);
            active_simplex_ids_.push_back(add_simplex({vertex_id}));
            continue;
        }

        do {
            std::vector<double> point = base;
            std::vector<std::int64_t> simplex;
            simplex.reserve(ndim_ + 1);
            simplex.push_back(get_or_add_vertex(point));
            for (const int axis : perm) {
                point[static_cast<size_t>(axis)] += step;
                simplex.push_back(get_or_add_vertex(point));
            }
            active_simplex_ids_.push_back(add_simplex(simplex));
        } while (std::next_permutation(perm.begin(), perm.end()));
    }
}

std::int64_t Geometry::get_or_add_vertex(const std::vector<double> &point) {
    const std::string key = point_key_from_ptr(point.data(), ndim_, tol_);
    auto it = vertex_lookup_.find(key);
    if (it != vertex_lookup_.end()) {
        return it->second;
    }
    const std::int64_t vertex_id = static_cast<std::int64_t>(vertex_count_);
    vertices_.insert(vertices_.end(), point.begin(), point.end());
    vertex_lookup_.emplace(key, vertex_id);
    ++vertex_count_;
    return vertex_id;
}

std::int64_t Geometry::add_simplex(
    const std::vector<std::int64_t> &vertex_ids,
    std::int64_t parent_id,
    size_t level,
    bool active
) {
    const std::int64_t simplex_id = static_cast<std::int64_t>(simplex_records_.size());
    SimplexRecord record;
    record.simplex_id = simplex_id;
    record.vertex_ids = vertex_ids;
    record.parent_id = parent_id;
    record.level = level;
    record.active = active;
    simplex_records_.push_back(std::move(record));
    simplex_volume_cache_.push_back(-1.0);
    return simplex_id;
}

const SimplexRecord &Geometry::simplex_record(std::int64_t simplex_id) const {
    if (simplex_id < 0 || static_cast<size_t>(simplex_id) >= simplex_records_.size()) {
        throw std::runtime_error("Geometry: simplex_id out of range");
    }
    return simplex_records_[static_cast<size_t>(simplex_id)];
}

void Geometry::ensure_volume_cache_size() {
    if (simplex_volume_cache_.size() < simplex_records_.size()) {
        simplex_volume_cache_.resize(simplex_records_.size(), -1.0);
    }
}

double Geometry::simplex_volume_impl(std::int64_t simplex_id) const {
    if (ndim_ == 0) {
        return 1.0;
    }
    return simplex_rules::simplex_volume_from_flat(
        simplex_points_flat(simplex_id),
        ndim_ + 1,
        ndim_
    );
}

std::pair<std::int64_t, std::int64_t> Geometry::longest_edge(std::int64_t simplex_id) const {
    if (ndim_ == 0) {
        throw std::runtime_error("Geometry: zero-dimensional simplices cannot be bisected");
    }
    const std::vector<double> points = simplex_points_flat(simplex_id);
    std::pair<std::int64_t, std::int64_t> best{0, 1};
    double best_length = -1.0;
    for (size_t i = 0; i < ndim_ + 1; ++i) {
        for (size_t j = i + 1; j < ndim_ + 1; ++j) {
            double length = 0.0;
            for (size_t axis = 0; axis < ndim_; ++axis) {
                const double delta = points[i * ndim_ + axis] - points[j * ndim_ + axis];
                length += delta * delta;
            }
            if (length > best_length) {
                best_length = length;
                best = {static_cast<std::int64_t>(i), static_cast<std::int64_t>(j)};
            }
        }
    }
    return best;
}

const std::vector<std::int64_t> &Geometry::ensure_children_impl(std::int64_t simplex_id) {
    auto &record = simplex_records_[static_cast<size_t>(simplex_id)];
    if (!record.children.empty()) {
        return record.children;
    }

    const size_t parent_level = record.level;
    const std::vector<std::int64_t> parent_vertex_ids = record.vertex_ids;
    const auto [edge_i, edge_j] = longest_edge(simplex_id);
    const std::vector<double> points = simplex_points_flat(simplex_id);
    std::vector<double> midpoint(ndim_, 0.0);
    for (size_t axis = 0; axis < ndim_; ++axis) {
        midpoint[axis] =
            0.5 * (
                points[static_cast<size_t>(edge_i) * ndim_ + axis] +
                points[static_cast<size_t>(edge_j) * ndim_ + axis]
            );
    }
    const std::int64_t midpoint_id = get_or_add_vertex(midpoint);

    std::vector<std::int64_t> child_a = parent_vertex_ids;
    std::vector<std::int64_t> child_b = parent_vertex_ids;
    child_a[static_cast<size_t>(edge_i)] = midpoint_id;
    child_b[static_cast<size_t>(edge_j)] = midpoint_id;
    const std::int64_t child_a_id = add_simplex(child_a, simplex_id, parent_level + 1, false);
    const std::int64_t child_b_id = add_simplex(child_b, simplex_id, parent_level + 1, false);

    auto &updated = simplex_records_[static_cast<size_t>(simplex_id)];
    updated.children = {child_a_id, child_b_id};
    updated.midpoint_vertex_id = midpoint_id;
    updated.split_edge = {edge_i, edge_j};
    return updated.children;
}

void Geometry::descendant_leaves_impl(
    std::int64_t simplex_id,
    std::int64_t levels,
    std::vector<std::int64_t> &out
) {
    if (levels <= 0) {
        out.push_back(simplex_id);
        return;
    }
    const auto &children = ensure_children_impl(simplex_id);
    if (levels == 1) {
        out.insert(out.end(), children.begin(), children.end());
        return;
    }
    for (const auto child_id : children) {
        descendant_leaves_impl(child_id, levels - 1, out);
    }
}

RefinementBatch Geometry::refine_marked(const std::vector<std::int64_t> &marked_ids) {
    std::unordered_set<std::int64_t> marked(marked_ids.begin(), marked_ids.end());
    RefinementBatch batch;
    if (marked.empty()) {
        return batch;
    }

    std::vector<std::int64_t> refined_active;
    refined_active.reserve(active_simplex_ids_.size() * 2);

    for (const auto simplex_id : active_simplex_ids_) {
        if (marked.find(simplex_id) == marked.end()) {
            refined_active.push_back(simplex_id);
            continue;
        }

        ++batch.refinements;
        simplex_records_[static_cast<size_t>(simplex_id)].active = false;
        const auto &children = ensure_children_impl(simplex_id);
        const auto &parent = simplex_records_[static_cast<size_t>(simplex_id)];
        for (const auto child_id : children) {
            simplex_records_[static_cast<size_t>(child_id)].active = true;
            refined_active.push_back(child_id);
        }

        batch.parent_ids.push_back(simplex_id);
        batch.child_ids.insert(batch.child_ids.end(), children.begin(), children.end());
        batch.child_offsets.push_back(static_cast<std::int64_t>(batch.child_ids.size()));
        batch.parent_vertex_ids.insert(
            batch.parent_vertex_ids.end(),
            parent.vertex_ids.begin(),
            parent.vertex_ids.end()
        );
        for (const auto child_id : children) {
            const auto &child_vertices = simplex_records_[static_cast<size_t>(child_id)].vertex_ids;
            batch.child_vertex_ids.insert(
                batch.child_vertex_ids.end(),
                child_vertices.begin(),
                child_vertices.end()
            );
        }
        batch.midpoint_ids.push_back(parent.midpoint_vertex_id);
        batch.bisected_edges.push_back(parent.split_edge[0]);
        batch.bisected_edges.push_back(parent.split_edge[1]);
    }

    active_simplex_ids_ = std::move(refined_active);
    ++generation_;
    return batch;
}

}  // namespace meanfi::zero_temp
