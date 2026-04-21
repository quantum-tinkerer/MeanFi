#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include "native_charge.h"
#include "native_density.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace meanfi::zero_temp_native {

NB_MODULE(_zero_temp_native, m) {
    m.doc() = "Native runtime for meanfi.zero_temp";

    nb::class_<ChargeSolveOptions>(m, "ChargeSolveOptions")
        .def(nb::init<>())
        .def_rw("mu_guess", &ChargeSolveOptions::mu_guess)
        .def_rw("charge_tol", &ChargeSolveOptions::charge_tol)
        .def_rw("mu_xtol", &ChargeSolveOptions::mu_xtol)
        .def_rw("max_mu_iterations", &ChargeSolveOptions::max_mu_iterations)
        .def_rw("max_subdivisions", &ChargeSolveOptions::max_subdivisions)
        .def_rw("bulk_theta", &ChargeSolveOptions::bulk_theta);

    nb::class_<ChargeSolveResult>(m, "ChargeSolveResult")
        .def_prop_ro("mu", [](const ChargeSolveResult &self) { return self.mu; })
        .def_prop_ro("charge", [](const ChargeSolveResult &self) { return self.charge; })
        .def_prop_ro("charge_error", [](const ChargeSolveResult &self) { return self.charge_error; })
        .def_prop_ro("dcharge_dmu", [](const ChargeSolveResult &self) { return self.dcharge_dmu; })
        .def_prop_ro("root_iterations", [](const ChargeSolveResult &self) { return self.root_iterations; })
        .def_prop_ro(
            "charge_integration_calls",
            [](const ChargeSolveResult &self) { return self.charge_integration_calls; }
        )
        .def_prop_ro("evaluator_evals", [](const ChargeSolveResult &self) { return self.evaluator_evals; })
        .def_prop_ro("subdivisions", [](const ChargeSolveResult &self) { return self.subdivisions; })
        .def_prop_ro("n_leaves", [](const ChargeSolveResult &self) { return self.n_leaves; })
        .def_prop_ro("n_leaf_nodes", [](const ChargeSolveResult &self) { return self.n_leaf_nodes; })
        .def_prop_ro("converged", [](const ChargeSolveResult &self) { return self.converged; })
        .def_prop_ro(
            "error_estimate_available",
            [](const ChargeSolveResult &self) { return self.error_estimate_available; }
        );

    nb::class_<DensityIntegrateOptions>(m, "DensityIntegrateOptions")
        .def(nb::init<>())
        .def_rw("density_atol", &DensityIntegrateOptions::density_atol)
        .def_rw("density_rtol", &DensityIntegrateOptions::density_rtol)
        .def_rw("max_subdivisions", &DensityIntegrateOptions::max_subdivisions)
        .def_rw("bulk_theta", &DensityIntegrateOptions::bulk_theta);

    nb::class_<DensityIntegrateResult>(m, "DensityIntegrateResult")
        .def(
            "estimate_array",
            [](const DensityIntegrateResult &self) {
                return make_array(
                    std::vector<std::complex<double>>(self.estimate),
                    {self.estimate.size()}
                );
            }
        )
        .def(
            "error_vector_array",
            [](const DensityIntegrateResult &self) {
                return make_array(std::vector<double>(self.error_vector), {self.error_vector.size()});
            }
        )
        .def_prop_ro("error_scalar", [](const DensityIntegrateResult &self) { return self.error_scalar; })
        .def_prop_ro("evaluator_evals", [](const DensityIntegrateResult &self) { return self.evaluator_evals; })
        .def_prop_ro("subdivisions", [](const DensityIntegrateResult &self) { return self.subdivisions; })
        .def_prop_ro("n_leaves", [](const DensityIntegrateResult &self) { return self.n_leaves; })
        .def_prop_ro("n_leaf_nodes", [](const DensityIntegrateResult &self) { return self.n_leaf_nodes; })
        .def_prop_ro("converged", [](const DensityIntegrateResult &self) { return self.converged; })
        .def_prop_ro(
            "error_estimate_available",
            [](const DensityIntegrateResult &self) {
                return self.error_estimate_available;
            }
        );

    nb::class_<NativeRefinementDescriptor>(m, "NativeRefinementDescriptor")
        .def_prop_ro("parent_id", [](const NativeRefinementDescriptor &self) { return self.parent_id; })
        .def_prop_ro("child_ids", &NativeRefinementDescriptor::child_ids_array)
        .def_prop_ro("parent_vertex_ids", &NativeRefinementDescriptor::parent_vertex_ids_array)
        .def_prop_ro("child_vertex_ids", &NativeRefinementDescriptor::child_vertex_ids_array)
        .def_prop_ro(
            "new_midpoint_vertex_id",
            [](const NativeRefinementDescriptor &self) { return self.new_midpoint_vertex_id; }
        )
        .def_prop_ro("bisected_edge", &NativeRefinementDescriptor::bisected_edge_array);

    nb::class_<NativeGeometry>(m, "NativeGeometry")
        .def_static("root", &NativeGeometry::root, "ndim"_a, "root_subcells_per_axis"_a = 2, "tol"_a = 1e-14)
        .def_prop_ro("ndim", &NativeGeometry::ndim)
        .def_prop_ro("root_subcells_per_axis", &NativeGeometry::root_subcells_per_axis)
        .def_prop_ro("n_vertices", &NativeGeometry::n_vertices)
        .def_prop_ro("n_simplices", &NativeGeometry::n_simplices)
        .def_prop_ro("n_active", &NativeGeometry::n_active)
        .def("vertices_array", &NativeGeometry::vertices_array)
        .def("active_simplex_ids", &NativeGeometry::active_simplex_ids)
        .def("simplex_vertex_ids", &NativeGeometry::simplex_vertex_ids, "simplex_id"_a)
        .def("simplex_points", &NativeGeometry::simplex_points, "simplex_id"_a)
        .def("simplex_volume", &NativeGeometry::simplex_volume, "simplex_id"_a)
        .def("ensure_children", &NativeGeometry::ensure_children, "simplex_id"_a)
        .def("descendant_leaves", &NativeGeometry::descendant_leaves, "simplex_id"_a, "levels"_a)
        .def("refine", &NativeGeometry::refine, "marked_ids"_a);

    nb::class_<NativeFrontier>(m, "NativeFrontier")
        .def_static("from_geometry", &NativeFrontier::from_geometry, "geometry"_a)
        .def("sync_from_geometry", &NativeFrontier::sync_from_geometry)
        .def("apply_refinement", nb::overload_cast<Int1D, Int1D, Int1D>(&NativeFrontier::apply_refinement), "parent_ids"_a, "child_offsets"_a, "child_ids"_a)
        .def_prop_ro("n_active", &NativeFrontier::n_active)
        .def_prop_ro("generation", &NativeFrontier::generation)
        .def_prop_ro("n_leaf_vertices", &NativeFrontier::n_leaf_vertices)
        .def("active_simplex_ids", &NativeFrontier::active_simplex_ids)
        .def("vertex_ids", &NativeFrontier::vertex_ids)
        .def("volumes", &NativeFrontier::volumes);

    nb::class_<NativeTightBindingModel>(m, "NativeTightBindingModel")
        .def(nb::init<Int2D, Complex3D>(), "keys"_a, "matrices"_a)
        .def_prop_ro("ndim", &NativeTightBindingModel::ndim)
        .def_prop_ro("ndof", &NativeTightBindingModel::ndof)
        .def_prop_ro("nterms", &NativeTightBindingModel::nterms)
        .def("keys_array", &NativeTightBindingModel::keys_array)
        .def("matrices_array", &NativeTightBindingModel::matrices_array)
        .def("evaluate_point", &NativeTightBindingModel::evaluate_point, "point"_a)
        .def("evaluate_many", &NativeTightBindingModel::evaluate_many, "points"_a)
        .def("spectral_bound", &NativeTightBindingModel::spectral_bound);

    nb::class_<NativeSpectralCache>(m, "NativeSpectralCache")
        .def(nb::init<std::shared_ptr<NativeTightBindingModel>, double>(), "model"_a, "tol"_a = 1e-14)
        .def("clear", &NativeSpectralCache::clear)
        .def("invalidate", &NativeSpectralCache::invalidate)
        .def_prop_ro("generation", &NativeSpectralCache::generation)
        .def_prop_ro("ndim", &NativeSpectralCache::ndim)
        .def_prop_ro("ndof", &NativeSpectralCache::ndof)
        .def_prop_ro("size", &NativeSpectralCache::size)
        .def_prop_ro("n_kernel_evals", &NativeSpectralCache::n_kernel_evals)
        .def_prop_ro("n_reduced_point_lookups", &NativeSpectralCache::n_reduced_point_lookups)
        .def_prop_ro("n_geometry_vertex_lookups", &NativeSpectralCache::n_geometry_vertex_lookups)
        .def_prop_ro("geometry_vertex_cache_size", &NativeSpectralCache::geometry_vertex_cache_size)
        .def("evaluate_many", &NativeSpectralCache::evaluate_many, "points"_a)
        .def("get_many", &NativeSpectralCache::get_many, "points"_a)
        .def("get_many_values", &NativeSpectralCache::get_many_values, "points"_a);

    nb::class_<NativeChargeEvaluator>(m, "NativeChargeEvaluator")
        .def(nb::init<NativeGeometry &, NativeSpectralCache &, double>(), "geometry"_a, "spectral_cache"_a, "tol"_a = 1e-14)
        .def_prop_ro("preview_depth", &NativeChargeEvaluator::preview_depth)
        .def("evaluate", &NativeChargeEvaluator::evaluate, "frontier"_a, "mu"_a, "levels"_a = 0)
        .def("simplex_charge", &NativeChargeEvaluator::simplex_charge, "simplex_id"_a, "mu"_a, "levels"_a = 0)
        .def("solve_mu_and_refine", &NativeChargeEvaluator::solve_mu_and_refine, "frontier"_a, "filling"_a, "options"_a);

    nb::class_<NativeDensityEvaluator>(m, "NativeDensityEvaluator")
        .def(nb::init<NativeGeometry &, NativeSpectralCache &, Float2D, double>(), "geometry"_a, "spectral_cache"_a, "keys"_a, "tol"_a = 1e-14)
        .def("clear", &NativeDensityEvaluator::clear)
        .def_prop_ro("phase_cache_size", &NativeDensityEvaluator::phase_cache_size)
        .def_prop_ro("cached_simplex_value_count", &NativeDensityEvaluator::cached_simplex_value_count)
        .def_prop_ro("leaf_build_count", &NativeDensityEvaluator::leaf_build_count)
        .def("evaluate", &NativeDensityEvaluator::evaluate, "frontier"_a, "mu"_a, "levels"_a = 0)
        .def("evaluate_many", &NativeDensityEvaluator::evaluate_many, "simplex_ids"_a, "mu"_a)
        .def("integrate_adaptive", &NativeDensityEvaluator::integrate_adaptive, "frontier"_a, "mu"_a, "options"_a);
}

}  // namespace meanfi::zero_temp_native
