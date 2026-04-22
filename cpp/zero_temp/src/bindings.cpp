#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include "adaptive_integrator.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace meanfi::zero_temp {

NB_MODULE(_zero_temp_ext, m) {
    m.doc() = "Compiled runtime for meanfi.zero_temp";

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
            [](const DensityIntegrateResult &self) { return self.error_estimate_available; }
        );

    nb::class_<Geometry>(m, "Geometry")
        .def_static("root", &Geometry::root, "ndim"_a, "root_subcells_per_axis"_a = 2, "tol"_a = 1e-14)
        .def_prop_ro("ndim", &Geometry::ndim)
        .def_prop_ro("root_subcells_per_axis", &Geometry::root_subcells_per_axis)
        .def_prop_ro("generation", &Geometry::generation)
        .def_prop_ro("n_vertices", &Geometry::n_vertices)
        .def_prop_ro("n_simplices", &Geometry::n_simplices)
        .def_prop_ro("n_active", &Geometry::n_active)
        .def_prop_ro("n_leaf_vertices", &Geometry::n_leaf_vertices)
        .def("vertices_array", &Geometry::vertices_array)
        .def("active_simplex_ids", &Geometry::active_simplex_ids)
        .def("simplex_vertex_ids", &Geometry::simplex_vertex_ids, "simplex_id"_a)
        .def("simplex_points", &Geometry::simplex_points, "simplex_id"_a)
        .def("simplex_volume", &Geometry::simplex_volume, "simplex_id"_a)
        .def("ensure_children", &Geometry::ensure_children, "simplex_id"_a)
        .def("descendant_leaves", &Geometry::descendant_leaves, "simplex_id"_a, "levels"_a)
        .def("refine", &Geometry::refine, "marked_ids"_a);

    nb::class_<TightBindingModel>(m, "TightBindingModel")
        .def(nb::init<Int2D, Complex3D>(), "keys"_a, "matrices"_a)
        .def_prop_ro("ndim", &TightBindingModel::ndim)
        .def_prop_ro("ndof", &TightBindingModel::ndof)
        .def_prop_ro("nterms", &TightBindingModel::nterms)
        .def("keys_array", &TightBindingModel::keys_array)
        .def("matrices_array", &TightBindingModel::matrices_array)
        .def("evaluate_point", &TightBindingModel::evaluate_point, "point"_a)
        .def("evaluate_many", &TightBindingModel::evaluate_many, "points"_a)
        .def("spectral_bound", &TightBindingModel::spectral_bound);

    nb::class_<VertexCache>(m, "VertexCache")
        .def(nb::init<std::shared_ptr<TightBindingModel>, double>(), "model"_a, "tol"_a = 1e-14)
        .def("clear", &VertexCache::clear)
        .def("invalidate", &VertexCache::invalidate)
        .def_prop_ro("generation", &VertexCache::generation)
        .def_prop_ro("ndim", &VertexCache::ndim)
        .def_prop_ro("ndof", &VertexCache::ndof)
        .def_prop_ro("size", &VertexCache::size)
        .def_prop_ro("n_kernel_evals", &VertexCache::n_kernel_evals);

    nb::class_<AdaptiveIntegrator>(m, "AdaptiveIntegrator")
        .def(nb::init<Geometry &, VertexCache &, Float2D, double>(), "geometry"_a, "vertex_cache"_a, "keys"_a, "tol"_a = 1e-14)
        .def("clear", &AdaptiveIntegrator::clear)
        .def_prop_ro("preview_depth", &AdaptiveIntegrator::preview_depth)
        .def_prop_ro("phase_cache_size", &AdaptiveIntegrator::phase_cache_size)
        .def_prop_ro("cached_simplex_value_count", &AdaptiveIntegrator::cached_simplex_value_count)
        .def_prop_ro("leaf_build_count", &AdaptiveIntegrator::leaf_build_count)
        .def("evaluate_charge", &AdaptiveIntegrator::evaluate_charge, "mu"_a, "levels"_a = 0)
        .def("evaluate_density", &AdaptiveIntegrator::evaluate_density, "mu"_a, "levels"_a = 0)
        .def("solve_mu_and_refine", &AdaptiveIntegrator::solve_mu_and_refine, "filling"_a, "options"_a)
        .def("integrate_density", &AdaptiveIntegrator::integrate_density, "mu"_a, "options"_a);
}

}  // namespace meanfi::zero_temp
