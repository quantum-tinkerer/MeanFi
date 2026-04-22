#pragma once

#include "types.h"

#include <Eigen/Dense>

namespace meanfi::zero_temp_native {

class TightBindingModel {
public:
    TightBindingModel(Int2D keys, Complex3D matrices);

    size_t ndim() const noexcept { return ndim_; }
    size_t ndof() const noexcept { return ndof_; }
    size_t nterms() const noexcept { return nterms_; }

    nb::ndarray<nb::numpy, std::int64_t> keys_array() const;
    nb::ndarray<nb::numpy, std::complex<double>> matrices_array() const;
    nb::ndarray<nb::numpy, std::complex<double>> evaluate_many(Float2D points) const;
    nb::ndarray<nb::numpy, std::complex<double>> evaluate_point(Float1D point) const;

    Eigen::MatrixXcd evaluate_point_raw(const double *point) const;
    double spectral_bound() const;

private:
    size_t ndim_ = 0;
    size_t ndof_ = 0;
    size_t nterms_ = 0;
    std::vector<std::int64_t> keys_;
    std::vector<std::complex<double>> matrices_;
};

}  // namespace meanfi::zero_temp_native
