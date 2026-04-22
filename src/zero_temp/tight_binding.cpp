#include "tight_binding.h"

#include <stdexcept>

namespace meanfi::zero_temp_native {

TightBindingModel::TightBindingModel(Int2D keys, Complex3D matrices) {
    ndim_ = keys.shape(1);
    nterms_ = keys.shape(0);
    if (matrices.shape(0) != nterms_) {
        throw std::runtime_error("TightBindingModel: term axis mismatch");
    }
    if (matrices.shape(1) != matrices.shape(2)) {
        throw std::runtime_error("TightBindingModel: matrices must be square");
    }
    ndof_ = matrices.shape(1);
    keys_.assign(keys.data(), keys.data() + nterms_ * ndim_);
    matrices_.assign(matrices.data(), matrices.data() + nterms_ * ndof_ * ndof_);
}

nb::ndarray<nb::numpy, std::int64_t> TightBindingModel::keys_array() const {
    return make_array(std::vector<std::int64_t>(keys_), {nterms_, ndim_});
}

nb::ndarray<nb::numpy, std::complex<double>> TightBindingModel::matrices_array() const {
    return make_array(
        std::vector<std::complex<double>>(matrices_),
        {nterms_, ndof_, ndof_}
    );
}

nb::ndarray<nb::numpy, std::complex<double>> TightBindingModel::evaluate_many(Float2D points) const {
    const auto n_points = points.shape(0);
    if (points.shape(1) != ndim_) {
        throw std::runtime_error("TightBindingModel: point dimension mismatch");
    }
    std::vector<std::complex<double>> out(n_points * ndof_ * ndof_);
    const double *point_ptr = points.data();
    for (size_t point_index = 0; point_index < n_points; ++point_index) {
        const Eigen::MatrixXcd h = evaluate_point_raw(point_ptr + point_index * ndim_);
        for (size_t row = 0; row < ndof_; ++row) {
            for (size_t col = 0; col < ndof_; ++col) {
                out[(point_index * ndof_ + row) * ndof_ + col] = h(row, col);
            }
        }
    }
    return make_array(std::move(out), {n_points, ndof_, ndof_});
}

nb::ndarray<nb::numpy, std::complex<double>> TightBindingModel::evaluate_point(Float1D point) const {
    if (point.shape(0) != ndim_) {
        throw std::runtime_error("TightBindingModel: point dimension mismatch");
    }
    const Eigen::MatrixXcd h = evaluate_point_raw(point.data());
    std::vector<std::complex<double>> out(ndof_ * ndof_);
    for (size_t row = 0; row < ndof_; ++row) {
        for (size_t col = 0; col < ndof_; ++col) {
            out[row * ndof_ + col] = h(row, col);
        }
    }
    return make_array(std::move(out), {ndof_, ndof_});
}

Eigen::MatrixXcd TightBindingModel::evaluate_point_raw(const double *point) const {
    Eigen::MatrixXcd h = Eigen::MatrixXcd::Zero(ndof_, ndof_);
    for (size_t term = 0; term < nterms_; ++term) {
        double phase_arg = 0.0;
        for (size_t axis = 0; axis < ndim_; ++axis) {
            phase_arg += point[axis] * static_cast<double>(keys_[term * ndim_ + axis]);
        }
        const std::complex<double> phase = std::exp(std::complex<double>(0.0, -phase_arg));
        for (size_t row = 0; row < ndof_; ++row) {
            for (size_t col = 0; col < ndof_; ++col) {
                h(row, col) += phase * matrices_[(term * ndof_ + row) * ndof_ + col];
            }
        }
    }
    return h;
}

double TightBindingModel::spectral_bound() const {
    double bound = 0.0;
    for (size_t term = 0; term < nterms_; ++term) {
        Eigen::MatrixXcd matrix(ndof_, ndof_);
        for (size_t row = 0; row < ndof_; ++row) {
            for (size_t col = 0; col < ndof_; ++col) {
                matrix(row, col) = matrices_[(term * ndof_ + row) * ndof_ + col];
            }
        }
        bound += matrix.norm();
    }
    return bound;
}

}  // namespace meanfi::zero_temp_native
