#pragma once

// #include <gtsam/nonlinear/NonlinearFactor.h>
// #include <gtsam/nonlinear/Values.h>
// #include <gtsam/linear/JacobianFactor.h>
// #include <gtsam/geometry/Pose3.h>

#include <boost/make_shared.hpp>
#include <boost/optional.hpp>

#include <ceres/dynamic_autodiff_cost_function.h>

#include "semantic_slam/Utils.h"
#include "semantic_slam/keypoints/geometry.h"

using geometry::ObjectModelBasis;

class StructureCostTerm
{
  private:
    typedef StructureCostTerm This;

  public:
    typedef boost::shared_ptr<This> shared_ptr;
    typedef shared_ptr Ptr;

    StructureCostTerm(const ObjectModelBasis& model,
                      const Eigen::VectorXd& weights,
                      double lambda = 1.0);

    static ceres::CostFunction* Create(const ObjectModelBasis& model,
                                       const Eigen::VectorXd& weights,
                                       double lambda = 1.0);

    void setWeights(const Eigen::VectorXd& weights) { weights_ = weights; }

    template<typename T>
    bool unwhitenedError(T const* const* parameters, T* residuals_ptr) const;

    // computes residuals
    template<typename T>
    bool operator()(T const* const* parameters, T* residuals_ptr) const;

    size_t dim() const { return 3 * m_ + k_; }

    size_t m() { return m_; }

    size_t k() { return k_; }

  private:
    ObjectModelBasis model_;

    double lambda_; // regularization factor

    size_t m_, k_;

    // gtsam::noiseModel::Base::shared_ptr noise_model_;

    Eigen::VectorXd weights_;

    template<typename T>
    Eigen::Matrix<T, 3, Eigen::Dynamic> structure(
      T const* const* parameters) const;

    template<typename T>
    bool whitenError(T* residuals) const;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

ceres::CostFunction*
StructureCostTerm::Create(const ObjectModelBasis& model,
                          const Eigen::VectorXd& weights,
                          double lambda)
{
    StructureCostTerm* cost_term =
      new StructureCostTerm(model, weights, lambda);
    ceres::DynamicAutoDiffCostFunction<StructureCostTerm, 4>* cost_function =
      new ceres::DynamicAutoDiffCostFunction<StructureCostTerm, 4>(cost_term);

    cost_function->AddParameterBlock(4);
    cost_function->AddParameterBlock(3);
    for (int i = 0; i < cost_term->m(); ++i) {
        cost_function->AddParameterBlock(3);
    }
    if (cost_term->k() > 0) {
        cost_function->AddParameterBlock(cost_term->k());
    }

    cost_function->SetNumResiduals(cost_term->dim());

    return cost_function;
}

StructureCostTerm::StructureCostTerm(const ObjectModelBasis& model,
                                     const Eigen::VectorXd& weights,
                                     double lambda)
  : model_(model)
  , lambda_(lambda)
{
    m_ = model_.mu.cols();
    k_ = model_.pc.rows() / 3;

    setWeights(weights);
}

template<typename T>
Eigen::Matrix<T, 3, Eigen::Dynamic>
StructureCostTerm::structure(T const* const* parameters) const
{
    Eigen::Matrix<T, 3, Eigen::Dynamic> S =
      Eigen::Matrix<T, 3, Eigen::Dynamic>::Zero(3, m_);

    S += model_.mu;

    if (k_ == 0)
        return S;

    Eigen::Map<const Eigen::Matrix<T, -1, 1>> c(parameters[2 + m_], k_);

    for (size_t i = 0; i < k_; ++i) {
        S += c[i] * model_.pc.block(3 * i, 0, 3, m_);
    }

    return S;
}

// #include <iostream>
// using std::cout; using std::endl;

// Compute residual vector [P - R*S - t] and jacobians
template<typename T>
bool
StructureCostTerm::unwhitenedError(T const* const* parameters,
                                   T* residuals_ptr) const
{
    Eigen::Map<const Eigen::Quaternion<T>> q(parameters[0]);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(parameters[1]);

    Eigen::Map<Eigen::Matrix<T, -1, 1>> residuals(residuals_ptr, dim());

    Eigen::Matrix<T, 3, Eigen::Dynamic> S = structure(parameters);

    for (size_t i = 0; i < m_; ++i) {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p(parameters[2 + i]);

        residuals.template segment<3>(3 * i) = p - q * S.col(i) - t;
        // cout << "Index " << i << ", key " <<
        // gtsam::DefaultKeyFormatter(landmark_keys_[i]) <<
        //     ", p = " << p.transpose() << ", err = " << (p - R*S.col(i) -
        //     t).transpose() << endl;
    }

    if (k_ > 0) {
        Eigen::Map<const Eigen::Matrix<T, -1, 1>> c(parameters[2 + m_], k_);

        residuals.template tail(k_) = c;
    }

    // std::cout << "Computed residuals " << residuals.transpose() << std::endl;

    return true;
}

template<typename T>
bool
StructureCostTerm::whitenError(T* residuals_ptr) const
{
    Eigen::Map<Eigen::Matrix<T, -1, 1>> residuals(residuals_ptr, dim());

    for (size_t i = 0; i < m_; ++i) {
        residuals.template segment<3>(3 * i) =
          residuals.template segment<3>(3 * i) * std::sqrt(weights_(i));
    }

    residuals.template tail(k_) =
      residuals.template tail(k_) * std::sqrt(lambda_);

    return true;
}

template<typename T>
bool
StructureCostTerm::operator()(T const* const* parameters, T* residuals) const
{
    // parameters p:
    // p[0] = object orientation (quaternion)
    // p[1] = object position
    // p[2] through p[(2 + m_) - 1] = keypoint positions
    // p[2 + m_] = structure coefficients
    // T r0 = residuals[0];

    unwhitenedError(parameters, residuals);
    whitenError(residuals);

    // std::cout << "Residual 0: " << r0 << " -> " << residuals[0] << std::endl;
    return true;
}
