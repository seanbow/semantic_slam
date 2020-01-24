#pragma once

#include <ceres/autodiff_cost_function.h>
#include <eigen3/Eigen/Core>

#include "semantic_slam/VectorNode.h"

template<int Dim>
class VectorNormPriorCostTerm
{
  public:
    VectorNormPriorCostTerm(double norm, double norm_covariance);

    template<typename T>
    bool operator()(const T* const data, T* residual_ptr) const;

    static ceres::CostFunction* Create(double norm, double norm_covariance);

  private:
    double sqrt_information_;

    double norm_prior_;
};

template<int Dim>
VectorNormPriorCostTerm<Dim>::VectorNormPriorCostTerm(double norm,
                                                      double norm_covariance)
{
    sqrt_information_ = 1.0 / std::sqrt(norm_covariance);
    norm_prior_ = norm;
}

template<int Dim>
template<typename T>
bool
VectorNormPriorCostTerm<Dim>::operator()(const T* const data_ptr,
                                         T* residual_ptr) const
{
    Eigen::Map<const Eigen::Matrix<T, Dim, 1>> x(data_ptr);

    residual_ptr[0] = sqrt_information_ * (x.norm() - norm_prior_);

    return true;
}

template<int Dim>
ceres::CostFunction*
VectorNormPriorCostTerm<Dim>::Create(double norm, double norm_covariance)
{
    return new ceres::
      AutoDiffCostFunction<VectorNormPriorCostTerm<Dim>, 1, Dim>(
        new VectorNormPriorCostTerm(norm, norm_covariance));
}
