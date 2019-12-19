#pragma once

#include <ceres/autodiff_cost_function.h>
#include <eigen3/Eigen/Core>

#include "semantic_slam/Pose3.h"

class PosePriorCostTerm
{
  public:
    PosePriorCostTerm(const Pose3& prior_pose,
                      const Eigen::MatrixXd& prior_covariance);

    template<typename T>
    bool operator()(const T* const data, T* residual_ptr) const;

    static ceres::CostFunction* Create(const Pose3& prior_pose,
                                       const Eigen::MatrixXd& prior_covariance);

  private:
    Eigen::Matrix<double, 6, 6> sqrt_information_;

    Eigen::Quaterniond q_prior_inverse_;
    Eigen::Vector3d p_prior_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};
