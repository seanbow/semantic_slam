#pragma once

#include <ceres/autodiff_cost_function.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include "semantic_slam/pose_math.h"
#include "semantic_slam/quaternion_math.h"

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

PosePriorCostTerm::PosePriorCostTerm(const Pose3& prior_pose,
                                     const Eigen::MatrixXd& prior_covariance)
{
    // Compute sqrt of information matrix
    Eigen::MatrixXd sqrtC = prior_covariance.llt().matrixL();
    sqrt_information_.setIdentity();
    sqrtC.triangularView<Eigen::Lower>().solveInPlace(sqrt_information_);

    // q_prior_inverse_ = math::quat_inv(prior_pose.rotation());
    q_prior_inverse_ = prior_pose.rotation().conjugate();
    p_prior_ = prior_pose.translation();
}

template<typename T>
bool
PosePriorCostTerm::operator()(const T* const data_ptr, T* residual_ptr) const
{
    Eigen::Map<const Eigen::Quaternion<T>> q(data_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p(data_ptr + 4);

    Eigen::Map<Eigen::Matrix<T, 6, 1>> residual(residual_ptr);

    // Eigen::Quaternion<T> dq =
    // Eigen::Map<Eigen::Quaterniond>(q_prior_inverse_.data()).cast<T>() * q;
    Eigen::Quaternion<T> dq = q_prior_inverse_.cast<T>() * q;
    Eigen::Matrix<T, 3, 1> dp = p - p_prior_;

    residual.template head<3>() = T(2.0) * dq.vec().template head<3>();
    residual.template tail<3>() = dp;

    residual.applyOnTheLeft(sqrt_information_);

    return true;
}

ceres::CostFunction*
PosePriorCostTerm::Create(const Pose3& prior_pose,
                          const Eigen::MatrixXd& prior_covariance)
{
    return new ceres::AutoDiffCostFunction<PosePriorCostTerm, 6, 7>(
      new PosePriorCostTerm(prior_pose, prior_covariance));
}
