#pragma once

#include "semantic_slam/Pose3.h"
#include <ceres/sized_cost_function.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

class FrontOfCameraConstraint : ceres::SizedCostFunction<1, 4, 3, 3>
{
  public:
    FrontOfCameraConstraint(
      Eigen::Quaterniond body_q_sensor = math::identity_quaternion(),
      Eigen::Vector3d body_p_sensor = Eigen::Vector3d::Zero());

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const;

    static ceres::CostFunction* Create(
      Eigen::Quaterniond body_q_sensor = math::identity_quaternion(),
      Eigen::Vector3d body_p_sensor = Eigen::Vector3d::Zero());

  private:
    Pose3 body_T_sensor_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};
