#pragma once

#include <ceres/sized_cost_function.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include "semantic_slam/pose_math.h"

class FrontOfCameraConstraint : ceres::SizedCostFunction<1, 4, 3, 3>
{
public:
  FrontOfCameraConstraint(Eigen::Quaterniond body_q_sensor = math::identity_quaternion(),
                          Eigen::Vector3d body_p_sensor = Eigen::Vector3d::Zero());

  bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

  static ceres::CostFunction* Create(Eigen::Quaterniond body_q_sensor = math::identity_quaternion(),
                                     Eigen::Vector3d body_p_sensor = Eigen::Vector3d::Zero());

private:
  Pose3 body_T_sensor_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

FrontOfCameraConstraint::FrontOfCameraConstraint(Eigen::Quaterniond body_q_sensor,
                                     Eigen::Vector3d body_p_sensor)
  : body_T_sensor_(body_q_sensor, body_p_sensor)
{
}

/**
 * Parameters: 
 * p[0] : robot orientation in global frame
 * p[1] : robot position in global frame
 * p[2] : object position in global frame
 */
bool FrontOfCameraConstraint::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
{
  Eigen::Map<const Eigen::Quaterniond> map_q_body(parameters[0]);
  Eigen::Map<const Eigen::Vector3d> map_p_body(parameters[1]);
  Eigen::Map<const Eigen::Vector3d> map_p_obj(parameters[2]);

  Pose3 map_T_body(map_q_body, map_p_body);

  // Transform p into the local camera frame
  Eigen::Vector3d body_p_obj = map_T_body.inverse() * map_p_obj;
  Eigen::Vector3d camera_p_obj = body_T_sensor_.inverse() * body_p_obj;

  // Never a "cost" here... TODO
  residuals[0] = 0;

  // Fill in Jacobians with all zeros
  if (jacobians) {
    if (jacobians[0]) memset(jacobians[0], 0, sizeof(double) * 4);
    if (jacobians[1]) memset(jacobians[1], 0, sizeof(double) * 3);
    if (jacobians[2]) memset(jacobians[2], 0, sizeof(double) * 3);
  }

  // Check if it is in front of the camera. If not return false i.e. infeasible
  if (camera_p_obj(2) > 0) {
      return true;
  } else {
      return false;
  }
}

ceres::CostFunction* FrontOfCameraConstraint::Create(Eigen::Quaterniond body_q_sensor,
                                     Eigen::Vector3d body_p_sensor)
{
  return new FrontOfCameraConstraint(body_q_sensor, body_p_sensor);
}
