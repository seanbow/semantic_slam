#pragma once

#include "semantic_slam/Pose3.h"

#include <ceres/autodiff_cost_function.h>

#include "semantic_slam/CameraCalibration.h"

class ProjectionCostTerm
{
  public:
    ProjectionCostTerm(const Eigen::Vector2d& measured,
                       const Eigen::Matrix2d& msmt_covariance,
                       const Eigen::Quaterniond& body_q_camera,
                       const Eigen::Vector3d& body_p_camera,
                       boost::shared_ptr<CameraCalibration> camera_calibration);

    template<typename T>
    bool operator()(const T* const map_x_body_ptr,
                    const T* const map_pt_ptr,
                    T* residual_ptr) const;

    static ceres::CostFunction* Create(
      const Eigen::Vector2d& measured,
      const Eigen::Matrix2d& msmt_covariance,
      const Eigen::Quaterniond& body_q_camera,
      const Eigen::Vector3d& body_p_camera,
      boost::shared_ptr<CameraCalibration> camera_calibration);

    void setCameraCalibration(double fx, double fy, double cx, double cy);

  private:
    Eigen::Vector2d measured_;
    Eigen::Matrix2d sqrt_information_;
    Eigen::Quaterniond body_q_camera_;
    Eigen::Vector3d body_p_camera_;
    boost::shared_ptr<CameraCalibration> camera_calibration_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};
