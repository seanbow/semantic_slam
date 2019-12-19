
#include "semantic_slam/ceres_cost_terms/ceres_projection.h"

ProjectionCostTerm::ProjectionCostTerm(
  const Eigen::Vector2d& measured,
  const Eigen::Matrix2d& msmt_covariance,
  const Eigen::Quaterniond& body_q_camera,
  const Eigen::Vector3d& body_p_camera,
  boost::shared_ptr<CameraCalibration> camera_calibration)
  : measured_(measured)
  , body_q_camera_(body_q_camera)
  , body_p_camera_(body_p_camera)
  , camera_calibration_(camera_calibration)
{
    Eigen::MatrixXd sqrtC = msmt_covariance.llt().matrixL();
    sqrt_information_.setIdentity();
    sqrtC.triangularView<Eigen::Lower>().solveInPlace(sqrt_information_);
}

template<typename T>
bool
ProjectionCostTerm::operator()(const T* const map_x_body_ptr,
                               const T* const map_pt_ptr,
                               T* residual_ptr) const
{
    Eigen::Map<const Eigen::Quaternion<T>> map_q_body(map_x_body_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> map_p_body(map_x_body_ptr + 4);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> map_pt(map_pt_ptr);

    // Transform point to camera coordinates
    Eigen::Matrix<T, 3, 1> body_pt =
      map_q_body.template conjugate() * (map_pt - map_p_body);
    Eigen::Matrix<T, 3, 1> sensor_pt = body_q_camera_.cast<T>().conjugate() *
                                       (body_pt - body_p_camera_.cast<T>());

    // Project to camera
    // double fx = 1, fy = 1;
    // double u0 = 0, v0 = 0;
    // if (camera_calibration_) {
    //   fx = camera_calibration_->fx();
    //   fy = camera_calibration_->fy();
    //   u0 = camera_calibration_->u0();
    //   v0 = camera_calibration_->v0();
    // }

    // Eigen::Matrix<T, 2, 1> zhat;
    // zhat(0) = sensor_pt(0) / sensor_pt(2);
    // zhat(0) = fx * zhat(0) + u0;

    // zhat(1) = sensor_pt(1) / sensor_pt(2);
    // zhat(1) = fy * zhat(1) + v0;

    Eigen::Matrix<T, 2, 1> zhat_normalized(sensor_pt(0) / sensor_pt(2),
                                           sensor_pt(1) / sensor_pt(2));
    Eigen::Matrix<T, 2, 1> zhat;

    if (camera_calibration_) {
        zhat = camera_calibration_->uncalibrate(zhat_normalized);
    } else {
        zhat = zhat_normalized;
    }

    // Compute residual
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residual(residual_ptr);
    residual = sqrt_information_ * (measured_ - zhat);

    return true;
}

ceres::CostFunction*
ProjectionCostTerm::Create(
  const Eigen::Vector2d& measured,
  const Eigen::Matrix2d& msmt_covariance,
  const Eigen::Quaterniond& body_q_camera,
  const Eigen::Vector3d& body_p_camera,
  boost::shared_ptr<CameraCalibration> camera_calibration)
{
    return new ceres::AutoDiffCostFunction<ProjectionCostTerm, 2, 7, 3>(
      new ProjectionCostTerm(measured,
                             msmt_covariance,
                             body_q_camera,
                             body_p_camera,
                             camera_calibration));
}