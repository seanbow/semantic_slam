// math utilities

#include <ros/ros.h>
#include <ceres/ceres.h>
// #include <gtsam/geometry/Pose3.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <sensor_msgs/Imu.h>

#include <boost/optional.hpp>

#include "semantic_slam/Utils.h"

double clamp_angle(double angle) {
  static constexpr double pi = 3.1415926536;

  if (std::isnan(angle) || std::isnan(-angle) || std::isinf(angle)) return angle;

  while (angle > pi) {
    angle -= 2*pi;
  }

  while (angle <= -pi) {
    angle += 2*pi;
  }

  return angle;
}

double sgn_fn(double x) {
    return (0.0 < x) - (x < 0.0);
}

Eigen::Matrix3d findRotation(const Eigen::MatrixXd& S1, const Eigen::MatrixXd& S2) {
  Eigen::Matrix3d M = S1 * S2.transpose();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);

  Eigen::Matrix3d R1 = svd.matrixU() * svd.matrixV().transpose();

  Eigen::Vector3d reflect;
  reflect << 1, 1, sgn_fn(R1.determinant());

  return svd.matrixU() * reflect.asDiagonal() * svd.matrixV().transpose();
}

void findSimilarityTransform(const Eigen::MatrixXd& S1, const Eigen::MatrixXd& S2, Eigen::Matrix3d& R, Eigen::Vector3d& T, boost::optional<double&> w) {
  Eigen::Vector3d T1 = S1.rowwise().mean();
  Eigen::MatrixXd S1_sub = S1.colwise() - T1;

  Eigen::Vector3d T2 = S2.rowwise().mean();
  Eigen::MatrixXd S2_sub = S2.colwise() - T2;

  R = findRotation(S1_sub, S2_sub);

  S2_sub = R * S2_sub;

  if (w) {
    *w = (S1_sub.transpose() * S2_sub).trace() / (S2_sub.transpose() * S2_sub).trace();
    T = T1 - (*w)*R*T2;
  } else {
    T = T1 - R*T2;
  }
}

void FromROSMsg(const geometry_msgs::PoseWithCovariance& msg,
                Eigen::Vector3d& position,
                Eigen::Quaterniond& orientation,
                Eigen::Matrix<double, 6, 6>& covariance) {

    position << msg.pose.position.x, msg.pose.position.y, msg.pose.position.z;

    orientation.x() = msg.pose.orientation.x;
    orientation.y() = msg.pose.orientation.y;
    orientation.z() = msg.pose.orientation.z;
    orientation.w() = msg.pose.orientation.w;

    boostArrayToEigen<6,6>(msg.covariance, covariance);
}

// void FromROSMsg(const geometry_msgs::PoseWithCovariance& msg,
//                 gtsam::Pose3& pose,
//                 Eigen::Matrix<double, 6, 6>& covariance) {
//     Eigen::Vector3d position;
//     Eigen::Quaterniond orientation;

//     FromROSMsg(msg, position, orientation, covariance);

//     pose = gtsam::Pose3(gtsam::Rot3(orientation.toRotationMatrix()), gtsam::Point3(position));
// }

void FromROSMsg(const sensor_msgs::Imu& msg,
                Eigen::Vector3d& omega,
                Eigen::Matrix3d& omega_cov,
                Eigen::Vector3d& accel,
                Eigen::Matrix3d& accel_cov) {

    // orientation.x() = msg.orientation.x;
    // orientation.y() = msg.orientation.y;
    // orientation.z() = msg.orientation.z;
    // orientation.w() = msg.orientation.w;

    omega << msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z;

    accel << msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z;

    boostArrayToEigen<3,3>(msg.angular_velocity_covariance, omega_cov);
    boostArrayToEigen<3,3>(msg.linear_acceleration_covariance, accel_cov);

}

Eigen::Matrix<double, 2, 9> computeProjectionJacobian(const Pose3& G_T_I,
                                                      const Pose3& I_T_C,
                                                      const Eigen::Vector3d& G_l)
{
  Eigen::MatrixXd Hpose1, Hpoint1;
  Eigen::Vector3d I_p = G_T_I.transform_to(G_l, Hpose1, Hpoint1);

  Eigen::MatrixXd Hpoint2;
  Eigen::Vector3d C_p = I_T_C.transform_from(I_p, boost::none, Hpoint2);

  Eigen::Matrix<double, 2, 3> Hcam;
  Hcam(0,0) = 1 / C_p(2);
  Hcam(1,1) = 1 / C_p(2);
  Hcam(0,1) = 0;
  Hcam(1,0) = 0;
  Hcam(0,2) = -C_p(0) / (C_p(2)*C_p(2));
  Hcam(1,2) = -C_p(1) / (C_p(2)*C_p(2));

  // Eigen::MatrixXd Hpose = Hcam * Hpoint2 * Hpose1;
  // Eigen::MatrixXd Hpoint = Hcam * Hpoint2 * Hpoint1;

  // Hpose is in the *ambient* (4-dimensional) quaternion space.
  // Want it in the *tangent* (3-dimensional) space.
  Eigen::Matrix<double, 4, 3, Eigen::RowMajor> Hquat_space;
  ceres::EigenQuaternionParameterization local_param;
  local_param.ComputeJacobian(G_T_I.rotation_data(), Hquat_space.data());

  Eigen::Matrix<double, 2, 9> H;
  H.block<2,3>(0,0) = Hcam * Hpoint2 * Hpoint1;
  H.block<2,3>(0,3) = Hcam * Hpoint2 * Hpose1.block<3,4>(0,0) * Hquat_space;
  H.block<2,3>(0,6) = Hcam * Hpoint2 * Hpose1.block<3,3>(0,4);

  return H;
}

// Eigen::Matrix<double, 2, 9> computeProjectionJacobian(const Eigen::Matrix3d& G_R_I,
//                                           const Eigen::Vector3d& G_t_I,
//                                           const Eigen::Matrix3d& I_R_C,
//                                           const Eigen::Vector3d& G_l)
// {
//   Eigen::Matrix3d G_R_C = G_R_I * I_R_C;

//   // assume G_t_I = G_t_C TODO

//   Eigen::Vector3d C_p = G_R_C.transpose() * (G_l - G_t_I);

//   Eigen::Matrix<double, 2, 3> Hcam;
//   Hcam(0,0) = 1 / C_p(2);
//   Hcam(1,1) = 1 / C_p(2);
//   Hcam(0,1) = 0;
//   Hcam(1,0) = 0;
//   Hcam(0,2) = -C_p(0) / (C_p(2)*C_p(2));
//   Hcam(1,2) = -C_p(1) / (C_p(2)*C_p(2));

//   Eigen::Matrix3d Hq = I_R_C.transpose() * skewsymm(G_R_I.transpose() * (G_l - G_t_I));
//   Eigen::Matrix3d Hp = -G_R_C.transpose();
//   Eigen::Matrix3d Hl = G_R_C.transpose();

//   Eigen::Matrix<double, 2, 9> H;

//   H.block<2,3>(0,0) = Hcam * Hl;
//   H.block<2,3>(0,3) = Hcam * Hq;
//   H.block<2,3>(0,6) = Hcam * Hp;

//   return H;
// }
