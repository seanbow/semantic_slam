// math utilities

#include <ros/ros.h>
#include <gtsam/geometry/Pose3.h>
#include <Eigen/Geometry>

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

void FromROSMsg(const geometry_msgs::PoseWithCovariance& msg,
                gtsam::Pose3& pose,
                Eigen::Matrix<double, 6, 6>& covariance) {
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;

    FromROSMsg(msg, position, orientation, covariance);

    pose = gtsam::Pose3(gtsam::Rot3(orientation.toRotationMatrix()), gtsam::Point3(position));
}

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