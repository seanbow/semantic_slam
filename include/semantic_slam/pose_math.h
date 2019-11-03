#pragma once

#include "semantic_slam/quaternion_math.h"

#include <boost/optional.hpp>

class Pose3
{
public:
  Pose3();
  Pose3(math::Quaterniond q, Eigen::Vector3d p);

  static Pose3 Identity();

  const math::Quaterniond& rotation() const { return q_; }
  math::Quaterniond& rotation() { return q_; }

  const Eigen::Vector3d& translation() const { return p_; }
  Eigen::Vector3d& translation() { return p_; }

  double* rotation_data() { return q_.data(); }
  double* translation_data() { return p_.data(); }

  const double* rotation_data() const { return q_.data(); }
  const double* translation_data() const { return p_.data(); }

  Eigen::Vector3d transform_from(
    const Eigen::Vector3d& p,
    boost::optional<Eigen::MatrixXd&> Hpose = boost::none,
    boost::optional<Eigen::MatrixXd&> Hpoint = boost::none) const;

  Eigen::Vector3d transform_to(
    const Eigen::Vector3d& p,
    boost::optional<Eigen::MatrixXd&> Hpose = boost::none,
    boost::optional<Eigen::MatrixXd&> Hpoint = boost::none) const;

  Pose3 inverse(boost::optional<Eigen::MatrixXd&> H = boost::none) const;

  Pose3 compose(const Pose3& other,
                boost::optional<Eigen::MatrixXd&> H1 = boost::none,
                boost::optional<Eigen::MatrixXd&> H2 = boost::none) const;

  Pose3 operator*(const Pose3& other) const;
  Eigen::Vector3d operator*(const Eigen::Vector3d& other_pt) const;
  
  Pose3 between(const Pose3& other,
                boost::optional<Eigen::MatrixXd&> Hpose1 = boost::none,
                boost::optional<Eigen::MatrixXd&> Hpose2 = boost::none) const;

private:
  math::Quaterniond q_;
  Eigen::Vector3d p_;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

Pose3::Pose3()
  : q_(math::identity_quaternion())
  , p_(Eigen::Vector3d::Zero())
{
}

Pose3::Pose3(math::Quaterniond q, Eigen::Vector3d p)
  : q_(q)
  , p_(p)
{
}

Pose3
Pose3::Identity()
{
  return Pose3();
}

Pose3
Pose3::inverse(boost::optional<Eigen::MatrixXd&> H) const
{
  Pose3 inverted(math::quat_inv(q_), -math::quat2rot(q_).transpose() * p_);

  if (H) {
    Eigen::Matrix4d Dq_dq = -Eigen::Matrix4d::Identity();
    Dq_dq(3, 3) = 1.0;

    Eigen::Matrix<double, 4, 3> Dq_dp = Eigen::Matrix<double, 4, 3>::Zero();
    Eigen::Matrix<double, 3, 4> Dp_dq =
      -math::Dpoint_transform_transpose_dq(q_, p_);
    Eigen::Matrix<double, 3, 3> Dp_dp = -math::quat2rot(q_).transpose();

    *H = Eigen::MatrixXd(7, 7);
    H->block<4, 4>(0, 0) = Dq_dq;
    H->block<3, 4>(4, 0) = Dp_dq;
    H->block<4, 3>(0, 4) = Dq_dp;
    H->block<3, 3>(4, 4) = Dp_dp;
  }

  return inverted;
}

Pose3
Pose3::compose(const Pose3& other, boost::optional<Eigen::MatrixXd&> H1,
               boost::optional<Eigen::MatrixXd&> H2) const
{
  const math::Quaterniond& q2 = other.rotation();
  const Eigen::Vector3d& p2 = other.translation();
  Pose3 new_pose(math::quat_mul(q_, q2), p_ + math::quat2rot(q_) * p2);

  if (H1) {
    *H1 = Eigen::MatrixXd(7, 7);
    Eigen::Matrix4d Dq_dq1;
    Dq_dq1 << q2(3), -q2(2), q2(1), q2(0), q2(2), q2(3), -q2(0), q2(1), -q2(1),
      q2(0), q2(3), q2(2), -q2(0), -q2(1), -q2(2), q2(3);

    Eigen::Matrix<double, 4, 3> Dq_dp1 = Eigen::Matrix<double, 4, 3>::Zero();

    Eigen::Matrix<double, 3, 4> Dp_dq1 = math::Dpoint_transform_dq(q_, p2);
    Eigen::Matrix3d Dp_dp1 = Eigen::Matrix3d::Identity();

    H1->block<4, 4>(0, 0) = Dq_dq1;
    H1->block<3, 4>(4, 0) = Dp_dq1;

    H1->block<4, 3>(0, 4) = Dq_dp1;
    H1->block<3, 3>(4, 4) = Dp_dp1;
  }

  if (H2) {
    *H2 = Eigen::MatrixXd(7, 7);
    Eigen::Matrix4d Dq_dq2;
    Dq_dq2 << q_(3), q_(2), -q_(1), q_(0), -q_(2), q_(3), q_(0), q_(1), q_(1),
      -q_(0), q_(3), q_(2), -q_(0), -q_(1), -q_(2), q_(3);

    Eigen::Matrix<double, 4, 3> Dq_dp2 = Eigen::Matrix<double, 4, 3>::Zero();
    Eigen::Matrix<double, 3, 4> Dp_dq2 = Eigen::Matrix<double, 3, 4>::Zero();
    Eigen::Matrix3d Dp_dp2 = math::quat2rot(q_);

    H2->block<4, 4>(0, 0) = Dq_dq2;
    H2->block<3, 4>(4, 0) = Dp_dq2;

    H2->block<4, 3>(0, 4) = Dq_dp2;
    H2->block<3, 3>(4, 4) = Dp_dp2;
  }

  return new_pose;
}

Pose3 Pose3::operator*(const Pose3& other) const
{
  return this->compose(other);
}

Eigen::Vector3d Pose3::operator*(const Eigen::Vector3d& other_pt) const
{
  return this->transform_from(other_pt);
}

Eigen::Vector3d
Pose3::transform_from(const Eigen::Vector3d& p,
                      boost::optional<Eigen::MatrixXd&> Hpose,
                      boost::optional<Eigen::MatrixXd&> Hpoint) const
{
  Eigen::Vector3d result = math::quat2rot(q_) * p + p_;

  if (Hpose) {
    *Hpose = Eigen::MatrixXd(3, 7);
    Hpose->block<3, 4>(0, 0) = math::Dpoint_transform_dq(q_, p);
    Hpose->block<3, 3>(0, 4) = Eigen::Matrix3d::Identity();
  }

  if (Hpoint) {
    *Hpoint = math::quat2rot(q_);
  }

  return result;
}

Eigen::Vector3d
Pose3::transform_to(const Eigen::Vector3d& p,
                    boost::optional<Eigen::MatrixXd&> Hpose,
                    boost::optional<Eigen::MatrixXd&> Hpoint) const
{
  Eigen::Vector3d result = math::quat2rot(q_).transpose() * (p - p_);

  if (Hpose) {
    *Hpose = Eigen::MatrixXd(3, 7);
    Hpose->block<3, 4>(0, 0) = math::Dpoint_transform_transpose_dq(q_, p - p_);
    Hpose->block<3, 3>(0, 4) = -math::quat2rot(q_).transpose();
  }

  if (Hpoint) {
    *Hpoint = math::quat2rot(q_).transpose();
  }

  return result;
}

Pose3
Pose3::between(const Pose3& other,
               boost::optional<Eigen::MatrixXd&> Hpose1,
               boost::optional<Eigen::MatrixXd&> Hpose2) const
{
    math::Quaterniond dq = math::quat_mul( math::quat_inv(q_), other.rotation() );
    Eigen::Vector3d dp = math::quat2rot(q_).transpose() * (other.translation() - p_);
    Pose3 result(dq, dp);

    if (Hpose1) {
        *Hpose1 = Eigen::MatrixXd(7, 7);

        Eigen::Matrix4d Dquat_inversion = -Eigen::Matrix4d::Identity();
        Dquat_inversion(3, 3) = 1.0;
        Eigen::Matrix4d Dq_dq1 = math::Dquat_mul_dq1(q_,other.rotation()) * Dquat_inversion;

        Eigen::Matrix<double, 4, 3> Dq_dp1 = Eigen::Matrix<double,4,3>::Zero();

        Eigen::Matrix<double, 3, 4> Dp_dq1 = math::Dpoint_transform_transpose_dq(q_, other.translation() - p_);
        Eigen::Matrix3d Dp_dp1 = -math::quat2rot(q_).transpose();

        Hpose1->block<4,4>(0,0) = Dq_dq1;
        Hpose1->block<4,3>(0,4) = Dq_dp1;
        Hpose1->block<3,4>(4,0) = Dp_dq1;
        Hpose1->block<3,3>(4,4) = Dp_dp1;
    }

    if (Hpose2) {
        *Hpose2 = Eigen::MatrixXd(7,7);

        Eigen::Matrix4d Dq_dq2 = math::Dquat_mul_dq2(math::quat_inv(q_), other.rotation());
        Eigen::Matrix<double,4,3> Dq_dp2 = Eigen::Matrix<double,4,3>::Zero();
        Eigen::Matrix<double,3,4> Dp_dq2 = Eigen::Matrix<double,3,4>::Zero();
        Eigen::Matrix3d Dp_dp2 = math::quat2rot(q_).transpose();

        Hpose2->block<4,4>(0,0) = Dq_dq2;
        Hpose2->block<4,3>(0,4) = Dq_dp2;
        Hpose2->block<3,4>(4,0) = Dp_dq2;
        Hpose2->block<3,3>(4,4) = Dp_dp2;
    }

    return result;
}