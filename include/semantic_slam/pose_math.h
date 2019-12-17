#pragma once

#include "semantic_slam/quaternion_math.h"

#include <boost/optional.hpp>

#include <gtsam/geometry/Pose3.h>

class Pose3
{
  public:
    Pose3();
    Pose3(Eigen::Quaterniond q, Eigen::Vector3d p);

    Pose3(const gtsam::Pose3& pose);

    Pose3(const Pose3& other);
    Pose3& operator=(const Pose3& other);

    // Do we need these?
    Pose3(Pose3&& other);
    Pose3& operator=(Pose3&& other);

    Pose3(const Eigen::VectorXd& data_vec);

    static Pose3 Identity();

    const Eigen::Map<Eigen::Quaterniond>& rotation() const { return q_; }
    Eigen::Map<Eigen::Quaterniond>& rotation() { return q_; }

    const Eigen::Map<Eigen::Vector3d>& translation() const { return p_; }
    Eigen::Map<Eigen::Vector3d>& translation() { return p_; }

    double* data() { return data_vector_.data(); }
    double* rotation_data() { return data_vector_.data(); }
    double* translation_data() { return data_vector_.data() + 4; }

    const double* data() const { return data_vector_.data(); }
    const double* rotation_data() const { return data_vector_.data(); }
    const double* translation_data() const { return data_vector_.data() + 4; }

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

    bool equals(const Pose3& other, double tolerance) const;

    double x() const { return translation()[0]; }
    double y() const { return translation()[1]; }
    double z() const { return translation()[2]; }

    friend std::ostream& operator<<(std::ostream& os, const Pose3& p);

    operator gtsam::Pose3() const;

  private:
    Eigen::Matrix<double, 7, 1> data_vector_;
    Eigen::Map<Eigen::Quaterniond> q_;
    Eigen::Map<Eigen::Vector3d> p_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

Pose3::Pose3()
  : q_(data_vector_.data())
  , p_(data_vector_.data() + 4)
{
    data_vector_.setZero();
    data_vector_(3) = 1.0; // identity quaternion initialization
}

Pose3::Pose3(Eigen::Quaterniond q, Eigen::Vector3d p)
  : q_(data_vector_.data())
  , p_(data_vector_.data() + 4)
{
    q_ = q;
    p_ = p;
}

Pose3::Pose3(const Pose3& other)
  : q_(data_vector_.data())
  , p_(data_vector_.data() + 4)
{
    q_ = other.rotation();
    p_ = other.translation();
}

Pose3::Pose3(const gtsam::Pose3& pose)
  : q_(data_vector_.data())
  , p_(data_vector_.data() + 4)
{
    q_ = pose.rotation().toQuaternion();
    p_ = pose.translation();
}

Pose3&
Pose3::operator=(const Pose3& other)
{
    if (this != &other) {
        q_ = other.rotation();
        p_ = other.translation();
    }
    return *this;
}

Pose3::Pose3(Pose3&& other)
  : data_vector_(std::move(other.data_vector_))
  , q_(data_vector_.data())
  , p_(data_vector_.data() + 4)
{}

Pose3&
Pose3::operator=(Pose3&& other)
{
    q_ = other.rotation();
    p_ = other.translation();

    return *this;
}

Pose3::Pose3(const Eigen::VectorXd& data_vec)
  : q_(data_vector_.data())
  , p_(data_vector_.data() + 4)
{
    data_vector_ = data_vec;
}

Pose3
Pose3::Identity()
{
    return Pose3();
}

Pose3
Pose3::inverse(boost::optional<Eigen::MatrixXd&> H) const
{
    Eigen::Quaterniond q_inv = q_.inverse();
    Pose3 inverted(q_inv, -(q_inv * p_));

    if (H) {
        Eigen::Matrix4d Dq_dq = math::Dquat_inv(q_);

        Eigen::Matrix<double, 4, 3> Dq_dp = Eigen::Matrix<double, 4, 3>::Zero();
        Eigen::Matrix<double, 3, 4> Dp_dq =
          -math::Dpoint_transform_transpose_dq(q_, p_);
        Eigen::Matrix<double, 3, 3> Dp_dp = -q_inv.toRotationMatrix();

        *H = Eigen::MatrixXd(7, 7);
        H->block<4, 4>(0, 0) = Dq_dq;
        H->block<3, 4>(4, 0) = Dp_dq;
        H->block<4, 3>(0, 4) = Dq_dp;
        H->block<3, 3>(4, 4) = Dp_dp;
    }

    return inverted;
}

Pose3
Pose3::compose(const Pose3& other,
               boost::optional<Eigen::MatrixXd&> H1,
               boost::optional<Eigen::MatrixXd&> H2) const
{
    const Eigen::Quaterniond& q2 = other.rotation();
    const Eigen::Vector3d& p2 = other.translation();
    Pose3 new_pose(q_ * q2, p_ + q_ * p2);

    if (H1) {
        *H1 = Eigen::MatrixXd(7, 7);
        Eigen::Matrix4d Dq_dq1 = math::Dquat_mul_dq1(q_, q2);

        Eigen::Matrix<double, 4, 3> Dq_dp1 =
          Eigen::Matrix<double, 4, 3>::Zero();

        Eigen::Matrix<double, 3, 4> Dp_dq1 = math::Dpoint_transform_dq(q_, p2);
        Eigen::Matrix3d Dp_dp1 = Eigen::Matrix3d::Identity();

        H1->block<4, 4>(0, 0) = Dq_dq1;
        H1->block<3, 4>(4, 0) = Dp_dq1;

        H1->block<4, 3>(0, 4) = Dq_dp1;
        H1->block<3, 3>(4, 4) = Dp_dp1;
    }

    if (H2) {
        *H2 = Eigen::MatrixXd(7, 7);
        Eigen::Matrix4d Dq_dq2 = math::Dquat_mul_dq2(q_, q2);

        Eigen::Matrix<double, 4, 3> Dq_dp2 =
          Eigen::Matrix<double, 4, 3>::Zero();
        Eigen::Matrix<double, 3, 4> Dp_dq2 =
          Eigen::Matrix<double, 3, 4>::Zero();
        Eigen::Matrix3d Dp_dp2 = q_.toRotationMatrix();

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
    Eigen::Vector3d result = q_ * p + p_;

    if (Hpose) {
        *Hpose = Eigen::MatrixXd(3, 7);
        Hpose->block<3, 4>(0, 0) = math::Dpoint_transform_dq(q_, p);
        Hpose->block<3, 3>(0, 4) = Eigen::Matrix3d::Identity();
    }

    if (Hpoint) {
        *Hpoint = q_.toRotationMatrix();
    }

    return result;
}

Eigen::Vector3d
Pose3::transform_to(const Eigen::Vector3d& p,
                    boost::optional<Eigen::MatrixXd&> Hpose,
                    boost::optional<Eigen::MatrixXd&> Hpoint) const
{
    Eigen::Vector3d result = q_.conjugate() * (p - p_);

    if (Hpose) {
        *Hpose = Eigen::MatrixXd(3, 7);
        Hpose->block<3, 4>(0, 0) =
          math::Dpoint_transform_transpose_dq(q_, p - p_);
        Hpose->block<3, 3>(0, 4) = -q_.conjugate().toRotationMatrix();
    }

    if (Hpoint) {
        *Hpoint = q_.conjugate().toRotationMatrix();
    }

    return result;
}

Pose3
Pose3::between(const Pose3& other,
               boost::optional<Eigen::MatrixXd&> Hpose1,
               boost::optional<Eigen::MatrixXd&> Hpose2) const
{
    /* HAMILTON */
    Eigen::Quaterniond q_inv = q_.conjugate();
    Eigen::Quaterniond dq = q_inv * other.rotation();
    Eigen::Vector3d dp = q_inv * (other.translation() - p_);
    Pose3 result(dq, dp);

    if (Hpose1) {
        *Hpose1 = Eigen::MatrixXd(7, 7);

        Eigen::Matrix4d Dquat_inversion = math::Dquat_inv(q_);
        Eigen::Matrix4d Dq_dq1 =
          math::Dquat_mul_dq1(q_inv, other.rotation()) * Dquat_inversion;

        Eigen::Matrix<double, 4, 3> Dq_dp1 =
          Eigen::Matrix<double, 4, 3>::Zero();

        Eigen::Matrix<double, 3, 4> Dp_dq1 =
          math::Dpoint_transform_transpose_dq(q_, other.translation() - p_);
        Eigen::Matrix3d Dp_dp1 = -q_inv.toRotationMatrix();

        Hpose1->block<4, 4>(0, 0) = Dq_dq1;
        Hpose1->block<4, 3>(0, 4) = Dq_dp1;
        Hpose1->block<3, 4>(4, 0) = Dp_dq1;
        Hpose1->block<3, 3>(4, 4) = Dp_dp1;
    }

    if (Hpose2) {
        *Hpose2 = Eigen::MatrixXd(7, 7);

        Eigen::Matrix4d Dq_dq2 = math::Dquat_mul_dq2(q_inv, other.rotation());
        Eigen::Matrix<double, 4, 3> Dq_dp2 =
          Eigen::Matrix<double, 4, 3>::Zero();
        Eigen::Matrix<double, 3, 4> Dp_dq2 =
          Eigen::Matrix<double, 3, 4>::Zero();
        Eigen::Matrix3d Dp_dp2 = q_inv.toRotationMatrix();

        Hpose2->block<4, 4>(0, 0) = Dq_dq2;
        Hpose2->block<4, 3>(0, 4) = Dq_dp2;
        Hpose2->block<3, 4>(4, 0) = Dp_dq2;
        Hpose2->block<3, 3>(4, 4) = Dp_dp2;
    }

    return result;
}

bool
Pose3::equals(const Pose3& other, double tolerance) const
{
    double tol2 = tolerance * tolerance;
    return (p_ - other.translation()).squaredNorm() < tol2 &&
           (q_.coeffs() - other.rotation().coeffs()).squaredNorm() < tol2;
}

std::ostream&
operator<<(std::ostream& os, const Pose3& pose)
{
    const Eigen::Quaterniond& q = pose.rotation();
    os << "q: [" << q.x() << ", " << q.y() << ", " << q.z() << ", " << q.w()
       << "];\n";

    const Eigen::Vector3d& t = pose.translation();
    os << "t: [" << t(0) << ", " << t(1) << ", " << t(2) << "]\';\n";
    return os;
}

Pose3::operator gtsam::Pose3() const
{
    return gtsam::Pose3(gtsam::Rot3(rotation()), translation());
}
