#pragma once

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

namespace math {
// Quaternion storage order = [x y z w]
// using Quaternion = Eigen::Vector4d;

using Eigen::Quaterniond;

inline Quaterniond
identity_quaternion()
{
  return Quaterniond(1,0,0,0);
}

inline Eigen::Matrix3d
skewsymm(const Eigen::Vector3d& x)
{
  Eigen::Matrix3d omega;
  omega << 0, -x(2), x(1), x(2), 0, -x(0), -x(1), x(0), 0;
  return omega;
}

inline Eigen::Matrix4d
Dquat_inv(const Quaterniond& q)
{
    Eigen::Matrix4d Dquat_inversion = -Eigen::Matrix4d::Identity();
    Dquat_inversion(3, 3) = 1.0;
    return Dquat_inversion;
}

// 4x4 Jacobian of q1 * q2 w.r.t. q1
inline Eigen::Matrix4d
Dquat_mul_dq1(const Quaterniond& qa, const Quaterniond& qb)
{
    Eigen::Matrix4d Dq_dq1;
    Dq_dq1  <<  qb.w(),  qb.z(), -qb.y(), qb.x(),
               -qb.z(),  qb.w(),  qb.x(), qb.y(),
                qb.y(), -qb.x(),  qb.w(), qb.z(),
               -qb.x(), -qb.y(), -qb.z(), qb.w();
    return Dq_dq1;
}

// 4x4 Jacobian of q1 * q2 w.r.t. q2
inline Eigen::Matrix4d
Dquat_mul_dq2(const Quaterniond& qa, const Quaterniond& qb)
{
    Eigen::Matrix4d Dq_dq2;
    Dq_dq2 << qa.w(), -qa.z(),  qa.y(), qa.x(),
              qa.z(),  qa.w(), -qa.x(), qa.y(),
             -qa.y(),  qa.x(),  qa.w(), qa.z(),
             -qa.x(), -qa.y(), -qa.z(), qa.w();

    return Dq_dq2;
}

// 3x4 Jacobian of R(q)*p w.r.t. q
inline Eigen::Matrix<double, 3, 4>
Dpoint_transform_dq(const Quaterniond& q, const Eigen::Vector3d& p)
{
  Eigen::Matrix<double, 3, 4> D;

  // Computed in mathematica
    D << 2*(p(1)*q.y() + p(2)*q.z()),
        2*(p(1)*q.x() - 2*p(0)*q.y() + p(2)*q.w()),
        2*p(2)*q.x() - 4*p(0)*q.z() - 2*p(1)*q.w(),
        2*p(2)*q.y() - 2*p(1)*q.z(),

        -4*p(1)*q.x() + 2*p(0)*q.y() - 2*p(2)*q.w(),
        2*(p(0)*q.x() + p(2)*q.z()),
        2*(p(2)*q.y() - 2*p(1)*q.z() + p(0)*q.w()),
        -2*p(2)*q.x() + 2*p(0)*q.z(),

        2*(-2*p(2)*q.x() + p(0)*q.z() + p(1)*q.w()),
        -4*p(2)*q.y() + 2*p(1)*q.z() - 2*p(0)*q.w(),
        2*(p(0)*q.x() + p(1)*q.y()),
        2*p(1)*q.x() - 2*p(0)*q.y();

  return D;
}
 
// 3x4 Jacobian of R^T(q)*p w.r.t. q
inline Eigen::Matrix<double, 3, 4>
Dpoint_transform_transpose_dq(const Quaterniond& qa, const Eigen::Vector3d& p)
{
    Eigen::Matrix<double, 3, 4> D;

    // Computed in mathematica  
    D << 2*(p(1)*qa.y() + p(2)*qa.z()),
        2*p(1)*qa.x() - 4*p(0)*qa.y() - 2*p(2)*qa.w(),
        2*(p(2)*qa.x() - 2*p(0)*qa.z() + p(1)*qa.w()),
        -2*p(2)*qa.y() + 2*p(1)*qa.z(),

        2*(-2*p(1)*qa.x() + p(0)*qa.y() + p(2)*qa.w()),
        2*(p(0)*qa.x() + p(2)*qa.z()),
        2*p(2)*qa.y() - 4*p(1)*qa.z() - 2*p(0)*qa.w(),
        2*p(2)*qa.x() - 2*p(0)*qa.z(),

        -4*p(2)*qa.x() + 2*p(0)*qa.z() - 2*p(1)*qa.w(),
        2*(-2*p(2)*qa.y() + p(1)*qa.z() + p(0)*qa.w()),
        2*(p(0)*qa.x() + p(1)*qa.y()),
        -2*p(1)*qa.x() + 2*p(0)*qa.y();

    return D;
}

// 3x3 Jacobian of R(q)*p w.r.t. p
inline Eigen::Matrix3d
Dpoint_transform_dp(const Quaterniond& q, const Eigen::Vector3d& p)
{
    return q.toRotationMatrix();
}

// 3x3 Jacobian of R^T(q)*p w.r.t. p
inline Eigen::Matrix3d
Dpoint_transform_transpose_dp(const Quaterniond& q, const Eigen::Vector3d& p)
{
    return q.toRotationMatrix().transpose();
}

} // namespace math