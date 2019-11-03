#pragma once

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

namespace math {
// Quaternion storage order = [x y z w]

template <typename T>
using Quaternion = Eigen::Matrix<T,4,1>;

using Quaterniond = Quaternion<double>;

inline Quaterniond
identity_quaternion()
{
  Quaterniond q;
  q << 0, 0, 0, 1;
  return q;
}

inline Eigen::Matrix3d
skewsymm(const Eigen::Vector3d& x)
{
  //    function C = skewsymm(X1)
  //    % generates skew symmetric matrix
  //    C = [0      , -X1(3) ,  X1(2)
  //        X1(3) , 0      , -X1(1)
  //        -X1(2) , X1(1)  ,     0];
  Eigen::Matrix3d omega;
  omega << 0, -x(2), x(1), x(2), 0, -x(0), -x(1), x(0), 0;
  return omega;
}

// template code is such a mess sometimes 

template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 4, 3>
quat_Xi_mat(const Eigen::MatrixBase<Derived>& q)
{
  Eigen::Matrix<typename Derived::Scalar, 4, 3> Xi;
  // Xi << q[3], -q[2], q[1], q[2], q[3], -q[0], -q[1], q[0], q[3], -q[0],
  // -q[1], -q[2];
  // return Xi;

  Xi.template topRows<3>() =
    q(3) * Eigen::Matrix<typename Derived::Scalar,3,3>::Identity() + math::skewsymm(q.template head<3>());
  Xi.template bottomRows<1>() = -q.template head<3>().transpose();
  return Xi;
}

template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 4, 3>
quat_Psi_mat(const Eigen::MatrixBase<Derived>& q)
{
  Eigen::Matrix<typename Derived::Scalar, 4, 3> Psi;

  Psi.template topRows<3>() =
    q(3) * Eigen::Matrix<typename Derived::Scalar,3,3>::Identity() - math::skewsymm(q.template head<3>());
  Psi.template bottomRows<1>() = -q.template head<3>().transpose();
  return Psi;
}

template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 4, 4>
quat_R_mat(const Eigen::MatrixBase<Derived>& q)
{
  Eigen::Matrix<typename Derived::Scalar,4,4> R;
  R.template leftCols<3>() = quat_Xi_mat(q);
  R.template rightCols<1>() = q;
  return R;
}

template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 4, 4>
quat_L_mat(const Eigen::MatrixBase<Derived>& q)
{
  Eigen::Matrix<typename Derived::Scalar,4,4> L;
  L.template leftCols<3>() = quat_Psi_mat(q);
  L.template rightCols<1>() = q;
  return L;
}

template <typename Derived1, typename Derived2>
inline Eigen::Matrix<typename Derived1::Scalar, 4, 1>
quat_mul(const Eigen::MatrixBase<Derived1>& q1, const Eigen::MatrixBase<Derived2>& q2)
{
  //   Eigen::MatrixXd Q1_MATR(4, 4);
  //   Q1_MATR(0, 0) = q1(3);
  //   Q1_MATR(0, 1) = q1(2);
  //   Q1_MATR(0, 2) = -q1(1);
  //   Q1_MATR(0, 3) = q1(0);

  //   Q1_MATR(1, 0) = -q1(2);
  //   Q1_MATR(1, 1) = q1(3);
  //   Q1_MATR(1, 2) = q1(0);
  //   Q1_MATR(1, 3) = q1(1);

  //   Q1_MATR(2, 0) = q1(1);
  //   Q1_MATR(2, 1) = -q1(0);
  //   Q1_MATR(2, 2) = q1(3);
  //   Q1_MATR(2, 3) = q1(2);

  //   Q1_MATR(3, 0) = -q1(0);
  //   Q1_MATR(3, 1) = -q1(1);
  //   Q1_MATR(3, 2) = -q1(2);
  //   Q1_MATR(3, 3) = q1(3);
  //   Quaternion q = Q1_MATR * q2;
  Eigen::Matrix<typename Derived1::Scalar, 4, 1> q;
  q = quat_L_mat(q1) * q2;
//   if (q(3) < 0) {
//     q = -q;
//   }
  q = q / q.norm();
  return q;
}

// This is hacky
// can't have the q(3) < 0 in autodifferentiating code so specialize for plain Quaterniond
// probably won't work like I want it to
template <>
inline Quaterniond
quat_mul<Quaterniond, Quaterniond>(const Eigen::MatrixBase<Quaterniond>& q1, 
                                   const Eigen::MatrixBase<Quaterniond>& q2)
{
  Quaterniond q = quat_L_mat(q1) * q2;
  if (q(3) < 0) {
    q = -q;
  }
  q = q / q.norm();
  return q;
}

/*Quaternion to Rotation Orthonormal Matrix as defined in Nikolas t.r*/
// template <typename T>
// inline Eigen::Matrix<T, 3, 3>
template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3>
quat2rot(const Eigen::MatrixBase<Derived>& q)
{
  //   Eigen::Matrix<double, 3, 3> A;
  //   A(0, 0) = q(0) * q(0) - q(1) * q(1) - q(2) * q(2) + q(3) * q(3);
  //   A(0, 1) = 2 * (q(0) * q(1) + q(2) * q(3));
  //   A(0, 2) = 2 * (q(0) * q(2) - q(1) * q(3));

  //   A(1, 0) = 2 * (q(0) * q(1) - q(2) * q(3));
  //   A(1, 1) = -q(0) * q(0) + q(1) * q(1) - q(2) * q(2) + q(3) * q(3);
  //   A(1, 2) = 2 * (q(1) * q(2) + q(0) * q(3));

  //   A(2, 0) = 2 * (q(0) * q(2) + q(1) * q(3));
  //   A(2, 1) = 2 * (q(1) * q(2) - q(0) * q(3));
  //   A(2, 2) = -q(0) * q(0) - q(1) * q(1) + q(2) * q(2) + q(3) * q(3);
  //   return A;

  Eigen::Matrix<typename Derived::Scalar,3,3> res;

  using T = typename Derived::Scalar;

  const T tx = T(2.0) * q(0);
  const T ty = T(2.0) * q(1);
  const T tz = T(2.0) * q(2);
  const T twx = tx * q(3);
  const T twy = ty * q(3);
  const T twz = tz * q(3);
  const T txx = tx * q(0);
  const T txy = ty * q(0);
  const T txz = tz * q(0);
  const T tyy = ty * q(1);
  const T tyz = tz * q(1);
  const T tzz = tz * q(2);

  res.coeffRef(0, 0) = T(1.0) - (tyy + tzz);
  res.coeffRef(0, 1) = txy - twz;
  res.coeffRef(0, 2) = txz + twy;
  res.coeffRef(1, 0) = txy + twz;
  res.coeffRef(1, 1) = T(1.0) - (txx + tzz);
  res.coeffRef(1, 2) = tyz - twx;
  res.coeffRef(2, 0) = txz - twy;
  res.coeffRef(2, 1) = tyz + twx;
  res.coeffRef(2, 2) = T(1.0) - (txx + tyy);

  return res;
}
//
//  rot2quat
//% converts a rotational matrix to a unit quaternion, according to JPL
//% procedure (Breckenridge Memo)
template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 4, 1>
// rot2quat(const Eigen::Matrix<Type,3,3>& rot)
rot2quat(const Eigen::MatrixBase<Derived>& rot)
{
  assert(rot.cols() == 3);
  assert(rot.rows() == 3);
  Eigen::Matrix<typename Derived::Scalar, 4, 1> q;
  double T = rot.trace();
  if ((rot(0, 0) > T) && (rot(0, 0) > rot(1, 1)) && (rot(0, 0) > rot(2, 2))) {
    q(0) = sqrt((1 + (2 * rot(0, 0)) - T) / 4);
    q(1) = (1 / (4 * q(0))) * (rot(0, 1) + rot(1, 0));
    q(2) = (1 / (4 * q(0))) * (rot(0, 2) + rot(2, 0));
    q(3) = (1 / (4 * q(0))) * (rot(1, 2) - rot(2, 1));
  } else if ((rot(1, 1) > T) && (rot(1, 1) > rot(0, 0)) &&
             (rot(1, 1) > rot(2, 2))) {
    q(1) = sqrt((1 + (2 * rot(1, 1)) - T) / 4);
    q(0) = (1 / (4 * q(1))) * (rot(0, 1) + rot(1, 0));
    q(2) = (1 / (4 * q(1))) * (rot(1, 2) + rot(2, 1));
    q(3) = (1 / (4 * q(1))) * (rot(2, 0) - rot(0, 2));
  } else if ((rot(2, 2) > T) && (rot(2, 2) > rot(0, 0)) &&
             (rot(2, 2) > rot(1, 1))) {
    q(2) = sqrt((1 + (2 * rot(2, 2)) - T) / 4);
    q(0) = (1 / (4 * q(2))) * (rot(0, 2) + rot(2, 0));
    q(1) = (1 / (4 * q(2))) * (rot(1, 2) + rot(2, 1));
    q(3) = (1 / (4 * q(2))) * (rot(0, 1) - rot(1, 0));
  } else {
    q(3) = sqrt((1 + T) / 4);
    q(0) = (1 / (4 * q(3))) * (rot(1, 2) - rot(2, 1));
    q(1) = (1 / (4 * q(3))) * (rot(2, 0) - rot(0, 2));
    q(2) = (1 / (4 * q(3))) * (rot(0, 1) - rot(1, 0));
  }
  if (q(3) < 0) {
    q = -q;
  }
  q = q / q.norm();
  return q;
}

/* Inverse Quaternion */
template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 4, 1>
quat_inv(const Eigen::MatrixBase<Derived>& x)
{
  Eigen::Matrix<typename Derived::Scalar, 4, 1> out;
  out(0) = -x(0);
  out(1) = -x(1);
  out(2) = -x(2);
  out(3) = x(3);
  return out;
}

// 4x4 Jacobian of q1 * q2 w.r.t. q1
template <typename T>
inline Eigen::Matrix<T,4,4>
Dquat_mul_dq1(const Quaternion<T>& q1, const Quaternion<T>& q2)
{
    // Eigen::Matrix4d Dq_dq1;
    // Dq_dq1 << q2(3), -q2(2),  q2(1), q2(0), 
    //           q2(2),  q2(3), -q2(0), q2(1), 
    //          -q2(1),  q2(0),  q2(3), q2(2), 
    //          -q2(0), -q2(1), -q2(2), q2(3);
    // return Dq_dq1;
    return quat_R_mat(q2);
}

// 4x4 Jacobian of q1 * q2 w.r.t. q2
template <typename T>
inline Eigen::Matrix<T,4,4>
Dquat_mul_dq2(const Quaternion<T>& q1, const Quaternion<T>& q2)
{
    // Eigen::Matrix4d Dq_dq2;
    // Dq_dq2 << q1(3),  q1(2), -q1(1), q1(0), 
    //          -q1(2),  q1(3),  q1(0), q1(1), 
    //           q1(1), -q1(0),  q1(3), q1(2), 
    //          -q1(0), -q1(1), -q1(2), q1(3);
    // return Dq_dq2;
    return quat_L_mat(q1);
}

// 3x4 Jacobian of R(q)*p w.r.t. q
template <typename T>
inline Eigen::Matrix<T, 3, 4>
Dpoint_transform_dq(const Quaternion<T>& q, const Eigen::Matrix<T,3,1>& p)
{
  Eigen::Matrix<T, 3, 4> D;

  // Computed in mathematica
  D(0, 0) = 2 * (p(1) * q(1) + p(2) * q(2));
  D(0, 1) = 2 * (p(1) * q(0) - 2 * p(0) * q(1) + p(2) * q(3));
  D(0, 2) = 2 * p(2) * q(0) - 4 * p(0) * q(2) - 2 * p(1) * q(3);
  D(0, 3) = 2 * p(2) * q(1) - 2 * p(1) * q(2);

  D(1, 0) = -4 * p(1) * q(0) + 2 * p(0) * q(1) - 2 * p(2) * q(3);
  D(1, 1) = 2 * (p(0) * q(0) + p(2) * q(2));
  D(1, 2) = 2 * (p(2) * q(1) - 2 * p(1) * q(2) + p(0) * q(3));
  D(1, 3) = -2 * p(2) * q(0) + 2 * p(0) * q(2);

  D(2, 0) = 2 * (-2 * p(2) * q(0) + p(0) * q(2) + p(1) * q(3));
  D(2, 1) = -4 * p(2) * q(1) + 2 * p(1) * q(2) - 2 * p(0) * q(3);
  D(2, 2) = 2 * (p(0) * q(0) + p(1) * q(1));
  D(2, 3) = 2 * p(1) * q(0) - 2 * p(0) * q(1);

  //   2 * p(0) * q(0) + 2 * p(1) * q(1) + 2 * p(2) * q(2);
  //   D(0, 1) = 2 * p(1) * q(0) - 2 * p(0) * q(1) - 2 * p(2) * q(3);
  //   D(0, 2) = 2 * p(2) * q(0) - 2 * p(0) * q(2) + 2 * p(1) * q(3);
  //   D(0, 3) = -2 * p(2) * q(1) + 2 * p(1) * q(2) + 2 * p(0) * q(3);

  //   D(1, 0) = -2 * p(1) * q(0) + 2 * p(0) * q(1) + 2 * p(2) * q(3);
  //   D(1, 1) = 2 * p(0) * q(0) + 2 * p(1) * q(1) + 2 * p(2) * q(2);
  //   D(1, 2) = 2 * p(2) * q(1) - 2 * p(1) * q(2) - 2 * p(0) * q(3);
  //   D(1, 3) = 2 * p(2) * q(0) - 2 * p(0) * q(2) + 2 * p(1) * q(3);

  //   D(2, 0) = -2 * p(2) * q(0) + 2 * p(0) * q(2) - 2 * p(1) * q(3);
  //   D(2, 1) = -2 * p(2) * q(1) + 2 * p(1) * q(2) + 2 * p(0) * q(3);
  //   D(2, 2) = 2 * p(0) * q(0) + 2 * p(1) * q(1) + 2 * p(2) * q(2);
  //   D(2, 3) = -2 * p(1) * q(0) + 2 * p(0) * q(1) + 2 * p(2) * q(3);

  return D;
}
 
// 3x4 Jacobian of R^T(q)*p w.r.t. q
inline Eigen::Matrix<double, 3, 4>
Dpoint_transform_transpose_dq(const Quaterniond& q, const Eigen::Vector3d& p)
{
  Eigen::Matrix<double, 3, 4> D;

  // Computed in mathematica
  D(0, 0) = 2 * (p(1) * q(1) + p(2) * q(2));
  D(0, 1) = 2 * p(1) * q(0) - 4 * p(0) * q(1) - 2 * p(2) * q(3);
  D(0, 2) = 2 * (p(2) * q(0) - 2 * p(0) * q(2) + p(1) * q(3));
  D(0, 3) = -2 * p(2) * q(1) + 2 * p(1) * q(2);

  D(1, 0) = 2 * (-2 * p(1) * q(0) + p(0) * q(1) + p(2) * q(3));
  D(1, 1) = 2 * (p(0) * q(0) + p(2) * q(2));
  D(1, 2) = 2 * p(2) * q(1) - 4 * p(1) * q(2) - 2 * p(0) * q(3);
  D(1, 3) = 2 * p(2) * q(0) - 2 * p(0) * q(2);

  D(2, 0) = -4 * p(2) * q(0) + 2 * p(0) * q(2) - 2 * p(1) * q(3);
  D(2, 1) = 2 * (-2 * p(2) * q(1) + p(1) * q(2) + p(0) * q(3));
  D(2, 2) = 2 * (p(0) * q(0) + p(1) * q(1));
  D(2, 3) = -2 * p(1) * q(0) + 2 * p(0) * q(1);

  //  2 * (p(0) * q(0) + p(1) * q(1) + p(2) * q(2));
  //   D(0, 1) = 2 * (p(1) * q(0) - p(0) * q(1) + p(2) * q(3));
  //   D(0, 2) = -2 * (-p(2) * q(0) + p(0) * q(2) + p(1) * q(3));
  //   D(0, 3) = 2 * (p(2) * q(1) - p(1) * q(2) + p(0) * q(3));

  //   D(1, 0) = -2 * (p(1) * q(0) - p(0) * q(1) + p(2) * q(3));
  //   D(1, 1) = 2 * (p(0) * q(0) + p(1) * q(1) + p(2) * q(2));
  //   D(1, 2) = 2 * (p(2) * q(1) - p(1) * q(2) + p(0) * q(3));
  //   D(1, 3) = 2 * (-p(2) * q(0) + p(0) * q(2) + p(1) * q(3));

  //   D(2, 0) = 2 * (-p(2) * q(0) + p(0) * q(2) + p(1) * q(3));
  //   D(2, 1) = -2 * (p(2) * q(1) - p(1) * q(2) + p(0) * q(3));
  //   D(2, 2) = 2 * (p(0) * q(0) + p(1) * q(1) + p(2) * q(2));
  //   D(2, 3) = 2 * (p(1) * q(0) - p(0) * q(1) + p(2) * q(3));

  return D;
}

// 3x3 Jacobian of R(q)*p w.r.t. p
inline Eigen::Matrix3d
Dpoint_transform_dp(const Quaterniond& q, const Eigen::Vector3d& p)
{
  return quat2rot(q);
}

// 3x3 Jacobian of R^T(q)*p w.r.t. p
inline Eigen::Matrix3d
Dpoint_transform_transpose_dp(const Quaterniond& q, const Eigen::Vector3d& p)
{
  return quat2rot(q).transpose();
}

} // namespace math