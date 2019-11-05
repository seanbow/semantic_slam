#pragma once

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

namespace math {
// Quaternion storage order = [x y z w]
using Quaternion = Eigen::Vector4d;

inline Quaternion
identity_quaternion()
{
  Quaternion q;
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

inline Eigen::Matrix<double, 4, 3>
quat_Xi_mat(const Quaternion& q)
{
  Eigen::Matrix<double, 4, 3> Xi;
  // Xi << q[3], -q[2], q[1], q[2], q[3], -q[0], -q[1], q[0], q[3], -q[0],
  // -q[1], -q[2];
  // return Xi;

  Xi.topRows<3>() =
    q(3) * Eigen::Matrix3d::Identity() + math::skewsymm(q.head<3>());
  Xi.bottomRows<1>() = -q.head<3>().transpose();
  return Xi;
}

inline Eigen::Matrix<double, 4, 3>
quat_Psi_mat(const Quaternion& q)
{
  Eigen::Matrix<double, 4, 3> Psi;

  Psi.topRows<3>() =
    q(3) * Eigen::Matrix3d::Identity() - math::skewsymm(q.head<3>());
  Psi.bottomRows<1>() = -q.head<3>().transpose();
  return Psi;
}

inline Eigen::Matrix4d
quat_R_mat(const Quaternion& q)
{
  Eigen::Matrix4d R;
  R.leftCols<3>() = quat_Xi_mat(q);
  R.rightCols<1>() = q;
  return R;
}

inline Eigen::Matrix4d
quat_L_mat(const Quaternion& q)
{
  Eigen::Matrix4d L;
  L.leftCols<3>() = quat_Psi_mat(q);
  L.rightCols<1>() = q;
  return L;
}

inline Quaternion
quat_mul(const Quaternion& q1, const Quaternion& q2)
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

    // From Eigen
//   Quaternion q;
//   q << q1(3) * q2(0) + q1(0) * q2(3) + q1(1) * q2(2) - q1(2) * q2(1),
//        q1(3) * q2(1) + q1(1) * q2(3) + q1(2) * q2(0) - q1(0) * q2(2),
//        q1(3) * q2(2) + q1(2) * q2(3) + q1(0) * q2(1) - q1(1) * q2(0),
//        q1(3) * q2(3) - q1(0) * q2(0) - q1(1) * q2(1) - q1(2) * q2(2);
  
  Quaternion q = quat_L_mat(q1) * q2;
  if (q(3) < 0) {
    q = -q;
  }
  q = q / q.norm();
  return q;
}

/*Quaternion to Rotation Orthonormal Matrix as defined in Nikolas t.r*/
inline Eigen::Matrix<double, 3, 3>
quat2rot(Quaternion q)
{
    Eigen::Matrix<double, 3, 3> A;
    A(0, 0) = q(0) * q(0) - q(1) * q(1) - q(2) * q(2) + q(3) * q(3);
    A(0, 1) = 2 * (q(0) * q(1) + q(2) * q(3));
    A(0, 2) = 2 * (q(0) * q(2) - q(1) * q(3));

    A(1, 0) = 2 * (q(0) * q(1) - q(2) * q(3));
    A(1, 1) = -q(0) * q(0) + q(1) * q(1) - q(2) * q(2) + q(3) * q(3);
    A(1, 2) = 2 * (q(1) * q(2) + q(0) * q(3));

    A(2, 0) = 2 * (q(0) * q(2) + q(1) * q(3));
    A(2, 1) = 2 * (q(1) * q(2) - q(0) * q(3));
    A(2, 2) = -q(0) * q(0) - q(1) * q(1) + q(2) * q(2) + q(3) * q(3);
    return A;

//   Eigen::Matrix3d res;

//   const double tx = 2 * q(0);
//   const double ty = 2 * q(1);
//   const double tz = 2 * q(2);
//   const double twx = tx * q(3);
//   const double twy = ty * q(3);
//   const double twz = tz * q(3);
//   const double txx = tx * q(0);
//   const double txy = ty * q(0);
//   const double txz = tz * q(0);
//   const double tyy = ty * q(1);
//   const double tyz = tz * q(1);
//   const double tzz = tz * q(2);

//   res.coeffRef(0, 0) = 1 - (tyy + tzz);
//   res.coeffRef(0, 1) = txy - twz;
//   res.coeffRef(0, 2) = txz + twy;

//   res.coeffRef(1, 0) = txy + twz;
//   res.coeffRef(1, 1) = 1 - (txx + tzz);
//   res.coeffRef(1, 2) = tyz - twx;

//   res.coeffRef(2, 0) = txz - twy;
//   res.coeffRef(2, 1) = tyz + twx;
//   res.coeffRef(2, 2) = 1 - (txx + tyy);

//   return res;
}
//
//  rot2quat
//% converts a rotational matrix to a unit quaternion, according to JPL
//% procedure (Breckenridge Memo)
inline Quaternion
rot2quat(const Eigen::MatrixXd& rot)
{
  assert(rot.cols() == 3);
  assert(rot.rows() == 3);
  Quaternion q;
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
inline Quaternion
quat_inv(const Quaternion& x)
{
  Quaternion out;
  out(0) = -x(0);
  out(1) = -x(1);
  out(2) = -x(2);
  out(3) = x(3);
  return out;
}

inline Eigen::Matrix4d
Dquat_inv(const Quaternion& q)
{
    Eigen::Matrix4d Dquat_inversion = -Eigen::Matrix4d::Identity();
    Dquat_inversion(3, 3) = 1.0;
    return Dquat_inversion;
}

// 4x4 Jacobian of q1 * q2 w.r.t. q1
inline Eigen::Matrix4d
Dquat_mul_dq1(const Quaternion& q1, const Quaternion& q2)
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
inline Eigen::Matrix4d
Dquat_mul_dq2(const Quaternion& q1, const Quaternion& q2)
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
inline Eigen::Matrix<double, 3, 4>
Dpoint_transform_dq(const Quaternion& q, const Eigen::Vector3d& p)
{
  Eigen::Matrix<double, 3, 4> D;

  // Computed in mathematica
    D << 2*(p(0)*q(0) + p(1)*q(1) + p(2)*q(2)),
        -2*(-(p(1)*q(0)) + p(0)*q(1) + p(2)*q(3)),
        2*(p(2)*q(0) - p(0)*q(2) + p(1)*q(3)),
        2*(-(p(2)*q(1)) + p(1)*q(2) + p(0)*q(3)),

        2*(-(p(1)*q(0)) + p(0)*q(1) + p(2)*q(3)),
        2*(p(0)*q(0) + p(1)*q(1) + p(2)*q(2)),
        -2*(-(p(2)*q(1)) + p(1)*q(2) + p(0)*q(3)),
        2*(p(2)*q(0) - p(0)*q(2) + p(1)*q(3)),

        -2*(p(2)*q(0) - p(0)*q(2) + p(1)*q(3)),
        2*(-(p(2)*q(1)) + p(1)*q(2) + p(0)*q(3)),
        2*(p(0)*q(0) + p(1)*q(1) + p(2)*q(2)),
        2*(-(p(1)*q(0)) + p(0)*q(1) + p(2)*q(3));

  // Computed in mathematica
  // HAMILTON CONVENTIONS. wrong.
    // D << 2*(p(1)*q(1) + p(2)*q(2)),
    //     2*(p(1)*q(0) - 2*p(0)*q(1) + p(2)*q(3)),
    //     2*p(2)*q(0) - 4*p(0)*q(2) - 2*p(1)*q(3),
    //     2*p(2)*q(1) - 2*p(1)*q(2),
    //     -4*p(1)*q(0) + 2*p(0)*q(1) - 2*p(2)*q(3),
    //     2*(p(0)*q(0) + p(2)*q(2)),
    //     2*(p(2)*q(1) - 2*p(1)*q(2) + p(0)*q(3)),
    //     -2*p(2)*q(0) + 2*p(0)*q(2),
    //     2*(-2*p(2)*q(0) + p(0)*q(2) + p(1)*q(3)),
    //     -4*p(2)*q(1) + 2*p(1)*q(2) - 2*p(0)*q(3),
    //     2*(p(0)*q(0) + p(1)*q(1)),
    //     2*p(1)*q(0) - 2*p(0)*q(1);

  return D;
}
 
// 3x4 Jacobian of R^T(q)*p w.r.t. q
inline Eigen::Matrix<double, 3, 4>
Dpoint_transform_transpose_dq(const Quaternion& q, const Eigen::Vector3d& p)
{
  Eigen::Matrix<double, 3, 4> D;

    D << 2*(p(0)*q(0) + p(1)*q(1) + p(2)*q(2)),
        2*(p(1)*q(0) - p(0)*q(1) + p(2)*q(3)),
        -2*(-(p(2)*q(0)) + p(0)*q(2) + p(1)*q(3)),
        2*(p(2)*q(1) - p(1)*q(2) + p(0)*q(3)),

        -2*(p(1)*q(0) - p(0)*q(1) + p(2)*q(3)),
        2*(p(0)*q(0) + p(1)*q(1) + p(2)*q(2)),
        2*(p(2)*q(1) - p(1)*q(2) + p(0)*q(3)),
        2*(-(p(2)*q(0)) + p(0)*q(2) + p(1)*q(3)),

        2*(-(p(2)*q(0)) + p(0)*q(2) + p(1)*q(3)),
        -2*(p(2)*q(1) - p(1)*q(2) + p(0)*q(3)),
        2*(p(0)*q(0) + p(1)*q(1) + p(2)*q(2)),
        2*(p(1)*q(0) - p(0)*q(1) + p(2)*q(3));

  // Computed in mathematica
  // HAMILTON CONVENTIONS. wrong.
// D << 2*(p(1)*q(1) + p(2)*q(2)),
//     2*p(1)*q(0) - 4*p(0)*q(1) - 2*p(2)*q(3),
//     2*(p(2)*q(0) - 2*p(0)*q(2) + p(1)*q(3)),
//     -2*p(2)*q(1) + 2*p(1)*q(2),
//     2*(-2*p(1)*q(0) + p(0)*q(1) + p(2)*q(3)),
//     2*(p(0)*q(0) + p(2)*q(2)),
//     2*p(2)*q(1) - 4*p(1)*q(2) - 2*p(0)*q(3),
//     2*p(2)*q(0) - 2*p(0)*q(2),
//     -4*p(2)*q(0) + 2*p(0)*q(2) - 2*p(1)*q(3),
//     2*(-2*p(2)*q(1) + p(1)*q(2) + p(0)*q(3)),
//     2*(p(0)*q(0) + p(1)*q(1)),
//     -2*p(1)*q(0) + 2*p(0)*q(1);

  return D;
}

// 3x3 Jacobian of R(q)*p w.r.t. p
inline Eigen::Matrix3d
Dpoint_transform_dp(const Quaternion& q, const Eigen::Vector3d& p)
{
  return quat2rot(q);
}

// 3x3 Jacobian of R^T(q)*p w.r.t. p
inline Eigen::Matrix3d
Dpoint_transform_transpose_dp(const Quaternion& q, const Eigen::Vector3d& p)
{
  return quat2rot(q).transpose();
}

} // namespace math