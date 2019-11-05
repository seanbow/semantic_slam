#include <ceres/ceres.h>
#include <ceres/local_parameterization.h>
#include <semantic_slam/quaternion_math.h>
#include <eigen3/Eigen/Core>

class QuaternionLocalParameterization : public ceres::LocalParameterization
{
public:
  using Jacobian = Eigen::Matrix<double, 4, 3, Eigen::RowMajor>;
  using LiftJacobian = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;

  virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const;
  virtual bool ComputeJacobian(const double* x, double* jacobian) const;
  
  virtual int GlobalSize() const
  {
    return 4;
  }
  virtual int LocalSize() const
  {
    return 3;
  }
};

bool QuaternionLocalParameterization::Plus(const double* x, const double* delta, double* x_plus_delta) const
{
  math::Quaternion dq;
  dq << 0.5 * delta[0], 0.5 * delta[1], 0.5 * delta[2], 1.0;
  dq.normalize();

  Eigen::Map<const math::Quaternion> q(x);
  Eigen::Map<math::Quaternion> result(x_plus_delta);

  result = math::quat_mul(dq, q);

  return true;
}

bool QuaternionLocalParameterization::ComputeJacobian(const double* x, double* jacobian) const
{
  Eigen::Map<const math::Quaternion> q(x);
  Eigen::Map<Jacobian> J(jacobian);

  // Jacobian of Plus(q, dtheta) at dtheta = 0
  // Plus(q, dtheta) = R(q)*dq, R = right quaternion matrix
  // dq = [dtheta/2, 1]
  // -> Plus(q, dtheta) = R(x)*[dtheta/2; 1]
  // = [Xi(x) x]*[dtheta/2 ; 1] = Xi(x)*dtheta/2 + x
  // -> dPlus/dtheta = 0.5*Xi(x)
  J = 0.5 * math::quat_Xi_mat(q);

  return true;
}

