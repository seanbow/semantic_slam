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

bool QuaternionLocalParameterization::Plus(const double* x_ptr, const double* delta, double* x_plus_delta_ptr) const
{  
    // Eigen::Map<Eigen::Quaterniond> x_plus_delta(x_plus_delta_ptr);
    // Eigen::Map<const Eigen::Quaterniond> x(x_ptr);

    // const double norm_delta =
    // sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
    // if (norm_delta > 0.0) {
    //     const double sin_delta_by_delta = sin(norm_delta) / norm_delta;

    //     // Note, in the constructor w is first.
    //     Eigen::Quaterniond delta_q(cos(norm_delta),
    //                                 sin_delta_by_delta * delta[0],
    //                                 sin_delta_by_delta * delta[1],
    //                                 sin_delta_by_delta * delta[2]);
    //     x_plus_delta = delta_q * x;
    // } else {
    //     x_plus_delta = x;
    // }

    Eigen::Quaterniond dq(1.0, 0.5 * delta[0], 0.5 * delta[1], 0.5 * delta[2]);
    dq.normalize();

    Eigen::Map<const Eigen::Quaterniond> q(x_ptr);
    Eigen::Map<Eigen::Quaterniond> x_plus_delta(x_plus_delta_ptr);

    x_plus_delta = (dq * q).normalized();
    
    return true;
}

bool QuaternionLocalParameterization::ComputeJacobian(const double* x, double* jacobian) const
{
  // Computed in Mathematica
  Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> J(jacobian);

  J <<  x[3],  x[2], -x[1],
       -x[2],  x[3],  x[0], 
        x[1], -x[0],  x[3],
       -x[0], -x[1], -x[2];

  J *= 0.5;

  return true;
}

