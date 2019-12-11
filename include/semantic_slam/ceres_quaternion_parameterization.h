#include <ceres/ceres.h>
#include <ceres/local_parameterization.h>
#include <eigen3/Eigen/Core>
#include <semantic_slam/quaternion_math.h>

class QuaternionLocalParameterization : public ceres::LocalParameterization
{
  public:
    using Jacobian = Eigen::Matrix<double, 4, 3, Eigen::RowMajor>;
    using LiftJacobian = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;

    bool Plus(const double* x, const double* delta, double* x_plus_delta) const;
    bool ComputeJacobian(const double* x, double* jacobian) const;

    int GlobalSize() const { return 4; }
    int LocalSize() const { return 3; }
};

class SE3LocalParameterization : public ceres::LocalParameterization
{
  public:
    SE3LocalParameterization();
    ~SE3LocalParameterization();

    bool Plus(const double* x, const double* delta, double* x_plus_delta) const;
    bool ComputeJacobian(const double* x, double* jacobian) const;

    int GlobalSize() const { return 7; }
    int LocalSize() const { return 6; }

  private:
    ceres::LocalParameterization* quaternion_parameterization_;
};

SE3LocalParameterization::SE3LocalParameterization()
{
    quaternion_parameterization_ = new QuaternionLocalParameterization;
}

SE3LocalParameterization::~SE3LocalParameterization()
{
    delete quaternion_parameterization_;
}

bool
SE3LocalParameterization::Plus(const double* x,
                               const double* delta,
                               double* x_plus_delta) const
{
    // fill in the rotation parts
    quaternion_parameterization_->Plus(x, delta, x_plus_delta);

    // and the translation parts
    x_plus_delta[4] = x[4] + delta[3];
    x_plus_delta[5] = x[5] + delta[4];
    x_plus_delta[6] = x[6] + delta[5];

    return true;
}

bool
SE3LocalParameterization::ComputeJacobian(const double* x,
                                          double* jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> J(
      jacobian, 7, 6);
    J.setZero();

    // Dq_dq...
    Eigen::Matrix<double, 4, 3, Eigen::RowMajor> Dq_dq;
    quaternion_parameterization_->ComputeJacobian(x, Dq_dq.data());
    J.block<4, 3>(0, 0) = Dq_dq;

    // Dq_dp and Dp_dq are zero

    // Dp_dp
    J.block<3, 3>(4, 3) = Eigen::Matrix3d::Identity();

    return true;
}

bool
QuaternionLocalParameterization::Plus(const double* x_ptr,
                                      const double* delta,
                                      double* x_plus_delta_ptr) const
{
    Eigen::Quaterniond dq(1.0, 0.5 * delta[0], 0.5 * delta[1], 0.5 * delta[2]);
    dq.normalize();

    Eigen::Map<const Eigen::Quaterniond> q(x_ptr);
    Eigen::Map<Eigen::Quaterniond> x_plus_delta(x_plus_delta_ptr);

    x_plus_delta = (dq * q).normalized();

    return true;
}

bool
QuaternionLocalParameterization::ComputeJacobian(const double* x,
                                                 double* jacobian) const
{
    // Computed in Mathematica
    Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> J(jacobian);

    J << x[3], x[2], -x[1], -x[2], x[3], x[0], x[1], -x[0], x[3], -x[0], -x[1],
      -x[2];

    J *= 0.5;

    return true;
}
