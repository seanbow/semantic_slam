#include <ceres/ceres.h>
#include <ceres/local_parameterization.h>
#include <eigen3/Eigen/Core>

class QuaternionLocalParameterization : public ceres::LocalParameterization
{
  public:
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
