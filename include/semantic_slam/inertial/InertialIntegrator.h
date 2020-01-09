#pragma once

#include "semantic_slam/Common.h"

#include <Eigen/Core>
#include <functional>
#include <iostream>
#include <vector>

class InertialIntegrator
{
  public:
    InertialIntegrator();
    InertialIntegrator(const Eigen::Vector3d& gravity);

    void addData(double t,
                 const Eigen::Vector3d& accel,
                 const Eigen::Vector3d& omega);

    template<typename T>
    Eigen::Matrix<T, -1, 1> integrateInertial(
      double t1,
      double t2,
      const Eigen::Matrix<T, -1, 1>& qvp0,
      const Eigen::Matrix<T, 3, 1>& gyro_bias,
      const Eigen::Matrix<T, 3, 1>& accel_bias);

    // Performs runge-kutta numerical integration of a function ydot = f(t, y)
    template<typename T>
    T integrateRK4(const std::function<T(double, const T&)>& f,
                   double t1,
                   double t2,
                   const T& y0,
                   double step_size = 0.1);

    template<typename T>
    T rk4_iteration(const std::function<T(double, const T&)>& f,
                    double t1,
                    double t2,
                    const T& y0);

    template<typename T>
    Eigen::Matrix<T, -1, 1> statedot(double t,
                                     const Eigen::Matrix<T, -1, 1>& state,
                                     const Eigen::Matrix<T, 3, 1>& gyro_bias,
                                     const Eigen::Matrix<T, 3, 1>& accel_bias);

  private:
    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, -1, -1> quaternionMatrixPhi(
      const Eigen::MatrixBase<Derived>& q);

    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, -1, -1> quaternionMatrixXi(
      const Eigen::MatrixBase<Derived>& q);

    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, -1, -1> quaternionMatrixOmega(
      const Eigen::MatrixBase<Derived>& w);

    Eigen::Vector3d interpolateData(
      double t,
      const std::vector<double>& times,
      const aligned_vector<Eigen::Vector3d>& data);

    Eigen::Vector3d gravity_;

    Eigen::Vector3d gyro_bias_;
    Eigen::Vector3d accel_bias_;

    std::vector<double> imu_times_;
    aligned_vector<Eigen::Vector3d> omegas_;
    aligned_vector<Eigen::Vector3d> accels_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template<typename T>
T
InertialIntegrator::integrateRK4(const std::function<T(double, const T&)>& f,
                                 double t1,
                                 double t2,
                                 const T& y0,
                                 double step_size)
{
    double t = t1;
    T y = y0;

    while (t + step_size <= t2) {
        y = rk4_iteration(f, t, t + step_size, y);
        t += step_size;
    }

    if (t < t2) {
        y = rk4_iteration(f, t, t2, y);
    }

    return y;
}

template<typename T>
T
InertialIntegrator::rk4_iteration(const std::function<T(double, const T&)>& f,
                                  double t1,
                                  double t2,
                                  const T& y0)
{
    double h = t2 - t1;
    T k1 = h * f(t1, y0);
    T k2 = h * f(t1 + h / 2, y0 + k1 / 2);
    T k3 = h * f(t1 + h / 2, y0 + k2 / 2);
    T k4 = h * f(t1 + h, y0 + k3);
    return y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
}

template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, -1, -1>
InertialIntegrator::quaternionMatrixOmega(const Eigen::MatrixBase<Derived>& w)
{
    Eigen::Matrix<typename Derived::Scalar, -1, -1> Omega(4, 4);

    // clang-format off
    Omega <<  0,   w(2),  -w(1), w(0),
            -w(2),  0,     w(0), w(1),
             w(1), -w(0),   0,   w(2),
            -w(0), -w(1), -w(2),  0  ;
    // clang-format on

    return Omega;
}

template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, -1, -1>
InertialIntegrator::quaternionMatrixPhi(const Eigen::MatrixBase<Derived>& q)
{
    Eigen::Matrix<typename Derived::Scalar, -1, -1> Phi(4, 3);

    // clang-format off
    Phi << q(3),  q(2), -q(1), 
          -q(2),  q(3),  q(0), 
           q(1), -q(0),  q(3), 
          -q(0), -q(1), -q(2);
    // clang-format on

    return Phi;
}

template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, -1, -1>
InertialIntegrator::quaternionMatrixXi(const Eigen::MatrixBase<Derived>& q)
{
    Eigen::Matrix<typename Derived::Scalar, -1, -1> Xi(4, 3);

    // clang-format off
    Xi << q(3), -q(2),  q(1), 
          q(2),  q(3), -q(0), 
         -q(1),  q(0),  q(3), 
         -q(0), -q(1), -q(2);
    // clang-format on

    return Xi;
}

template<typename T>
Eigen::Matrix<T, -1, 1>
InertialIntegrator::statedot(double t,
                             const Eigen::Matrix<T, -1, 1>& state,
                             const Eigen::Matrix<T, 3, 1>& gyro_bias,
                             const Eigen::Matrix<T, 3, 1>& accel_bias)
{
    Eigen::Matrix<T, -1, 1> xdot(10, 1);

    Eigen::Matrix<T, 4, 1> quat = state.template head<4>();

    // Quaternion derivative
    // xdot.template head<4>() =
    //   0.5 * quaternionMatrixXi(quat) * interpolateData(t, imu_times_,
    //   omegas_);

    xdot.template head<4>() =
      0.5 *
      quaternionMatrixOmega(interpolateData(t, imu_times_, omegas_) -
                            gyro_bias) *
      quat;

    quat.normalize();
    if (quat(3) < 0)
        quat = -quat;

    // Velocity derivative
    Eigen::Quaternion<T> q(
      quat(3), quat(0), quat(1), quat(2)); // w, x, y, z order

    xdot.template segment<3>(4) =
      q.toRotationMatrix() *
        (interpolateData(t, imu_times_, accels_) - accel_bias) +
      gravity_;

    // Position derivative
    xdot.template tail<3>() = state.template segment<3>(4);

    return xdot;
}

template<typename T>
Eigen::Matrix<T, -1, 1>
InertialIntegrator::integrateInertial(double t1,
                                      double t2,
                                      const Eigen::Matrix<T, -1, 1>& qvp0,
                                      const Eigen::Matrix<T, 3, 1>& gyro_bias,
                                      const Eigen::Matrix<T, 3, 1>& accel_bias)
{
    std::function<Eigen::Matrix<T, -1, 1>(double,
                                          const Eigen::Matrix<T, -1, 1>&)>
      f = [&](double t, const Eigen::Matrix<T, -1, 1>& x) {
          return this->statedot(t, x, gyro_bias, accel_bias);
      };

    return this->integrateRK4(f, t1, t2, qvp0, 0.01);
}
