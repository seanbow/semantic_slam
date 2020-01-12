#pragma once

#include "semantic_slam/Common.h"

#include <Eigen/Core>
#include <ceres/jet.h>
#include <functional>
#include <iostream>
#include <type_traits>
#include <vector>

class InertialIntegrator
{
  public:
    InertialIntegrator();
    InertialIntegrator(const Eigen::Vector3d& gravity);

    void setAdditiveMeasurementNoise(double gyro_sigma, double accel_sigma);
    void setBiasRandomWalkNoise(double gyro_sigma, double accel_sigma);

    void addData(double t,
                 const Eigen::Vector3d& accel,
                 const Eigen::Vector3d& omega);

    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, -1, 1> integrateInertial(
      double t1,
      double t2,
      const Eigen::Matrix<typename Derived::Scalar, -1, 1>& qvp0,
      const Eigen::MatrixBase<Derived>& gyro_accel_bias);

    template<typename Derived>
    std::pair<Eigen::Matrix<typename Derived::Scalar, -1, 1>, Eigen::MatrixXd>
    integrateInertialWithCovariance(
      double t1,
      double t2,
      const Eigen::Matrix<typename Derived::Scalar, -1, 1>& qvp0,
      const Eigen::MatrixBase<Derived>& gyro_accel_bias);

    // Performs runge-kutta numerical integration of a function ydot = f(t, y)
    // Rely on x being something like Eigen::Matrix<Scalar, -1, 1>
    template<typename Scalar>
    Eigen::Matrix<Scalar, -1, 1> integrateRK4(
      const std::function<
        Eigen::Matrix<Scalar, -1, 1>(double,
                                     const Eigen::Matrix<Scalar, -1, 1>&)>& f,
      double t1,
      double t2,
      const Eigen::Matrix<Scalar, -1, 1>& y0,
      double step_size = 0.1);

    template<typename Scalar>
    Eigen::Matrix<Scalar, -1, 1> rk4_iteration(
      const std::function<
        Eigen::Matrix<Scalar, -1, 1>(double,
                                     const Eigen::Matrix<Scalar, -1, 1>&)>& f,
      double t1,
      double t2,
      const Eigen::Matrix<Scalar, -1, 1>& y0);

    // TODO this is messy we're duplicating code
    // Performs runge-kutta integration of a function [xdot, ydot] = f(t, x, y)
    template<typename Scalar>
    std::pair<Eigen::Matrix<Scalar, -1, 1>, Eigen::MatrixXd> integrateRK4(
      const std::function<
        std::pair<Eigen::Matrix<Scalar, -1, 1>, Eigen::MatrixXd>(
          double,
          const Eigen::Matrix<Scalar, -1, 1>&,
          const Eigen::MatrixXd&)>& f,
      double t1,
      double t2,
      const Eigen::Matrix<Scalar, -1, 1>& x0,
      const Eigen::MatrixXd& y0,
      double step_size = 0.1);

    template<typename Scalar>
    std::pair<Eigen::Matrix<Scalar, -1, 1>, Eigen::MatrixXd> rk4_iteration(
      const std::function<
        std::pair<Eigen::Matrix<Scalar, -1, 1>, Eigen::MatrixXd>(
          double,
          const Eigen::Matrix<Scalar, -1, 1>&,
          const Eigen::MatrixXd&)>& f,
      double t1,
      double t2,
      const Eigen::Matrix<Scalar, -1, 1>& x0,
      const Eigen::MatrixXd& y0);

    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, -1, 1> statedot(
      double t,
      const Eigen::Matrix<typename Derived::Scalar, -1, 1>& state,
      const Eigen::MatrixBase<Derived>& gyro_accel_bias);

    template<typename Derived>
    Eigen::MatrixXd Pdot(
      double t,
      const Eigen::Matrix<typename Derived::Scalar, -1, 1>& state,
      const Eigen::MatrixXd& P,
      const Eigen::MatrixBase<Derived>& gyro_accel_bias);

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

    std::vector<double> imu_times_;
    aligned_vector<Eigen::Vector3d> omegas_;
    aligned_vector<Eigen::Vector3d> accels_;

    Eigen::MatrixXd Q_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template<typename Scalar>
Eigen::Matrix<Scalar, -1, 1>
InertialIntegrator::integrateRK4(
  const std::function<
    Eigen::Matrix<Scalar, -1, 1>(double, const Eigen::Matrix<Scalar, -1, 1>&)>&
    f,
  double t1,
  double t2,
  const Eigen::Matrix<Scalar, -1, 1>& y0,
  double step_size)
{
    double t = t1;
    Eigen::Matrix<Scalar, -1, 1> y = y0;

    while (t + step_size <= t2) {
        y = rk4_iteration(f, t, t + step_size, y);
        t += step_size;
    }

    if (t < t2) {
        y = rk4_iteration(f, t, t2, y);
    }

    return y;
}

template<typename Scalar>
Eigen::Matrix<Scalar, -1, 1>
InertialIntegrator::rk4_iteration(
  const std::function<
    Eigen::Matrix<Scalar, -1, 1>(double, const Eigen::Matrix<Scalar, -1, 1>&)>&
    f,
  double t1,
  double t2,
  const Eigen::Matrix<Scalar, -1, 1>& y0)
{
    double h = t2 - t1;
    Eigen::Matrix<Scalar, -1, 1> k1 = h * f(t1, y0);
    Eigen::Matrix<Scalar, -1, 1> k2 =
      h * f(t1 + h / 2.0, y0 + k1 / Scalar(2.0));
    Eigen::Matrix<Scalar, -1, 1> k3 =
      h * f(t1 + h / 2.0, y0 + k2 / Scalar(2.0));
    Eigen::Matrix<Scalar, -1, 1> k4 = h * f(t1 + h, y0 + k3);
    return y0 + (k1 + Scalar(2.0) * k2 + Scalar(2.0) * k3 + k4) / Scalar(6.0);
}

template<typename Scalar>
std::pair<Eigen::Matrix<Scalar, -1, 1>, Eigen::MatrixXd>
InertialIntegrator::integrateRK4(
  const std::function<std::pair<Eigen::Matrix<Scalar, -1, 1>, Eigen::MatrixXd>(
    double,
    const Eigen::Matrix<Scalar, -1, 1>&,
    const Eigen::MatrixXd&)>& f,
  double t1,
  double t2,
  const Eigen::Matrix<Scalar, -1, 1>& x0,
  const Eigen::MatrixXd& y0,
  double step_size)
{
    double t = t1;
    Eigen::Matrix<Scalar, -1, 1> x = x0;
    Eigen::MatrixXd y = y0;

    while (t + step_size <= t2) {
        auto xy = rk4_iteration(f, t, t + step_size, x, y);
        x = xy.first;
        y = xy.second;
        t += step_size;
    }

    if (t < t2) {
        auto xy = rk4_iteration(f, t, t2, x, y);
        x = xy.first;
        y = xy.second;
    }

    return std::make_pair(x, y);
}

// Rely on x being something like Eigen::Matrix<Scalar, -1, 1>
template<typename Scalar>
std::pair<Eigen::Matrix<Scalar, -1, 1>, Eigen::MatrixXd>
InertialIntegrator::rk4_iteration(
  const std::function<std::pair<Eigen::Matrix<Scalar, -1, 1>, Eigen::MatrixXd>(
    double,
    const Eigen::Matrix<Scalar, -1, 1>&,
    const Eigen::MatrixXd&)>& f,
  double t1,
  double t2,
  const Eigen::Matrix<Scalar, -1, 1>& x0,
  const Eigen::MatrixXd& y0)
{
    double h = t2 - t1;
    auto k1_h = f(t1, x0, y0);
    auto k2_h = f(
      t1 + h / 2, x0 + h * k1_h.first / Scalar(2.0), y0 + h * k1_h.second / 2);
    auto k3_h = f(
      t1 + h / 2, x0 + h * k2_h.first / Scalar(2.0), y0 + h * k2_h.second / 2);
    auto k4_h = f(t1 + h, x0 + h * k3_h.first, y0 + h * k3_h.second);

    return std::make_pair(x0 + (h * k1_h.first + Scalar(2.0) * h * k2_h.first +
                                Scalar(2.0) * h * k3_h.first + h * k4_h.first) /
                                 Scalar(6.0),
                          y0 + (h * k1_h.second + 2.0 * h * k2_h.second +
                                2.0 * h * k3_h.second + h * k4_h.second) /
                                 6.0);
}

template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, -1, -1>
InertialIntegrator::quaternionMatrixOmega(const Eigen::MatrixBase<Derived>& w)
{
    using T = typename Derived::Scalar;

    Eigen::Matrix<T, -1, -1> Omega(4, 4);

    // clang-format off
    Omega << T(0.0),  w(2),  -w(1),   w(0),
             -w(2),  T(0.0),  w(0),   w(1),
              w(1),  -w(0),  T(0.0),  w(2),
             -w(0),  -w(1),  -w(2),  T(0.0);
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

template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, -1, 1>
InertialIntegrator::statedot(
  double t,
  const Eigen::Matrix<typename Derived::Scalar, -1, 1>& state,
  const Eigen::MatrixBase<Derived>& gyro_accel_bias)
{
    using T = typename Derived::Scalar;

    Eigen::Matrix<T, -1, 1> xdot(10, 1);

    Eigen::Matrix<T, 4, 1> quat = state.template head<4>();

    // Quaternion derivative
    // xdot.template head<4>() =
    //   0.5 * quaternionMatrixXi(quat) * interpolateData(t, imu_times_,
    //   omegas_);

    xdot.template head<4>() =
      0.5 *
      quaternionMatrixOmega(interpolateData(t, imu_times_, omegas_) -
                            gyro_accel_bias.template head<3>()) *
      quat;

    quat.normalize();
    if (quat(3) < 0.0)
        quat = -quat;

    // Velocity derivative
    Eigen::Quaternion<T> q(
      quat(3), quat(0), quat(1), quat(2)); // w, x, y, z order

    xdot.template segment<3>(4) =
      q.toRotationMatrix() * (interpolateData(t, imu_times_, accels_) -
                              gyro_accel_bias.template tail<3>()) +
      gravity_;

    // Position derivative
    xdot.template tail<3>() = state.template segment<3>(4);

    return xdot;
}

template<typename Derived>
typename std::enable_if<std::is_arithmetic<typename Derived::Scalar>::value,
                        Eigen::MatrixXd>::type
removeJet(const Eigen::MatrixBase<Derived>& x)
{
    return x;
}

template<typename Derived>
typename std::enable_if<std::is_class<typename Derived::Scalar>::value,
                        Eigen::MatrixXd>::type
removeJet(const Eigen::MatrixBase<Derived>& x)
{
    // extract just the scalar portion from the Jet object
    Eigen::MatrixXd double_x(x.rows(), x.cols());

    for (int i = 0; i < x.rows(); ++i) {
        for (int j = 0; j < x.cols(); ++j) {
            double_x(i, j) = x(i, j).a;
        }
    }

    return double_x;
}

template<typename Derived>
Eigen::MatrixXd
InertialIntegrator::Pdot(
  double t,
  const Eigen::Matrix<typename Derived::Scalar, -1, 1>& state,
  const Eigen::MatrixXd& P,
  const Eigen::MatrixBase<Derived>& gyro_accel_bias)
{
    // Use continuous time kalman filter equation:
    //   Pdot = F*P + P*F' + Q
    // First compute error state transition matrix F
    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(15, 15);

    // Need the rotation
    Eigen::VectorXd quat_double = removeJet(state.template head<4>());

    Eigen::Quaterniond quat(
      quat_double(3), quat_double(0), quat_double(1), quat_double(2));
    quat.normalize();

    Eigen::VectorXd biases = removeJet(gyro_accel_bias);

    F.block<3, 3>(0, 0) =
      -skewsymm(interpolateData(t, imu_times_, omegas_) - biases.head<3>());

    F.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

    F.block<3, 3>(6, 0) =
      -quat.toRotationMatrix() *
      skewsymm(interpolateData(t, imu_times_, accels_) - biases.tail<3>());

    F.block<3, 3>(6, 9) = -quat.toRotationMatrix();

    F.block<3, 3>(12, 6) = Eigen::Matrix3d::Identity();

    Eigen::MatrixXd integration_covariance = Eigen::MatrixXd::Zero(15, 15);
    integration_covariance.block<3, 3>(12, 12) =
      1e-6 * Eigen::Matrix3d::Identity();
    // Eigen::MatrixXd integration_covariance =
    //   1e-4 * Eigen::MatrixXd::Identity(15, 15);

    // Noise matrix
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(15, 12);
    G.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
    G.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
    G.block<3, 3>(6, 6) = -quat.toRotationMatrix();
    G.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();

    return F * P + P * F.transpose() + G * Q_ * G.transpose() +
           integration_covariance;
}

template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, -1, 1>
InertialIntegrator::integrateInertial(
  double t1,
  double t2,
  const Eigen::Matrix<typename Derived::Scalar, -1, 1>& qvp0,
  const Eigen::MatrixBase<Derived>& gyro_accel_bias)
{
    using Scalar = typename Derived::Scalar;
    using StateType = Eigen::Matrix<Scalar, -1, 1>;

    std::function<StateType(double, const StateType&)> f =
      [&](double t, const StateType& x) {
          return this->statedot(t, x, gyro_accel_bias);
      };

    return this->integrateRK4(f, t1, t2, qvp0, 0.01);
}

template<typename Derived>
std::pair<Eigen::Matrix<typename Derived::Scalar, -1, 1>, Eigen::MatrixXd>
InertialIntegrator::integrateInertialWithCovariance(
  double t1,
  double t2,
  const Eigen::Matrix<typename Derived::Scalar, -1, 1>& qvp0,
  const Eigen::MatrixBase<Derived>& gyro_accel_bias)
{
    using Scalar = typename Derived::Scalar;
    using StateType = Eigen::Matrix<Scalar, -1, 1>;

    std::function<std::pair<StateType, Eigen::MatrixXd>(
      double, const StateType&, const Eigen::MatrixXd&)>
      f = [&](double t, const StateType& x, const Eigen::MatrixXd& P) {
          return std::make_pair(this->statedot(t, x, gyro_accel_bias),
                                this->Pdot(t, x, P, gyro_accel_bias));
      };

    Eigen::MatrixXd P0 = Eigen::MatrixXd::Zero(15, 15);

    return this->integrateRK4(f, t1, t2, qvp0, P0, 0.01);
}