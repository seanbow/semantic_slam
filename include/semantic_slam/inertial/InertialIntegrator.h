#pragma once

#include "semantic_slam/Common.h"

#include "semantic_slam/LocalParameterizations.h"
#include "semantic_slam/quaternion_math.h"

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

    Eigen::Vector3d gravity() const { return gravity_; }

    void setAdditiveMeasurementNoise(double gyro_sigma, double accel_sigma);
    void setBiasRandomWalkNoise(double gyro_sigma, double accel_sigma);

    Eigen::MatrixXd randomWalkCovariance() const { return Q_random_walk_; }

    void addData(double t,
                 const Eigen::Vector3d& accel,
                 const Eigen::Vector3d& omega);

    Eigen::VectorXd integrateInertial(double t1,
                                      double t2,
                                      const Eigen::VectorXd& qvp0,
                                      const Eigen::VectorXd& gyro_accel_bias);

    aligned_vector<Eigen::MatrixXd> integrateInertialWithCovariance(
      double t1,
      double t2,
      const Eigen::VectorXd& qvp0,
      const Eigen::VectorXd& gyro_accel_bias);

    Eigen::VectorXd preintegrateInertial(
      double t1,
      double t2,
      const Eigen::VectorXd& gyro_accel_bias);

    aligned_vector<Eigen::MatrixXd> preintegrateInertialWithJacobian(
      double t1,
      double t2,
      const Eigen::VectorXd& gyro_accel_bias);

    aligned_vector<Eigen::MatrixXd>
    preintegrateInertialWithJacobianAndCovariance(
      double t1,
      double t2,
      const Eigen::VectorXd& gyro_accel_bias);

    // Performs runge-kutta numerical integration of a function ydot = f(t, y)
    Eigen::VectorXd integrateRK4(
      const std::function<Eigen::VectorXd(double, const Eigen::VectorXd&)>& f,
      double t1,
      double t2,
      const Eigen::VectorXd& y0,
      double step_size = 0.01);

    Eigen::VectorXd rk4_iteration(
      const std::function<Eigen::VectorXd(double, const Eigen::VectorXd&)>& f,
      double t1,
      double t2,
      const Eigen::VectorXd& y0);

    // TODO this is slightly messy we're duplicating code
    // Performs runge-kutta integration of a function of N matrices
    // [x1dot, x2dot, ...] = f(t, x1, x2, ...)
    aligned_vector<Eigen::MatrixXd> integrateRK4(
      const std::function<aligned_vector<
        Eigen::MatrixXd>(double, const aligned_vector<Eigen::MatrixXd>&)>& f,
      double t1,
      double t2,
      const aligned_vector<Eigen::MatrixXd>& x0,
      double step_size = 0.01);

    aligned_vector<Eigen::MatrixXd> rk4_iteration(
      const std::function<aligned_vector<
        Eigen::MatrixXd>(double, const aligned_vector<Eigen::MatrixXd>&)>& f,
      double t1,
      double t2,
      const aligned_vector<Eigen::MatrixXd>& x0);

    Eigen::VectorXd statedot(double t,
                             const Eigen::VectorXd& state,
                             const Eigen::VectorXd& gyro_accel_bias);

    Eigen::VectorXd statedot_preint(double t,
                                    const Eigen::VectorXd& state,
                                    const Eigen::VectorXd& gyro_accel_bias);

    Eigen::MatrixXd Pdot(double t,
                         const Eigen::VectorXd& state,
                         const Eigen::MatrixXd& P,
                         const Eigen::VectorXd& gyro_accel_bias);

    Eigen::MatrixXd quaternionMatrixOmega(const Eigen::VectorXd& w);

    Eigen::MatrixXd Dqdot_dnoise(const Eigen::VectorXd& quat);

    Eigen::MatrixXd Dbias_sensitivity_dt(
      double t,
      const Eigen::VectorXd& state,
      const Eigen::MatrixXd& sensitivities,
      const Eigen::VectorXd& gyro_accel_bias);

    Eigen::MatrixXd Dstatedot_dbias(double t,
                                    const Eigen::VectorXd& state,
                                    const Eigen::VectorXd& gyro_accel_bias);

    Eigen::MatrixXd Dstatedot_dstate(double t,
                                     const Eigen::VectorXd& state,
                                     const Eigen::VectorXd& gyro_accel_bias);

    Eigen::MatrixXd Dstatedot_dnoise(double t,
                                     const Eigen::VectorXd& state,
                                     const Eigen::VectorXd& gyro_accel_bias);

    Eigen::Vector3d interpolateData(
      double t,
      const std::vector<double>& times,
      const aligned_vector<Eigen::Vector3d>& data);

    Eigen::Vector3d gravity_;

    std::vector<double> imu_times_;
    aligned_vector<Eigen::Vector3d> omegas_;
    aligned_vector<Eigen::Vector3d> accels_;

    Eigen::MatrixXd Q_;
    Eigen::MatrixXd Q_random_walk_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
