#pragma once

#include <Eigen/Core>
#include <boost/shared_ptr.hpp>
#include <ceres/autodiff_cost_function.h>

#include "semantic_slam/inertial/InertialIntegrator.h"

class InertialCostTerm
{
  public:
    InertialCostTerm(double t0,
                     double t1,
                     boost::shared_ptr<InertialIntegrator> integrator);

    template<typename T>
    bool operator()(const T* const map_x_body0_ptr,
                    const T* const map_v_body0_ptr,
                    const T* const bias0_ptr,
                    const T* const map_x_body1_ptr,
                    const T* const map_v_body1_ptr,
                    const T* const bias1_ptr,
                    T* residual_ptr) const;

    static ceres::CostFunction* Create(
      double t0,
      double t1,
      boost::shared_ptr<InertialIntegrator> integrator);

  private:
    double t0_;
    double t1_;
    boost::shared_ptr<InertialIntegrator> integrator_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

template<typename T>
bool
InertialCostTerm::operator()(const T* const map_x_body0_ptr,
                             const T* const map_v_body0_ptr,
                             const T* const bias0_ptr,
                             const T* const map_x_body1_ptr,
                             const T* const map_v_body1_ptr,
                             const T* const bias1_ptr,
                             T* residual_ptr) const
{
    Eigen::Map<const Eigen::Quaternion<T>> map_q_body0(map_x_body0_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> map_p_body0(map_x_body0_ptr + 4);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> map_v_body0(map_v_body0_ptr);
    Eigen::Map<const Eigen::Matrix<T, 6, 1>> bias0(bias0_ptr);

    Eigen::Map<const Eigen::Quaternion<T>> map_q_body1(map_x_body1_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> map_p_body1(map_x_body1_ptr + 4);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> map_v_body1(map_v_body1_ptr);
    Eigen::Map<const Eigen::Matrix<T, 6, 1>> bias1(bias1_ptr);

    Eigen::Map<Eigen::Matrix<T, 15, 1>> residual(residual_ptr);

    // Integrate the inertial measurements given the current state estimate
    Eigen::Matrix<T, -1, 1> qvp0(10);
    qvp0.template head<4>() = map_q_body0.coeffs() / map_q_body0.norm();
    qvp0.template segment<3>(4) = map_v_body0;
    qvp0.template tail<3>() = map_p_body0;

    auto xP_est =
      integrator_->integrateInertialWithCovariance(t0_, t1_, qvp0, bias0);
    auto map_qvp_x1_hat = xP_est.first;
    auto P01 = xP_est.second;

    // auto map_qvp_x1_hat = integrator_->integrateInertial(t0_, t1_, qvp0,
    // bias0);

    // Relative state from current estimate
    // Eigen::Quaternion<T> body0_q_map = map_q_body0.inverse();
    // Eigen::Quaternion<T> x0_q_x1 = body0_q_map * map_q_body1;
    // Eigen::Matrix<T, 3, 1> x0_v_x1 = body0_q_map * (map_v_body1 -
    // map_v_body0); Eigen::Matrix<T, 3, 1> x0_p_x1 = body0_q_map * (map_p_body1
    // - map_p_body0);

    Eigen::Quaternion<T> map_q_x1_hat(map_qvp_x1_hat.template head<4>());
    map_q_x1_hat.normalize();

    Eigen::Quaternion<T> deltaq = map_q_x1_hat.conjugate() * map_q_body1;

    // Eigen::Quaternion<T> deltaq = x0_q_x1_hat.conjugate() * x0_q_x1;

    // // Residual ordering is [q bw v bv p]

    residual.template segment<3>(0) = T(2.0) * deltaq.vec();
    residual.template segment<3>(3) =
      bias1.template head<3>() - bias0.template head<3>();
    residual.template segment<3>(6) =
      map_v_body1 - map_qvp_x1_hat.template segment<3>(4);
    residual.template segment<3>(9) =
      bias1.template tail<3>() - bias1.template tail<3>();
    residual.template segment<3>(12) =
      map_p_body1 - map_qvp_x1_hat.template tail<3>();

    // residual.template segment<3>(0) = T(2.0) * deltaq.vec();
    // residual.template segment<3>(3) =
    //   bias1.template head<3>() - bias0.template head<3>();
    // residual.template segment<3>(6) =
    //   x0_v_x1 - x0_qvp_x1_hat.template segment<3>(4);
    // residual.template segment<3>(9) =
    //   bias1.template tail<3>() - bias1.template tail<3>();
    // residual.template segment<3>(12) =
    //   x0_p_x1 - x0_qvp_x1_hat.template segment<3>(7);

    // std::cout << "Residual before P = \n" << residual.transpose() <<
    // std::endl;
    // std::cout << "P = \n" << P01 << std::endl;

    // // This is the same ordering that P is in so we can just apply it
    // // directly...
    auto sqrtP = P01.llt().matrixL();
    Eigen::MatrixXd sqrtPinv = Eigen::MatrixXd::Identity(15, 15);
    sqrtP.solveInPlace(sqrtPinv);
    residual.applyOnTheLeft(sqrtPinv);
    // sqrtP.solveInPlace(residual);

    // auto sqrtP =
    //   Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(P01).operatorSqrt();
    // residual = (sqrtP * residual).eval();

    // std::cout << "Residual after P = \n" << residual.transpose() <<
    // std::endl;

    return true;
}