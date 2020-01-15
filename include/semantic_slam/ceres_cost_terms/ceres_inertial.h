#pragma once

#include <Eigen/Core>
#include <boost/shared_ptr.hpp>
#include <ceres/sized_cost_function.h>

#include "semantic_slam/inertial/InertialIntegrator.h"

class InertialCostTerm : ceres::SizedCostFunction<15, 7, 3, 6, 7, 3, 6>
{
  public:
    InertialCostTerm(double t0,
                     double t1,
                     boost::shared_ptr<InertialIntegrator> integrator,
                     const Eigen::VectorXd& bias_estimate);

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const;

    void preintegrate(const Eigen::VectorXd& bias0);

    // template<typename T>
    // bool operator()(const T* const map_x_body0_ptr,
    //                 const T* const map_v_body0_ptr,
    //                 const T* const bias0_ptr,
    //                 const T* const map_x_body1_ptr,
    //                 const T* const map_v_body1_ptr,
    //                 const T* const bias1_ptr,
    //                 T* residual_ptr) const;

    static ceres::CostFunction* Create(
      double t0,
      double t1,
      boost::shared_ptr<InertialIntegrator> integrator,
      const Eigen::VectorXd& bias_estimate);

  private:
    double t0_;
    double t1_;
    boost::shared_ptr<InertialIntegrator> integrator_;

    mutable Eigen::VectorXd preint_x_;
    mutable Eigen::MatrixXd preint_Jbias_;
    mutable Eigen::MatrixXd preint_P_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

// template<typename T>
// bool
// InertialCostTerm::operator()(const T* const map_x_body0_ptr,
//                              const T* const map_v_body0_ptr,
//                              const T* const bias0_ptr,
//                              const T* const map_x_body1_ptr,
//                              const T* const map_v_body1_ptr,
//                              const T* const bias1_ptr,
//                              T* residual_ptr) const
// {
//     Eigen::Map<const Eigen::Quaternion<T>> map_q_body0(map_x_body0_ptr);
//     Eigen::Map<const Eigen::Matrix<T, 3, 1>> map_p_body0(map_x_body0_ptr +
//     4); Eigen::Map<const Eigen::Matrix<T, 3, 1>>
//     map_v_body0(map_v_body0_ptr); Eigen::Map<const Eigen::Matrix<T, 6, 1>>
//     bias0(bias0_ptr);

//     Eigen::Map<const Eigen::Quaternion<T>> map_q_body1(map_x_body1_ptr);
//     Eigen::Map<const Eigen::Matrix<T, 3, 1>> map_p_body1(map_x_body1_ptr +
//     4); Eigen::Map<const Eigen::Matrix<T, 3, 1>>
//     map_v_body1(map_v_body1_ptr); Eigen::Map<const Eigen::Matrix<T, 6, 1>>
//     bias1(bias1_ptr);

//     Eigen::Map<Eigen::Matrix<T, 15, 1>> residual(residual_ptr);

//     // Preintegration...
//     auto preint = integrator_->preintegrateInertial(t0_, t1_, bias0);

//     double dt = t1_ - t0_;
//     Eigen::Quaternion<T> body0_q_map = map_q_body0.inverse();
//     Eigen::Quaternion<T> body0_q_body1 = body0_q_map * map_q_body1;

//     Eigen::Matrix<T, -1, 1> vpre_hat =
//       body0_q_map.toRotationMatrix() *
//       (map_v_body1 - map_v_body0 - dt * integrator_->gravity());

//     Eigen::Matrix<T, -1, 1> ppre_hat =
//       body0_q_map.toRotationMatrix() *
//       (map_p_body1 - map_p_body0 - dt * map_v_body0 -
//        0.5 * dt * dt * integrator_->gravity());

//     Eigen::Quaternion<T> body0_q_body1_preint(preint.template head<4>());
//     Eigen::Quaternion<T> deltaq =
//       body0_q_body1_preint.template conjugate() * body0_q_body1;

//     residual.template segment<3>(0) = T(2.0) * deltaq.vec();
//     residual.template segment<3>(3) =
//       bias1.template head<3>() - bias0.template head<3>();
//     residual.template segment<3>(6) = vpre_hat - preint.template
//     segment<3>(4); residual.template segment<3>(9) =
//       bias1.template tail<3>() - bias1.template tail<3>();
//     residual.template segment<3>(12) = ppre_hat - preint.template tail<3>();

//     return true;
// }