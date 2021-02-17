#pragma once

#define CERES_BETWEEN_AUTODIFF 0

#include "semantic_slam/Pose3.h"

#include <ceres/sized_cost_function.h>
#include <eigen3/Eigen/Core>

class BetweenCostTerm
#if !(CERES_BETWEEN_AUTODIFF)
  : public ceres::SizedCostFunction<6, 7, 7>
#endif
{
  private:
    typedef BetweenCostTerm This;

  public:
    using shared_ptr = boost::shared_ptr<This>;
    using Ptr = shared_ptr;

    BetweenCostTerm(const Pose3& between, const Eigen::MatrixXd& cov);

#if CERES_BETWEEN_AUTODIFF
    template<typename T>
    bool operator()(const T* const qp1,
                    const T* const qp2,
                    T* residual_ptr) const;
#else
    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const;
#endif

    static ceres::CostFunction* Create(const Pose3& between,
                                       const Eigen::MatrixXd& cov);

  private:
    Eigen::Matrix<double, 6, 6> sqrt_information_;

    Eigen::Quaterniond dq_;
    Eigen::Vector3d dp_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
