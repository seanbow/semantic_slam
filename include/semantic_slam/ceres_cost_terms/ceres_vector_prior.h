#pragma once

#include <ceres/dynamic_autodiff_cost_function.h>
#include <eigen3/Eigen/Core>

template<typename Vector>
class VectorPriorCostTerm
{
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(Vector);

  public:
    VectorPriorCostTerm(const Vector& prior_vec,
                        const Eigen::MatrixXd& prior_covariance);

    template<typename T>
    bool operator()(T const* const* parameters, T* residual_ptr) const;

    static ceres::CostFunction* Create(const Vector& prior_vec,
                                       const Eigen::MatrixXd& prior_covariance);

  private:
    Eigen::MatrixXd sqrt_information_;

    Vector prior_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

template<typename Vector>
VectorPriorCostTerm<Vector>::VectorPriorCostTerm(
  const Vector& prior,
  const Eigen::MatrixXd& prior_covariance)
  : prior_(prior)
{
    // Compute sqrt of information matrix
    Eigen::MatrixXd sqrtC = prior_covariance.llt().matrixL();
    sqrt_information_.setIdentity(prior.size(), prior.size());
    sqrtC.triangularView<Eigen::Lower>().solveInPlace(sqrt_information_);
}

template<typename Vector>
template<typename T>
bool
VectorPriorCostTerm<Vector>::operator()(T const* const* parameters,
                                        T* residual_ptr) const
{
    Eigen::Map<const Eigen::Matrix<T, Vector::RowsAtCompileTime, 1>> x(
      parameters[0]);

    Eigen::Map<Eigen::Matrix<T, Vector::RowsAtCompileTime, 1>> residual(
      residual_ptr);

    residual = x - prior_;

    residual.applyOnTheLeft(sqrt_information_);

    return true;
}

template<typename Vector>
ceres::CostFunction*
VectorPriorCostTerm<Vector>::Create(const Vector& prior,
                                    const Eigen::MatrixXd& prior_covariance)
{
    auto cf =
      new ceres::DynamicAutoDiffCostFunction<VectorPriorCostTerm<Vector>>(
        new VectorPriorCostTerm<Vector>(prior, prior_covariance));

    cf->AddParameterBlock(prior.size());
    cf->SetNumResiduals(prior.size());

    return cf;
}
