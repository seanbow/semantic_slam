#pragma once

#include <ceres/sized_cost_function.h>
#include <cmath>

class DoublePriorCostTerm : ceres::SizedCostFunction<1, 1>
{
  public:
    DoublePriorCostTerm(double prior, double prior_covariance);

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const;

    static ceres::CostFunction* Create(double prior, double prior_covariance);

  private:
    double sqrt_information_;

    double prior_;
};

DoublePriorCostTerm::DoublePriorCostTerm(double prior, double prior_covariance)
  : prior_(prior)
{
    // Compute sqrt of information "matrix"
    sqrt_information_ = 1.0 / std::sqrt(prior_covariance);
}

bool
DoublePriorCostTerm::Evaluate(double const* const* parameters,
                              double* residual,
                              double** jacobians) const
{
    double const* value = parameters[0];

    residual[0] = sqrt_information_ * (prior_ - value[0]);

    if (jacobians && jacobians[0]) {
        jacobians[0][0] = -sqrt_information_;
    }

    return true;
}

ceres::CostFunction*
DoublePriorCostTerm::Create(double prior, double prior_covariance)
{
    return new DoublePriorCostTerm(prior, prior_covariance);
}
