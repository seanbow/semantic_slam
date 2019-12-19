#pragma once

// #include <gtsam/nonlinear/NonlinearFactor.h>
// #include <gtsam/nonlinear/Values.h>
// #include <gtsam/linear/JacobianFactor.h>
// #include <gtsam/geometry/Pose3.h>

#include <boost/make_shared.hpp>
#include <boost/optional.hpp>

#include "semantic_slam/Pose3.h"
#include "semantic_slam/Utils.h"
#include "semantic_slam/keypoints/geometry.h"
// #include "omnigraph/keypoints/quaternion_math.h"

#include "ceres/cost_function.h"
#include "ceres/dynamic_autodiff_cost_function.h"

using geometry::ObjectModelBasis;
/** AUTODIFF VERSION **/

class StructureProjectionCostTerm
{
  private:
    typedef StructureProjectionCostTerm This;

  public:
    typedef boost::shared_ptr<This> shared_ptr;
    typedef shared_ptr Ptr;

    StructureProjectionCostTerm(Eigen::MatrixXd normalized_measurements,
                                ObjectModelBasis model,
                                const Eigen::VectorXd& weights,
                                double lambda = 1.0);

    static ceres::CostFunction* Create(Eigen::MatrixXd normalized_measurements,
                                       ObjectModelBasis model,
                                       const Eigen::VectorXd& weights,
                                       double lambda = 1.0);

    void setWeights(const Eigen::VectorXd& weights) { weights_ = weights; }

    template<typename T>
    bool unwhitenedError(T const* const* parameters, T* residuals_ptr) const;

    // computes residuals
    template<typename T>
    bool operator()(T const* const* parameters, T* residuals_ptr) const;

    // number of residuals
    size_t dim() const { return 3 * m_ + k_; }

    size_t m() { return m_; }

    size_t k() { return k_; }

  private:
    Eigen::MatrixXd measurements_;
    ObjectModelBasis model_;

    double lambda_; // regularization factor

    size_t m_, k_;

    // gtsam::noiseModel::Base::shared_ptr noise_model_;

    Eigen::VectorXd weights_;

    template<typename T>
    Eigen::Matrix<T, 3, Eigen::Dynamic> structure(
      T const* const* parameters) const;

    template<typename T>
    bool whitenError(T* residuals) const;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
