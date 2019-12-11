#pragma once

// #include <gtsam/nonlinear/NonlinearFactor.h>
// #include <gtsam/nonlinear/Values.h>
// #include <gtsam/linear/JacobianFactor.h>
// #include <gtsam/geometry/Pose3.h>

#include <boost/make_shared.hpp>
#include <boost/optional.hpp>

#include "semantic_slam/Utils.h"
#include "semantic_slam/keypoints/geometry.h"
#include "semantic_slam/pose_math.h"
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

ceres::CostFunction*
StructureProjectionCostTerm::Create(Eigen::MatrixXd normalized_measurements,
                                    ObjectModelBasis model,
                                    const Eigen::VectorXd& weights,
                                    double lambda)
{
    StructureProjectionCostTerm* cost_term = new StructureProjectionCostTerm(
      normalized_measurements, model, weights, lambda);

    auto cost_function =
      new ceres::DynamicAutoDiffCostFunction<StructureProjectionCostTerm, 4>(
        cost_term);

    cost_function->AddParameterBlock(7); // object pose
    // cost_function->AddParameterBlock(cost_term->m()); // landmark Z values
    for (int i = 0; i < cost_term->m(); ++i) {
        cost_function->AddParameterBlock(1); // landmark Z
    }
    if (cost_term->k() > 0) {
        cost_function->AddParameterBlock(cost_term->k()); // basis coefficients
    }

    cost_function->SetNumResiduals(cost_term->dim());

    return cost_function;
}

StructureProjectionCostTerm::StructureProjectionCostTerm(
  Eigen::MatrixXd normalized_measurements,
  ObjectModelBasis model,
  const Eigen::VectorXd& weights,
  double lambda)
  : measurements_(normalized_measurements)
  , model_(model)
  , lambda_(lambda)
{
    m_ = model_.mu.cols();
    k_ = model_.pc.rows() / 3;

    setWeights(weights);
}

template<typename T>
Eigen::Matrix<T, 3, Eigen::Dynamic>
StructureProjectionCostTerm::structure(T const* const* parameters) const
{
    Eigen::Matrix<T, 3, Eigen::Dynamic> S =
      Eigen::Matrix<T, 3, Eigen::Dynamic>::Zero(3, m_);

    S += model_.mu;

    if (k_ == 0)
        return S;

    Eigen::Map<const Eigen::Matrix<T, -1, 1>> c(parameters[1 + m_], k_);

    for (size_t i = 0; i < k_; ++i) {
        S += c[i] * model_.pc.block(3 * i, 0, 3, m_);
    }

    return S;
}

// #include <iostream>
// using std::cout; using std::endl;

// Compute residual vector [P - R*S - t] and jacobians
template<typename T>
bool
StructureProjectionCostTerm::unwhitenedError(T const* const* parameters,
                                             T* residuals_ptr) const
{
    Eigen::Map<const Eigen::Quaternion<T>> q(parameters[0]);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(parameters[0] + 4);

    Eigen::Map<Eigen::Matrix<T, -1, 1>> residuals(residuals_ptr, dim());

    Eigen::Matrix<T, 3, Eigen::Dynamic> S = structure(parameters);

    for (size_t i = 0; i < m_; ++i) {
        T const* Z = parameters[1 + i];
        Eigen::Matrix<T, 3, 1> p = measurements_.col(i) * Z[0];

        residuals.template segment<3>(3 * i) = p - q * S.col(i) - t;
        // cout << "Index " << i << ", key " <<
        // gtsam::DefaultKeyFormatter(landmark_keys_[i]) <<
        //     ", p = " << p.transpose() << ", err = " << (p - R*S.col(i)
        //     -t).transpose() << endl;
    }

    if (k_ > 0) {
        Eigen::Map<const Eigen::Matrix<T, -1, 1>> c(parameters[1 + m_], k_);
        residuals.template tail(k_) = c;
    }

    return true;
}

template<typename T>
bool
StructureProjectionCostTerm::whitenError(T* residuals_ptr) const
{
    Eigen::Map<Eigen::Matrix<T, -1, 1>> residuals(residuals_ptr, dim());

    for (size_t i = 0; i < m_; ++i) {
        residuals.template segment<3>(3 * i) =
          residuals.template segment<3>(3 * i) * std::sqrt(weights_(i));
    }

    if (k_ > 0) {
        residuals.template tail(k_) =
          residuals.template tail(k_) * std::sqrt(lambda_);
    }

    return true;
}

template<typename T>
bool
StructureProjectionCostTerm::operator()(T const* const* parameters,
                                        T* residuals) const
{
    // parameters p:
    // p[0] = object orientation (quaternion)
    // p[1] = object position
    // p[2] through p[(2 + m_) - 1] = keypoint positions
    // p[2 + m_] = structure coefficients
    unwhitenedError(parameters, residuals);
    whitenError(residuals);
    return true;
}

/** ANALYTIC DERIV VERSION **/

/*

class StructureProjectionCostTerm : public ceres::CostFunction
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

  void setWeights(const Eigen::VectorXd& weights);

  bool unwhitenedError(double const* const* parameters,
                       double* residuals_ptr) const;

  // computes residuals & jacobians
  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const;
  //   template <typename T>
  //   bool operator()(T const* const* parameters, T* residuals_ptr) const;

  size_t dim() const { return 3 * m_ + k_; }

  const std::vector<int32_t>& parameter_block_sizes();

  size_t m() { return m_; }

  size_t k() { return k_; }

private:
  Eigen::MatrixXd measurements_;
  ObjectModelBasis model_;

  double lambda_; // regularization factor
  double sqrt_lambda_;

  size_t m_, k_;

  Eigen::VectorXd sqrt_weights_;

  std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> Bi_;

  Eigen::Matrix<double, 3, Eigen::Dynamic> structure(
    double const* const* parameters) const;

  bool whitenError(double* residuals) const;

  bool computeObjectOrientationJacobian(double const* const* parameters,
                                        double* jacobian) const;
  bool computeObjectTranslationJacobian(double const* const* parameters,
                                        double* jacobian) const;

  bool computeLandmarkJacobian(double const* const* parameters, size_t index,
                               double* jacobian) const;

  bool computeBasisJacobian(double const* const* parameters,
                            double* jacobian) const;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

void
StructureProjectionCostTerm::setWeights(const Eigen::VectorXd& weights)
{
  sqrt_weights_ = weights.array().sqrt();
}

ceres::CostFunction*
StructureProjectionCostTerm::Create(Eigen::MatrixXd normalized_measurements,
                                    ObjectModelBasis model,
                                    const Eigen::VectorXd& weights,
                                    double lambda)
{
  return new StructureProjectionCostTerm(normalized_measurements, model,
                                         weights, lambda);
}

StructureProjectionCostTerm::StructureProjectionCostTerm(
  Eigen::MatrixXd normalized_measurements, ObjectModelBasis model,
  const Eigen::VectorXd& weights, double lambda)
  : measurements_(normalized_measurements)
  , model_(model)
  , lambda_(lambda)
{
  m_ = model_.mu.cols();
  k_ = model_.pc.rows() / 3;

  sqrt_lambda_ = std::sqrt(lambda);

  setWeights(weights);

  // Preconstruct basis vector matrices
  // Bi = [b0 b1 ... bk] for ith keypoint
  if (k_ > 0) {
    for (size_t i = 0; i < m_; ++i) {
      Bi_.emplace_back(3, k_);
      for (size_t j = 0; j < k_; ++j) {
        Bi_.back().block<3, 1>(0, j) = model_.pc.block<3, 1>(3 * j, i);
      }
    }
  }

  std::vector<int32_t>* block_sizes = mutable_parameter_block_sizes();

  block_sizes->push_back(4); // object orientation
  block_sizes->push_back(3); // object position
  for (int i = 0; i < m_; ++i) {
    block_sizes->push_back(1); // landmark Z
  }
  if (k_ > 0) {
    block_sizes->push_back(k_); // basis coefficients
  }

  set_num_residuals(dim());
}

Eigen::Matrix<double, 3, Eigen::Dynamic>
StructureProjectionCostTerm::structure(double const* const* parameters) const
{
  Eigen::Matrix<double, 3, Eigen::Dynamic> S =
    Eigen::Matrix<double, 3, Eigen::Dynamic>::Zero(3, m_);

  S += model_.mu;

  Eigen::Map<const Eigen::VectorXd> c(parameters[2 + m_], k_);

  for (size_t i = 0; i < k_; ++i) {
    S += c[i] * model_.pc.block(3 * i, 0, 3, m_);
  }

  return S;
}

// #include <iostream>
// using std::cout; using std::endl;

// Compute residual vector [P - R*S - t] and jacobians
bool
StructureProjectionCostTerm::unwhitenedError(double const* const* parameters,
                                             double* residuals_ptr) const
{
  Eigen::Map<const Eigen::Quaterniond> q(parameters[0]);
  Eigen::Map<const Eigen::Vector3d> t(parameters[1]);
  Eigen::Map<const Eigen::VectorXd> c(parameters[2 + m_], k_);

  Eigen::Map<Eigen::VectorXd> residuals(residuals_ptr, num_residuals());

  Eigen::Matrix<double, 3, Eigen::Dynamic> S = structure(parameters);

  for (size_t i = 0; i < m_; ++i) {
    double const* Z = parameters[2 + i];
    Eigen::Vector3d p = measurements_.col(i) * Z[0];

    residuals.segment<3>(3 * i) = p - q * S.col(i) - t;
    // cout << "Index " << i << ", key " <<
    // gtsam::DefaultKeyFormatter(landmark_keys_[i]) <<
    //     ", p = " << p.transpose() << ", err = " << (p - R*S.col(i) -
    //     t).transpose() << endl;
  }

  residuals.tail(k_) = c;

  return true;
}

bool
StructureProjectionCostTerm::whitenError(double* residuals_ptr) const
{
  Eigen::Map<Eigen::VectorXd> residuals(residuals_ptr, dim());

  for (size_t i = 0; i < m_; ++i) {
    residuals.segment<3>(3 * i) =
      residuals.segment<3>(3 * i) * sqrt_weights_(i);
  }

  residuals.tail(k_) = residuals.tail(k_) * std::sqrt(lambda_);

  return true;
}

bool
StructureProjectionCostTerm::Evaluate(double const* const* parameters,
                                      double* residuals,
                                      double** jacobians) const
{
  // parameters p:
  // p[0] = object orientation (quaternion)
  // p[1] = object position
  // p[2] through p[(2 + m_) - 1] = keypoint positions
  // p[2 + m_] = structure coefficients
  unwhitenedError(parameters, residuals);
  whitenError(residuals);

  if (jacobians) {
    computeObjectOrientationJacobian(parameters, jacobians[0]);
    computeObjectTranslationJacobian(parameters, jacobians[1]);

    for (size_t i = 0; i < m_; ++i) {
      computeLandmarkJacobian(parameters, i, jacobians[2 + i]);
    }

    if (k_ > 0) {
      computeBasisJacobian(parameters, jacobians[2 + m_]);
    }
  }
  return true;
}

bool
StructureProjectionCostTerm::computeObjectOrientationJacobian(
  double const* const* parameters, double* jacobian) const
{
  if (jacobian == NULL)
    return true;

  Eigen::Map<const Eigen::Quaterniond> q(parameters[0]);

  Eigen::Map<Eigen::Matrix<double, -1, 4, Eigen::RowMajor>> H(
    jacobian, num_residuals(), 4);

  Eigen::MatrixXd S = structure(parameters);

  // Eigen::Matrix3d R = q.toRotationMatrix();

  for (size_t i = 0; i < m_; ++i) {
    // compute d(residual_i) / d(dtheta)
    H.block<3, 4>(3 * i, 0) =
      -sqrt_weights_(i) * math::Dpoint_transform_dq(q, S.col(i));
  }

  if (k_ > 0)
    H.bottomRows(k_) = Eigen::MatrixXd::Zero(k_, 4);

  return true;
}

bool
StructureProjectionCostTerm::computeObjectTranslationJacobian(
  double const* const* parameters, double* jacobian) const
{
  if (jacobian == NULL)
    return true;

  // Eigen::Map<const Eigen::Quaterniond> q(parameters[0]);

  Eigen::Map<Eigen::Matrix<double, -1, 3, Eigen::RowMajor>> H(
    jacobian, num_residuals(), 3);

  //   Eigen::MatrixXd S = structure(parameters);

  // Eigen::Matrix3d R = q.toRotationMatrix();

  for (size_t i = 0; i < m_; ++i) {
    // compute d(residual_i) / d(translation)
    H.block<3, 3>(3 * i, 0) = -sqrt_weights_(i) * Eigen::Matrix3d::Identity();
  }

  if (k_ > 0)
    H.bottomRows(k_) = Eigen::MatrixXd::Zero(k_, 3);

  return true;
}

bool
StructureProjectionCostTerm::computeLandmarkJacobian(
  double const* const* parameters, size_t index, double* jacobian) const
{
  if (jacobian == NULL)
    return true;

  // Eigen::Map<const Eigen::Quaterniond> q(parameters[0]);
  // Eigen::Map<const Eigen::Vector3d> t(parameters[1]);
  // Eigen::Map<const Eigen::VectorXd> c(parameters[2 + m_], k_);

  Eigen::Map<Eigen::VectorXd> Hp(jacobian, num_residuals(), 1);

  Hp.setZero();

  Hp.segment<3>(3 * index) = sqrt_weights_(index) * measurements_.col(index);

  return true;
}

bool
StructureProjectionCostTerm::computeBasisJacobian(
  double const* const* parameters, double* jacobian) const
{
  if (jacobian == NULL)
    return true;

  Eigen::Map<const Eigen::Quaterniond> q(parameters[0]);
  //   Eigen::Map<const Eigen::Vector3d> t(parameters[1]);
  //   Eigen::Map<const Eigen::VectorXd> c(parameters[2 + m_], k_);

  // this will only be called when k_ > 0

  Eigen::Map<
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
    Hc(jacobian, num_residuals(), k_);

  Eigen::Matrix3d R = q.toRotationMatrix();

  for (size_t i = 0; i < m_; ++i) {
    Hc.block(3 * i, 0, 3, k_) = -sqrt_weights_(i) * R * Bi_[i];
  }

  Hc.bottomRows(k_) = sqrt_lambda_ * Eigen::MatrixXd::Identity(k_, k_);

  return true;
}

*/
