
#include "semantic_slam/keypoints/gtsam/StructureFactor.h"

#include <gtsam/geometry/Pose3.h>

namespace semslam
{
StructureFactor::StructureFactor(gtsam::Key object_key, const std::vector<gtsam::Key>& landmark_keys,
                                 const gtsam::Key& coefficient_key, const ObjectModelBasis& model,
                                 const Eigen::VectorXd& weights, double lambda)
  : object_key_(object_key)
  , landmark_keys_(landmark_keys)
  , coefficient_key_(coefficient_key)
  , model_(model)
  , lambda_(lambda)
{
  m_ = model_.mu.cols();
  k_ = model_.pc.rows() / 3;

  setWeights(weights);

  // Preconstruct basis vector matrices
  // Bi = [b0 b1 ... bk] for ith keypoint
  for (size_t i = 0; i < m_; ++i)
  {
    Bi_.emplace_back(3, k_);
    for (size_t j = 0; j < k_; ++j)
    {
      Bi_.back().block<3, 1>(0, j) = model_.pc.block<3, 1>(3 * j, i);
    }
  }

  // Turn the weights into a noise model
  // Diagonal matrix of dimension  3*m_ + k_
  // weights(i) repeated thrice along the diagonal plus sqrt(lambda) for regularizer
  // Because we need to specify it as variances not weights, use inverse
  // Eigen::VectorXd rep_weights(3*m_ + k_);
  // for (size_t i = 0; i < m_; ++i) {
  //     rep_weights.segment<3>(3*i) = Eigen::Vector3d::Constant(1 / weights(i));
  // }
  // rep_weights.tail(k_) = Eigen::VectorXd::Constant(k_, 1 / lambda);

  // noise_model_ = gtsam::noiseModel::Diagonal::Variances(rep_weights);

  // Ordering of keys: object landmarks coefficients
  keys_.push_back(object_key);
  for (auto& key : landmark_keys)
    keys_.push_back(key);
  
  if (k_ > 0) keys_.push_back(coefficient_key);
}

Eigen::MatrixXd StructureFactor::structure(const gtsam::Values& values) const
{
  if (k_ == 0) return model_.mu;

  Eigen::MatrixXd S = model_.mu;

  gtsam::Vector c = values.at<gtsam::Vector>(coefficient_key_);

  for (size_t i = 0; i < k_; ++i)
  {
    S += c(i) * model_.pc.block(3 * i, 0, 3, m_);
  }

  return S;
}

// #include <iostream>
// using std::cout; using std::endl;

// Compute residual vector [P - R*S - t] and jacobians
gtsam::Vector StructureFactor::unwhitenedError(const gtsam::Values& values,
                                               boost::optional<std::vector<gtsam::Matrix>&> H) const
{
  Eigen::VectorXd e = Eigen::VectorXd::Zero(3 * m_ + k_);

  Eigen::MatrixXd S = structure(values);

  gtsam::Pose3 pose = values.at<gtsam::Pose3>(object_key_);
  Eigen::Matrix3d R = pose.rotation().matrix();
  Eigen::Vector3d t = pose.translation().vector();

  for (size_t i = 0; i < m_; ++i)
  {
    Eigen::Vector3d p = values.at<gtsam::Point3>(landmark_keys_[i]).vector();
    e.segment<3>(3 * i) = p - R * S.col(i) - t;
    // cout << "Index " << i << ", key " << gtsam::DefaultKeyFormatter(landmark_keys_[i]) <<
    //     ", p = " << p.transpose() << ", err = " << (p - R*S.col(i) - t).transpose() << endl;
  }

  if (k_ > 0) e.tail(k_) = values.at<Eigen::VectorXd>(coefficient_key_);

  if (H)
  {
    // ordering: object landmarks coefficients
    H->push_back(Dobject(values));

    for (size_t i = 0; i < m_; ++i)
    {
      H->push_back(Dlandmark(values, i));
    }

    H->push_back(Dcoefficients(values));
  }

  return e;
}

void StructureFactor::whitenError(gtsam::Vector& e) const
{
  for (size_t i = 0; i < m_; ++i)
  {
    e.segment<3>(3 * i) = e.segment<3>(3 * i) * std::sqrt(weights_(i));
  }

  e.tail(k_) = e.tail(k_) * std::sqrt(lambda_);
}

void StructureFactor::whitenJacobians(std::vector<gtsam::Matrix>& vec) const
{
  for (auto& H : vec)
  {
    whitenJacobian(H);
  }
}

void StructureFactor::whitenJacobian(gtsam::Matrix& H) const
{
  for (size_t i = 0; i < m_; ++i)
  {
    H.block(3 * i, 0, 3, H.cols()) *= std::sqrt(weights_(i));
  }

  H.bottomRows(k_) *= std::sqrt(lambda_);
}

gtsam::Vector StructureFactor::whitenedError(const gtsam::Values& values) const
{
  gtsam::Vector e = unwhitenedError(values);
  whitenError(e);
  return e;
}

double StructureFactor::error(const gtsam::Values& values) const
{
  gtsam::Vector e = whitenedError(values);
  return 0.5 * e.transpose() * e;
  // return 0.5 * noise_model_->distance(e);
}

boost::shared_ptr<gtsam::GaussianFactor> StructureFactor::linearize(const gtsam::Values& values) const
{
  std::vector<gtsam::Matrix> A;
  gtsam::Vector e = -unwhitenedError(values, A);

  // noise_model_->WhitenSystem(A, e);
  whitenJacobians(A);
  whitenError(e);

  // Fill in terms needed to create a JacobianFactor
  std::vector<std::pair<gtsam::Key, gtsam::Matrix>> terms(size());
  for (size_t i = 0; i < size(); ++i)
  {
    terms[i].first = keys()[i];
    terms[i].second.swap(A[i]);
  }

  // std::vector<std::pair<gtsam::Key, gtsam::Matrix>> terms;
  // terms.push_back(std::make_pair(object_key_, A.front()));
  // terms.push_back(std::make_pair(coefficient_key_, A.back()));

  return gtsam::GaussianFactor::shared_ptr(new gtsam::JacobianFactor(terms, e));
}

Eigen::MatrixXd StructureFactor::Dobject(const gtsam::Values& values) const
{
  Eigen::MatrixXd H(3 * m_ + k_, 6);

  Eigen::MatrixXd S = structure(values);

  gtsam::Pose3 pose = values.at<gtsam::Pose3>(object_key_);
  Eigen::Matrix3d R = pose.rotation().matrix();
  // const auto& t = pose.translation().vector();

  for (size_t i = 0; i < m_; ++i)
  {
    // Eigen::Vector3d p = values.at<gtsam::Point3>(landmark_keys_[i]).vector();

    // compute d(residual_i) / d(dtheta)
    // H.block<3,3>(3*i, 0) = -skewsymm(R*S.col(i));
    H.block<3, 3>(3 * i, 0) = R * skewsymm(S.col(i));

    // compute d(residual_i) / d(translation)
    H.block<3, 3>(3 * i, 3) = -R;  // TODO why is this not identity
  }

  H.bottomRows(k_) = Eigen::MatrixXd::Zero(k_, 6);

  return H;
}

Eigen::MatrixXd StructureFactor::Dlandmark(const gtsam::Values& values, size_t landmark_index) const
{
  Eigen::MatrixXd Hp(3 * m_ + k_, 3);

  Hp = Eigen::MatrixXd::Zero(3 * m_ + k_, 3);

  Hp.block<3, 3>(3 * landmark_index, 0) = Eigen::Matrix3d::Identity();

  return Hp;
}

Eigen::MatrixXd StructureFactor::Dcoefficients(const gtsam::Values& values) const
{
  Eigen::MatrixXd Hc(3 * m_ + k_, k_);

  gtsam::Pose3 pose = values.at<gtsam::Pose3>(object_key_);
  Eigen::Matrix3d R = pose.rotation().matrix();
  // const auto& t = pose.translation().vector();

  // const gtsam::Vector& c = values.at<gtsam::Vector>(coefficient_key_);

  // Eigen::MatrixXd S = structure(values);

  for (size_t i = 0; i < m_; ++i)
  {
    // Eigen::Vector3d p = values.at<gtsam::Point3>(landmark_keys_[i]).vector();

    // Eigen::MatrixXd RBi = R*Bi_[i];
    // Hc.block(i, 0, 1, k_) = -2.0 * (RBi.transpose() * (p - t - R*model_.mu.col(i) - RBi*c) ).transpose();
    Hc.block(3 * i, 0, 3, k_) = -R * Bi_[i];
  }

  Hc.bottomRows(k_) = Eigen::MatrixXd::Identity(k_, k_);

  return Hc;
}

}  // namespace semslam
