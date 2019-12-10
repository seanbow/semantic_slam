
#include "semantic_slam/keypoints/gtsam/StructureProjectionFactor.h"

namespace semslam {
StructureProjectionFactor::StructureProjectionFactor(
  const Eigen::MatrixXd& normalized_measurements,
  gtsam::Key object_key,
  const std::vector<gtsam::Key>& landmark_keys,
  const gtsam::Key& coefficient_key,
  const ObjectModelBasis& model,
  const Eigen::VectorXd& weights,
  double lambda)
  : measurements_(normalized_measurements)
  , object_key_(object_key)
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
    if (k_ > 0) {
        for (size_t i = 0; i < m_; ++i) {
            Bi_.emplace_back(3, k_);
            for (size_t j = 0; j < k_; ++j) {
                Bi_.back().block<3, 1>(0, j) = model_.pc.block<3, 1>(3 * j, i);
            }
        }
    }

    // Ordering of keys: object landmarks coefficients
    keys_.push_back(object_key);
    for (auto& key : landmark_keys)
        keys_.push_back(key);

    if (k_ > 0)
        keys_.push_back(coefficient_key);
}

Eigen::MatrixXd
StructureProjectionFactor::structure(const gtsam::Values& values) const
{
    if (k_ == 0)
        return model_.mu;

    Eigen::MatrixXd S = model_.mu;

    gtsam::Vector c = values.at<gtsam::Vector>(coefficient_key_);

    for (size_t i = 0; i < k_; ++i) {
        S += c(i) * model_.pc.block(3 * i, 0, 3, m_);
    }

    return S;
}

// Compute residual vector [WZ - R*S - t] and jacobians
gtsam::Vector
StructureProjectionFactor::unweightedError(
  const gtsam::Values& values,
  boost::optional<std::vector<gtsam::Matrix>&> H) const
{
    Eigen::VectorXd e = Eigen::VectorXd::Zero(3 * m_ + k_);

    Eigen::MatrixXd S = structure(values);

    gtsam::Pose3 pose = values.at<gtsam::Pose3>(object_key_);
    Eigen::Matrix3d R = pose.rotation().matrix();
    Eigen::Vector3d t = pose.translation().vector();

    for (size_t i = 0; i < m_; ++i) {
        double depth = values.at<double>(landmark_keys_[i]);
        Eigen::Vector3d p = measurements_.col(i) * depth;

        e.segment<3>(3 * i) = p - R * S.col(i) - t;
    }

    if (k_ > 0)
        e.tail(k_) = values.at<Eigen::VectorXd>(coefficient_key_);

    if (H) {
        // ordering: object landmarks coefficients
        H->push_back(Dobject(values));

        for (size_t i = 0; i < m_; ++i) {
            H->push_back(Dlandmark(values, i));
        }

        if (k_ > 0)
            H->push_back(Dcoefficients(values));
    }

    return e;
}

void
StructureProjectionFactor::weightError(gtsam::Vector& e) const
{
    for (size_t i = 0; i < m_; ++i) {
        e.segment<3>(3 * i) = e.segment<3>(3 * i) * std::sqrt(weights_(i));
    }

    if (k_ > 0)
        e.tail(k_) = e.tail(k_) * std::sqrt(lambda_);
}

void
StructureProjectionFactor::weightJacobians(
  std::vector<gtsam::Matrix>& vec) const
{
    for (auto& H : vec)
        weightJacobian(H);
}

void
StructureProjectionFactor::weightJacobian(gtsam::Matrix& H) const
{
    for (size_t i = 0; i < m_; ++i) {
        H.block(3 * i, 0, 3, H.cols()) *= std::sqrt(weights_(i));
    }

    if (k_ > 0)
        H.bottomRows(k_) *= std::sqrt(lambda_);
}

gtsam::Vector
StructureProjectionFactor::weightedError(const gtsam::Values& values) const
{
    gtsam::Vector e = unweightedError(values);
    weightError(e);
    return e;
}

double
StructureProjectionFactor::error(const gtsam::Values& values) const
{
    gtsam::Vector e = weightedError(values);
    return 0.5 * e.transpose() * e;
}

boost::shared_ptr<gtsam::GaussianFactor>
StructureProjectionFactor::linearize(const gtsam::Values& values) const
{
    std::vector<gtsam::Matrix> A;
    gtsam::Vector e = -unweightedError(values, A);

    weightJacobians(A);
    weightError(e);

    // Fill in terms needed to create a JacobianFactor
    std::vector<std::pair<gtsam::Key, gtsam::Matrix>> terms(size());
    for (size_t i = 0; i < size(); ++i) {
        terms[i].first = keys()[i];
        terms[i].second.swap(A[i]);
    }

    return gtsam::GaussianFactor::shared_ptr(
      new gtsam::JacobianFactor(terms, e));
}

Eigen::MatrixXd
StructureProjectionFactor::Dobject(const gtsam::Values& values) const
{
    Eigen::MatrixXd H(3 * m_ + k_, 6);

    Eigen::MatrixXd S = structure(values);

    gtsam::Pose3 pose = values.at<gtsam::Pose3>(object_key_);
    Eigen::Matrix3d R = pose.rotation().matrix();

    for (size_t i = 0; i < m_; ++i) {
        // compute d(residual_i) / d(dtheta)
        H.block<3, 3>(3 * i, 0) = R * skewsymm(S.col(i));

        // compute d(residual_i) / d(translation)
        H.block<3, 3>(3 * i, 3) = -R; // TODO why is this not identity
    }

    if (k_ > 0)
        H.bottomRows(k_) = Eigen::MatrixXd::Zero(k_, 6);

    return H;
}

Eigen::VectorXd
StructureProjectionFactor::Dlandmark(const gtsam::Values& values,
                                     size_t landmark_index) const
{
    Eigen::VectorXd Hp = Eigen::VectorXd::Zero(3 * m_ + k_);

    Hp.segment<3>(3 * landmark_index) = measurements_.col(landmark_index);

    return Hp;
}

Eigen::MatrixXd
StructureProjectionFactor::Dcoefficients(const gtsam::Values& values) const
{
    // this will only be called when k_ > 0

    Eigen::MatrixXd Hc(3 * m_ + k_, k_);

    gtsam::Pose3 pose = values.at<gtsam::Pose3>(object_key_);
    Eigen::Matrix3d R = pose.rotation().matrix();

    for (size_t i = 0; i < m_; ++i) {
        Hc.block(3 * i, 0, 3, k_) = -R * Bi_[i];
    }

    Hc.bottomRows(k_) = Eigen::MatrixXd::Identity(k_, k_);

    return Hc;
}

} // namespace semslam
