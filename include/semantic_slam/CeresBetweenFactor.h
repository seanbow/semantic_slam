#pragma once 

#include "semantic_slam/SE3Node.h"
#include "semantic_slam/pose_math.h"

class CeresBetweenFactor : public CeresFactor
{
public:
    CeresBetweenFactor(SE3NodePtr node1, SE3NodePtr node2, Pose3 between, Eigen::MatrixXd covariance, int tag=0);

    void addToProblem(boost::shared_ptr<ceres::Problem> problem);

    using This = CeresBetweenFactor;
    using Ptr = boost::shared_ptr<This>;

private: 
    ceres::CostFunction* cf_;

    SE3NodePtr node1_;
    SE3NodePtr node2_;

};

using CeresBetweenFactorPtr = CeresBetweenFactor::Ptr;

class BetweenCostTerm
{
private:
    typedef BetweenCostTerm This;

public:
    using shared_ptr = boost::shared_ptr<This>;
    using Ptr = shared_ptr;

    BetweenCostTerm(const Pose3& between, const Eigen::MatrixXd& cov);

    template <typename T>
    bool operator()(const T* const q1, 
                    const T* const p1, 
                    const T* const q2, 
                    const T* const p2, 
                    T* residual_ptr) const;

    static ceres::CostFunction* Create(const Pose3& between, const Eigen::MatrixXd& cov);

private:
    Eigen::Matrix<double, 6, 6> sqrt_information_;

    math::Quaterniond dq_;
    Eigen::Vector3d dp_;
};

CeresBetweenFactor::CeresBetweenFactor(SE3NodePtr node1, 
                                       SE3NodePtr node2, 
                                       Pose3 between, 
                                       Eigen::MatrixXd covariance,
                                       int tag)
    : CeresFactor(FactorType::ODOMETRY, tag),
      node1_(node1),
      node2_(node2)
{
    cf_ = BetweenCostTerm::Create(between, covariance);
}

void
CeresBetweenFactor::addToProblem(boost::shared_ptr<ceres::Problem> problem)
{
    problem->AddResidualBlock(cf_, 
                              NULL, 
                              node1_->pose().rotation_data(), 
                              node1_->pose().translation_data(),
                              node2_->pose().rotation_data(),
                              node2_->pose().translation_data());
}

BetweenCostTerm::BetweenCostTerm(const Pose3& between, const Eigen::MatrixXd& cov)
{
    // Sqrt of information matrix
    Eigen::MatrixXd sqrtC = cov.llt().matrixL();
    sqrt_information_.setIdentity();
    sqrtC.triangularView<Eigen::Lower>().solveInPlace(sqrt_information_);

    dq_ = between.rotation();
    dp_ = between.translation();
}

template <typename T>
bool BetweenCostTerm::operator()(const T* const q1_ptr, 
                                 const T* const p1_ptr, 
                                 const T* const q2_ptr, 
                                 const T* const p2_ptr, 
                                 T* residual_ptr) const
{
    Eigen::Map<const math::Quaternion<T>> q1(q1_ptr);
    Eigen::Map<const Eigen::Matrix<T,3,1>> p1(p1_ptr);

    Eigen::Map<const math::Quaternion<T>> q2(q2_ptr);
    Eigen::Map<const Eigen::Matrix<T,3,1>> p2(p2_ptr);

    // Need to make pose3 generic...
    math::Quaternion<T> q_between = math::quat_mul(math::quat_inv(q1), q2);
    Eigen::Matrix<T,3,1> p_between = math::quat2rot(q1).transpose() * (p2 - p1);

    math::Quaternion<T> q_error = math::quat_mul( math::quat_inv(dq_.cast<T>()), q_between);
    // Eigen::Quaternion<T> q_error = q_between * dq_.inverse().cast<T>();
    Eigen::Matrix<T, 3, 1> p_error = p_between - dp_.cast<T>();

    Eigen::Map<Eigen::Matrix<T,6,1>> residual(residual_ptr);

    residual.template head<3>() = T(2.0) * q_error.template head<3>();
    residual.template tail<3>() = p_error;

    residual.applyOnTheLeft(sqrt_information_);

    return true;
}

ceres::CostFunction* BetweenCostTerm::Create(const Pose3& between, const Eigen::MatrixXd& cov)
{
    BetweenCostTerm* term = new BetweenCostTerm(between, cov);
    return new ceres::AutoDiffCostFunction<BetweenCostTerm, 6, 4, 3, 4, 3>(term);
}