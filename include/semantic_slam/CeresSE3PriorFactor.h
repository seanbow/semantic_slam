#pragma once

#include "semantic_slam/Common.h"

#include "semantic_slam/CeresFactor.h"
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/ceres_cost_terms/ceres_pose_prior.h"
#include "semantic_slam/pose_math.h"

class CeresSE3PriorFactor : public CeresFactor
{
  public:
    CeresSE3PriorFactor(boost::shared_ptr<SE3Node> node,
                        const Pose3& prior,
                        const Eigen::MatrixXd& covariance,
                        int tag = 0);
    ~CeresSE3PriorFactor();

    void addToProblem(boost::shared_ptr<ceres::Problem> problem);

    boost::shared_ptr<SE3Node> node() const
    {
        return boost::static_pointer_cast<SE3Node>(nodes_[0]);
    }

    // Returns a new factor that is identical to this one except in which node
    // it operates on
    CeresFactor::Ptr clone() const;

    using Ptr = boost::shared_ptr<CeresSE3PriorFactor>;

  private:
    Pose3 prior_;
    Eigen::MatrixXd covariance_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

using CeresSE3PriorFactorPtr = CeresSE3PriorFactor::Ptr;

CeresSE3PriorFactor::CeresSE3PriorFactor(boost::shared_ptr<SE3Node> node,
                                         const Pose3& prior,
                                         const Eigen::MatrixXd& covariance,
                                         int tag)
  : CeresFactor(FactorType::PRIOR, tag)
  , prior_(prior)
  , covariance_(covariance)
{
    nodes_.push_back(node);
    // ceres::Problem will take ownership of this cost function
    cf_ = PosePriorCostTerm::Create(prior, covariance);
}

CeresSE3PriorFactor::~CeresSE3PriorFactor()
{
    delete cf_;
}

CeresFactor::Ptr
CeresSE3PriorFactor::clone() const
{
    auto fac = util::allocate_aligned<CeresSE3PriorFactor>(
      nullptr, prior_, covariance_, tag_);
    return fac;
}

void
CeresSE3PriorFactor::addToProblem(boost::shared_ptr<ceres::Problem> problem)
{
    // assume the node has already been added to the problem!!
    // TODO do this more intelligently
    ceres::ResidualBlockId residual_id =
      problem->AddResidualBlock(cf_, NULL, node()->pose().data());
    residual_ids_[problem.get()] = residual_id;

    active_ = true;
}
