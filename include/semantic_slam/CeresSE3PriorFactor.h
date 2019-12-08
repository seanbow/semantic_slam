#pragma once

#include "semantic_slam/Common.h"

#include "semantic_slam/CeresFactor.h"
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/pose_math.h"
#include "semantic_slam/ceres_cost_terms/ceres_pose_prior.h"

class CeresSE3PriorFactor : public CeresFactor
{
public:
    CeresSE3PriorFactor(SE3NodePtr node, const Pose3& prior, const Eigen::MatrixXd& covariance, int tag=0);
    ~CeresSE3PriorFactor();

    void addToProblem(boost::shared_ptr<ceres::Problem> problem);

    using Ptr = boost::shared_ptr<CeresSE3PriorFactor>;

private:
    SE3NodePtr node_;
};

using CeresSE3PriorFactorPtr = CeresSE3PriorFactor::Ptr;

CeresSE3PriorFactor::CeresSE3PriorFactor(SE3NodePtr node, 
                                         const Pose3& prior, 
                                         const Eigen::MatrixXd& covariance,
                                         int tag)
    : CeresFactor(FactorType::PRIOR, tag),
      node_(node)
{
    // ceres::Problem will take ownership of this cost function
    cf_ = PosePriorCostTerm::Create(prior, covariance);
}

CeresSE3PriorFactor::~CeresSE3PriorFactor()
{
    delete cf_;
}

void CeresSE3PriorFactor::addToProblem(boost::shared_ptr<ceres::Problem> problem)
{
    // assume the node has already been added to the problem!!
    // TODO do this more intelligently
    ceres::ResidualBlockId residual_id = problem->AddResidualBlock(cf_, NULL, node_->pose().rotation_data(), node_->pose().translation_data());
    residual_ids_[problem.get()] = residual_id;
}
