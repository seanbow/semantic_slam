#pragma once

#include "semantic_slam/Common.h"

#include "semantic_slam/CeresFactor.h"
#include "semantic_slam/ceres_vector_prior.h"
#include "semantic_slam/VectorNode.h"

template <typename Vector>
class CeresVectorPriorFactor : public CeresFactor
{
public:
    CeresVectorPriorFactor(VectorNodePtr<Vector> node, 
                           const Vector& prior, 
                           const Eigen::MatrixXd& covariance, 
                           int tag=0);

    void addToProblem(boost::shared_ptr<ceres::Problem> problem);
    void removeFromProblem(boost::shared_ptr<ceres::Problem> problem);

    using This = CeresVectorPriorFactor<Vector>;
    using Ptr = boost::shared_ptr<This>;

private:
    ceres::CostFunction* cf_;
    ceres::ResidualBlockId residual_id_;

    VectorNodePtr<Vector> node_;
};

template <typename Vector>
using CeresVectorPriorFactorPtr = typename CeresVectorPriorFactor<Vector>::Ptr;

template <typename Vector>
CeresVectorPriorFactor<Vector>::CeresVectorPriorFactor(VectorNodePtr<Vector> node, 
                                                const Vector& prior, 
                                                const Eigen::MatrixXd& covariance,
                                                int tag)
    : CeresFactor(FactorType::PRIOR, tag),
      node_(node)
{
    // ceres::Problem will take ownership of this cost function
    cf_ = VectorPriorCostTerm<Vector>::Create(prior, covariance);
}

template <typename Vector>
void CeresVectorPriorFactor<Vector>::addToProblem(boost::shared_ptr<ceres::Problem> problem)
{
    residual_id_ = problem->AddResidualBlock(cf_, NULL, node_->vector().data());
}

template <typename Vector>
void CeresVectorPriorFactor<Vector>::addToProblem(boost::shared_ptr<ceres::Problem> problem)
{
    problem->RemoveResidualBlock(residual_id_);
}

using CeresVector2dPriorFactor = CeresVectorPriorFactor<Eigen::Vector2d>;
using CeresVector2dPriorFactorPtr = CeresVectorPriorFactorPtr<Eigen::Vector2d>;

using CeresVector3dPriorFactor = CeresVectorPriorFactor<Eigen::Vector3d>;
using CeresVector3dPriorFactorPtr = CeresVectorPriorFactorPtr<Eigen::Vector3d>;