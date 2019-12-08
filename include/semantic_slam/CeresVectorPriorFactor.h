#pragma once

#include "semantic_slam/Common.h"

#include "semantic_slam/CeresFactor.h"
#include "semantic_slam/ceres_vector_prior.h"
#include "semantic_slam/VectorNode.h"

template <int Dim>
class CeresVectorPriorFactor : public CeresFactor
{
public:
    CeresVectorPriorFactor(VectorNodePtr<Dim> node, 
                           const typename VectorNode<Dim>::VectorType& prior, 
                           const Eigen::MatrixXd& covariance, 
                           int tag=0);

    void addToProblem(boost::shared_ptr<ceres::Problem> problem);

    using This = CeresVectorPriorFactor<Dim>;
    using Ptr = boost::shared_ptr<This>;

private:
    VectorNodePtr<Dim> node_;
};

template <int Dim>
using CeresVectorPriorFactorPtr = typename CeresVectorPriorFactor<Dim>::Ptr;

template <int Dim>
CeresVectorPriorFactor<Dim>::CeresVectorPriorFactor(VectorNodePtr<Dim> node, 
                                                const typename VectorNode<Dim>::VectorType& prior, 
                                                const Eigen::MatrixXd& covariance,
                                                int tag)
    : CeresFactor(FactorType::PRIOR, tag),
      node_(node)
{
    // ceres::Problem will take ownership of this cost function
    cf_ = VectorPriorCostTerm<typename VectorNode<Dim>::VectorType>::Create(prior, covariance);
}

template <int Dim>
void CeresVectorPriorFactor<Dim>::addToProblem(boost::shared_ptr<ceres::Problem> problem)
{
    ceres::ResidualBlockId residual_id = problem->AddResidualBlock(cf_, NULL, node_->vector().data());
    residual_ids_.emplace(problem.get(), residual_id);
}

using CeresVector2dPriorFactor = CeresVectorPriorFactor<2>;
using CeresVector2dPriorFactorPtr = CeresVectorPriorFactorPtr<2>;

using CeresVector3dPriorFactor = CeresVectorPriorFactor<3>;
using CeresVector3dPriorFactorPtr = CeresVectorPriorFactorPtr<3>;