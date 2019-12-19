#pragma once

#include "semantic_slam/Common.h"

#include "semantic_slam/CeresFactor.h"
#include "semantic_slam/VectorNode.h"
#include "semantic_slam/ceres_cost_terms/ceres_vector_prior.h"

template<int Dim>
class CeresVectorPriorFactor : public CeresFactor
{
  public:
    CeresVectorPriorFactor(VectorNodePtr<Dim> node,
                           const typename VectorNode<Dim>::VectorType& prior,
                           const Eigen::MatrixXd& covariance,
                           int tag = 0);

    CeresFactor::Ptr clone() const;

    void addToProblem(boost::shared_ptr<ceres::Problem> problem);

    VectorNodePtr<Dim> node() const
    {
        return boost::static_pointer_cast<VectorNode<Dim>>(nodes_[0]);
    }

    using This = CeresVectorPriorFactor<Dim>;
    using Ptr = boost::shared_ptr<This>;

  private:
    Eigen::MatrixXd covariance_;
    typename VectorNode<Dim>::VectorType prior_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

template<int Dim>
using CeresVectorPriorFactorPtr = typename CeresVectorPriorFactor<Dim>::Ptr;

template<int Dim>
CeresVectorPriorFactor<Dim>::CeresVectorPriorFactor(
  VectorNodePtr<Dim> node,
  const typename VectorNode<Dim>::VectorType& prior,
  const Eigen::MatrixXd& covariance,
  int tag)
  : CeresFactor(FactorType::PRIOR, tag)
  , prior_(prior)
  , covariance_(covariance)
{
    // ceres::Problem will take ownership of this cost function
    cf_ = VectorPriorCostTerm<typename VectorNode<Dim>::VectorType>::Create(
      prior, covariance);

    nodes_.push_back(node);
}

template<int Dim>
CeresFactor::Ptr
CeresVectorPriorFactor<Dim>::clone() const
{
    return util::allocate_aligned<This>(nullptr, prior_, covariance_, tag_);
}

template<int Dim>
void
CeresVectorPriorFactor<Dim>::addToProblem(
  boost::shared_ptr<ceres::Problem> problem)
{
    ceres::ResidualBlockId residual_id =
      problem->AddResidualBlock(cf_, NULL, node()->vector().data());

    residual_ids_.emplace(problem.get(), residual_id);

    active_ = true;
}

using CeresVector2dPriorFactor = CeresVectorPriorFactor<2>;
using CeresVector2dPriorFactorPtr = CeresVectorPriorFactorPtr<2>;

using CeresVector3dPriorFactor = CeresVectorPriorFactor<3>;
using CeresVector3dPriorFactorPtr = CeresVectorPriorFactorPtr<3>;