#pragma once

#include "semantic_slam/Common.h"

#include "semantic_slam/CeresFactor.h"
#include "semantic_slam/VectorNode.h"
#include "semantic_slam/ceres_cost_terms/ceres_vector_norm.h"

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/sam/RangeFactor.h>

#include "semantic_slam/keypoints/gtsam/NormFactor.h"

template<int Dim>
class CeresVectorNormPriorFactor : public CeresFactor
{
  public:
    CeresVectorNormPriorFactor(VectorNodePtr<Dim> node,
                               double prior,
                               double covariance,
                               int tag = 0);

    CeresFactor::Ptr clone() const;

    void addToProblem(boost::shared_ptr<ceres::Problem> problem);

    VectorNodePtr<Dim> node() const
    {
        return boost::static_pointer_cast<VectorNode<Dim>>(nodes_[0]);
    }

    void addToGtsamGraph(
      boost::shared_ptr<gtsam::NonlinearFactorGraph> graph) const;

    using This = CeresVectorNormPriorFactor<Dim>;
    using Ptr = boost::shared_ptr<This>;

  private:
    double prior_;
    double covariance_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

template<int Dim>
using CeresVectorNormPriorFactorPtr =
  typename CeresVectorNormPriorFactor<Dim>::Ptr;

template<int Dim>
CeresVectorNormPriorFactor<Dim>::CeresVectorNormPriorFactor(
  VectorNodePtr<Dim> node,
  double prior,
  double covariance,
  int tag)
  : CeresFactor(FactorType::PRIOR, tag)
  , prior_(prior)
  , covariance_(covariance)
{
    // ceres::Problem will take ownership of this cost function
    cf_ = VectorNormPriorCostTerm<Dim>::Create(prior, covariance);

    nodes_.push_back(node);
}

template<int Dim>
CeresFactor::Ptr
CeresVectorNormPriorFactor<Dim>::clone() const
{
    return util::allocate_aligned<This>(nullptr, prior_, covariance_, tag_);
}

template<int Dim>
void
CeresVectorNormPriorFactor<Dim>::addToProblem(
  boost::shared_ptr<ceres::Problem> problem)
{
    ceres::ResidualBlockId residual_id =
      problem->AddResidualBlock(cf_, NULL, node()->vector().data());

    residual_ids_.emplace(problem.get(), residual_id);

    active_ = true;
}

template<int Dim>
void
CeresVectorNormPriorFactor<Dim>::addToGtsamGraph(
  boost::shared_ptr<gtsam::NonlinearFactorGraph> graph) const
{
    auto gtsam_noise = gtsam::noiseModel::Isotropic::Sigma(1, covariance_);
    auto fac = util::allocate_aligned<gtsam::NormFactor<Eigen::Vector3d>>(
      node()->key(), prior_, gtsam_noise);
    graph->push_back(fac);
}