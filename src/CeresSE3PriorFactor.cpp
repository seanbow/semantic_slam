
#include "semantic_slam/CeresSE3PriorFactor.h"

#include "semantic_slam/SE3Node.h"
#include "semantic_slam/ceres_cost_terms/ceres_pose_prior.h"
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/PriorFactor.h>

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

void
CeresSE3PriorFactor::addToGtsamGraph(
  boost::shared_ptr<gtsam::NonlinearFactorGraph> graph) const
{
    auto gtsam_noise = gtsam::noiseModel::Gaussian::Covariance(covariance_);

    auto gtsam_fac = util::allocate_aligned<gtsam::PriorFactor<gtsam::Pose3>>(
      node()->key(), gtsam::Pose3(prior_), gtsam_noise);

    graph->push_back(gtsam_fac);
}