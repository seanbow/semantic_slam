#include "semantic_slam/CeresBetweenFactor.h"
#include "semantic_slam/ceres_cost_terms/ceres_between.h"

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>

CeresBetweenFactor::CeresBetweenFactor(SE3NodePtr node0,
                                       SE3NodePtr node1,
                                       Pose3 between,
                                       Eigen::MatrixXd covariance,
                                       int tag)
  : CeresFactor(FactorType::ODOMETRY, tag)
  , between_(between)
  , covariance_(covariance)
{
    cf_ = BetweenCostTerm::Create(between, covariance);

    nodes_.push_back(node0);
    nodes_.push_back(node1);

    auto gtsam_noise = gtsam::noiseModel::Gaussian::Covariance(covariance_);

    gtsam_factor_ = util::allocate_aligned<gtsam::BetweenFactor<gtsam::Pose3>>(
      nodes_[0]->key(),
      nodes_[1]->key(),
      gtsam::Pose3(gtsam::Rot3(between.rotation()), between.translation()),
      gtsam_noise);
}

CeresBetweenFactor::~CeresBetweenFactor()
{
    delete cf_;
}

CeresFactor::Ptr
CeresBetweenFactor::clone() const
{
    return util::allocate_aligned<CeresBetweenFactor>(
      nullptr, nullptr, between_, covariance_, tag_);
}

void
CeresBetweenFactor::addToProblem(boost::shared_ptr<ceres::Problem> problem)
{
    ceres::ResidualBlockId residual_id =
      problem->AddResidualBlock(cf_,
                                NULL,
                                node0()->pose().rotation_data(),
                                node0()->pose().translation_data(),
                                node1()->pose().rotation_data(),
                                node1()->pose().translation_data());
    residual_ids_[problem.get()] = residual_id;

    active_ = true;
}

boost::shared_ptr<gtsam::NonlinearFactor>
CeresBetweenFactor::getGtsamFactor() const
{
    return gtsam_factor_;
}

void
CeresBetweenFactor::addToGtsamGraph(
  boost::shared_ptr<gtsam::NonlinearFactorGraph> graph) const
{
    graph->push_back(getGtsamFactor());
}