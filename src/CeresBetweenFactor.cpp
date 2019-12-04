#include "semantic_slam/CeresBetweenFactor.h"
#include "semantic_slam/ceres_cost_terms/ceres_between.h"

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

CeresBetweenFactor::CeresBetweenFactor(SE3NodePtr node1, 
                                       SE3NodePtr node2, 
                                       Pose3 between, 
                                       Eigen::MatrixXd covariance,
                                       int tag)
    : CeresFactor(FactorType::ODOMETRY, tag),
      node1_(node1),
      node2_(node2),
      between_(between),
      covariance_(covariance)
{
    cf_ = BetweenCostTerm::Create(between, covariance);

    nodes_.push_back(node1);
    nodes_.push_back(node2);

    auto gtsam_noise = gtsam::noiseModel::Gaussian::Covariance(covariance_);

    gtsam_factor_ = util::allocate_aligned<gtsam::BetweenFactor<gtsam::Pose3>>(
                    node1->key(),
                    node2->key(),
                    gtsam::Pose3(gtsam::Rot3(between.rotation()), between.translation()),
                    gtsam_noise
    );
}

CeresBetweenFactor::~CeresBetweenFactor()
{
    delete cf_;
}

void
CeresBetweenFactor::addToProblem(boost::shared_ptr<ceres::Problem> problem)
{
    active_ = true;
    residual_id_ = problem->AddResidualBlock(cf_, 
                                            NULL, 
                                            node1_->pose().rotation_data(), 
                                            node1_->pose().translation_data(),
                                            node2_->pose().rotation_data(),
                                            node2_->pose().translation_data());
}

void
CeresBetweenFactor::removeFromProblem(boost::shared_ptr<ceres::Problem> problem)
{
    active_ = false;
    problem->RemoveResidualBlock(residual_id_);
}


boost::shared_ptr<gtsam::NonlinearFactor> 
CeresBetweenFactor::getGtsamFactor() const
{
    return gtsam_factor_;
}

void 
CeresBetweenFactor::addToGtsamGraph(boost::shared_ptr<gtsam::NonlinearFactorGraph> graph) const
{
    graph->push_back(getGtsamFactor());
}