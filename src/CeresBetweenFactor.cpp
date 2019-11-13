#include "semantic_slam/CeresBetweenFactor.h"
#include "semantic_slam/ceres_cost_terms/ceres_between.h"

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
