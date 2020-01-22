#pragma once

#include "semantic_slam/CeresVectorPriorFactor.h"
#include "semantic_slam/ImuBiasNode.h"
#include <gtsam/navigation/CombinedImuFactor.h>

// Unfortunate class design that this needs to exist... needed for GTSAM
// interoperability right now. TODO refactor things
class CeresImuBiasPriorFactor : public CeresVectorPriorFactor<6>
{
  public:
    using Base = CeresVectorPriorFactor<6>;

    CeresImuBiasPriorFactor(boost::shared_ptr<ImuBiasNode> node,
                            const ImuBiasNode::VectorType& prior,
                            const Eigen::MatrixXd& covariance,
                            int tag = 0)
      : Base(node, prior, covariance, tag)
    {}

    virtual void addToGtsamGraph(
      boost::shared_ptr<gtsam::NonlinearFactorGraph> graph) const
    {
        // Need to swap the covariance quadrants because ordering is opposite in
        // gtsam
        Eigen::MatrixXd gtsam_cov(6, 6);
        gtsam_cov.topLeftCorner<3, 3>() = covariance_.bottomRightCorner<3, 3>();
        gtsam_cov.topRightCorner<3, 3>() = covariance_.bottomLeftCorner<3, 3>();
        gtsam_cov.bottomRightCorner<3, 3>() = covariance_.topLeftCorner<3, 3>();
        gtsam_cov.bottomLeftCorner<3, 3>() = covariance_.topRightCorner<3, 3>();

        auto gtsam_noise = gtsam::noiseModel::Gaussian::Covariance(gtsam_cov);

        gtsam::imuBias::ConstantBias gtsam_prior(prior_.tail<3>(),
                                                 prior_.head<3>());

        auto gtsam_fac = util::allocate_aligned<
          gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
          node()->key(), gtsam_prior, gtsam_noise);

        graph->push_back(gtsam_fac);
    }
};