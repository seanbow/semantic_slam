#include "semantic_slam/ImuBiasNode.h"

#include <gtsam/navigation/CombinedImuFactor.h>

ImuBiasNode::ImuBiasNode(Symbol sym, boost::optional<ros::Time> time)
  : Base(sym, time)
{}

boost::shared_ptr<gtsam::Value>
ImuBiasNode::getGtsamValue() const
{
    gtsam::imuBias::ConstantBias gtsam_bias(vector().tail<3>(),
                                            vector().head<3>());

    return util::allocate_aligned<
      gtsam::GenericValue<gtsam::imuBias::ConstantBias>>(gtsam_bias);
}