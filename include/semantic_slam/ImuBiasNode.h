#pragma once

#include "semantic_slam/VectorNode.h"

// A class for storing IMU bias values; needed to interface well with GTSAM
class ImuBiasNode : public VectorNode<6>
{
  public:
    ImuBiasNode(Symbol sym, boost::optional<ros::Time> time = boost::none);

    using Base = VectorNode<6>;

    virtual boost::shared_ptr<gtsam::Value> getGtsamValue() const;
};