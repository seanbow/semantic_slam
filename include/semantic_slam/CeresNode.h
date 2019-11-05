#pragma once 

#include "semantic_slam/Common.h"

#include <ceres/ceres.h>
#include <gtsam/inference/Symbol.h>
#include <ros/ros.h>

class CeresNode {
public:

    virtual void addToProblem(boost::shared_ptr<ceres::Problem> problem) = 0;

    gtsam::Symbol symbol() const { return gtsam::Symbol(key_); }
    unsigned char chr() const { return gtsam::Symbol(key_).chr(); }
    size_t index() const { return gtsam::Symbol(key_).index(); }
    gtsam::Key key() const { return key_; }

    boost::optional<ros::Time> time() const { return time_; }

    std::vector<double*> parameter_blocks() { return parameter_blocks_; }

    using Ptr = boost::shared_ptr<CeresNode>;

protected:

    CeresNode(gtsam::Key key, boost::optional<ros::Time> time=boost::none);

    std::vector<double*> parameter_blocks_;

    gtsam::Key key_;
    boost::optional<ros::Time> time_;
};

using CeresNodePtr = CeresNode::Ptr;

CeresNode::CeresNode(gtsam::Key key, boost::optional<ros::Time> time)
    : key_(key),
      time_(time)
{
    
}