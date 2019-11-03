#pragma once 

#include "semantic_slam/Common.h"

#include <ceres/ceres.h>
#include <gtsam/inference/Symbol.h>
#include <ros/ros.h>

class CeresNode {
public:

    virtual void addToProblem(boost::shared_ptr<ceres::Problem> problem) = 0;

    gtsam::Symbol symbol() { return gtsam::Symbol(key_); }
    unsigned char chr() { return gtsam::Symbol(key_).chr(); }
    size_t index() { return gtsam::Symbol(key_).index(); }
    gtsam::Key key() { return key_; }

    boost::optional<ros::Time> time() { return time_; }

    using Ptr = boost::shared_ptr<CeresNode>;

protected:

    CeresNode(gtsam::Key key, boost::optional<ros::Time> time=boost::none);

private:
    gtsam::Key key_;
    boost::optional<ros::Time> time_;
};

using CeresNodePtr = CeresNode::Ptr;

CeresNode::CeresNode(gtsam::Key key, boost::optional<ros::Time> time)
    : key_(key),
      time_(time)
{
    
}