#pragma once

#include "semantic_slam/Common.h"

#include "semantic_slam/FactorGraph.h"
#include "semantic_slam/CeresNode.h"

#include <condition_variable>

class Handler {
public:

    Handler();

    virtual ~Handler() { };

    virtual void setup() = 0;
    virtual void update() = 0;

    virtual bool isOdometry() { return false; }
    virtual CeresNodePtr getSpineNode(ros::Time time) { return nullptr; }

    // non-odometry handlers may need to know of the odometry handler to access the 
    // "spine" nodes of the factor graph
    // TODO this is messy
    virtual void setOdometryHandler(boost::shared_ptr<Handler> odom) {
        odometry_handler_ = odom;
    }

    void setGraph(boost::shared_ptr<FactorGraph> graph) { graph_ = graph; }
    void setCv(boost::shared_ptr<std::condition_variable> cv) { cv_ = cv; }

protected:
    boost::shared_ptr<FactorGraph> graph_;
    boost::shared_ptr<Handler> odometry_handler_;

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    boost::shared_ptr<std::condition_variable> cv_;
};

Handler::Handler()
    : nh_(),
      pnh_("~")
{
}