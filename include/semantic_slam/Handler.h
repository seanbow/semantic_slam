#pragma once

#include "semantic_slam/Common.h"

#include "semantic_slam/FactorGraph.h"

#include <condition_variable>

class Handler {
public:

    Handler(boost::shared_ptr<FactorGraph> graph, boost::shared_ptr<std::condition_variable> cv);

    virtual ~Handler() { };

    virtual void setup() = 0;
    virtual void update() = 0;

protected:
    boost::shared_ptr<FactorGraph> graph_;

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    boost::shared_ptr<std::condition_variable> cv_;
};

Handler::Handler(boost::shared_ptr<FactorGraph> graph, boost::shared_ptr<std::condition_variable> cv)
    : graph_(graph),
      nh_(),
      pnh_("~"),
      cv_(cv)
{
}