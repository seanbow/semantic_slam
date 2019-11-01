#pragma once

#include "semantic_slam/Common.h"
#include "semantic_slam/FactorGraph.h"

#include <condition_variable>

class Presenter {
public:

    Presenter(boost::shared_ptr<FactorGraph> graph, boost::shared_ptr<std::condition_variable> cv);

    virtual ~Presenter() { };

    virtual void setup() = 0;
    virtual void present() = 0;

protected:
    boost::shared_ptr<FactorGraph> graph_;

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    boost::shared_ptr<std::condition_variable> cv_;
};

Presenter::Presenter(boost::shared_ptr<FactorGraph> graph, boost::shared_ptr<std::condition_variable> cv)
    : graph_(graph),
      nh_(),
      pnh_("~"),
      cv_(cv)
{
}