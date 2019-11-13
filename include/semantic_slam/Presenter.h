#pragma once

#include "semantic_slam/Common.h"
#include "semantic_slam/FactorGraph.h"

#include <condition_variable>

class Presenter {
public:

    Presenter();

    virtual ~Presenter() { };

    virtual void setup() = 0;
    virtual void present() = 0;

    void setGraph(boost::shared_ptr<FactorGraph> graph) { graph_ = graph; }

protected:
    boost::shared_ptr<FactorGraph> graph_;

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
};

Presenter::Presenter()
    : nh_(),
      pnh_("~")
{
}