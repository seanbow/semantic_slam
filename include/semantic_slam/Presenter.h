#pragma once

#include "semantic_slam/Common.h"

class FactorGraph;
class SemanticKeyframe;
class EstimatedObject;

class Presenter {
public:

    Presenter();

    virtual ~Presenter() { };

    virtual void setup() { }
    
    virtual void present(const std::vector<boost::shared_ptr<SemanticKeyframe>>& keyframes,
                         const std::vector<boost::shared_ptr<EstimatedObject>>& objects) = 0;

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