#pragma once

#include "semantic_slam/Common.h"
#include "semantic_slam/FactorGraph.h"
#include "semantic_slam/SemanticKeyframe.h"
#include "semantic_slam/keypoints/EstimatedObject.h"

#include <condition_variable>

class Presenter {
public:

    Presenter();

    virtual ~Presenter() { };

    virtual void setup() { }
    
    virtual void present(const std::vector<SemanticKeyframe::Ptr>& keyframes,
                         const std::vector<EstimatedObject::Ptr>& objects) = 0;

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