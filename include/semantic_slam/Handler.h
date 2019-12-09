#pragma once

#include "semantic_slam/Common.h"

#include "semantic_slam/CeresNode.h"
#include "semantic_slam/FactorGraph.h"

#include <condition_variable>

class SemanticMapper;

class Handler
{
  public:
    Handler();

    virtual ~Handler(){};

    virtual void setup() = 0;
    virtual void update() {}

    // virtual CeresNodePtr getSpineNode(ros::Time time) { return nullptr; }

    void setGraph(boost::shared_ptr<FactorGraph> graph) { graph_ = graph; }
    void setEssentialGraph(boost::shared_ptr<FactorGraph> graph)
    {
        essential_graph_ = graph;
    }
    void setMapper(SemanticMapper* mapper) { mapper_ = mapper; }

  protected:
    boost::shared_ptr<FactorGraph> graph_;
    boost::shared_ptr<FactorGraph> essential_graph_;

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    SemanticMapper* mapper_;

    // boost::shared_ptr<std::condition_variable> cv_;
};

Handler::Handler()
  : nh_()
  , pnh_("~")
{}