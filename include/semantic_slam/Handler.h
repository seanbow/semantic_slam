#pragma once

#include "semantic_slam/FactorGraph.h"

class Handler {

    Handler(boost::shared_ptr<FactorGraph> graph);

private:
    boost::shared_ptr<FactorGraph> factor_graph_;
};

Handler::Handler(boost::shared_ptr<FactorGraph> graph)
    : factor_graph_(graph)
{
    
}