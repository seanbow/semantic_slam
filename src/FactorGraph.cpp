#include "semantic_slam/FactorGraph.h"

#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

FactorGraph::FactorGraph()
  : modified_(false)
{
    graph_ = boost::make_shared<gtsam::NonlinearFactorGraph>();
    values_ = boost::make_shared<gtsam::Values>();
}


void FactorGraph::addFactor(FactorInfo fac)
{
    factors_.push_back(fac);

    graph_->push_back(fac.factor());

    modified_ = true;
}

boost::optional<NodeInfo&>
FactorGraph::findNodeBySymbol(gtsam::Symbol sym)
{
    for (auto& node : nodes_) {
        if (node.symbol() == sym) {
            return node;
        }
    }

    return boost::none;
}

bool FactorGraph::solve()
{
    gtsam::LevenbergMarquardtParams lm_params;
    // lm_params.setVerbosityLM("SUMMARY");
    // lm_params.setVerbosityLM("DAMPED");
    lm_params.diagonalDamping = true;

    try {
        gtsam::LevenbergMarquardtOptimizer optimizer(*graph_, *values_, lm_params);
        *values_ = optimizer.optimize();
    } catch (std::exception& e) {
        ROS_WARN("Error when optimizing factor graph");
        ROS_WARN_STREAM(e.what());
        return false;
    }

    modified_ = false;

    return true;

}


boost::optional<NodeInfo&> FactorGraph::findLastNodeBeforeTime(unsigned char symbol_chr, ros::Time time)
{
    ros::Time last_time(0);
    bool found = false;
    size_t node_index = 0;

    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i].chr() != symbol_chr) continue;

        if (!nodes_[i].time()) continue;

        if (nodes_[i].time() > last_time && nodes_[i].time() <= time) {
            last_time = *nodes_[i].time();
            found = true;
            node_index = i;
        }
    }

    if (found) return nodes_[node_index];
    else return boost::none;
}