#pragma once

#include "semantic_slam/Common.h"

#include <ros/ros.h>

#include <boost/shared_ptr.hpp>
#include <boost/optional.hpp>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <fmt/format.h>

#include "semantic_slam/FactorInfo.h"
#include "semantic_slam/NodeInfo.h"

class FactorGraph {
public:
    FactorGraph();

    void addFactor(FactorInfo fac);

    template <typename T>
    void addNode(NodeInfo node, const T& initial_value);

    boost::optional<NodeInfo&> findNodeBySymbol(gtsam::Symbol sym);

    boost::optional<NodeInfo&> findLastNodeBeforeTime(unsigned char symbol_chr, ros::Time time);

    template <typename T>
    bool getEstimate(gtsam::Symbol sym, T& value);

    bool solve();

private:
    boost::shared_ptr<gtsam::NonlinearFactorGraph> graph_;

    std::vector<FactorInfo> factors_;
    std::vector<NodeInfo> nodes_;

    boost::shared_ptr<gtsam::Values> values_;

    bool modified_; //< Whether or not the graph has been modified since the last solving
};

template <typename T>
bool FactorGraph::getEstimate(gtsam::Symbol sym, T& value) {
    if (values_->find(sym) == values_->end()) {
        return false;
    }

    if (modified_) {
        ROS_WARN("Getting value from modified factor graph before solution.");
    }

    value = values_->at<T>(sym);
    return true;
}

template <typename T>
void FactorGraph::addNode(NodeInfo node, const T& initial_estimate)
{
    // Make sure we don't already have a node with this symbol
    auto existing_node = findNodeBySymbol(node.symbol());
    if (existing_node) {
        throw std::runtime_error(
                fmt::format("Tried to add already existing node with symbol {} to graph",
                            gtsam::DefaultKeyFormatter(node.key())));
    }

    nodes_.push_back(node);

    values_->insert(node.symbol(), initial_estimate);
}