#pragma once

#include "semantic_slam/Common.h"

#include <ros/ros.h>

#include <boost/shared_ptr.hpp>
#include <boost/optional.hpp>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Marginals.h>
#include <fmt/format.h>
#include <condition_variable>

#include "semantic_slam/FactorInfo.h"
#include "semantic_slam/NodeInfo.h"

class FactorGraph {
public:
    FactorGraph();

    void addFactor(FactorInfoPtr fac);

    template <typename T>
    void addNode(NodeInfoPtr node, const T& initial_value);

    NodeInfoPtr findNodeBySymbol(gtsam::Symbol sym);

    NodeInfoPtr findLastNodeBeforeTime(unsigned char symbol_chr, ros::Time time);
    NodeInfoPtr findFirstNodeAfterTime(unsigned char symbol_chr, ros::Time time);
    NodeInfoPtr findLastNode(unsigned char symbol_chr);

    template <typename T>
    bool getEstimate(gtsam::Symbol sym, T& value);

    bool marginalCovariance(gtsam::Key key, Eigen::MatrixXd& cov);

    bool solve();

    size_t num_nodes() { return nodes_.size(); }
    size_t num_factors() { return factors_.size(); }

    bool setModified() { modified_ = true; }

    bool modified() { return modified_; }

private:
    boost::shared_ptr<gtsam::NonlinearFactorGraph> graph_;

    std::vector<FactorInfoPtr> factors_;
    std::vector<NodeInfoPtr> nodes_;

    boost::shared_ptr<gtsam::Values> values_;

    bool modified_; //< Whether or not the graph has been modified since the last solving

    std::mutex mutex_;

    boost::shared_ptr<gtsam::Marginals> marginals_;
};

template <typename T>
bool FactorGraph::getEstimate(gtsam::Symbol sym, T& value) {
    if (values_->find(sym) == values_->end()) {
        return false;
    }

    // if (modified_) {
    //     ROS_WARN("Getting value from modified factor graph before solution.");
    // }

    value = values_->at<T>(sym);
    return true;
}

template <typename T>
void FactorGraph::addNode(NodeInfoPtr node, const T& initial_estimate)
{
    // Make sure we don't already have a node with this symbol
    auto existing_node = findNodeBySymbol(node->symbol());
    if (existing_node) {
        throw std::runtime_error(
                fmt::format("Tried to add already existing node with symbol {} to graph",
                            gtsam::DefaultKeyFormatter(node->key())));
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        nodes_.push_back(node);
        values_->insert(node->symbol(), initial_estimate);
    }
}