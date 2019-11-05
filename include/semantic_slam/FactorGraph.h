#pragma once

#include "semantic_slam/Common.h"

#include <ros/ros.h>

#include <boost/shared_ptr.hpp>
#include <boost/optional.hpp>
// #include <gtsam/nonlinear/NonlinearFactorGraph.h>
// #include <gtsam/nonlinear/Marginals.h>
#include <fmt/format.h>
#include <condition_variable>

#include "semantic_slam/Symbol.h"
#include "semantic_slam/CeresFactorGraph.h"
#include "semantic_slam/CeresNode.h"
#include "semantic_slam/CeresFactor.h"
// #include "semantic_slam/FactorInfo.h"
// #include "semantic_slam/NodeInfo.h"

class FactorGraph {
public:
    FactorGraph();

    void addFactor(CeresFactorPtr fac);

    void addNode(CeresNodePtr node);

    CeresNodePtr getNode(Symbol sym);

    CeresNodePtr findLastNodeBeforeTime(unsigned char symbol_chr, ros::Time time);
    CeresNodePtr findFirstNodeAfterTime(unsigned char symbol_chr, ros::Time time);
    CeresNodePtr findLastNode(unsigned char symbol_chr);

    template <typename T>
    bool getEstimate(Symbol sym, T& value);

    // bool marginalCovariance(Key key, Eigen::MatrixXd& cov);

    bool solve(bool verbose=false);

    size_t num_nodes() { return nodes_.size(); }
    size_t num_factors() { return factors_.size(); }

    bool setModified() { modified_ = true; }

    bool modified() { return modified_; }

private:
    boost::shared_ptr<CeresFactorGraph> graph_;

    std::vector<CeresFactorPtr> factors_;
    std::vector<CeresNodePtr> nodes_;

    bool modified_; //< Whether or not the graph has been modified since the last solving

    std::mutex mutex_;

    // boost::shared_ptr<gtsam::Marginals> marginals_;
};

// template <typename T>
// bool FactorGraph::getEstimate(Symbol sym, T& value) {
//     if (values_->find(sym) == values_->end()) {
//         return false;
//     }

//     // if (modified_) {
//     //     ROS_WARN("Getting value from modified factor graph before solution.");
//     // }

//     value = values_->at<T>(sym);
//     return true;
// }

void FactorGraph::addNode(CeresNodePtr node)
{
    // Make sure we don't already have a node with this symbol
    auto existing_node = getNode(node->symbol());
    if (existing_node) {
        throw std::runtime_error(
                fmt::format("Tried to add already existing node with symbol {} to graph",
                            DefaultKeyFormatter(node->key())));
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        graph_->addNode(node);
        nodes_.push_back(node);
        // values_->insert(node->symbol(), initial_estimate);
    }
}