#pragma once

#include "semantic_slam/Common.h"

#include <ceres/ceres.h>

#include "semantic_slam/CeresNode.h"
#include "semantic_slam/CeresFactor.h"

#include <unordered_map>
#include <mutex>
#include <rosfmt/rosfmt.h>

class FactorGraph
{
public:
    FactorGraph();

    void addNode(CeresNodePtr node);

    void addFactor(CeresFactorPtr factor);
    void addFactors(std::vector<CeresFactorPtr> factors);

    void removeNode(CeresNodePtr node);

    void removeFactor(CeresFactorPtr factor);

    bool solve(bool verbose=false);

    size_t num_nodes() { return nodes_.size(); }
    size_t num_factors() { return factors_.size(); }

    bool setModified() { modified_ = true; }

    bool modified() { return modified_; }

    bool setNodeConstant(CeresNodePtr node);

    template <typename NodeType=CeresNode>
    boost::shared_ptr<NodeType> getNode(Symbol sym) const;

    bool computeMarginalCovariance(const std::vector<Key>& keys);
    bool computeMarginalCovariance(const std::vector<CeresNodePtr>& nodes);

    Eigen::MatrixXd getMarginalCovariance(const Key& key) const;
    Eigen::MatrixXd getMarginalCovariance(const Key& key1, const Key& key2) const;
    Eigen::MatrixXd getMarginalCovariance(CeresNodePtr node) const;
    Eigen::MatrixXd getMarginalCovariance(CeresNodePtr node1, CeresNodePtr node2) const;

    CeresNodePtr findLastNodeBeforeTime(unsigned char symbol_chr, ros::Time time);
    CeresNodePtr findFirstNodeAfterTime(unsigned char symbol_chr, ros::Time time);
    CeresNodePtr findNearestNode(unsigned char symbol_chr, ros::Time time);

    template <typename NodeType=CeresNode>
    boost::shared_ptr<NodeType> findLastNode(unsigned char symbol_chr);

    std::vector<Key> keys();

    const ceres::Problem& problem() const { return *problem_; }

private:
    boost::shared_ptr<ceres::Problem> problem_;

    std::unordered_map<Key, CeresNodePtr> nodes_;
    std::vector<CeresFactorPtr> factors_;

    bool modified_; //< Whether or not the graph has been modified since the last solving

    ceres::Solver::Options solver_options_;

    ceres::Covariance::Options covariance_options_;
    boost::shared_ptr<ceres::Covariance> covariance_;

    std::mutex mutex_;
};

template <typename NodeType>
boost::shared_ptr<NodeType>
FactorGraph::getNode(Symbol sym) const
{
    auto node = nodes_.find(sym.key());

    if (node == nodes_.end()) return nullptr;
    else if (node->second) return boost::dynamic_pointer_cast<NodeType>(node->second);
    else return nullptr;

    // for (auto& key_node : nodes_) {
    //     if (key_node.second->symbol() == sym) {
    //         return boost::dynamic_pointer_cast<NodeType>(key_node.second);
    //     }
    // }
    // return nullptr;
}

template <typename NodeType>
boost::shared_ptr<NodeType> 
FactorGraph::findLastNode(unsigned char symbol_chr)
{
    CeresNodePtr result = nullptr;

    ros::Time last_time = ros::Time(0);

    for (auto& key_node : nodes_) {
        if (key_node.second->chr() != symbol_chr) continue;

        if (!key_node.second->time()) continue;

        if (key_node.second->time() > last_time) {
            last_time = *key_node.second->time();
            result = key_node.second;
        }
    }

    return boost::dynamic_pointer_cast<NodeType>(result);
}
