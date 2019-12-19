#pragma once

#include "semantic_slam/Common.h"

#include <ceres/ceres.h>

#include "semantic_slam/CeresFactor.h"
#include "semantic_slam/CeresNode.h"

#include <mutex>
#include <rosfmt/rosfmt.h>
#include <unordered_map>

namespace gtsam {
class NonlinearFactorGraph;
class Values;
}

class IterationCallbackWrapper;

class FactorGraph
{
  public:
    FactorGraph();

    void addNode(CeresNodePtr node);

    void addFactor(CeresFactorPtr factor);
    void addFactors(std::vector<CeresFactorPtr> factors);

    void removeNode(CeresNodePtr node);

    void removeFactor(CeresFactorPtr factor);

    bool containsNode(CeresNodePtr node);
    bool containsNode(Key key);
    bool containsFactor(CeresFactorPtr factor);

    bool solve(bool verbose = false,
               boost::optional<ceres::Solver::Summary&> summary = boost::none);

    // Returns an identical copy of this factor graph except
    // that it operates on a new cloned set of nodes
    boost::shared_ptr<FactorGraph> clone() const;

    size_t num_nodes() { return nodes_.size(); }
    size_t num_factors() { return factors_.size(); }

    const std::unordered_map<Key, CeresNodePtr>& nodes() const
    {
        return nodes_;
    }

    void setModified() { modified_ = true; }

    bool modified() { return modified_; }

    bool setNodeConstant(Key key);
    bool setNodeConstant(CeresNodePtr node);
    bool setNodeVariable(Key key);
    bool setNodeVariable(CeresNodePtr node);

    bool isNodeConstant(CeresNodePtr node) const;

    template<typename NodeType = CeresNode>
    boost::shared_ptr<NodeType> getNode(Symbol sym) const;

    bool computeMarginalCovariance(const std::vector<Key>& keys);
    bool computeMarginalCovariance(const std::vector<CeresNodePtr>& nodes);

    void setSolverOptions(ceres::Solver::Options opts);

    ceres::Solver::Options& solver_options() { return solver_options_; }

    void setNumThreads(int n_threads);

    // Adds a callback function that is called at each iteration
    // of the optimization once solve() is called
    using IterationCallbackType =
      std::function<ceres::CallbackReturnType(ceres::IterationSummary)>;
    void addIterationCallback(IterationCallbackType callback);

    Eigen::MatrixXd getMarginalCovariance(const Key& key) const;
    Eigen::MatrixXd getMarginalCovariance(const Key& key1,
                                          const Key& key2) const;
    // Eigen::MatrixXd getMarginalCovariance(CeresNodePtr node) const;
    Eigen::MatrixXd getMarginalCovariance(
      const std::vector<CeresNodePtr>& nodes) const;

    CeresNodePtr findLastNodeBeforeTime(unsigned char symbol_chr,
                                        ros::Time time);
    CeresNodePtr findFirstNodeAfterTime(unsigned char symbol_chr,
                                        ros::Time time);
    CeresNodePtr findNearestNode(unsigned char symbol_chr, ros::Time time);

    template<typename NodeType = CeresNode>
    boost::shared_ptr<NodeType> findLastNode(unsigned char symbol_chr);

    std::vector<Key> keys();

    const ceres::Problem& problem() const { return *problem_; }

    boost::shared_ptr<gtsam::NonlinearFactorGraph> getGtsamGraph() const;
    boost::shared_ptr<gtsam::Values> getGtsamValues() const;

  private:
    boost::shared_ptr<ceres::Problem> problem_;

    std::unordered_map<Key, CeresNodePtr> nodes_;
    std::vector<CeresFactorPtr> factors_;

    bool modified_; //< Whether or not the graph has been modified since the
                    // last solving

    ceres::Solver::Options solver_options_;

    ceres::Covariance::Options covariance_options_;
    boost::shared_ptr<ceres::Covariance> covariance_;

    mutable std::mutex mutex_;

    std::vector<boost::shared_ptr<IterationCallbackWrapper>>
      iteration_callbacks_;

    void addFactorInternal(CeresFactorPtr factor);
};

template<typename NodeType>
boost::shared_ptr<NodeType>
FactorGraph::getNode(Symbol sym) const
{
    auto node = nodes_.find(sym.key());

    if (node == nodes_.end())
        return nullptr;
    else if (node->second)
        return boost::dynamic_pointer_cast<NodeType>(node->second);
    else
        return nullptr;

    // for (auto& key_node : nodes_) {
    //     if (key_node.second->symbol() == sym) {
    //         return boost::dynamic_pointer_cast<NodeType>(key_node.second);
    //     }
    // }
    // return nullptr;
}

template<typename NodeType>
boost::shared_ptr<NodeType>
FactorGraph::findLastNode(unsigned char symbol_chr)
{
    CeresNodePtr result = nullptr;

    size_t last_index = 0;

    for (auto& key_node : nodes_) {
        if (key_node.second->chr() != symbol_chr)
            continue;

        if (key_node.second->index() > last_index) {
            last_index = key_node.second->index();
            result = key_node.second;
        }
    }

    return boost::dynamic_pointer_cast<NodeType>(result);
}

// Helper class to support passing in simple lambda functions etc
// for iteration callbacks. Ceres needs them to be derived from a specific class
// rather than being a generic functor
class IterationCallbackWrapper : public ceres::IterationCallback
{
  public:
    IterationCallbackWrapper(
      const FactorGraph::IterationCallbackType& callback_function)
      : callback_fn_(callback_function)
    {}

    ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary)
    {
        return callback_fn_(summary);
    }

  private:
    FactorGraph::IterationCallbackType callback_fn_;
};
