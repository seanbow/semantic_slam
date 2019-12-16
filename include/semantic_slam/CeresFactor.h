#pragma once

#include "semantic_slam/CeresNode.h"
#include "semantic_slam/Common.h"
#include "semantic_slam/registry.h"

#include <ceres/ceres.h>

#include <unordered_map>

namespace gtsam {
class NonlinearFactor;
class NonlinearFactorGraph;
}

class CeresFactor
{
  public:
    CeresFactor(FactorType type, int tag = 0);

    virtual void addToProblem(boost::shared_ptr<ceres::Problem> problem) = 0;
    virtual void removeFromProblem(boost::shared_ptr<ceres::Problem> problem);

    FactorType type() const { return type_; }
    int tag() const { return tag_; }

    bool active() const { return active_; }

    virtual void setNodes(std::vector<CeresNodePtr> new_nodes)
    {
        nodes_ = new_nodes;
    }

    std::vector<Key> keys() const;

    // Returns a cloned copy of this factor, except where all
    // nodes are set to NULL. must set the nodes prior to use.
    virtual boost::shared_ptr<CeresFactor> clone() const = 0;

    const std::vector<boost::shared_ptr<CeresNode>>& nodes() const
    {
        return nodes_;
    }

    virtual boost::shared_ptr<gtsam::NonlinearFactor> getGtsamFactor() const
    {
        throw std::logic_error("unimplemented");
    }

    virtual void addToGtsamGraph(
      boost::shared_ptr<gtsam::NonlinearFactorGraph> graph) const
    {
        throw std::logic_error("unimplemented");
    }

    virtual bool operator==(const CeresFactor& other) const;

  protected:
    FactorType type_;
    int tag_;

    bool active_;

    std::vector<boost::shared_ptr<CeresNode>> nodes_;

    ceres::CostFunction* cf_;

    // We may end up getting added to multiple problems, so we need to keep
    // track of our ID in each
    std::unordered_map<ceres::Problem*, ceres::ResidualBlockId> residual_ids_;

  public:
    using Ptr = boost::shared_ptr<CeresFactor>;
    using ConstPtr = boost::shared_ptr<const CeresFactor>;
};

using CeresFactorPtr = CeresFactor::Ptr;
using CeresFactorConstPtr = CeresFactor::ConstPtr;

CeresFactor::CeresFactor(FactorType type, int tag)
  : type_(type)
  , tag_(tag)
  , active_(false)
{}

bool
CeresFactor::operator==(const CeresFactor& other) const
{
    // At its core this class is a wrapper around ceres::CostFunction objects,
    // so equality can be assumed based on the underlying cost function
    // equality...

    return this->cf_ == other.cf_;
}

void
CeresFactor::removeFromProblem(boost::shared_ptr<ceres::Problem> problem)
{
    auto it = residual_ids_.find(problem.get());
    if (it != residual_ids_.end()) {
        problem->RemoveResidualBlock(it->second);
        residual_ids_.erase(it);
    }

    active_ = !residual_ids_.empty();
}

std::vector<Key>
CeresFactor::keys() const
{
    std::vector<Key> keys;
    for (const auto& node : nodes_) {
        keys.push_back(node->key());
    }
    return keys;
}