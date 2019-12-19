
#include "semantic_slam/CeresFactor.h"

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