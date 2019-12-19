
#include "semantic_slam/CeresNode.h"

CeresNode::CeresNode(Key key, boost::optional<ros::Time> time)
  : active_(false)
  , key_(key)
  , time_(time)
{}

void
CeresNode::addParameterBlock(double* values,
                             int size,
                             ceres::LocalParameterization* local_param)
{
    parameter_blocks_.push_back(values);
    parameter_block_sizes_.push_back(size);
    local_parameterizations_.push_back(local_param);

    if (local_param) {
        parameter_block_local_sizes_.push_back(local_param->LocalSize());
    } else {
        parameter_block_local_sizes_.push_back(size);
    }
}

void
CeresNode::addToProblem(boost::shared_ptr<ceres::Problem> problem)
{
    for (size_t i = 0; i < parameter_blocks_.size(); ++i) {
        problem->AddParameterBlock(parameter_blocks_[i],
                                   parameter_block_sizes_[i],
                                   local_parameterizations_[i]);
    }

    active_problems_.push_back(problem.get());

    active_ = true;
}

void
CeresNode::removeFromProblem(boost::shared_ptr<ceres::Problem> problem)
{
    auto it = std::find(
      active_problems_.begin(), active_problems_.end(), problem.get());

    if (it != active_problems_.end()) {
        for (double* block : parameter_blocks_) {
            problem->RemoveParameterBlock(block);
        }

        active_problems_.erase(it);
    }

    active_ = !active_problems_.empty();
}

void
CeresNode::addToOrderingGroup(
  std::shared_ptr<ceres::ParameterBlockOrdering> ordering,
  int group) const
{
    for (double* block : parameter_blocks_) {
        ordering->AddElementToGroup(block, group);
    }
}