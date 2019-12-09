#pragma once

#include "semantic_slam/Common.h"
#include "semantic_slam/Symbol.h"

#include <ceres/ceres.h>
// #include <gtsam/inference/Symbol.h>
#include <ros/ros.h>

// #include <gtsam/base/Value.h>

namespace gtsam {
class Value;
}

class CeresNode
{
  public:
    virtual ~CeresNode() {}

    virtual void addToProblem(boost::shared_ptr<ceres::Problem> problem);
    virtual void removeFromProblem(boost::shared_ptr<ceres::Problem> problem);

    Symbol symbol() const { return Symbol(key_); }
    unsigned char chr() const { return Symbol(key_).chr(); }
    size_t index() const { return Symbol(key_).index(); }
    Key key() const { return key_; }

    virtual size_t dim() const = 0;
    virtual size_t local_dim() const = 0;

    virtual boost::shared_ptr<gtsam::Value> getGtsamValue() const
    {
        throw std::logic_error("unimplemented");
    }

    bool active() const { return active_; }

    boost::optional<ros::Time> time() const { return time_; }

    void addParameterBlock(double* values,
                           int size,
                           ceres::LocalParameterization* local_param = nullptr);

    const std::vector<double*>& parameter_blocks() const
    {
        return parameter_blocks_;
    }
    const std::vector<size_t>& parameter_block_sizes() const
    {
        return parameter_block_sizes_;
    }
    const std::vector<size_t>& parameter_block_local_sizes() const
    {
        return parameter_block_local_sizes_;
    }
    const std::vector<ceres::LocalParameterization*> local_parameterizations()
      const
    {
        return local_parameterizations_;
    }

    using Ptr = boost::shared_ptr<CeresNode>;

  protected:
    CeresNode(Key key, boost::optional<ros::Time> time = boost::none);

    std::vector<double*> parameter_blocks_;
    std::vector<size_t> parameter_block_sizes_;
    std::vector<size_t> parameter_block_local_sizes_;
    std::vector<ceres::LocalParameterization*> local_parameterizations_;

    std::vector<ceres::Problem*> active_problems_;

    bool active_;

    Key key_;
    boost::optional<ros::Time> time_;
};

using CeresNodePtr = CeresNode::Ptr;

CeresNode::CeresNode(Key key, boost::optional<ros::Time> time)
  : key_(key)
  , time_(time)
  , active_(false)
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
    for (int i = 0; i < parameter_blocks_.size(); ++i) {
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