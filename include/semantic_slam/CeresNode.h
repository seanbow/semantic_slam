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

    void addToOrderingGroup(
      std::shared_ptr<ceres::ParameterBlockOrdering> ordering,
      int group) const;

    virtual boost::shared_ptr<CeresNode> clone() const = 0;

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
