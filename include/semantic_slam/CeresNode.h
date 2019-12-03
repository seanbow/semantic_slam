#pragma once 

#include "semantic_slam/Common.h"
#include "semantic_slam/Symbol.h"

#include <ceres/ceres.h>
// #include <gtsam/inference/Symbol.h>
#include <ros/ros.h>

class CeresNode {
public:

    virtual ~CeresNode() { }

    virtual void addToProblem(boost::shared_ptr<ceres::Problem> problem) = 0;
    virtual void removeFromProblem(boost::shared_ptr<ceres::Problem> problem);

    Symbol symbol() const { return Symbol(key_); }
    unsigned char chr() const { return Symbol(key_).chr(); }
    size_t index() const { return Symbol(key_).index(); }
    Key key() const { return key_; }

    virtual size_t dim() const = 0;
    virtual size_t local_dim() const = 0;

    bool active() const { return active_; }

    boost::optional<ros::Time> time() const { return time_; }

    const std::vector<double*>& parameter_blocks() const { return parameter_blocks_; }
    const std::vector<size_t>& parameter_block_sizes() const { return parameter_block_sizes_; }
    const std::vector<size_t>& parameter_block_local_sizes() const { return parameter_block_local_sizes_; }
    const std::vector<ceres::LocalParameterization*> local_parameterizations() const { return local_parameterizations_; }

    using Ptr = boost::shared_ptr<CeresNode>;

protected:

    CeresNode(Key key, boost::optional<ros::Time> time=boost::none);

    std::vector<double*> parameter_blocks_;
    std::vector<size_t> parameter_block_sizes_;
    std::vector<size_t> parameter_block_local_sizes_;
    std::vector<ceres::LocalParameterization*> local_parameterizations_;

    bool active_;

    Key key_;
    boost::optional<ros::Time> time_;
};

using CeresNodePtr = CeresNode::Ptr;

CeresNode::CeresNode(Key key, boost::optional<ros::Time> time)
    : key_(key),
      time_(time),
      active_(false)
{
    
}

void CeresNode::removeFromProblem(boost::shared_ptr<ceres::Problem> problem)
{
    for (double* block : parameter_blocks_) {
        problem->RemoveParameterBlock(block);
    }

    active_ = false;
}