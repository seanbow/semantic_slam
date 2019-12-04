#pragma once

#include "semantic_slam/Common.h"
#include "semantic_slam/registry.h"

#include <ceres/ceres.h>

namespace gtsam {
class NonlinearFactor;
class NonlinearFactorGraph;
}

class CeresNode;

class CeresFactor
{
public:
    CeresFactor(FactorType type, int tag=0);

    virtual void addToProblem(boost::shared_ptr<ceres::Problem> problem) = 0;
    virtual void removeFromProblem(boost::shared_ptr<ceres::Problem> problem) = 0;

    FactorType type() const { return type_; }
    int tag() const { return tag_; }

    bool active() const { return active_; }

    const std::vector<boost::shared_ptr<CeresNode>>& nodes() const { return nodes_; }

    virtual boost::shared_ptr<gtsam::NonlinearFactor> getGtsamFactor() const { 
        throw std::logic_error("unimplemented");
    }

    virtual void addToGtsamGraph(boost::shared_ptr<gtsam::NonlinearFactorGraph> graph) const { 
        throw std::logic_error("unimplemented");
    }

    virtual bool operator==(const CeresFactor& other) const;

protected:
    FactorType type_;
    int tag_;

    bool active_;

    std::vector<boost::shared_ptr<CeresNode>> nodes_;

    ceres::CostFunction* cf_;
    ceres::ResidualBlockId residual_id_;

public:
    using Ptr = boost::shared_ptr<CeresFactor>;
    using ConstPtr = boost::shared_ptr<const CeresFactor>;
};

using CeresFactorPtr = CeresFactor::Ptr;
using CeresFactorConstPtr = CeresFactor::ConstPtr;

CeresFactor::CeresFactor(FactorType type, int tag)
    : type_(type),
      tag_(tag),
      active_(false),
      residual_id_(NULL)
{

}

bool CeresFactor::operator==(const CeresFactor& other) const
{  
    // At its core this class is a wrapper around ceres::CostFunction objects, so
    // equality can be assumed based on the underlying cost function equality...

    return this->cf_ == other.cf_;
}