#pragma once

#include "semantic_slam/Common.h"
#include "semantic_slam/registry.h"

class CeresFactor
{
public:
    CeresFactor(FactorType type, int tag=0);

    virtual void addToProblem(boost::shared_ptr<ceres::Problem> problem) = 0;

private:
    FactorType type_;
    int tag_;


public:
    using Ptr = boost::shared_ptr<CeresFactor>;
    using ConstPtr = boost::shared_ptr<const CeresFactor>;
};

using CeresFactorPtr = CeresFactor::Ptr;
using CeresFactorConstPtr = CeresFactor::ConstPtr;

CeresFactor::CeresFactor(FactorType type, int tag)
    : type_(type),
      tag_(tag)
{

}