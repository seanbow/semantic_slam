#pragma once

#include "semantic_slam/Common.h"
#include "semantic_slam/CeresNode.h"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include "semantic_slam/ceres_quaternion_parameterization.h"
#include "semantic_slam/pose_math.h"


class SE3Node : public CeresNode
{
public:
    SE3Node(Symbol sym, boost::optional<ros::Time> time=boost::none);

    void addToProblem(boost::shared_ptr<ceres::Problem> problem);

    void setParametersConstant(boost::shared_ptr<ceres::Problem> problem);

    size_t dim() const { return 7; }
    size_t local_dim() const { return 6; }

    const Pose3& pose() const { return pose_; }
    Pose3& pose() { return pose_; }

    const Eigen::Quaterniond& rotation() const { return pose_.rotation(); }
    Eigen::Quaterniond& rotation() { return pose_.rotation(); }

    const Eigen::Vector3d& translation() const { return pose_.translation(); }
    Eigen::Vector3d& translation() { return pose_.translation(); }

    using Ptr = boost::shared_ptr<SE3Node>;

private:
    Pose3 pose_;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

using SE3NodePtr = SE3Node::Ptr;

SE3Node::SE3Node(Symbol sym, boost::optional<ros::Time> time)
    : CeresNode(sym, time)
{
    pose_ = Pose3::Identity();
    
    parameter_blocks_.push_back(pose_.rotation_data());
    parameter_block_sizes_.push_back(4);
    parameter_block_local_sizes_.push_back(3);
    local_parameterizations_.push_back(new QuaternionLocalParameterization);
    // local_parameterizations_.push_back(new ceres::EigenQuaternionParameterization);

    parameter_blocks_.push_back(pose_.translation_data());
    parameter_block_sizes_.push_back(3);
    parameter_block_local_sizes_.push_back(3);
    local_parameterizations_.push_back(nullptr);
}

void
SE3Node::addToProblem(boost::shared_ptr<ceres::Problem> problem)
{
    problem->AddParameterBlock(pose_.rotation_data(), 4);

    // ceres::Problem takes ownership of the new parameterization
    problem->SetParameterization(pose_.rotation_data(), local_parameterizations_[0]);
    // problem->SetParameterization(pose_.rotation_data(), new ceres::EigenQuaternionParameterization);

    problem->AddParameterBlock(pose_.translation_data(), 3);

    active_ = true;
}
