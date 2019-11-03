#pragma once

#include "semantic_slam/CeresNode.h"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include "semantic_slam/ceres_quaternion_parameterization.h"
#include "semantic_slam/pose_math.h"


class SE3Node : public CeresNode
{
public:
    SE3Node(gtsam::Symbol sym, boost::optional<ros::Time> time=boost::none);

    void addToProblem(boost::shared_ptr<ceres::Problem> problem);

    const Pose3& pose() const { return pose_; }
    Pose3& pose() { return pose_; }

    const math::Quaterniond& rotation() const { return pose_.rotation(); }
    math::Quaterniond& rotation() { return pose_.rotation(); }

    const Eigen::Vector3d& translation() const { return pose_.translation(); }
    Eigen::Vector3d& translation() { return pose_.translation(); }

    using Ptr = boost::shared_ptr<SE3Node>;

private:
    Pose3 pose_;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

using SE3NodePtr = SE3Node::Ptr;

SE3Node::SE3Node(gtsam::Symbol sym, boost::optional<ros::Time> time)
    : CeresNode(sym, time)
{
}

void
SE3Node::addToProblem(boost::shared_ptr<ceres::Problem> problem)
{
    problem->AddParameterBlock(pose_.rotation_data(), 4);

    // ceres::Problem takes ownership of the new parameterization
    problem->SetParameterization(pose_.rotation_data(), new QuaternionLocalParameterization);

    problem->AddParameterBlock(pose_.translation_data(), 3);
}