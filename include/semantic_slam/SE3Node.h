#pragma once

#include "semantic_slam/CeresNode.h"
#include "semantic_slam/Common.h"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include "semantic_slam/ceres_quaternion_parameterization.h"
#include "semantic_slam/pose_math.h"

#include <gtsam/base/GenericValue.h>
#include <gtsam/geometry/Pose3.h>

class SE3Node : public CeresNode
{
  public:
    SE3Node(Symbol sym, boost::optional<ros::Time> time = boost::none);

    size_t dim() const { return 7; }
    size_t local_dim() const { return 6; }

    const Pose3& pose() const { return pose_; }
    Pose3& pose() { return pose_; }

    const Eigen::Quaterniond& rotation() const { return pose_.rotation(); }
    Eigen::Quaterniond& rotation() { return pose_.rotation(); }

    const Eigen::Vector3d& translation() const { return pose_.translation(); }
    Eigen::Vector3d& translation() { return pose_.translation(); }

    boost::shared_ptr<gtsam::Value> getGtsamValue() const;

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

    // parameter_blocks_.push_back(pose_.rotation_data());
    // parameter_block_sizes_.push_back(4);
    // parameter_block_local_sizes_.push_back(3);
    // local_parameterizations_.push_back(new QuaternionLocalParameterization);

    // parameter_blocks_.push_back(pose_.translation_data());
    // parameter_block_sizes_.push_back(3);
    // parameter_block_local_sizes_.push_back(3);
    // local_parameterizations_.push_back(nullptr);

    addParameterBlock(
      pose_.rotation_data(), 4, new QuaternionLocalParameterization);
    addParameterBlock(pose_.translation_data(), 3);
}

boost::shared_ptr<gtsam::Value>
SE3Node::getGtsamValue() const
{
    return util::allocate_aligned<gtsam::GenericValue<gtsam::Pose3>>(
      gtsam::Pose3(gtsam::Rot3(rotation()), translation()));
}
