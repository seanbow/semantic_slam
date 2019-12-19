#pragma once

#include "semantic_slam/Common.h"

#include "semantic_slam/CeresNode.h"
#include "semantic_slam/LocalParameterizations.h"
#include "semantic_slam/Pose3.h"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

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

    boost::shared_ptr<CeresNode> clone() const;

    const Eigen::Map<Eigen::Quaterniond>& rotation() const
    {
        return pose_.rotation();
    }
    Eigen::Map<Eigen::Quaterniond>& rotation() { return pose_.rotation(); }

    const Eigen::Map<Eigen::Vector3d>& translation() const
    {
        return pose_.translation();
    }
    Eigen::Map<Eigen::Vector3d>& translation() { return pose_.translation(); }

    boost::shared_ptr<gtsam::Value> getGtsamValue() const;

    static ceres::LocalParameterization* Parameterization()
    {
        return new SE3LocalParameterization;
    }

    using Ptr = boost::shared_ptr<SE3Node>;

  private:
    Pose3 pose_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

using SE3NodePtr = SE3Node::Ptr;

inline
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

    addParameterBlock(pose_.data(), 7, SE3Node::Parameterization());
}

inline
boost::shared_ptr<CeresNode>
SE3Node::clone() const
{
    auto node = util::allocate_aligned<SE3Node>(symbol(), time());
    node->pose() = pose_;
    return node;
}

inline
boost::shared_ptr<gtsam::Value>
SE3Node::getGtsamValue() const
{
    return util::allocate_aligned<gtsam::GenericValue<gtsam::Pose3>>(
      gtsam::Pose3(gtsam::Rot3(rotation()), translation()));
}
