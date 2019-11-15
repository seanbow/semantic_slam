#pragma once

#include "semantic_slam/Common.h"
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/CeresFactor.h"
#include "semantic_slam/keypoints/EstimatedObject.h"

#include <ros/ros.h>
#include <vector>
#include <eigen3/Eigen/Core>

class SemanticKeyframe {
public:
    SemanticKeyframe(Key key, ros::Time time);

    Key key() const { return key_; }
    int index() const { return Symbol(key_).index(); }
    unsigned char chr() const { return Symbol(key_).chr(); }

    ros::Time time() const { return time_; }

    Pose3& pose() { return pose_; }
    const Pose3& pose() const { return pose_; }

    Eigen::MatrixXd& covariance() { return pose_covariance_; }
    const Eigen::MatrixXd& covariance() const { return pose_covariance_; }

    Pose3& odometry() { return odometry_; }
    const Pose3& odometry() const { return odometry_; }

    CeresFactorPtr& spine_factor() { return spine_factor_; }
    const CeresFactorPtr& spine_factor() const { return spine_factor_; }

    SE3NodePtr& graph_node() { return graph_node_; }
    const SE3NodePtr& graph_node() const { return graph_node_; }

    aligned_vector<ObjectMeasurement> measurements;

    // std::vector<ObjectMeasurement>& measurements() { return measurements_; }
    std::vector<EstimatedObject::Ptr>& visible_objects() { return visible_objects_; }

    using Ptr = boost::shared_ptr<SemanticKeyframe>;

private:
    Key key_;
    ros::Time time_;

    Pose3 odometry_;
    Pose3 pose_;
    Eigen::MatrixXd pose_covariance_;

    SE3NodePtr graph_node_;

    CeresFactorPtr spine_factor_;

    // aligned_vector<ObjectMeasurement> measurements_;
    std::vector<EstimatedObject::Ptr> visible_objects_;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

SemanticKeyframe::SemanticKeyframe(Key key, ros::Time time)
    : key_(key),
      time_(time)
{
    graph_node_ = util::allocate_aligned<SE3Node>(key, time);
    pose_covariance_ = Eigen::MatrixXd::Zero(6,6);
}