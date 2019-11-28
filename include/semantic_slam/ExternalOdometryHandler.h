#pragma once

#include "semantic_slam/Common.h"
#include "semantic_slam/OdometryHandler.h"
#include "semantic_slam/pose_math.h"
#include "semantic_slam/SemanticKeyframe.h"

#include <nav_msgs/Odometry.h>
#include <mutex>
#include <deque>
#include <unordered_map>
// #include <gtsam/geometry/Pose3.h>

class ExternalOdometryHandler : public OdometryHandler
{
public:
    void setup();

    void msgCallback(const nav_msgs::Odometry::ConstPtr& msg);

    SemanticKeyframe::Ptr originKeyframe(ros::Time time);

    SemanticKeyframe::Ptr createKeyframe(ros::Time time);

    SemanticKeyframe::Ptr findNearestKeyframe(ros::Time time);

    bool getRelativePoseEstimate(ros::Time t1, ros::Time t2, Pose3& T12);

    // bool getRelativePoseJacobianEstimate(ros::Time t1, ros::Time t2, Eigen::MatrixXd& J);

    // inherit constructor
    using OdometryHandler::OdometryHandler;

private:
    ros::Subscriber subscriber_;

	std::deque<nav_msgs::Odometry> msg_queue_;

    std::vector<SemanticKeyframe::Ptr> keyframes_;

    std::mutex mutex_;

    Pose3 last_odom_;
    ros::Time last_time_;
    size_t last_keyframe_index_;
    
    size_t received_msgs_;
    size_t last_msg_seq_;

    unsigned char node_chr_;
    ros::Duration max_node_period_;

    Eigen::MatrixXd extractOdometryCovariance(const nav_msgs::Odometry& msg) const;

    Pose3 msgToPose3(const nav_msgs::Odometry& msg) const;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

};
