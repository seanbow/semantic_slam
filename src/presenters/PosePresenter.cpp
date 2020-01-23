#include "semantic_slam/presenters/PosePresenter.h"
#include "semantic_slam/Pose3.h"
#include "semantic_slam/SemanticKeyframe.h"
#include "semantic_slam/keypoints/EstimatedObject.h"

#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Odometry.h>

void
PosePresenter::setup()
{
    pub_pose_ =
      nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("pose", 10);
    
    pub_odom_ =
      nh_.advertise<nav_msgs::Odometry>("semslam/robot_odom", 10, true);
}

void
PosePresenter::present(const std::vector<SemanticKeyframe::Ptr>& keyframes,
                       const std::vector<EstimatedObject::Ptr>& objects)
{
    if (keyframes.empty())
        return;

    // Publish the last (most recent) pose
    // Assume that the keyframes are ordered
    auto keyframe = keyframes.back();

    Pose3 pose = keyframe->pose();

    Eigen::MatrixXd cov = keyframe->covariance();

    Eigen::Quaterniond q = pose.rotation();
    Eigen::Vector3d p = pose.translation();

    double Z_SCALE = 1.0;

    geometry_msgs::PoseWithCovarianceStampedPtr msg_pose(
      new geometry_msgs::PoseWithCovarianceStamped);
    nav_msgs::OdometryPtr msg_odom(new nav_msgs::Odometry);
    msg_pose->header.frame_id = "/map";
    msg_pose->header.stamp = keyframe->time();
    msg_pose->pose.pose.position.x = p(0);
    msg_pose->pose.pose.position.y = p(1);
    msg_pose->pose.pose.position.z = Z_SCALE * p(2);
    msg_pose->pose.pose.orientation.x = q.x();
    msg_pose->pose.pose.orientation.y = q.y();
    msg_pose->pose.pose.orientation.z = q.z();
    msg_pose->pose.pose.orientation.w = q.w();

    // note: ROS expects covariance of the form
    // [ Ppp Ppq ]
    // [ Pqp Pqq ]
    //
    // but GTSAM provides it as
    // [ Pqq Pqp ]
    // [ Ppq Ppp ]
    //
    // Make a new matrix in the right format
    Eigen::Matrix<double, 6, 6> P2;

    P2.block<3, 3>(0, 0) = cov.block<3, 3>(3, 3);
    P2.block<3, 3>(0, 3) = cov.block<3, 3>(3, 0);
    P2.block<3, 3>(3, 0) = cov.block<3, 3>(0, 3);
    P2.block<3, 3>(3, 3) = cov.block<3, 3>(0, 0);

    // ROS_INFO_STREAM("Covariance: \n" << P2);

    eigenToBoostArray<6, 6>(P2, msg_pose->pose.covariance);

    msg_odom->header.frame_id = "/map";
    msg_odom->header.stamp = keyframe->time();
    msg_odom->pose.pose = msg_pose->pose.pose;
    msg_odom->pose.covariance = msg_pose->pose.covariance;

    pub_pose_.publish(msg_pose);
    pub_odom_.publish(msg_odom);
}
