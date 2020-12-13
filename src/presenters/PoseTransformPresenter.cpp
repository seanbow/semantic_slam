#include "semantic_slam/presenters/PoseTransformPresenter.h"
#include "semantic_slam/Pose3.h"
#include "semantic_slam/SemanticKeyframe.h"
#include "semantic_slam/keypoints/EstimatedObject.h"

#include <geometry_msgs/TransformStamped.h>
#include <thread>

void
PoseTransformPresenter::setup()
{
    std::thread publish_thread(&PoseTransformPresenter::publishFunction,
                               this);

    publish_thread.detach();
}

void
PoseTransformPresenter::publishFunction()
{
    ros::Rate publish_rate(10); // parameter is in Hz

    while (ros::ok()) {
        geometry_msgs::TransformStamped transform;

        // transform.header.stamp = keyframe->time();
        transform.header.stamp = ros::Time::now();
        transform.header.frame_id = "odom";
        transform.child_frame_id = "base_link";

        // std::lock_guard<std::mutex> lock(transform_mutex_);

        transform.transform.translation.x = odom_T_pose_.translation()(0);
        transform.transform.translation.y = odom_T_pose_.translation()(1);
        transform.transform.translation.z = odom_T_pose_.translation()(2);

        transform.transform.rotation.x = odom_T_pose_.rotation().x();
        transform.transform.rotation.y = odom_T_pose_.rotation().y();
        transform.transform.rotation.z = odom_T_pose_.rotation().z();
        transform.transform.rotation.w = odom_T_pose_.rotation().w();

        broadcaster_.sendTransform(transform);

        publish_rate.sleep();
    }
}

void
PoseTransformPresenter::present(
  const std::vector<SemanticKeyframe::Ptr>& keyframes,
  const std::vector<EstimatedObject::Ptr>& objects)
{
    if (keyframes.empty())
        return;

    // Publish the last (most recent) transform
    // Assume that the keyframes are ordered
    auto keyframe = keyframes.back();

    odom_T_pose_ = keyframe->odometry();
}
