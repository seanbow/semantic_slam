#include "semantic_slam/presenters/OdometryTransformPresenter.h"
#include "semantic_slam/Pose3.h"
#include "semantic_slam/SemanticKeyframe.h"
#include "semantic_slam/keypoints/EstimatedObject.h"

#include <geometry_msgs/TransformStamped.h>
#include <thread>

void
OdometryTransformPresenter::setup()
{
    std::thread publish_thread(&OdometryTransformPresenter::publishFunction,
                               this);

    publish_thread.detach();
}

void
OdometryTransformPresenter::publishFunction()
{
    ros::Rate publish_rate(1); // parameter is in Hz

    while (ros::ok()) {
        geometry_msgs::TransformStamped transform;

        // transform.header.stamp = keyframe->time();
        transform.header.stamp = ros::Time::now();
        transform.header.frame_id = "map";
        transform.child_frame_id = "odom";

        // std::lock_guard<std::mutex> lock(transform_mutex_);

        transform.transform.translation.x = map_T_odom_.translation()(0);
        transform.transform.translation.y = map_T_odom_.translation()(1);
        transform.transform.translation.z = map_T_odom_.translation()(2);

        transform.transform.rotation.x = map_T_odom_.rotation().x();
        transform.transform.rotation.y = map_T_odom_.rotation().y();
        transform.transform.rotation.z = map_T_odom_.rotation().z();
        transform.transform.rotation.w = map_T_odom_.rotation().w();

        broadcaster_.sendTransform(transform);

        publish_rate.sleep();
    }
}

void
OdometryTransformPresenter::present(
  const std::vector<SemanticKeyframe::Ptr>& keyframes,
  const std::vector<EstimatedObject::Ptr>& objects)
{
    if (keyframes.empty())
        return;

    // Publish the last (most recent) transform
    // Assume that the keyframes are ordered
    auto keyframe = keyframes.back();

    Pose3 map_T_pose = keyframe->pose();
    Pose3 odom_T_pose = keyframe->odometry();

    // cannot afford the time it takes to wait on a mutex here...?
    // std::lock_guard<std::mutex> lock(transform_mutex_);
    map_T_odom_ = map_T_pose * odom_T_pose.inverse();
}
