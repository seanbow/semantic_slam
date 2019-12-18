
#include "semantic_slam/presenters/ObjectKeypointPresenter.h"
#include "semantic_slam/SemanticKeyframe.h"
#include "semantic_slam/keypoints/EstimatedKeypoint.h"
#include "semantic_slam/keypoints/EstimatedObject.h"

#include <visualization_msgs/MarkerArray.h>

void
ObjectKeypointPresenter::setup()
{
    pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
      "keypoint_objects/object_keypoint_markers", 10);

    if (!pnh_.getParam("object_keypoint_scale", scale_)) {
        ROS_ERROR(
          "[ObjectKeypointPresenter] Unable to read visualization parameters");
        scale_ = 0.5;
    }
}

void
ObjectKeypointPresenter::present(
  const std::vector<SemanticKeyframe::Ptr>& keyframes,
  const std::vector<EstimatedObject::Ptr>& objects)
{
    if (objects.empty())
        return;

    visualization_msgs::MarkerArray object_markers;
    visualization_msgs::Marker object_marker;
    object_marker.type = visualization_msgs::Marker::SPHERE;

    object_marker.header.frame_id = "map";
    object_marker.header.stamp = ros::Time::now();

    object_marker.scale.x = scale_;
    object_marker.scale.y = scale_;
    object_marker.scale.z = scale_;

    object_marker.color.r = 0;
    object_marker.color.g = 1;
    object_marker.color.b = 0;
    object_marker.color.a = 1.0f;

    object_marker.ns = "object_keypoints";
    object_marker.action = visualization_msgs::Marker::ADD;

    visualization_msgs::Marker delete_marker;
    delete_marker.header.frame_id = "map";
    delete_marker.header.stamp = ros::Time::now();
    delete_marker.ns = "object_keypoints";
    delete_marker.type = visualization_msgs::Marker::SPHERE;
    delete_marker.action = visualization_msgs::Marker::DELETE;

    size_t n_added = 0;

    for (const EstimatedObject::Ptr& obj : objects) {
        if (obj->bad()) {
            for (auto& kp : obj->keypoints()) {
                delete_marker.id = kp->id();
                object_markers.markers.push_back(delete_marker);
            }
            continue;
        }

        // if (!obj->inGraph()) continue;

        if (obj->inGraph()) {
            // blue
            object_marker.color.b = 1.0;
            object_marker.color.r = 0.0;
            object_marker.color.a = 1.0;
        } else {
            // red
            object_marker.color.b = 0.0;
            object_marker.color.r = 1.0;
            object_marker.color.a = 0.5;
        }

        for (const EstimatedKeypoint::Ptr& kp : obj->keypoints()) {
            object_marker.id = kp->id();

            object_marker.pose.position.x = kp->position().x();
            object_marker.pose.position.y = kp->position().y();
            object_marker.pose.position.z = kp->position().z();

            object_markers.markers.push_back(object_marker);

            ++n_added;
        }
    }

    pub_.publish(object_markers);
}
