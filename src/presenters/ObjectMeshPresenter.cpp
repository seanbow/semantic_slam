#include "semantic_slam/presenters/ObjectMeshPresenter.h"
#include "semantic_slam/SemanticKeyframe.h"
#include "semantic_slam/keypoints/EstimatedObject.h"
#include "semantic_slam/pose_math.h"

#include <fmt/format.h>

#include <visualization_msgs/MarkerArray.h>
// #include <gtsam/geometry/Point3.h>
// #include <gtsam/geometry/Quaternion.h>

void
ObjectMeshPresenter::setup()
{
    vis_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
      "keypoint_objects/object_markers", 10);

    if (!pnh_.param("show_object_labels", show_object_labels_, true)) {
        ROS_ERROR("Unable to read visualization parameters");
    }
}

void
ObjectMeshPresenter::present(
  const std::vector<SemanticKeyframe::Ptr>& keyframes,
  const std::vector<EstimatedObject::Ptr>& objects)
{
    if (objects.empty())
        return;

    visualization_msgs::MarkerArray object_markers;
    visualization_msgs::Marker object_marker;
    object_marker.type = visualization_msgs::Marker::MESH_RESOURCE;

    object_marker.header.frame_id = "map";
    object_marker.header.stamp = ros::Time::now();

    double model_scale = 1;

    object_marker.scale.x = model_scale;
    object_marker.scale.y = model_scale;
    object_marker.scale.z = model_scale;

    object_marker.color.r = 0;
    object_marker.color.g = 0;
    object_marker.color.b = 1;
    object_marker.color.a = 1.0f;

    object_marker.ns = "objects";
    object_marker.action = visualization_msgs::Marker::ADD;

    visualization_msgs::Marker delete_marker;
    delete_marker.header.frame_id = "map";
    delete_marker.header.stamp = object_marker.header.stamp;
    delete_marker.ns = "objects";
    delete_marker.type = visualization_msgs::Marker::MESH_RESOURCE;
    delete_marker.action = visualization_msgs::Marker::DELETE;

    visualization_msgs::Marker object_text;
    object_text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    object_text.header.frame_id = "map";
    object_text.header.stamp = object_marker.header.stamp;
    object_text.action = visualization_msgs::Marker::ADD;
    object_text.ns = "object_texts";

    object_text.scale.z = 2.5;

    object_text.color.r = 0;
    object_text.color.g = 0;
    object_text.color.b = 0;
    object_text.color.a = 1.0f;

    size_t n_added = 0;

    for (const EstimatedObject::Ptr& obj : objects) {
        if (obj->bad()) {
            delete_marker.id = obj->id();

            delete_marker.ns = "objects";
            object_markers.markers.push_back(delete_marker);

            delete_marker.ns = "object_texts";
            object_markers.markers.push_back(delete_marker);

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

        object_marker.action = visualization_msgs::Marker::ADD;
        // object_text.action = visualization_msgs::Marker::ADD;

        double model_scale = 1.0;

        if (obj->obj_name() == "chair") {
            model_scale = 1.5;
        } else if (obj->obj_name() == "gascan") {
            model_scale = 0.25;
        } else if (obj->obj_name() == "cart") {
            model_scale = 0.12;
        } else if (obj->obj_name() == "tableclosed") {
            model_scale = 9.0;
        } else if (obj->obj_name() == "ladder") {
            model_scale = 0.125;
        } else if (obj->obj_name() == "pelican") {
            model_scale = 0.25;
        } else if (obj->obj_name() == "car") {
            model_scale = 6;
        }

        object_marker.scale.x = model_scale;
        object_marker.scale.y = model_scale;
        object_marker.scale.z = model_scale;

        object_marker.pose.position.x = obj->pose().x();
        object_marker.pose.position.y = obj->pose().y();
        object_marker.pose.position.z = obj->pose().z();

        Eigen::Quaterniond q = obj->pose().rotation();

        object_marker.pose.orientation.x = q.x();
        object_marker.pose.orientation.y = q.y();
        object_marker.pose.orientation.z = q.z();
        object_marker.pose.orientation.w = q.w();

        object_marker.mesh_resource =
          std::string("package://semantic_slam/models/viz_meshes/") +
          obj->obj_name() + ".dae";
        // object_marker.mesh_resource =
        // std::string("package://semslam/models/viz_meshes/car.dae");

        // object_text.pose.position.x = obj->pose().translation().x();
        // object_text.pose.position.y = obj->pose().translation().y();
        // object_text.pose.position.z = Z_SCALE * obj->pose().translation().z()
        // + 1.5;

        object_marker.id = obj->id();
        object_marker.text = obj->obj_name();

        object_markers.markers.push_back(object_marker);

        if (show_object_labels_) {
            object_text.pose = object_marker.pose;
            object_text.pose.position.z += 1.5;

            object_text.id = obj->id();

            object_text.text = fmt::format("{}", obj->id());

            object_markers.markers.push_back(object_text);
        }

        std::vector<int64_t> keypoint_ids = obj->getKeypointIndices();
        // addKeypointMarkers(object_markers.markers, model_scale, obj, t);

        ++n_added;
    }

    vis_pub_.publish(object_markers);
}
