#include "semantic_slam/presenters/ObjectPosePresenter.h"
#include "semantic_slam/Pose3.h"
#include "semantic_slam/SemanticKeyframe.h"
#include "semantic_slam/keypoints/EstimatedObject.h"

#include <fmt/format.h>

#include <object_pose_interface_msgs/SemanticMapObjectArray.h>
#include <object_pose_interface_msgs/SemanticMapObject.h>
// #include <gtsam/geometry/Point3.h>
// #include <gtsam/geometry/Quaternion.h>

void
ObjectPosePresenter::setup()
{
    pub_landmarks_poses_ = nh_.advertise<object_pose_interface_msgs::SemanticMapObjectArray>(
      "semslam/landmarks_poses", 10);
}

void
ObjectPosePresenter::present(
  const std::vector<SemanticKeyframe::Ptr>& keyframes,
  const std::vector<EstimatedObject::Ptr>& objects)
{
    if (objects.empty())
        return;

    object_pose_interface_msgs::SemanticMapObjectArray objects_semantic;
    object_pose_interface_msgs::SemanticMapObject object_semantic;

    objects_semantic.header.frame_id = "map";
    objects_semantic.header.stamp = ros::Time::now();

    for (const EstimatedObject::Ptr& obj : objects) {
        if (obj->bad()) {
            continue;
        }

        object_semantic.pose.header = objects_semantic.header;
        object_semantic.pose.pose.position.x = obj->pose().x();
        object_semantic.pose.pose.position.y = obj->pose().y();
        object_semantic.pose.pose.position.z = obj->pose().z();

        Eigen::Quaterniond q = obj->pose().rotation();

        object_semantic.pose.pose.orientation.x = q.x();
        object_semantic.pose.pose.orientation.y = q.y();
        object_semantic.pose.pose.orientation.z = q.z();
        object_semantic.pose.pose.orientation.w = q.w();

        object_semantic.classification.type.name = obj->obj_name();
        object_semantic.classification.type.id = obj->id();

        if (obj->inGraph()) {
            objects_semantic.objects.push_back(object_semantic);
        }
    }

    pub_landmarks_poses_.publish(objects_semantic);
}
