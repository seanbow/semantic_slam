#include "semantic_slam/presenters/TrajectoryPresenter.h"
#include "semantic_slam/SemanticKeyframe.h"
#include "semantic_slam/keypoints/EstimatedObject.h"

#include "semantic_slam/Pose3.h"
#include <visualization_msgs/Marker.h>

void
TrajectoryPresenter::setup()
{
    publisher_ = nh_.advertise<visualization_msgs::Marker>("trajectory", 10);

    if (!pnh_.param("trajectory_width", trajectory_width_, 0.25)) {
        ROS_WARN("Unable to read trajectory width visualization parameter");
    }
}

void
TrajectoryPresenter::present(
  const std::vector<SemanticKeyframe::Ptr>& keyframes,
  const std::vector<EstimatedObject::Ptr>& objects)
{
    if (keyframes.empty())
        return;

    visualization_msgs::Marker traj;
    traj.type = visualization_msgs::Marker::LINE_STRIP;

    // initialize quaternion to identity. this isn't actually used anywhere but
    // it suppresses a warning
    traj.pose.orientation.x = 0;
    traj.pose.orientation.y = 0;
    traj.pose.orientation.z = 0;
    traj.pose.orientation.w = 1;

    traj.header.frame_id = "/map";
    traj.header.stamp = keyframes.back()->time();

    Eigen::Vector3d rgb_color(0, 1, 0);
    double Z_SCALE = 1.0;

    double scale = trajectory_width_;
    traj.scale.x = scale;
    traj.color.r = rgb_color(0);
    traj.color.g = rgb_color(1);
    traj.color.b = rgb_color(2);
    traj.color.a = 1.0f;

    traj.ns = "optimized_trajectory";
    traj.id = 0;
    traj.action = visualization_msgs::Marker::ADD;

    // create vertices for each pose
    // Assume that the keyframe vector is ordered...
    for (size_t i = 0; i < keyframes.size(); ++i) {
        // auto node = graph_->getNode<SE3Node>(Symbol('x', i));

        // if (!node) continue;

        Eigen::Vector3d p = keyframes[i]->pose().translation();

        geometry_msgs::Point p_msg;
        p_msg.x = p(0);
        p_msg.y = p(1);
        p_msg.z = Z_SCALE * p(2);

        traj.points.push_back(p_msg);
    }

    publisher_.publish(traj);
}
