#include "semantic_slam/TrajectoryPresenter.h"

// #include <gtsam/geometry/Point3.h>
// #include <gtsam/geometry/Quaternion.h>
#include "semantic_slam/pose_math.h"
#include "semantic_slam/SE3Node.h"
#include <visualization_msgs/Marker.h>

void TrajectoryPresenter::setup()
{
    publisher_ = nh_.advertise<visualization_msgs::Marker>("trajectory", 10);
}

void TrajectoryPresenter::present()
{
    // Find the last odometry node N and assume that all in [0,N] exist
    SE3NodePtr node = graph_->findLastNode<SE3Node>('x');
    size_t last_index = node->index();
    
    
    visualization_msgs::Marker traj;
    traj.type = visualization_msgs::Marker::LINE_STRIP;

    // initialize quaternion to identity. this isn't actually used anywhere but it suppresses a warning
    traj.pose.orientation.x = 0;
    traj.pose.orientation.y = 0;
    traj.pose.orientation.z = 0;
    traj.pose.orientation.w = 1;

    traj.header.frame_id = "/map";
    traj.header.stamp = *node->time();

    Eigen::Vector3d rgb_color(0, 1, 0);
    double Z_SCALE = 1.0;

    double scale = 0.25;
    traj.scale.x = scale;
    traj.scale.y = scale;
    traj.scale.z = scale;
    traj.color.r = rgb_color(0);
    traj.color.g = rgb_color(1);
    traj.color.b = rgb_color(2);
    traj.color.a = 1.0f;

    traj.ns = "optimized_trajectory";
    traj.id = 0;
    traj.action = visualization_msgs::Marker::ADD;

    // create vertices for each pose
    for (size_t i = 0; i < last_index; ++i) {
        auto node = graph_->getNode<SE3Node>(Symbol('x', i));

        if (!node) continue;

        Eigen::Vector3d p = node->pose().translation();

        geometry_msgs::Point p_msg;
        p_msg.x = p(0);
        p_msg.y = p(1);
        p_msg.z = Z_SCALE * p(2);

        traj.points.push_back(p_msg);
    }

    publisher_.publish(traj);
}
