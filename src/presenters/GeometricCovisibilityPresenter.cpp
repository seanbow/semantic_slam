#include "semantic_slam/presenters/GeometricCovisibilityPresenter.h"
#include "semantic_slam/SemanticKeyframe.h"

#include <fmt/format.h>

#include <visualization_msgs/Marker.h>
// #include <gtsam/geometry/Point3.h>
// #include <gtsam/geometry/Quaternion.h>

void GeometricCovisibilityPresenter::setup()
{
    vis_pub_ = nh_.advertise<visualization_msgs::Marker>("keypoint_objects/semantic_covisibility", 10);

    if (!pnh_.param("covisibility_width", line_width_, 0.1) ||
        !pnh_.param("draw_geometric_covisibility", present_, true)) {
        ROS_WARN("Unable to read trajectory width visualization parameter");
    }
}

void GeometricCovisibilityPresenter::present(const std::vector<SemanticKeyframe::Ptr>& keyframes,
                                  const std::vector<EstimatedObject::Ptr>& objects)
{
    if (keyframes.empty()) return;
    if (!present_) return;
    
    visualization_msgs::Marker line_list;
    line_list.type = visualization_msgs::Marker::LINE_LIST;

    line_list.header.frame_id = "map";
    line_list.header.stamp = ros::Time::now();

    line_list.scale.x = line_width_;

    line_list.color.r = 1;
    line_list.color.g = 1;
    line_list.color.b = 0;
    line_list.color.a = 0.5f;

    line_list.ns = "geometric_covisibility";
    line_list.action = visualization_msgs::Marker::ADD;

    line_list.id = 0;

    visualization_msgs::Marker delete_marker;
    delete_marker.header.frame_id = "map";
    delete_marker.header.stamp = line_list.header.stamp;
    delete_marker.ns = "geometric_covisibility";
    delete_marker.type = visualization_msgs::Marker::LINE_LIST;
    delete_marker.action = visualization_msgs::Marker::DELETE;

    size_t n_added = 0;

    for (const SemanticKeyframe::Ptr& frame : keyframes) {
        geometry_msgs::Point frame_pt;
        frame_pt.x = frame->pose().translation()(0);
        frame_pt.y = frame->pose().translation()(1);
        frame_pt.z = frame->pose().translation()(2);

        for (const auto& cov_frame : frame->geometric_neighbors()) {
            geometry_msgs::Point cov_pt;
            cov_pt.x = cov_frame.first->pose().translation()(0);
            cov_pt.y = cov_frame.first->pose().translation()(1);
            cov_pt.z = cov_frame.first->pose().translation()(2);

            line_list.points.push_back(frame_pt);
            line_list.points.push_back(cov_pt);

            ++n_added;
        }
    }

    vis_pub_.publish(line_list);

}
