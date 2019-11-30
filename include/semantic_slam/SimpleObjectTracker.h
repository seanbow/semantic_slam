#pragma once

#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/subscriber.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <darknet_ros_msgs/BoundingBoxes.h>

class SimpleObjectTracker {
public:

    struct TrackedObject {
        std::string object_name;
        size_t n_missed_detections;
        cv::Rect2d bounding_box;
        int id;
    };

    SimpleObjectTracker();

    void detectionCallback(const sensor_msgs::ImageConstPtr& image,
                           const darknet_ros_msgs::BoundingBoxes::ConstPtr& msg);
  
    int getTrackingIndex(const std::string& name, const cv::Rect2d& drect);
    
    void visualizeTracking(const sensor_msgs::ImageConstPtr& image,
                           const darknet_ros_msgs::BoundingBoxes& msg);

private:
    double calculateMatchRate(const cv::Rect2d& r1, const cv::Rect2d& r2);

    std::vector<TrackedObject> tracked_objects_;

    int next_object_id_;

    ros::Publisher det_pub_;
    image_transport::Publisher img_pub_;

    double det_conf_thresh_;
    double f2f_match_thresh_;
    int missed_detection_thresh_;

    boost::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> image_sub_;
    boost::shared_ptr<message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes>> det_sub_;
    boost::shared_ptr<message_filters::TimeSynchronizer<sensor_msgs::Image, darknet_ros_msgs::BoundingBoxes>> sync_;

    ros::NodeHandle nh_, pnh_;
};
