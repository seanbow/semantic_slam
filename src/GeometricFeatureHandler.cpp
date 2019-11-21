#include "semantic_slam/GeometricFeatureHandler.h"

void GeometricFeatureHandler::setup()
{  
    std::string image_topic;

    if (!pnh_.getParam("image_topic", image_topic)) {
        ROS_ERROR_STREAM("Error: unable to read feature tracker parameters");
    }

    last_img_seq_ = -1;

    img_sub_ = nh_.subscribe(image_topic, 100, &GeometricFeatureHandler::imageCallback, this);

    loadParameters();
}

void GeometricFeatureHandler::update()
{
    
}

void GeometricFeatureHandler::addKeyframeTime(ros::Time t)
{
    tracker_->addKeyframeTime(t);
}

void GeometricFeatureHandler::loadParameters()
{
    if (!pnh_.getParam("ransac_iterations", tracker_params_.ransac_iterations) ||
        !pnh_.getParam("feature_spacing", tracker_params_.feature_spacing) ||
        !pnh_.getParam("max_features_per_im", tracker_params_.max_features_per_im) ||
        !pnh_.getParam("sqrt_samp_thresh", tracker_params_.sqrt_samp_thresh)) {
        ROS_ERROR("Error: unable to read feature tracker parameters.");
        return;
    }

    double fx, fy, s, u0, v0, k1, k2, p1, p2;
    s = 0.0; //skew

    if (!pnh_.getParam("cam_cx", u0) || 
        !pnh_.getParam("cam_cy", v0) || 
        !pnh_.getParam("cam_d0", k1) ||
        !pnh_.getParam("cam_d1", k2) ||
        !pnh_.getParam("cam_d2", p1) ||
        !pnh_.getParam("cam_d3", p2) || 
        !pnh_.getParam("cam_fx", fx) ||
        !pnh_.getParam("cam_fy", fy)) {
        ROS_ERROR("Error: unable to read camera intrinsic parameters.");
        return;
    }

    tracker_ = util::allocate_aligned<FeatureTracker>(tracker_params_);

    tracker_->setCameraCalibration(fx, fy, s, u0, v0, k1, k2, p1, p2);
}

void GeometricFeatureHandler::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    if (msg->header.seq != last_img_seq_ + 1) {
        ROS_ERROR_STREAM("[Tracker] dropped image message, non-sequential sequence numbers, received " << msg->header.seq << ", expected " << last_img_seq_ + 1);
    }

    img_queue_.push_back(msg);
    // ROS_INFO_STREAM("IMAGE time = " << msg->header.stamp);

    last_img_seq_ = msg->header.seq;

    tracker_->addImage(msg);
}