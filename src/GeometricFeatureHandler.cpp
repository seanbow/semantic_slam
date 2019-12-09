#include "semantic_slam/GeometricFeatureHandler.h"

#include "semantic_slam/MultiProjectionFactor.h"
#include "semantic_slam/SmartProjectionFactor.h"

void GeometricFeatureHandler::setup()
{  
    std::string image_topic;

    if (!pnh_.getParam("image_topic", image_topic)) {
        ROS_ERROR_STREAM("Error: unable to read feature tracker parameters");
    } else {
        ROS_INFO_STREAM("[Tracker] Subscribing to " << image_topic);
    }

    last_img_seq_ = -1;

    image_transport::ImageTransport it(nh_);
    img_sub_ = it.subscribe(image_topic, 10000, &GeometricFeatureHandler::imageCallback, this);

    loadParameters();

    running_ = true;

    // work_thread_ = std::thread(&GeometricFeatureHandler::extractKeypointsThread, this);
}

void GeometricFeatureHandler::update()
{
    
}

void GeometricFeatureHandler::setExtrinsicCalibration(const Pose3& body_T_sensor)
{
    I_T_C_ = body_T_sensor;
}

void GeometricFeatureHandler::addKeyframe(const SemanticKeyframe::Ptr& frame)
{
    kfs_to_process_.push_back(frame);
}

void GeometricFeatureHandler::updateEssentialGraph()
{
    // remove *all* factors. we will re-add them later.
    if (use_smart_projection_factors_) {
        for (auto fac : factors_) {
            essential_graph_->removeFactor(fac.second);
        }

        // Work our way through the keyframes
        // For the next keyframe with no constraints in the essential graph, find the 
        // projection factor that ties the most frames together and add. Then repeat.

    }
}

void GeometricFeatureHandler::processPendingFrames()
{
    while (!kfs_to_process_.empty()) {

        auto frame = kfs_to_process_.front();

        std::vector<FeatureTracker::TrackedFeature> tracks;
        
        bool got_tracks = tracker_->addKeyframeTime(frame->image_time, tracks);

        if (!got_tracks) break;

        kfs_to_process_.pop_front();

        // For each feature tracked into this frame, either create or update its projection factor

        for (int i = 0; i < tracks.size(); ++i) {
            const auto& tf = tracks[i];

            auto feature_it = features_.find(tf.pt_id);
            boost::shared_ptr<GeometricFeature> feature;

            if (feature_it == features_.end()) {
                feature = util::allocate_aligned<GeometricFeature>();
                feature->id = tf.pt_id;
                features_.emplace(tf.pt_id, feature);
            } else {
                feature = feature_it->second;
            }

            feature->keyframe_observations.push_back(frame);
            frame->visible_geometric_features().push_back(feature);

            auto factor_it = factors_.find(tf.pt_id);

            if (use_smart_projection_factors_) {
                SmartProjectionFactor::Ptr factor;

                if (factor_it == factors_.end()) {
                    factor = util::allocate_aligned<SmartProjectionFactor>(I_T_C_,
                                                                        calibration_,
                                                                        reprojection_error_threshold_);
                    factors_.emplace(tf.pt_id, factor);
                    
                } else {
                    factor = boost::static_pointer_cast<SmartProjectionFactor>(factor_it->second);
                }


                Eigen::Vector2d msmt(tf.pt.x, tf.pt.y);
                Eigen::Matrix2d msmt_noise = cam_sigma_ * Eigen::Matrix2d::Identity();

                factor->addMeasurement(frame->graph_node(), msmt, msmt_noise);

                if (factor->nMeasurements() >= 5 && !factor->inGraph()) {
                    graph_->addFactor(factor);
                }

                feature->point = factor->point();
                feature->active = factor->active();

            } else {
                MultiProjectionFactor::Ptr factor;
                Vector3dNode::Ptr landmark_node;

                if (factor_it == factors_.end()) {
                    landmark_node = util::allocate_aligned<Vector3dNode>(Symbol('g', tf.pt_id));
                    factor = util::allocate_aligned<MultiProjectionFactor>(landmark_node,
                                                                        I_T_C_,
                                                                        calibration_,
                                                                        reprojection_error_threshold_);
                    landmark_nodes_.emplace(tf.pt_id, landmark_node);
                    factors_.emplace(tf.pt_id, factor);
                } else {
                    factor = boost::static_pointer_cast<MultiProjectionFactor>(factor_it->second);
                    landmark_node = landmark_nodes_[tf.pt_id];
                }

                Eigen::Vector2d msmt(tf.pt.x, tf.pt.y);
                Eigen::Matrix2d msmt_noise = cam_sigma_ * Eigen::Matrix2d::Identity();

                factor->addMeasurement(frame->graph_node(), msmt, msmt_noise);

                if (factor->nMeasurements() >= 5 && !factor->inGraph()) {
                    graph_->addNode(landmark_node);
                    graph_->addFactor(factor);
                }

                feature->point = landmark_node->vector();
                feature->active = factor->active();

            }
        }

        frame->updateGeometricConnections();
    }

    updateEssentialGraph();
}

void GeometricFeatureHandler::removeLandmark(int index)
{
    // this function only really makes sense if we're not using smart factors
    if (!use_smart_projection_factors_) {
        ROS_WARN_STREAM("Removing geom. landmark " << index);
        
        auto factor = factors_.find(index);
        if (factor != factors_.end()) graph_->removeFactor(factor->second);
        
        auto node = landmark_nodes_.find(index);
        if (node != landmark_nodes_.end()) graph_->removeNode(node->second);
    }
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

    if (!pnh_.getParam("reprojection_error_threshold", reprojection_error_threshold_) || 
        !pnh_.getParam("use_smart_projection_factors", use_smart_projection_factors_) || 
        !pnh_.getParam("cam_sigma", cam_sigma_)) {
        ROS_ERROR("Error: unable to get geometric feature handler parameters.");
        return;
    }

    tracker_ = util::allocate_aligned<FeatureTracker>(tracker_params_);

    tracker_->setCameraCalibration(fx, fy, s, u0, v0, k1, k2, p1, p2);

    calibration_ = boost::make_shared<CameraCalibration>(fx, fy, s, u0, v0, k1, k2, p1, p2);
}

void GeometricFeatureHandler::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    if (msg->header.seq != last_img_seq_ + 1) {
        ROS_ERROR_STREAM("[Tracker] dropped image message, non-sequential sequence numbers, received " << msg->header.seq << ", expected " << last_img_seq_ + 1);
    }

    // {
    //     std::lock_guard<std::mutex> lock(queue_mutex_);
    //     img_queue_.push_back(msg);
    // }

    // queue_cv_.notify_all();

    last_img_seq_ = msg->header.seq;

    FeatureTracker::Frame new_frame;
    new_frame.image = msg;
    tracker_->addImage(std::move(new_frame));
}

// void GeometricFeatureHandler::extractKeypointsThread()
// {
//     while (ros::ok() && running_) {

//         FeatureTracker::Frame new_frame;

//         {
//             std::unique_lock<std::mutex> lock(queue_mutex_);

//             queue_cv_.wait(lock, [&] { return !img_queue_.empty(); });

//             new_frame.image = img_queue_.front();
//             img_queue_.pop_front();
//         }

//         tracker_->addImage(std::move(new_frame));
//     }
// }