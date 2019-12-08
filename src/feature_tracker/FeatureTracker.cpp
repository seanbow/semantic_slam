

#include "semantic_slam/feature_tracker/FeatureTracker.h"
#include "semantic_slam/feature_tracker/FivePointRansac.h"
#include "semantic_slam/feature_tracker/ORBextractor.h"

// -- ROS -- 
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
// #include <sensor_msgs/CameraInfo.h>
#include <opencv/cv.h>
// #include <opencv2/features2d.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/features2d/features2d.hpp>

// -- general --
#include <string>
#include <chrono>
#include <numeric>
// #include <unsupported/Eigen/MatrixFunctions>


FeatureTracker::FeatureTracker(const Params& params)
    : params_(params),
      frames_received_(0),
      n_keyframes_(0),
      n_features_extracted_(0),
      received_bias_(false),
      last_keyframe_seq_(-1),
      last_img_seq_(-1),
      last_keyframe_time_(0),
      nh_("klt"),
      five_ransac_(params.ransac_iterations, params.sqrt_samp_thresh),
      two_ransac_(params.ransac_iterations, params.sqrt_samp_thresh)
{

    // orb_ = cv::Ptr<cv::ORB>( new cv::ORB(2000) );
    // orb_ = cv::ORB::create();

    // orb_ = boost::make_shared<ORB_SLAM2::ORBextractor>(2000, 1.2, 8, 15, 3);
    orb_ = boost::make_shared<ORB_SLAM2::ORBextractor>(1000, 1.2, 8, 20, 7);

    image_transport::ImageTransport it(nh_);
    pub_marked_images_ = it.advertise("marked_image", 1);

    // keyframe_dR_ = Eigen::Matrix3d::Identity();
}

void FeatureTracker::addImage(Frame&& new_frame)
{
    // extractKeypointsDescriptors(new_frame);

    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        image_buffer_.push_back(new_frame);
    }
}

bool
FeatureTracker::addKeyframeTime(ros::Time t, std::vector<FeatureTracker::TrackedFeature>& tracks)
{
    tracks.clear();

    // Don't do anything if this is our first keyframe...
    if (last_keyframe_time_ == ros::Time(0)) {
        last_keyframe_time_ = t;
        return true;
    }

    // Lookup image that corresponds to this time, and also to the previous time
    int last_kf_index = -1;
    int this_kf_index = -1;
    for (int i = 0; i < image_buffer_.size(); ++i) {
        if (image_buffer_[i].image->header.stamp == t) {
            this_kf_index = i;
        }

        if (image_buffer_[i].image->header.stamp == last_keyframe_time_) {
            last_kf_index = i;
        }
    }

    if (this_kf_index < 0) {
        ROS_ERROR_STREAM("Unable to find image for time " << t);
        // ROS_ERROR_STREAM("Last buffer time = " << image_buffer_.back().image->header.stamp);
        // last_keyframe_time_ = t;
        return false;
    }

    if (last_kf_index < 0) {
        ROS_ERROR_STREAM("Unable to find image for time " << last_keyframe_time_);
        // ROS_ERROR_STREAM("Last buffer time = " << image_buffer_.back().image->header.stamp);
        // last_keyframe_time_ = t;
        return false;
    }

    // Extract new features in the previous keyframe

    addNewKeyframeFeatures(image_buffer_[last_kf_index]);
    
    // And track them forward
    for (int i = last_kf_index; i < this_kf_index; ++i) {
        trackFeaturesForward(i);
    }

    tracks = image_buffer_[this_kf_index].feature_tracks;

    // Remove old unneeded frames
    std::lock_guard<std::mutex> lock(buffer_mutex_);

    image_buffer_.erase(image_buffer_.begin(), image_buffer_.begin() + this_kf_index);

    last_keyframe_time_ = t;

    return true;
}

void FeatureTracker::extractKeypointsDescriptors(Frame& frame)
{
    cv_bridge::CvImageConstPtr cv_image;
    cv::Mat img;

    // TIME_TIC;

    try {
        cv_image = cv_bridge::toCvShare(frame.image, "bgr8");
        cv::cvtColor(cv_image->image, img, CV_BGR2GRAY);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    (*orb_)(img, cv::Mat(), frame.keypoints, frame.descriptors);

    // ROS_INFO_STREAM("Extraction took " << TIME_TOC << " ms.");

    // ROS_INFO_STREAM("Extracted " << frame.keypoints.size() << " ORB keypoints");
}

void FeatureTracker::trackFeaturesForward(int idx1)
{
    // Make sure we have any features to track...
    if (idx1 >= image_buffer_.size() || image_buffer_[idx1].feature_tracks.size() == 0) {
        ROS_WARN_STREAM("Requested feature tracking with no active feature tracks");
        return;
    }

    int idx2 = idx1 + 1;

    const Frame& frame1 = image_buffer_[idx1];
    Frame& frame2 = image_buffer_[idx2];

    if (frame2.keypoints.size() == 0) {
        extractKeypointsDescriptors(frame2);
    }

    const auto& kps2 = frame2.keypoints;
    const auto& descriptors2 = frame2.descriptors;

    // Match keypoints from old image to new
    // First collect the old keypoints/descriptors from the actual tracks

    std::vector<cv::KeyPoint> kps1;
    cv::Mat descriptors1(frame1.feature_tracks.size(), frame1.descriptors.cols, frame1.descriptors.type());

    for (int i = 0; i < frame1.feature_tracks.size(); ++i) {
        kps1.push_back(frame1.feature_tracks[i].kp);
        frame1.feature_tracks[i].descriptor.copyTo(descriptors1.row(i));
    }

    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);

    std::vector<std::vector<cv::DMatch>> matches;
    matcher->knnMatch(descriptors1, descriptors2, matches, 1);

    std::vector<cv::Point2f> pts1, pts2;
    std::vector<int> indices1, indices2;
    for (int i = 0; i < matches.size(); ++i) {
        if (matches[i].size() == 0) continue; // no match found for keypoint i

        // knn with k = 1 so just use first match...

        pts1.push_back(kps1[matches[i][0].queryIdx].pt);
        pts2.push_back(kps2[matches[i][0].trainIdx].pt);

        indices1.push_back(matches[i][0].queryIdx);
        indices2.push_back(matches[i][0].trainIdx);
    }

    // Remove outliers 
    Eigen::Matrix<bool, 1, Eigen::Dynamic> inliers;
    size_t n_inliers = five_ransac_.computeInliers(pts1, pts2, inliers);

    // std::cout << "Proportion of inliers = " << n_inliers / (double)indices1.size() << std::endl;

    for (int i = 0; i < indices1.size(); ++i) {
        if (inliers(i)) {
            TrackedFeature tf;

            tf.kp = kps2[indices2[i]];
            tf.pt = tf.kp.pt;
            tf.descriptor = descriptors2.row( indices2[i] );

            tf.frame_id = frame2.image->header.seq; // TODO

            tf.n_images_in = frame1.feature_tracks[ indices1[i] ].n_images_in + 1;
            tf.pt_id = frame1.feature_tracks[ indices1[i] ].pt_id;

            frame2.feature_tracks.push_back(tf);
        }
    }

    // ROS_INFO_STREAM("RANSAC: From " << pts1.size() << " features to " << frame2.feature_tracks.size() << " inliers.");

    // publish marked image with new tracks
    
    cv::Mat color_img;
    try {
        color_img = cv_bridge::toCvCopy(frame2.image, "bgr8")->image;
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // mark image
    // cv::Scalar pt_color = cv::Scalar(0, 255, 0); // green
    for (size_t i = 0; i < frame2.feature_tracks.size(); ++i) {

        if (frame2.feature_tracks[i].n_images_in < 5) continue;

        auto color_it = pt_colors_.find(frame2.feature_tracks[i].pt_id);
        cv::Scalar color;

        if (color_it == pt_colors_.end()) {
            // color = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
            color = cv::Scalar(0, 255, 0);
            pt_colors_[frame2.feature_tracks[i].pt_id] = color;
        } else {
            color = color_it->second;
        }

        cv::circle(color_img, frame2.feature_tracks[i].pt, 4, color, -1);
    }

    cv_bridge::CvImage img_msg;
    img_msg.header = frame2.image->header;
    img_msg.image = color_img;
    img_msg.encoding = sensor_msgs::image_encodings::BGR8;

    pub_marked_images_.publish(img_msg.toImageMsg());

    // ROS_INFO_STREAM("Published image.");
}

void FeatureTracker::addNewKeyframeFeatures(Frame& frame) {

    if (frame.keypoints.size() == 0) {
        extractKeypointsDescriptors(frame);
    }

    // sort by keypoint strength to the best ones only
    std::vector<size_t> idx(frame.keypoints.size());
    std::iota(idx.begin(), idx.end(), 0);

    std::sort(idx.begin(), 
              idx.end(), 
              [&](size_t i1, size_t i2) { return frame.keypoints[i1].response > frame.keypoints[i2].response; });

    // throw out features we're already tracking
    // TODO do this smarter it's currently O(n^2)
    std::vector<uchar> good_detection(frame.keypoints.size(), 1);

    // const auto& tracked_feats = feature_tracks_[index];

    // std::vector<TrackedFeature> new_feats;

    // ROS_INFO_STREAM("Feature tracks size: " << feature_tracks_.size());
    // ROS_INFO_STREAM("Index = " << index);

    // std::cout << std::endl;

    // ROS_INFO_STREAM("New keyframe features, index " << index << ", kps size = " << kps.size()
    //     << ", tracked feats size = " << tracked_feats.size());

    for (int i = 0; i < frame.keypoints.size(); ++i) {
        // Compare against all features that have already been tracked into this frame
        for (int j = 0; j < frame.feature_tracks.size(); ++j) {
            if (cv::norm(frame.keypoints[idx[i]].pt - frame.feature_tracks[j].pt) < params_.feature_spacing) {
                good_detection[idx[i]] = 0;
                break;
            }
        }

        // Compare against new features that we've already added
        // Don't need to do this as it will already be in feature_tracks and accounted for in 
        // the above loop!!
        // for (int j = 0; j < i; ++j) {
        //     if (good_detection[idx[j]] && 
        //             cv::norm(frame.keypoints[idx[i]].pt - frame.keypoints[idx[j]].pt) < params_.feature_spacing) {
        //         good_detection[idx[i]] = 0;
        //         break;
        //     }
        // }

        if (good_detection[idx[i]]) { 
            TrackedFeature new_feat(frame.keypoints[idx[i]].pt, 
                                    frame.image->header.seq,
                                    n_features_extracted_++, 
                                    frame.keypoints[idx[i]].size);
            new_feat.kp = frame.keypoints[idx[i]];
            frame.descriptors.row(idx[i]).copyTo(new_feat.descriptor);

            frame.feature_tracks.push_back(new_feat);
        }

        if (frame.feature_tracks.size() >= params_.max_features_per_im) break;

    }
}

// void FeatureTracker::imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
//     imu_queue_.push_back(msg);

//     if (msg->header.seq != last_imu_seq_ + 1) {
//         ROS_ERROR_STREAM("[FeatureTracker] Error: dropped IMU message. Expected " << last_imu_seq_ + 1 << ", got " << msg->header.seq);
//     }

//     last_imu_seq_ = msg->header.seq;
//     last_imu_time_ = msg->header.stamp;

//     // ROS_INFO_STREAM("IMU time = " << last_imu_time_);

//     tryProcessNextImage();
// }

// void FeatureTracker::imgCallback(const sensor_msgs::ImageConstPtr& msg) {
//     if (msg->header.seq != last_img_seq_ + 1) {
//         ROS_ERROR_STREAM("[Tracker] dropped image message, non-sequential sequence numbers, received " << msg->header.seq << ", expected " << last_img_seq_ + 1);
//     }

//     img_queue_.push_back(msg);
//     // ROS_INFO_STREAM("IMAGE time = " << msg->header.stamp);

//     last_img_seq_ = msg->header.seq;

//     tryProcessNextImage();
// }

// Eigen::Matrix3d FeatureTracker::computeFrameRotation(ros::Time frame_time) {
//     Eigen::Quaterniond q = math::identity_quaternion();

//     Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

//     ros::Time t = imu_queue_.front()->header.stamp;

//     // ROS_INFO_STREAM("Integrating from " << last_integrated_imu_time_ << " to " << frame_time);

//     while (t < frame_time) {
//         auto msg = imu_queue_.front();
//         imu_queue_.pop_front();
//         t = msg->header.stamp;

//         Eigen::Vector3d omega;
//         omega << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;

//         omega -= gyro_bias_;

//         // use a zeroth-order integrator for now
//         // TODO first-order?

//         double w_norm = omega.norm();
//         Eigen::Vector3d w_unit = omega / w_norm;
//         Eigen::Matrix3d w_skew = skewsymm(w_unit);

//         Eigen::Quaterniond dq;

//         if (w_norm > 1e-10) {
//             dq.vec() = w_unit * std::sin( w_norm/2 * imu_dt_);
//             dq.w()   = std::cos(w_norm/2 * imu_dt_);

//             q = dq * q;
//         } else {
//             q.coeffs() = (Eigen::Matrix4d::Identity() + imu_dt_/2 * math::quat_Omega(omega)) * q.coeffs();
//         }

//         q.normalize();


//         last_integrated_imu_time_ = t;
//     }

//     if (t != frame_time) {
//         ROS_WARN_STREAM("Timing off in feature tracker IMU integration; integrated to "  << t << " but requested " << frame_time);
//     }

//     // return q;
//     // return rot2quat(R);

//     // std::cout << "delta T = " << rot_int.deltaTij() << std::endl;

//     // return I_T_C_.rotation().transpose() * rot_int.deltaRij().matrix() * I_T_C_.rotation().matrix();
//     // return I_T_C_.rotation().transpose() * R * I_T_C_.rotation().matrix();
//     return (I_q_C_.conjugate() * q * I_q_C_).toRotationMatrix();
// }

// void FeatureTracker::gyroBiasCallback(const geometry_msgs::Vector3Stamped::ConstPtr& msg) {
//     gyro_bias_(0) = msg->vector.x;
//     gyro_bias_(1) = msg->vector.y;
//     gyro_bias_(2) = msg->vector.z;

//     // ROS_INFO_STREAM("GYRO bias = " << gyro_bias_.transpose());

//     received_bias_ = true;
// }

void FeatureTracker::setCameraExtrinsics(std::vector<double> I_p_C, std::vector<double> I_q_C) {
    Eigen::Quaterniond q;
    q.x() = I_q_C[0]; q.y() = I_q_C[1]; q.z() = I_q_C[2]; q.w() = I_q_C[3];
    q.normalize();

    I_p_C_ = Eigen::Vector3d(I_p_C[0], I_p_C[1], I_p_C[2]);
    I_q_C_ = q;
}

// void FeatureTracker::tryProcessNextImage() {

//     if (img_queue_.empty()) return;

//     // if using 2pt ransac, to continue, require IMU messages >= last image time
//     if (params_.use_2pt_ransac && last_imu_time_ < img_queue_.front()->header.stamp) return;

//     // compute between frame IMU rotation
//     Eigen::Matrix3d R;
//     if (params_.use_2pt_ransac) R = computeFrameRotation(img_queue_.front()->header.stamp);
//     // Eigen::Matrix3d I2_R_I1 = quat2rot(I2_q_I1);

//     // ROS_INFO_STREAM("Rotation amount = " << 180*2*std::acos(q(3))/M_PI << " degrees.");

//     // keyframe_dR_ = R * keyframe_dR_;

//     auto msg = img_queue_.front();
//     img_queue_.pop_front();

//     cv::Mat color_img;

//     try {
//         color_img = cv_bridge::toCvCopy(msg, "bgr8")->image;
//         cv::cvtColor(color_img, new_img_, CV_BGR2GRAY);
//     } catch (cv_bridge::Exception& e) {
//         ROS_ERROR("cv_bridge exception: %s", e.what());
//     }

//     bool is_keyframe = (frames_received_ % params_.keyframe_spacing == 0);

//     // if (is_keyframe) {
//     //     ROS_INFO_STREAM("::KLT:: dR between last keyframe times " << last_keyframe_time_ << " and " << msg->header.stamp << ": ");
//     //     // std::cout << keyframe_dR_ << std::endl;
//     //     Eigen::Quaterniond dqq(keyframe_dR_);
//     //     std::cout << "[ " << dqq.x() << " " << dqq.y() << " " << dqq.z() << " " << dqq.w() << " ]" << std::endl;
//     //     keyframe_dR_ = Eigen::Matrix3d::Identity();
//     // }

//     // extract kps / descriptors
//     cv::Mat descriptors;
//     std::vector<cv::KeyPoint> kps;

//     (*orb_)(new_img_, cv::Mat(), kps, descriptors);
//     // orb_->detectAndCompute(new_img_, cv::Mat(), kps, descriptors);

//     trackFeatures(kps, descriptors, R);


//     std::vector<TrackedFeature> new_feats;
//     if (is_keyframe) extractFeatures(new_feats);

//     // mark image
//     // cv::Scalar pt_color = cv::Scalar(0, 255, 0); // green
//     for (size_t i = 0; i < tracked_features_.size(); ++i) {

//         if (tracked_features_[i].n_images_in < 5) continue;

//         auto color_it = pt_colors_.find(tracked_features_[i].pt_id);
//         cv::Scalar color;

//         if (color_it == pt_colors_.end()) {
//             // color = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
//             color = cv::Scalar(0, 255, 0);
//             pt_colors_[tracked_features_[i].pt_id] = color;
//         } else {
//             color = color_it->second;
//         }

//         cv::circle(color_img, tracked_features_[i].pt, 4, color, -1);
//     }

//     // add newly tracked detected to the list to track through next frame
//     for (auto it = new_feats.begin(); it != new_feats.end(); ++it) {
//         tracked_features_.push_back(*it);
//         tracked_feature_pts_.push_back(it->pt);
//     }

//     // Compute average track length for funsies
//     /*
//     long n_ims_sum = 0;
//     for (auto it = tracked_features_.begin(); it != tracked_features_.end(); ++it) {
//         n_ims_sum += it->n_images_in;
//     }
//     ROS_INFO_STREAM("Average track length = " << (double)n_ims_sum / tracked_features_.size() << " images.");
//     */

//     cv_bridge::CvImage img_msg;
//     img_msg.header = msg->header;
//     img_msg.image = color_img;
//     img_msg.encoding = sensor_msgs::image_encodings::BGR8;

//     pub_marked_images_.publish(img_msg.toImageMsg());

//     // forward the original message too to make it easier to process only keyframe images
//     // if (is_keyframe) pub_images_.publish(msg);

//     // if (is_keyframe) publishKeyframe(msg->header);

//     // last_img_ = img;
//     new_img_.copyTo(last_img_); 
//     // cv::swap(new_img_, last_img_); // <-- this is broken unfortunately

//     frames_received_++;
//     if (is_keyframe) {
//         n_keyframes_++;
//         last_keyframe_time_ = msg->header.stamp;
//     }
// }

// void FeatureTracker::camInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg) {
//     // Copy P into K
//     sensor_msgs::CameraInfo info_msg = *msg;

//     info_msg.K[0] = msg->P[0]; // fx
//     info_msg.K[2] = msg->P[2]; // cx
//     info_msg.K[4] = msg->P[5]; // fy
//     info_msg.K[5] = msg->P[6]; // cy
//     info_msg.K[8] = 1.0;

//     // pub_camera_info_.publish(info_msg);
// }

void FeatureTracker::setCameraCalibration(double fx, 
                        double fy, 
                        double s, 
                        double u0, 
                        double v0, 
                        double k1, 
                        double k2, 
                        double p1, 
                        double p2) {
    five_ransac_.setCameraCalibration(fx, fy, s, u0, v0, k1, k2, p1, p2);
    two_ransac_.setCameraCalibration(fx, fy, s, u0, v0, k1, k2, p1, p2);
}
