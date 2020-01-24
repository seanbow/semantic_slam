#ifndef KLT_TRACKER_FEATURETRACKER_H_
#define KLT_TRACKER_FEATURETRACKER_H_

#include "semantic_slam/Common.h"

#include <deque>
#include <mutex>
#include <unordered_map>

#include <geometry_msgs/Vector3Stamped.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>

#include <boost/optional.hpp>

#include <opencv/cv.h>
// #include <opencv2/features2d/features2d.hpp>
// #include <svo_msgs/Keyframe.h>

#include "semantic_slam/feature_tracker/FivePointRansac.h"
#include "semantic_slam/feature_tracker/TwoPointRansac.h"
// #include "semantic_slam/feature_tracker/ORBextractor.h"

namespace ORB_SLAM2 {
class ORBextractor;
}

class FeatureTracker
{
  public:
    struct Params
    {
        int ransac_iterations;
        int feature_spacing;
        int max_features_per_im;
        int keyframe_spacing;

        bool use_2pt_ransac;

        double feat_quality; // "quality" threshold within the detector. e.g.
                             // harris default is 0.01

        double sqrt_samp_thresh; // RANSAC reprojection error threshold

        std::string feature_type;

        Params()
          : ransac_iterations(100)
          , feature_spacing(10)
          , max_features_per_im(100)
          , keyframe_spacing(15)
          , use_2pt_ransac(true)
          , feat_quality(0.01)
          , sqrt_samp_thresh(5.0e-4)
          , feature_type("HARRIS")
        {}
    };

    struct TrackedFeature
    {
        TrackedFeature() {}

        TrackedFeature(cv::Point2f pt_in, size_t frame, size_t pt)
          : pt(pt_in)
          , frame_id(frame)
          , pt_id(pt)
          , n_images_in(1)
        {}

        TrackedFeature(cv::Point2f pt_in, size_t frame, size_t pt, float size)
          : pt(pt_in)
          , frame_id(frame)
          , pt_id(pt)
          , size(size)
          , n_images_in(1)
        {}

        cv::KeyPoint kp;
        cv::Mat descriptor;

        cv::Point2f pt;
        size_t frame_id;
        size_t pt_id;

        float size;
        double pixel_sigma2;

        size_t n_images_in;
    };

    struct Frame
    {
        ros::Time stamp;
        uint32_t seq;
        cv::Mat image;
        
        std::vector<TrackedFeature> feature_tracks;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
    };

    // FeatureTracker();
    FeatureTracker(const Params& params);

    void setImuDT(double dt) { imu_dt_ = dt; }

    void imgCallback(const sensor_msgs::ImageConstPtr& msg);
    // void camInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg);

    void addImage(Frame&& new_frame);

    bool addKeyframeTime(ros::Time t, std::vector<TrackedFeature>& tracks);

    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg);

    void gyroBiasCallback(const geometry_msgs::Vector3Stamped::ConstPtr& msg);

    Eigen::Matrix3d computeFrameRotation(ros::Time frame_time);

    // void publishKeyframe(const std_msgs::Header& header);

    void setCameraCalibration(double fx,
                              double fy,
                              double s,
                              double u0,
                              double v0,
                              double k1,
                              double k2,
                              double p1,
                              double p2);

    void setCameraExtrinsics(std::vector<double> I_p_C,
                             std::vector<double> I_q_C);

    void extractKeypointsDescriptors(Frame& frame);

    void trackFeaturesForward(size_t idx1);

    void setTrackingFramerate(double frame_rate);

  private:
    // void trackFeatures(const std::vector<cv::KeyPoint>& new_kps, const
    // cv::Mat& new_descriptors, boost::optional<Eigen::Matrix3d>
    // R=boost::none);

    // void extractDescriptors(const std::vector<TrackedFeature>& features,
    // cv::Mat& descriptors);

    void tryProcessNextImage();

    void addNewKeyframeFeatures(Frame& frame);

    Params params_;

    size_t frames_received_;
    size_t n_keyframes_;

    size_t n_features_extracted_;

    // cv::Ptr<cv::FeatureDetector> detector_;
    boost::shared_ptr<ORB_SLAM2::ORBextractor> orb_;

    std::deque<sensor_msgs::Imu::ConstPtr> imu_queue_;
    size_t last_imu_seq_;
    ros::Time last_imu_time_;
    ros::Time last_integrated_imu_time_;
    double imu_dt_;
    Eigen::Vector3d gyro_bias_;
    bool received_bias_;

    double tracking_framerate_;
    double image_period_;

    std::deque<Frame> image_buffer_;
    int last_keyframe_seq_;

    Eigen::Quaterniond I_q_C_;
    Eigen::Vector3d I_p_C_;

    // Eigen::Matrix3d keyframe_dR_;

    std::mutex buffer_mutex_;

    Eigen::Vector3d previous_omega_; //< should be always equal to omega at time
                                     // t = last_integrated_imu_time_;

    std::deque<sensor_msgs::ImageConstPtr> img_queue_;
    int64_t last_img_seq_;

    // cv::Mat new_img_;
    // cv::Mat last_img_;
    ros::Time last_keyframe_time_;

    // std::vector<TrackedFeature> tracked_features_;
    // std::vector<cv::Point2f> tracked_feature_pts_;

    ros::NodeHandle nh_;
    // image_transport::Publisher pub_images_;
    image_transport::Publisher pub_marked_images_;
    // ros::Publisher pub_camera_info_;
    // ros::Publisher pub_frame_features_;

    FivePointRansac five_ransac_;
    TwoPointRansac two_ransac_;

    std::unordered_map<size_t, cv::Scalar> pt_colors_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif
