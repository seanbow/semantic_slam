#pragma once

#include "semantic_slam/Common.h"
#include "semantic_slam/Handler.h"
#include "semantic_slam/pose_math.h"
#include "semantic_slam/keypoints/geometry.h"
#include "semantic_slam/CameraCalibration.h"
#include "semantic_slam/keypoints/EstimatedObject.h"
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/SemanticKeyframe.h"
#include "semantic_slam/Presenter.h"

#include <object_pose_interface_msgs/KeypointDetections.h>

#include <nav_msgs/Odometry.h>
#include <mutex>
// #include <shared_mutex>
#include <deque>
#include <unordered_map>
#include <unordered_set>
// #include <gtsam/geometry/Pose3.h>

class ExternalOdometryHandler;
class GeometricFeatureHandler;

class SemanticMapper
{
public:
    SemanticMapper();

    void setup();

    void anchorOrigin();

    bool haveNextKeyframe();
    bool tryFetchNextKeyframe();
    bool updateNextKeyframeObjects();
    void tryAddObjectsToGraph();
    bool tryOptimize();

    void processMessagesUpdateObjectsThread();
    void addObjectsAndOptimizeGraphThread();

    void computeCovariances();

    Eigen::MatrixXd getPlx(Key key1, Key key2);

    void start();

    void msgCallback(const object_pose_interface_msgs::KeypointDetections::ConstPtr& msg);

    aligned_vector<ObjectMeasurement>
    processObjectDetectionMessage(const object_pose_interface_msgs::KeypointDetections& msg, Key keyframe_key);
    
    void loadModelFiles(std::string path);

    bool loadCalibration();

    bool loadParameters();

    void visualizeObjectMeshes() const;
    void visualizeObjects() const;

    bool keepFrame(const object_pose_interface_msgs::KeypointDetections& msg);

    void setOdometryHandler(boost::shared_ptr<ExternalOdometryHandler> odom);
    void setGeometricFeatureHandler(boost::shared_ptr<GeometricFeatureHandler> odom);

    void addPresenter(boost::shared_ptr<Presenter> presenter);

    void prepareGraphNodes();
    void commitGraphSolution();

    std::vector<SemanticKeyframe::Ptr> addNewOdometryToGraph();

    void freezeNonCovisible(const std::vector<SemanticKeyframe::Ptr>& target_frames);
    void unfreezeAll();

    std::mutex& map_mutex() { return map_mutex_; }

    // double computeMahalanobisDistance(const ObjectMeasurement& msmt,
    //                                   const EstimatedObject::Ptr& obj);

    // double computeMahalanobisDistance(const KeypointMeasurement& msmt,
    //                                   const EstimatedKeypoint::Ptr& kp);

    const std::vector<SemanticKeyframe::Ptr>& keyframes() { return keyframes_; }

    bool needToComputeCovariances();

    SemanticKeyframe::Ptr getKeyframeByIndex(int index);
    SemanticKeyframe::Ptr getKeyframeByKey(Key key);

private:
    boost::shared_ptr<FactorGraph> graph_;
    std::mutex graph_mutex_; // wish this could be a shared_mutex
    std::mutex map_mutex_;

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    ros::Subscriber subscriber_;

	std::deque<object_pose_interface_msgs::KeypointDetections> msg_queue_;
    std::mutex queue_mutex_;

    size_t received_msgs_;
    size_t last_msg_seq_;

    size_t measurements_processed_;

    size_t n_landmarks_;

    unsigned char node_chr_;

    SemanticKeyframe::Ptr next_keyframe_;

    Eigen::MatrixXd last_kf_covariance_;
    ros::Time last_kf_covariance_time_;
    aligned_map<int, Eigen::MatrixXd> Plxs_;
    size_t Plxs_index_;
    ros::Time Plxs_time_;

    std::unordered_set<int> unfrozen_kfs_;
    std::unordered_set<int> unfrozen_objs_;

    aligned_map<std::string, geometry::ObjectModelBasis> object_models_;

    std::vector<EstimatedObject::Ptr> estimated_objects_;

    boost::shared_ptr<ExternalOdometryHandler> odometry_handler_;
    boost::shared_ptr<GeometricFeatureHandler> geom_handler_;

    // A list of tracking IDs that are associated with each estimated object
    // for example if the object tracks 3 and 8 are associated with the same physical object 2,
    // we should have object_track_ids_[3] = 2 and object_track_ids_[8] = 2.
    std::unordered_map<size_t, size_t> object_track_ids_;

    boost::shared_ptr<CameraCalibration> camera_calibration_;
    Pose3 I_T_C_;

    ObjectParams params_;

    std::vector<SemanticKeyframe::Ptr> keyframes_;

    std::vector<bool> 
    predictVisibleObjects(SemanticKeyframe::Ptr node);

    bool updateObjects(SemanticKeyframe::Ptr node,
                       const aligned_vector<ObjectMeasurement>& measurements,
                       const std::vector<size_t>& measurement_index,
                       const std::map<size_t, size_t>& known_das,
                       const Eigen::MatrixXd& weights,
                       const std::vector<size_t>& object_index);

    void processGeometricFeatureTracks(const std::vector<SemanticKeyframe::Ptr>& new_keyframes);

    ros::Publisher vis_pub_;

    std::vector<boost::shared_ptr<Presenter>> presenters_;

    bool running_;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};
