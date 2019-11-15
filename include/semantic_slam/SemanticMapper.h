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
// #include <gtsam/geometry/Pose3.h>

class OdometryHandler;

class SemanticMapper
{
public:
    SemanticMapper();

    void setup();

    bool updateObjects();
    void tryAddObjectsToGraph();
    bool tryOptimize();

    void processMessagesUpdateObjectsThread();
    void addObjectsAndOptimizeGraphThread();

    void computeLandmarkCovariances();

    void start();

    void msgCallback(const object_pose_interface_msgs::KeypointDetections::ConstPtr& msg);
    
    void loadModelFiles(std::string path);

    bool loadCalibration();

    bool loadParameters();

    void visualizeObjectMeshes() const;
    void visualizeObjects() const;

    bool keepFrame(ros::Time time);

    void setOdometryHandler(boost::shared_ptr<OdometryHandler> odom);

    void addPresenter(boost::shared_ptr<Presenter> presenter);

private:
    boost::shared_ptr<FactorGraph> graph_;
    std::mutex graph_mutex_; // wish this could be a shared_mutex

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

    aligned_map<std::string, geometry::ObjectModelBasis> object_models_;

    std::vector<EstimatedObject::Ptr> estimated_objects_;

    boost::shared_ptr<OdometryHandler> odometry_handler_;

    // A list of tracking IDs that are associated with each estimated object
    // for example if the object tracks 3 and 8 are associated with the same physical object 2,
    // we should have object_track_ids_[3] = 2 and object_track_ids_[8] = 2.
    std::unordered_map<size_t, size_t> object_track_ids_;

    boost::shared_ptr<CameraCalibration> camera_calibration_;
    Pose3 I_T_C_;

    ObjectParams params_;

    std::vector<SemanticKeyframe::Ptr> keyframes_;

    std::vector<bool> 
    getVisibleObjects(SE3Node::Ptr node);

    bool updateObjects(SE3Node::Ptr node,
                       const aligned_vector<ObjectMeasurement>& measurements,
                       const std::vector<size_t>& measurement_index,
                       const std::map<size_t, size_t>& known_das,
                       const Eigen::MatrixXd& weights,
                       const std::vector<size_t>& object_index);

    ros::Publisher vis_pub_;

    std::vector<boost::shared_ptr<Presenter>> presenters_;

    bool running_;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};
