#pragma once

#include "semantic_slam/Common.h"
#include "semantic_slam/Pose3.h"
#include "semantic_slam/keypoints/geometry.h"

#include <object_pose_interface_msgs/KeypointDetections.h>

#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

class OdometryHandler;
class GeometricFeatureHandler;
class LoopCloser;
class SemanticSmoother;
class EstimatedObject;
class SemanticKeyframe;
class Presenter;
class CameraCalibration;

class SemanticMapper
{
  public:
    SemanticMapper();

    enum class OperationMode
    {
        NORMAL,
        LOOP_CLOSURE_PENDING,
        LOOP_CLOSING
    };

    void setup();

    void anchorOrigin();

    bool haveNextKeyframe();
    boost::shared_ptr<SemanticKeyframe> tryFetchNextKeyframe();
    bool updateKeyframeObjects(boost::shared_ptr<SemanticKeyframe> frame);

    OperationMode operation_mode() { return operation_mode_; }
    void setOperationMode(OperationMode mode) { operation_mode_ = mode; }

    void computeDataAssociationWeights(
      boost::shared_ptr<SemanticKeyframe> frame);

    void processMessagesUpdateObjectsThread();

    Eigen::MatrixXd getPlx(Key key1, Key key2);

    Eigen::MatrixXd computePlxExact(Key l_key, Key x_key);

    void start();

    void msgCallback(
      const object_pose_interface_msgs::KeypointDetections::ConstPtr& msg);

    aligned_vector<ObjectMeasurement> processObjectDetectionMessage(
      const object_pose_interface_msgs::KeypointDetections& msg,
      Key keyframe_key);

    void loadModelFiles(std::string path);

    bool loadCalibration();

    bool loadParameters();

    bool keepFrame(const object_pose_interface_msgs::KeypointDetections& msg);

    void setOdometryHandler(boost::shared_ptr<OdometryHandler> odom);
    void setGeometricFeatureHandler(
      boost::shared_ptr<GeometricFeatureHandler> odom);

    void addPresenter(boost::shared_ptr<Presenter> presenter);

    std::mutex& map_mutex() { return map_mutex_; }
    std::mutex& geometric_map_mutex() { return present_mutex_; }

    const std::vector<boost::shared_ptr<SemanticKeyframe>>& keyframes()
    {
        return keyframes_;
    }

    boost::shared_ptr<SemanticKeyframe> getKeyframeByIndex(int index);
    boost::shared_ptr<SemanticKeyframe> getKeyframeByKey(Key key);

    boost::shared_ptr<SemanticKeyframe> getLastKeyframeInGraph();

    boost::shared_ptr<EstimatedObject> getObjectByKey(Key key);
    boost::shared_ptr<EstimatedObject> getObjectByIndex(int index);
    boost::shared_ptr<EstimatedObject> getObjectByKeypointKey(Key key);

    std::vector<boost::shared_ptr<EstimatedObject>> estimated_objects();

    bool checkLoopClosingDone();
    bool processLoopClosure();

  private:
    std::mutex map_mutex_;
    std::mutex present_mutex_;

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

    std::atomic<OperationMode> operation_mode_;
    std::atomic<bool> invalidate_local_optimization_;

    aligned_map<std::string, geometry::ObjectModelBasis> object_models_;

    std::vector<boost::shared_ptr<EstimatedObject>> estimated_objects_;

    boost::shared_ptr<OdometryHandler> odometry_handler_;

    bool include_geometric_features_;
    boost::shared_ptr<GeometricFeatureHandler> geom_handler_;

    bool verbose_optimization_;
    double covariance_delay_;
    double max_optimization_time_;
    int smoothing_length_;

    int loop_closure_threshold_;

    // A list of tracking IDs that are associated with each estimated object
    // for example if the object tracks 3 and 8 are associated with the same
    // physical object 2, we should have object_track_ids_[3] = 2 and
    // object_track_ids_[8] = 2.
    std::unordered_map<size_t, size_t> object_track_ids_;

    boost::shared_ptr<CameraCalibration> camera_calibration_;
    Pose3 I_T_C_;

    ObjectParams params_;

    std::vector<boost::shared_ptr<SemanticKeyframe>> keyframes_;

    std::deque<boost::shared_ptr<SemanticKeyframe>> pending_keyframes_;

    std::vector<bool> predictVisibleObjects(
      boost::shared_ptr<SemanticKeyframe> node);

    bool addMeasurementsToObjects(boost::shared_ptr<SemanticKeyframe> frame);
    bool removeMeasurementsFromObjects(
      boost::shared_ptr<SemanticKeyframe> frame);

    // returns number of keyframes processed
    int processPendingKeyframes();

    int createNewObject(const ObjectMeasurement& measurement,
                        const Pose3& map_T_camera,
                        double weight);

    ros::Publisher vis_pub_;

    std::vector<boost::shared_ptr<Presenter>> presenters_;

    bool running_;

    boost::shared_ptr<LoopCloser> loop_closer_;
    boost::shared_ptr<SemanticSmoother> smoother_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};
