#pragma once

#include "semantic_slam/CameraCalibration.h"
#include "semantic_slam/Common.h"
#include "semantic_slam/Handler.h"
#include "semantic_slam/Presenter.h"
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/SemanticKeyframe.h"
#include "semantic_slam/keypoints/EstimatedObject.h"
#include "semantic_slam/keypoints/geometry.h"
#include "semantic_slam/pose_math.h"

#include <object_pose_interface_msgs/KeypointDetections.h>

#include <memory>
#include <mutex>
#include <nav_msgs/Odometry.h>
// #include <shared_mutex>
#include <deque>
#include <unordered_map>
#include <unordered_set>

class GeometricFeatureHandler;
class LoopCloser;
class SemanticMapper;

class SemanticSmoother
{
  public:
    SemanticSmoother(ObjectParams params, SemanticMapper* mapper);

    void setOrigin(SemanticKeyframe::Ptr origin_frame);

    void tryAddObjectsToGraph();
    bool tryOptimize();

    void run();
    void stop();

    bool needToComputeCovariances();
    bool computeLatestCovariance();

    bool computeCovariances(const std::vector<SemanticKeyframe::Ptr>& frames);

    void prepareGraphNodes();
    void commitGraphSolution();

    std::vector<SemanticKeyframe::Ptr> addNewOdometryToGraph();

    void freezeNonCovisible(
      const std::vector<SemanticKeyframe::Ptr>& target_frames);
    void unfreezeAll();

    bool solveGraph();

    Eigen::MatrixXd mostRecentKeyframeCovariance()
    {
        return last_kf_covariance_;
    }

    ros::Time mostRecentCovarianceTime() { return last_kf_covariance_time_; }
    int mostRecentOptimizedKeyframeIndex() { return last_optimized_kf_index_; }

    void setGeometricFeatureHandler(
      boost::shared_ptr<GeometricFeatureHandler> geom);

    void setLoopCloser(boost::shared_ptr<LoopCloser> closer);

    // TODO make a parameter struct
    void setVerbose(bool verbose_optimization_);
    void setCovarianceDelay(double covariance_delay_);
    void setMaxOptimizationTime(double max_optimization_time_);
    void setSmoothingLength(int smoothing_length_);
    void setLoopClosureThreshold(int loop_closure_threshold_);

    void informLoopClosure() { invalidate_optimization_ = true; }

    // TODO we shouldn't need to expose these
    boost::shared_ptr<FactorGraph> graph() { return graph_; }
    boost::shared_ptr<FactorGraph> essential_graph()
    {
        return essential_graph_;
    }

  private:
    SemanticMapper* mapper_;

    ObjectParams params_;

    boost::shared_ptr<FactorGraph> graph_;
    boost::shared_ptr<FactorGraph> essential_graph_;
    std::mutex graph_mutex_;

    std::atomic<bool> invalidate_optimization_;

    Eigen::MatrixXd last_kf_covariance_;
    ros::Time last_kf_covariance_time_;
    int last_optimized_kf_index_;

    std::unordered_set<int> unfrozen_kfs_;
    std::unordered_set<int> unfrozen_objs_;

    bool include_geometric_features_;
    boost::shared_ptr<GeometricFeatureHandler> geom_handler_;

    boost::shared_ptr<LoopCloser> loop_closer_;

    bool verbose_optimization_;
    double covariance_delay_;
    double max_optimization_time_;
    int smoothing_length_;

    void processGeometricFeatureTracks(
      const std::vector<SemanticKeyframe::Ptr>& new_keyframes);

    int loop_closure_threshold_;

    bool computeCovariancesWithGtsam(
      const std::vector<SemanticKeyframe::Ptr>& frames);

    std::atomic<bool> running_;
};