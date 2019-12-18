#pragma once

#include "semantic_slam/CeresStructureFactor.h"
#include "semantic_slam/Common.h"
#include "semantic_slam/FactorGraph.h"
#include "semantic_slam/keypoints/EstimatedKeypoint.h"
#include "semantic_slam/keypoints/geometry.h"
#include "semantic_slam/pose_math.h"

#include <memory>
#include <mutex>

#include "semantic_slam/keypoints/StructureOptimizationProblem.h"

#include <boost/enable_shared_from_this.hpp>

#include <gtsam/nonlinear/Values.h>

class SemanticKeyframe;
class SemanticMapper;

class EstimatedObject : public boost::enable_shared_from_this<EstimatedObject>
{
  public:
    using Ptr = boost::shared_ptr<EstimatedObject>;

    static EstimatedObject::Ptr Create(
      boost::shared_ptr<FactorGraph> graph,
      boost::shared_ptr<FactorGraph> semantic_graph,
      const ObjectParams& params,
      geometry::ObjectModelBasis object_model,
      uint64_t object_id,
      uint64_t first_keypoint_id,
      const ObjectMeasurement& msmt,
      const Pose3& G_T_C,
      const Pose3& I_T_C,
      std::string platform,
      boost::shared_ptr<CameraCalibration> calibration,
      SemanticMapper* mapper);

    // double computeMeasurementLikelihood(const ObjectMeasurement& msmt) const;
    double computeMahalanobisDistance(const ObjectMeasurement& msmt) const;

    void addKeypointMeasurements(const ObjectMeasurement& msmt, double weight);

    // void updateAndCheck(uint64_t pose_id, const gtsam::Values& estimate);
    void update(boost::shared_ptr<SemanticKeyframe> keyframe);

    void updateGraphFactors();

    // bool checkMerge(const EstimatedObject& other_obj);

    // void mergeWith(EstimatedObject& obj2);

    void removeFromEstimation();

    void markBad() { is_bad_ = true; }

    // int64_t getWhenAddedToGraph() { if (inGraph()) return
    // pose_added_to_graph_; else return -1; }

    Pose3& pose() { return pose_; }
    const Pose3& pose() const { return pose_; }

    void commitGraphSolution();
    void commitGraphSolution(boost::shared_ptr<FactorGraph> graph);
    void prepareGraphNode();

    void commitGtsamSolution(const gtsam::Values& values);

    void applyTransformation(const Pose3& old_T_new);

    // void setPose(const Pose3 &pose)
    // {
    //   // pose_ = pose;
    //   graph_pose_node_->pose() = pose;
    // }

    bool inGraph() const;

    bool bad() const { return is_bad_; }

    uint64_t id() const { return id_; }

    void setClassid(uint64_t class_id) { classid_ = class_id; }
    uint64_t classid() const { return classid_; }

    std::string obj_name() const { return obj_name_; }

    size_t numKeypoints() const { return keypoints_.size(); }

    int first_seen() const { return first_seen_; }
    uint64_t last_seen() const { return last_seen_; }

    void setIsVisible(boost::shared_ptr<SemanticKeyframe> kf);

    // uint64_t lastVisible() const { return last_visible_; }

    std::vector<int64_t> getKeypointIndices() const;
    const std::vector<EstimatedKeypoint::Ptr>& getKeypoints() const;
    const std::vector<EstimatedKeypoint::Ptr>& keypoints() const;

    const aligned_vector<ObjectMeasurement>& getMeasurements() const
    {
        return measurements_;
    }

    void initializeFromMeasurement(const ObjectMeasurement& msmt,
                                   const Pose3& G_T_C);

    Eigen::MatrixXd getKeypointPositionMatrix() const;

    Eigen::VectorXd getKeypointOptimizationWeights() const;

    void optimizeStructure();

    bool readyToAddToGraph();

    void addToGraph();

    void setConstantInGraph();
    void setVariableInGraph();

    // double getStructureError() const;

    Eigen::MatrixXd getPlx(Key l_key, Key x_key) const;

    int64_t findKeypointByClass(uint64_t classid) const;
    int64_t findKeypointByKey(Key key) const;

    const std::vector<boost::shared_ptr<SemanticKeyframe>>&
    keyframe_observations() const
    {
        return keyframe_observations_;
    }

    Key key() const { return symbol_shorthand::O(id()); }

    // void createMarginals(const std::vector<Key>& extra_keys);

    // utils::ProjectionFactor::shared_ptr getProjectionFactor(const
    // KeypointMeasurement &kp_msmt) const;

  private:
    EstimatedObject(boost::shared_ptr<FactorGraph> graph,
                    boost::shared_ptr<FactorGraph> semantic_graph,
                    const ObjectParams& params,
                    geometry::ObjectModelBasis object_model,
                    uint64_t object_id,
                    uint64_t first_keypoint_id,
                    const ObjectMeasurement& msmt,
                    const Pose3& G_T_C,
                    const Pose3& I_T_C,
                    std::string platform,
                    boost::shared_ptr<CameraCalibration> calibration,
                    SemanticMapper* mapper);

    void initializePose(const ObjectMeasurement& msmt, const Pose3& G_T_C);
    void initializeKeypoints(const ObjectMeasurement& msmt);
    void initializeStructure(const ObjectMeasurement& msmt);

    // void updatePositionFromKeypoints();

    // void addKeypointToGraph(EstimatedKeypoint::Ptr& kp,
    // 						std::vector<FactorInfoPtr>&
    // new_factors, 						gtsam::Values&
    // values);

    // void sanityCheck(NodeInfoConstPtr node_info);

    size_t countObservedFeatures(const ObjectMeasurement& msmt) const;

    // bool checkObjectExploded() const;

    boost::shared_ptr<FactorGraph> graph_;
    boost::shared_ptr<FactorGraph> semantic_graph_;

    uint64_t id_; // ID within the *pose graph*
    uint64_t first_kp_id_;
    uint64_t classid_;
    std::string obj_name_;
    bool in_graph_;
    bool is_bad_;

    int first_seen_;
    uint64_t last_seen_;
    uint64_t last_visible_;
    uint64_t last_optimized_;

    // uint64_t pose_added_to_graph_;

    ObjectParams params_;

    SE3NodePtr graph_pose_node_;
    VectorXdNodePtr graph_coefficient_node_;

    Pose3 pose_;

    Pose3 I_T_C_;
    std::string platform_;
    boost::shared_ptr<CameraCalibration> camera_calibration_;

    aligned_vector<ObjectMeasurement> measurements_;

    // List of keyframes that observed this object
    std::vector<boost::shared_ptr<SemanticKeyframe>> keyframe_observations_;

    std::vector<EstimatedKeypoint::Ptr> keypoints_;

    /** Flag on whether the structure factor graph has been modified since the
     * last marginals computation */
    bool modified_;

    // FactorInfoPtr structure_factor_;
    CeresStructureFactorPtr structure_factor_;

    geometry::ObjectModelBasis model_;
    size_t m_, k_; // number of model keypoints and number of basis directions,
                   // respectively
    Eigen::VectorXd basis_coefficients_;

    boost::shared_ptr<StructureOptimizationProblem> structure_problem_;
    mutable std::mutex problem_mutex_;

    SemanticMapper* mapper_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  private:
    // copying breaks back-references from keypoints
    EstimatedObject(const EstimatedObject& other) = delete;
    EstimatedObject() = delete;
    EstimatedObject& operator=(const EstimatedObject& other) = delete;
};
