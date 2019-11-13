#pragma once

// #include "semslam/Common.h"
// #include "semslam/FactorInfo.h"
#include "semantic_slam/Common.h"
#include "semantic_slam/keypoints/EstimatedKeypoint.h"
// #include "omnigraph/keypoints/StructureFactor.h"
#include "semantic_slam/keypoints/geometry.h"
#include "semantic_slam/FactorGraph.h"
#include "semantic_slam/pose_math.h"
#include "semantic_slam/CeresStructureFactor.h"
// #include "omnigraph/omnigraph.h"

#include <memory>

// #include <gtsam/geometry/Pose3.h>
// #include <gtsam/nonlinear/Values.h>
// #include <gtsam/base/FastVector.h>
// #include <gtsam/nonlinear/Marginals.h>
// #include <gtsam/sam/BearingRangeFactor.h>

// #include <rcta_worldmodel_msgs/Object.h>

#include "semantic_slam/keypoints/StructureOptimizationProblem.h"

#include <boost/enable_shared_from_this.hpp>

// using semslam::StructureFactor;
// using namespace omnigraph;

// namespace gtsam
// {
// // class ISAM2;
// class Values;
// class Pose3;
// class NonlinearFactorGraph;
// } // namespace gtsam

class EstimatedObject : public boost::enable_shared_from_this<EstimatedObject>
{
public:
  using Ptr = boost::shared_ptr<EstimatedObject>;

  // EstimatedObject(boost::shared_ptr<PoseGraphHandler> graph,
  // 				const ObjectParams& params,
  // 				uint64_t object_id,
  // 				uint64_t first_keypoint_id,
  // 				const ObjectMeasurement& msmt,
  // 				const gtsam::Pose3& G_T_C,
  // 				const gtsam::Pose3& I_T_C,
  //                 boost::shared_ptr<gtsam::Cal3DS2> calibration);

  static EstimatedObject::Ptr create(boost::shared_ptr<FactorGraph> graph, const ObjectParams &params,
                                     geometry::ObjectModelBasis object_model, uint64_t object_id,
                                     uint64_t first_keypoint_id, const ObjectMeasurement &msmt,
                                     const Pose3 &G_T_C, const Pose3 &I_T_C, std::string platform,
                                     boost::shared_ptr<CameraCalibration> calibration);

  // double computeMeasurementLikelihood(const ObjectMeasurement& msmt) const;
  double computeMahalanobisDistance(const ObjectMeasurement &msmt) const;

  void addKeypointMeasurements(const ObjectMeasurement &msmt, double weight);

  // void updateAndCheck(uint64_t pose_id, const gtsam::Values& estimate);
  void update(CeresNodePtr spur_node);

  // bool checkMerge(const EstimatedObject& other_obj);

  // void mergeWith(EstimatedObject& obj2);

  void removeFromEstimation();

  void markBad()
  {
    is_bad_ = true;
  }

  // int64_t getWhenAddedToGraph() { if (inGraph()) return pose_added_to_graph_; else return -1; }

  // std::vector<Key> getKeypointKeys() const;
  // std::vector<Key> getKeypointKeysInGraph() const;
  // std::vector<Key> getAllKeys() const;
  // std::vector<Key> getKeysInGraph() const;

  // const gtsam::Pose3& pose() const { return pose_; }
  Pose3 pose() const;

  void setPose(const Pose3 &pose)
  {
    // pose_ = pose;
    graph_pose_node_->pose() = pose;
  }

  bool inGraph() const;

  bool bad() const
  {
    return is_bad_;
  }

  uint64_t id() const
  {
    return id_;
  }

  void setClassid(uint64_t class_id) { classid_ = class_id; }
  uint64_t classid() const { return classid_; }

  std::string obj_name() const
  {
    return obj_name_;
  }

  size_t numKeypoints() const
  {
    return keypoints_.size();
  }

  // uint64_t lastSeen() const { return last_seen_; }

  void setIsVisible(CeresNodePtr node)
  {
    last_visible_ = node->index();
  }

  // uint64_t lastVisible() const { return last_visible_; }

  std::vector<int64_t> getKeypointIndices() const;
  const std::vector<EstimatedKeypoint::Ptr> &getKeypoints() const;

  const aligned_vector<ObjectMeasurement> &getMeasurements() const
  {
    return measurements_;
  }

  void initializeFromMeasurement(const ObjectMeasurement &msmt, const Pose3 &G_T_C);

  Eigen::MatrixXd getKeypointPositionMatrix() const;

  Eigen::VectorXd getKeypointOptimizationWeights() const;

  // gtsam::JointMarginal jointMarginalCovariance(const ObjectMeasurement &msmt) const;

  void optimizeStructure();

  // double getStructureError() const;

  // const gtsam::NonlinearFactorGraph &getStructureGraph()
  // {
  //   if (modified_)
  //     optimizeStructure();
  //   return structure_graph_;
  // }
  // const gtsam::Values &getStructureValues()
  // {
  //   if (modified_)
  //     optimizeStructure();
  //   return structure_optimization_values_;
  // }

  // geometry::StructureResult
  // optimizeStructureFromMeasurementSet(const aligned_vector<ObjectMeasurement>& object_measurements,
  // 									boost::shared_ptr<gtsam::Cal3DS2> camera_calibration,
  // 									const gtsam::Pose3& I_T_C,
  // 									boost::shared_ptr<omnigraph::Omnigraph> graph,
  // 									const geometry::ObjectModelBasis& model,
  // 									gtsam::Pose3& object_pose,
  // 									bool compute_covariance=false);

  Eigen::MatrixXd getPlx(Key l_key, Key x_key);

  // void createMarginals(const std::vector<Key>& extra_keys);

  // utils::ProjectionFactor::shared_ptr getProjectionFactor(const KeypointMeasurement &kp_msmt) const;

  // void addCameraPoseConstraints(gtsam::NonlinearFactorGraph &G) const;
  // void addProjectionConstraints(gtsam::NonlinearFactorGraph &G) const;
  // void addDepthConstraints(gtsam::NonlinearFactorGraph &G) const;

  // void setWorldModelObject(rcta_worldmodel_msgs::Object obj);
  // void setWorldModelObject(const boost::shared_ptr<rcta_worldmodel_msgs::Object> &obj_ptr);
  // boost::shared_ptr<rcta_worldmodel_msgs::Object> getWorldModelObject() const;

  // bool inWorldModel() const { return in_worldmodel_; }
  // void setInWorldModel(bool value) { in_worldmodel_ = value; }

private:
  EstimatedObject(boost::shared_ptr<FactorGraph> graph, const ObjectParams &params,
                  geometry::ObjectModelBasis object_model, uint64_t object_id, uint64_t first_keypoint_id,
                  const ObjectMeasurement &msmt, const Pose3 &G_T_C, const Pose3 &I_T_C,
                  std::string platform, boost::shared_ptr<CameraCalibration> calibration);

  int64_t findKeypointByClass(uint64_t classid) const;
  int64_t findKeypointByKey(Key key) const;

  void initializePose(const ObjectMeasurement &msmt, const Pose3 &G_T_C);
  void initializeKeypoints(const ObjectMeasurement &msmt);
  void initializeStructure(const ObjectMeasurement &msmt);

  // void updatePositionFromKeypoints();

  // void addKeypointToGraph(EstimatedKeypoint::Ptr& kp,
  // 						std::vector<FactorInfoPtr>& new_factors,
  // 						gtsam::Values& values);

  void tryAddSelfToGraph(const ObjectMeasurement &msmt);

  // void scaleStructure(double scale);

  // bool checkSafeToAdd(const gtsam::NonlinearFactorGraph& G, const gtsam::Values& new_values, const gtsam::Values&
  // state_estimate);

  // bool checkSafe(const gtsam::NonlinearFactorGraph& G,
  // 	const gtsam::Values& new_values,
  // 	const gtsam::Values& state_estimate,
  // 	const gtsam::ISAM2& isam);

  // gtsam::FactorIndices getFactorIndicesForRemoval() const;

  // void sanityCheck(NodeInfoConstPtr node_info);

  size_t countObservedFeatures(const ObjectMeasurement &msmt) const;

  // bool checkObjectExploded() const;

  // void updatePoseFromKeypoints();

  // bool checkKeypointMerge(const EstimatedKeypoint::Ptr& kp1,
  //                         const EstimatedKeypoint::Ptr& kp2);

  // gtsam::NonlinearFactorGraph structure_graph_;
  // gtsam::Values structure_optimization_values_;
  // boost::shared_ptr<gtsam::Marginals> structure_marginals_;

  // gtsam::NonlinearFactorGraph full_factor_graph_;
  // gtsam::Values full_values_;

  boost::shared_ptr<FactorGraph> graph_;

  // boost::shared_ptr<rcta_worldmodel_msgs::Object> managed_wm_object_;

  uint64_t id_;            // ID within the *pose graph*
  // uint64_t worldmodel_id_; // ID within the *world model* -- different from id_!
  uint64_t first_kp_id_;
  uint64_t classid_;
  std::string obj_name_;
  bool in_graph_;
  bool is_bad_;
  // bool in_worldmodel_;

  uint64_t last_seen_;
  uint64_t last_visible_;
  uint64_t last_optimized_;

  // uint64_t pose_added_to_graph_;

  ObjectParams params_;

  SE3NodePtr graph_pose_node_;
  VectorXdNodePtr graph_coefficient_node_;

  // Pose3 pose_;

  Pose3 I_T_C_;
  std::string platform_;
  boost::shared_ptr<CameraCalibration> camera_calibration_;

  aligned_vector<ObjectMeasurement> measurements_;

  std::vector<EstimatedKeypoint::Ptr> keypoints_;

  // std::vector<std::pair<FactorInfo, gtsam::NonlinearFactor::shared_ptr>> new_factors_;
  // std::vector<NodeInfoPtr> new_node_infos_;
  // gtsam::Values new_values_;

  // FactorInfo structure_factor_info_;
  // gtsam::NonlinearFactor::shared_ptr structure_factor_;

/** Flag on whether the structure factor graph has been modified since the last marginals computation */
  bool modified_;

  // FactorInfoPtr structure_factor_;
  CeresStructureFactorPtr structure_factor_;

  geometry::ObjectModelBasis model_;
  size_t m_, k_; // number of model keypoints and number of basis directions, respectively
  Eigen::VectorXd basis_coefficients_;

  boost::shared_ptr<StructureOptimizationProblem> structure_problem_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  // copying breaks back-references from keypoints
  EstimatedObject(const EstimatedObject &other) = delete;
  EstimatedObject() = delete;
  EstimatedObject &operator=(const EstimatedObject &other) = delete;
};
