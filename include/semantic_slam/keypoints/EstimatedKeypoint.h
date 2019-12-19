#pragma once

#include "semantic_slam/Common.h"

#include "semantic_slam/CameraCalibration.h"
#include "semantic_slam/Pose3.h"

#include <gtsam/nonlinear/Values.h>

class EstimatedObject;
class SemanticMapper;
class CeresFactor;
class FactorGraph;

template<int Dim>
class VectorNode;

class EstimatedKeypoint
{
  public:
    using Ptr = boost::shared_ptr<EstimatedKeypoint>;

    EstimatedKeypoint(boost::shared_ptr<FactorGraph> graph,
                      boost::shared_ptr<FactorGraph> semantic_graph,
                      const ObjectParams& params,
                      size_t id,
                      size_t object_id,
                      size_t class_id,
                      Pose3 I_T_C,
                      std::string platform,
                      boost::shared_ptr<CameraCalibration> camera_calib,
                      boost::shared_ptr<EstimatedObject> parent,
                      SemanticMapper* mapper);

    bool inGraph() const { return in_graph_; }

    size_t id() const { return global_id_; }

    size_t classid() const { return classid_; }

    bool bad() const { return is_bad_; }

    void commitGraphSolution();
    void commitGraphSolution(boost::shared_ptr<FactorGraph> graph);
    void prepareGraphNode();

    void commitGtsamSolution(const gtsam::Values& values);

    bool triangulate(boost::optional<double&> condition_number = boost::none);

    bool initialized() const { return initialized_; }

    boost::shared_ptr<VectorNode<3>> graph_node() { return graph_node_; }

    uint64_t local_id;

    void addMeasurement(const KeypointMeasurement& msmt, double weight);

    double computeMahalanobisDistance(const KeypointMeasurement& msmt) const;

    std::vector<Key> getObservedKeys() const;

    // double computeMeasurementLikelihood(const KeypointMeasurement& msmt)
    // const;

    void addToGraph();

    void addToGraphForced();

    void setConstantInGraph();
    void setVariableInGraph();

    double totalMahalanobisError() const;
    double mahalDistanceThreshold() const;

    double measurementWeightSum() const;
    double detectionScoreSum() const { return detection_score_sum_; }

    bool checkSafeToAdd();

    void tryAddProjectionFactors();

    //   void serialize(const std::string& file_name) const;

    // void setObjectIsInGraph() { object_in_graph_ = true; }

    size_t nMeasurements() const;

    aligned_vector<KeypointMeasurement> measurements() const
    {
        return measurements_;
    }

    double maxMahalanobisDistance() const;

    const Eigen::Vector3d& position() const { return global_position_; }
    Eigen::Vector3d& position() { return global_position_; }

    const Eigen::Matrix3d& covariance() const { return global_covariance_; }
    Eigen::Matrix3d& covariance() { return global_covariance_; }

    void removeFromEstimation();

    void initializeFromMeasurement(const KeypointMeasurement& msmt);

    boost::shared_ptr<EstimatedObject> parent_object() { return parent_; }

    Key key() const;

  private:
    void initializePosition(const KeypointMeasurement& msmt);

    boost::shared_ptr<FactorGraph> graph_;
    boost::shared_ptr<FactorGraph> semantic_graph_;
    ObjectParams params_;
    uint64_t global_id_;
    uint64_t object_id_;
    uint64_t classid_;
    Pose3 I_T_C_;
    std::string platform_;
    boost::shared_ptr<CameraCalibration> camera_calibration_;

    boost::shared_ptr<VectorNode<3>> graph_node_;

    Eigen::Vector3d global_position_;
    Eigen::Matrix3d global_covariance_;

    bool in_graph_;
    bool is_bad_;
    // bool object_in_graph_;

    bool initialized_;

    double detection_score_sum_;

    // uint64_t last_seen_;

    std::vector<boost::shared_ptr<CeresFactor>> projection_factors_;
    std::vector<double> measurement_weights_;

    Eigen::Vector3d local_position_; // within containing object

    aligned_vector<KeypointMeasurement> measurements_;

    boost::shared_ptr<EstimatedObject> parent_;

    SemanticMapper* mapper_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};