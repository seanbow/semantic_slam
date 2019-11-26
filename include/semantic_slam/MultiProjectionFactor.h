#pragma once

#include "semantic_slam/Common.h"
#include "semantic_slam/CeresFactor.h"
#include "semantic_slam/CameraCalibration.h"
#include "semantic_slam/pose_math.h"
#include "semantic_slam/VectorNode.h"
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/CeresProjectionFactor.h"
#include "semantic_slam/CameraSet.h"

#include <ceres/ceres.h>
#include <eigen3/Eigen/Core>

class MultiProjectionFactor : public CeresFactor, public ceres::CostFunction
{
public:
    using This = MultiProjectionFactor;
    using Ptr = boost::shared_ptr<This>;

    MultiProjectionFactor(Vector3dNodePtr landmark_node, 
                         const Pose3& body_T_sensor,
                         boost::shared_ptr<CameraCalibration> calibration,
                         double reprojection_error_threshold,
                         int tag=0);

    void addToProblem(boost::shared_ptr<ceres::Problem> problem);
    void removeFromProblem(boost::shared_ptr<ceres::Problem> problem);

    void addMeasurement(SE3NodePtr body_pose_node,
                        const Eigen::Vector2d& pixel_coords, 
                        const Eigen::Matrix2d& msmt_covariance);

    int index() const { return landmark_node_->index(); }
    unsigned char chr() const { return landmark_node_->chr(); }
    Symbol symbol() const { return landmark_node_->symbol(); }

    Vector3dNodePtr landmark() { return landmark_node_; }

    size_t nMeasurements() const;

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

    bool inGraph() const { return in_graph_; }
    bool active() const { return active_; }

    void triangulate();

private:
    Vector3dNodePtr landmark_node_;
    Pose3 I_T_C_;
    boost::shared_ptr<CameraCalibration> calibration_;

    std::vector<SE3NodePtr> body_poses_;
    aligned_vector<Eigen::Vector2d> msmts_;
    // aligned_vector<Eigen::Matrix2d> covariances_;
    aligned_vector<Eigen::Matrix2d> sqrt_informations_;

    std::vector<double*> parameter_blocks_;

    // std::vector<CeresProjectionFactorPtr> projection_factors_;

    double reprojection_error_threshold_;

    bool in_graph_;
    bool active_;
    boost::shared_ptr<ceres::Problem> problem_;

    bool triangulation_good_;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};