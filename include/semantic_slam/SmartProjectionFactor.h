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

class SmartProjectionFactor : public CeresFactor, public ceres::CostFunction
{
public:
    using This = SmartProjectionFactor;
    using Ptr = boost::shared_ptr<This>;

    SmartProjectionFactor(const Pose3& body_T_sensor,
                         boost::shared_ptr<CameraCalibration> calibration,
                         double reprojection_error_threshold,
                         int tag=0);

    void addToProblem(boost::shared_ptr<ceres::Problem> problem);
    void removeFromProblem(boost::shared_ptr<ceres::Problem> problem);

    void addMeasurement(SE3NodePtr body_pose_node,
                        const Eigen::Vector2d& pixel_coords, 
                        const Eigen::Matrix2d& msmt_covariance);

    size_t nMeasurements() const;

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

    bool inGraph() const { return in_graph_; }

    void triangulate(const aligned_vector<Pose3>& body_poses) const;

    bool decideIfTriangulate(const aligned_vector<Pose3>& body_poses) const;

private:
    mutable Eigen::Vector3d landmark_position_;
    Pose3 I_T_C_;
    boost::shared_ptr<CameraCalibration> calibration_;

    std::vector<SE3NodePtr> body_poses_;
    aligned_vector<Eigen::Vector2d> msmts_;
    // aligned_vector<Eigen::Matrix2d> covariances_;
    aligned_vector<Eigen::Matrix2d> sqrt_informations_;

    mutable aligned_vector<Pose3> triangulation_poses_;

    std::vector<double*> parameter_blocks_;

    // std::vector<CeresProjectionFactorPtr> projection_factors_;

    double reprojection_error_threshold_;

    bool in_graph_;
    bool active_;
    boost::shared_ptr<ceres::Problem> problem_;

    mutable bool triangulation_good_;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};