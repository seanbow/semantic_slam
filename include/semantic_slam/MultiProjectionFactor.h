#pragma once

#include "semantic_slam/CeresFactor.h"
#include "semantic_slam/Common.h"
#include "semantic_slam/Pose3.h"
#include "semantic_slam/VectorNode.h"

#include <ceres/ceres.h>
#include <eigen3/Eigen/Core>

class CameraCalibration;
class CeresFactor;
class SE3Node;

namespace gtsam {
class Cal3DS2;

template<typename Pose, typename Landmark, typename Calibration>
class GenericProjectionFactor;
}

class MultiProjectionFactor
  : public CeresFactor
  , public ceres::CostFunction
{
  public:
    using This = MultiProjectionFactor;
    using Ptr = boost::shared_ptr<This>;

    MultiProjectionFactor(Vector3dNodePtr landmark_node,
                          const Pose3& body_T_sensor,
                          boost::shared_ptr<CameraCalibration> calibration,
                          double reprojection_error_threshold,
                          int tag = 0);

    boost::shared_ptr<CeresFactor> clone() const;

    void addToProblem(boost::shared_ptr<ceres::Problem> problem);
    void removeFromProblem(boost::shared_ptr<ceres::Problem> problem);

    void internalAddToProblem(boost::shared_ptr<ceres::Problem> problem);
    void internalRemoveFromProblem(boost::shared_ptr<ceres::Problem> problem);

    void addMeasurement(boost::shared_ptr<SE3Node> body_pose_node,
                        const Eigen::Vector2d& pixel_coords,
                        const Eigen::Matrix2d& msmt_covariance);

    bool decideIfTriangulate(const aligned_vector<Pose3>& body_poses) const;

    int index() const { return landmark()->index(); }
    unsigned char chr() const { return landmark()->chr(); }
    Symbol symbol() const { return landmark()->symbol(); }

    Vector3dNodePtr landmark() const
    {
        return boost::static_pointer_cast<Vector3dNode>(nodes_[0]);
    }

    boost::shared_ptr<SE3Node> camera_node(int i) const
    {
        return boost::static_pointer_cast<SE3Node>(nodes_[i + 1]);
    }

    size_t nMeasurements() const;

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const;

    bool inGraph() const { return in_graph_; }

    void triangulate(const aligned_vector<Pose3>& body_poses) const;

    bool triangulation_good() const { return triangulation_good_; }

    void createGtsamFactors() const;

    void addToGtsamGraph(
      boost::shared_ptr<gtsam::NonlinearFactorGraph> graph) const;

  private:
    Pose3 I_T_C_;
    boost::shared_ptr<CameraCalibration> calibration_;

    aligned_vector<Eigen::Vector2d> msmts_;
    aligned_vector<Eigen::Matrix2d> covariances_;
    aligned_vector<Eigen::Matrix2d> sqrt_informations_;

    std::vector<double*> parameter_blocks_;

    double reprojection_error_threshold_;

    bool in_graph_;
    std::vector<boost::shared_ptr<ceres::Problem>> problems_;

    mutable bool triangulation_good_;

    mutable aligned_vector<Pose3> triangulation_poses_;

    using GtsamFactorType = gtsam::
      GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3DS2>;
    mutable std::vector<boost::shared_ptr<GtsamFactorType>> gtsam_factors_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};