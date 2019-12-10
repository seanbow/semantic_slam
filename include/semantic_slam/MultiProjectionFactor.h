#pragma once

#include "semantic_slam/CameraCalibration.h"
#include "semantic_slam/CameraSet.h"
#include "semantic_slam/CeresFactor.h"
#include "semantic_slam/CeresProjectionFactor.h"
#include "semantic_slam/Common.h"
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/VectorNode.h"
#include "semantic_slam/pose_math.h"

#include <ceres/ceres.h>
#include <eigen3/Eigen/Core>

namespace gtsam {
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

    CeresFactor::Ptr clone() const;

    void addToProblem(boost::shared_ptr<ceres::Problem> problem);
    void removeFromProblem(boost::shared_ptr<ceres::Problem> problem);

    void addMeasurement(SE3NodePtr body_pose_node,
                        const Eigen::Vector2d& pixel_coords,
                        const Eigen::Matrix2d& msmt_covariance);

    bool decideIfTriangulate(const aligned_vector<Pose3>& body_poses) const;

    int index() const { return landmark()->index(); }
    unsigned char chr() const { return landmark()->chr(); }
    Symbol symbol() const { return landmark()->symbol(); }

    Vector3dNodePtr landmark() const { return boost::static_pointer_cast<Vector3dNode>(nodes_[0]); }

    SE3NodePtr camera_node(int i) const { return boost::static_pointer_cast<SE3Node>(nodes_[i + 1]); }

    size_t nMeasurements() const;

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const;

    bool inGraph() const { return in_graph_; }

    void triangulate(const aligned_vector<Pose3>& body_poses) const;

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
    boost::shared_ptr<ceres::Problem> problem_;

    mutable bool triangulation_good_;

    mutable aligned_vector<Pose3> triangulation_poses_;

    using GtsamFactorType = gtsam::
      GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3DS2>;
    mutable std::vector<boost::shared_ptr<GtsamFactorType>> gtsam_factors_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};