#pragma once

#include "semantic_slam/Common.h"

#include <vector>

#include <boost/shared_ptr.hpp>
#include <ceres/ceres.h>

// #include <gtsam/geometry/Cal3DS2.h>

#include "semantic_slam/CameraCalibration.h"

#include "semantic_slam/keypoints/geometry.h"
#include "semantic_slam/pose_math.h"

class SemanticKeyframe;
class SE3Node;

class StructureOptimizationProblem
{
  public:
    using CovarianceBlocks =
      std::vector<std::pair<const double*, const double*>>;

    StructureOptimizationProblem(
      geometry::ObjectModelBasis model,
      boost::shared_ptr<CameraCalibration> camera_calibration,
      Pose3 body_T_camera,
      Eigen::VectorXd weights,
      ObjectParams params);

    void initializeKeypointPosition(size_t kp_id, const Eigen::Vector3d& p);
    void initializePose(Pose3 pose);

    void addKeypointMeasurement(const KeypointMeasurement& kp_msmt);
    void addCamera(boost::shared_ptr<SemanticKeyframe> keyframe,
                   bool use_constant_camera_pose = true);
    void setBasisCoefficients(const Eigen::VectorXd& coefficientS);

    boost::shared_ptr<Eigen::Vector3d> getKeypoint(size_t index) const;
    Pose3 getObjectPose() const;

    // Here landmark_index is the *local class id* of the landmark within this
    // object camera_index is the *global* camera pose id
    Eigen::MatrixXd getPlx(size_t camera_index);

    Eigen::VectorXd getBasisCoefficients() const;

    void computeCovariances();

    void solve();

    // void addAllBlockPairs(const std::vector<const double*>& to_add,
    //                       CovarianceBlocks& blocks) const;

  private:
    geometry::ObjectModelBasis model_;
    size_t m_, k_;

    boost::shared_ptr<CameraCalibration> camera_calibration_;
    Pose3 body_T_camera_;

    ceres::Problem ceres_problem_;

    Pose3 object_pose_;

    std::vector<boost::shared_ptr<Eigen::Vector3d>> kps_;

    Eigen::VectorXd basis_coefficients_;

    Eigen::VectorXd weights_;

    ceres::CostFunction* structure_cf_;
    ceres::LocalParameterization* pose_parameterization_;
    std::vector<ceres::CostFunction*> projection_cfs_;
    std::vector<ceres::CostFunction*> depth_cfs_;

    // Map from camera poses in this object to camera poses in the outer
    // structure std::unordered_map<size_t, size_t> camera_pose_ids_; size_t
    // num_poses_;

    std::unordered_map<size_t, boost::shared_ptr<SemanticKeyframe>> keyframes_;
    std::unordered_map<size_t, boost::shared_ptr<SE3Node>> local_pose_nodes_;

    ObjectParams params_;

    // Collection of pointers for ceres parameters in the order of structure
    // cost function no ownership here
    std::vector<double*> ceres_parameters_;

    boost::shared_ptr<ceres::Covariance> covariance_;

    bool solved_;
    bool have_covariance_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};
