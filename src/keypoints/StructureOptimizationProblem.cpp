#include "semantic_slam/keypoints/StructureOptimizationProblem.h"

#include <algorithm>
#include "semantic_slam/Symbol.h"

#include "semantic_slam/ceres_cost_terms/ceres_pose_prior.h"
#include "semantic_slam/ceres_cost_terms/ceres_projection.h"
// #include "omnigraph/keypoints/ceres_quaternion_parameterization.h"
// #include "semantic_slam/ceres_range.h"
#include "semantic_slam/ceres_cost_terms/ceres_structure.h"

#include "semantic_slam/keypoints/ceres_camera_constraint.h"

#include "semantic_slam/keypoints/geometry.h"

#include "semantic_slam/SemanticKeyframe.h"

void
StructureOptimizationProblem::setBasisCoefficients(
  const Eigen::VectorXd& coeffs)
{
  basis_coefficients_ = coeffs;
}

Eigen::VectorXd
StructureOptimizationProblem::getBasisCoefficients() const
{
  return basis_coefficients_;
}

boost::shared_ptr<Eigen::Vector3d>
StructureOptimizationProblem::getKeypoint(size_t index) const
{
  return kps_[index];
}

Pose3
StructureOptimizationProblem::getObjectPose() const
{
  return object_pose_;
}

Eigen::MatrixXd
StructureOptimizationProblem::getPlx(size_t camera_index)
{
  size_t Plx_dim = 6 + 3*m_;

  Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(Plx_dim, Plx_dim);

  double buf[36];

  if (!have_covariance_) {
    computeCovariances();
  }

  // if (have_covariance_) {

  //   // Just get Pll for now...
  //   covariance_->GetCovarianceBlock(kps_[landmark_index]->data(),
  //                                   kps_[landmark_index]->data(), Pll);
  //   cov.block<3, 3>(0, 0) =
  //     Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(Pll);

  //   return cov;
  // } 
  
  if (!have_covariance_) {
    cov = Eigen::MatrixXd::Identity(Plx_dim, Plx_dim);
    cov.bottomRightCorner<3,3>() = Eigen::Matrix3d::Zero();
    return cov;
  }


  // what's crazy is that sometimes camera_index won't be in our list of 
  // optimized camera poses. 
  // for many purposes we just need one *near* it so use the closest one 
  // and warn about it
  if (keyframes_.find(camera_index) == keyframes_.end())
  {
    size_t closest_index = 0;
    size_t index_distance = std::numeric_limits<size_t>::max();

    for (auto& frame_pair : keyframes_) {
      // std::cout << "Index " << frame_pair.first << " has distance " << std::abs((int)camera_index - (int)frame_pair.first) << std::endl;
      if (std::abs((int)camera_index - (int)frame_pair.first) < index_distance) {
        closest_index = frame_pair.first;
        index_distance = std::abs((int)closest_index - (int)frame_pair.first);
      }
    }

    // ROS_WARN_STREAM("WARNING: Camera pose " << camera_index << " not in optimization values. Using " 
    //                   << closest_index << " instead.");

    camera_index = closest_index;
  }

  std::vector<double*> parameter_blocks;
  std::vector<size_t> block_sizes;
  for (auto& kp : kps_) {
    parameter_blocks.push_back(kp->data());
    block_sizes.push_back(3);
  }

  parameter_blocks.push_back(local_pose_nodes_[camera_index]->pose().rotation_data());
  block_sizes.push_back(3); // <-- 3 because these blocks are in the tangent space
  parameter_blocks.push_back(local_pose_nodes_[camera_index]->pose().translation_data());
  block_sizes.push_back(3);

  size_t index_i = 0;
  size_t index_j = 0;

  using RowMajorMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  for (int i = 0; i < parameter_blocks.size(); ++i) {
    for (int j = i; j < parameter_blocks.size(); ++j) {
      covariance_->GetCovarianceBlockInTangentSpace(parameter_blocks[i], parameter_blocks[j], buf);
      cov.block(index_i, index_j, block_sizes[i], block_sizes[j])
          = Eigen::Map<RowMajorMatrixXd>(buf, block_sizes[i], block_sizes[j]);

      index_j += block_sizes[j];
    }

    index_i += block_sizes[i];
    index_j = index_i;
  }

  return cov.selfadjointView<Eigen::Upper>();
}

StructureOptimizationProblem::StructureOptimizationProblem(
  geometry::ObjectModelBasis model, boost::shared_ptr<CameraCalibration> camera_calibration,
  Pose3 body_T_camera, Eigen::VectorXd weights, ObjectParams params)
  : model_(model)
  , camera_calibration_(camera_calibration)
  , body_T_camera_(body_T_camera)
  , object_pose_(Pose3::Identity())
  , weights_(weights)
  // , num_poses_(0)
  , params_(params)
  , solved_(false),
  have_covariance_(false)
{
  m_ = model_.mu.cols();
  k_ = model_.pc.rows() / 3;

  basis_coefficients_ = Eigen::VectorXd::Zero(k_);

  // ceres::Problem will take ownership of this pointer
  quaternion_parameterization_ = new QuaternionLocalParameterization;

  for (size_t i = 0; i < m_; ++i) {
    kps_.push_back(util::allocate_aligned<Eigen::Vector3d>());
  }

  // weights_ = Eigen::VectorXd::Ones(m_);

  structure_cf_ = StructureCostTerm::Create(
    model_, weights_, params_.structure_regularization_factor);

  // Accumulate data pointers for ceres parameters
  ceres_parameters_.push_back(object_pose_.rotation_data());
  ceres_parameters_.push_back(object_pose_.translation_data());
  for (size_t i = 0; i < m_; ++i) {
    ceres_parameters_.push_back(kps_[i]->data());
  }
  if (k_ > 0) {
    ceres_parameters_.push_back(basis_coefficients_.data());
  }

  ceres::LossFunction* structure_loss = new ceres::ScaledLoss(
    NULL, params_.structure_error_coefficient, ceres::TAKE_OWNERSHIP);

  ceres_problem_.AddResidualBlock(structure_cf_, structure_loss,
                                  ceres_parameters_);
  ceres_problem_.SetParameterization(object_pose_.rotation_data(),
                                     quaternion_parameterization_);
}

void
StructureOptimizationProblem::addKeypointMeasurement(
  const KeypointMeasurement& kp_msmt)
{
  // size_t pose_index;
  // if (camera_pose_ids_.find(kp_msmt.measured_symbol.index()) ==
  //     camera_pose_ids_.end()) {
  //   camera_pose_ids_[kp_msmt.measured_symbol.index()] = num_poses_;
  //   num_poses_++;
  // }

  // ceres::LossFunction* cauchy_loss =
  //   new ceres::CauchyLoss(params_.robust_estimator_parameter);

  ceres::LossFunction* huber_loss = 
      new ceres::HuberLoss(params_.robust_estimator_parameter);

  // double effective_sigma = kp_msmt.pixel_sigma / kp_msmt.score;

  double effective_sigma = kp_msmt.pixel_sigma;

  Eigen::Matrix2d msmt_covariance =
    effective_sigma * effective_sigma * Eigen::Matrix2d::Identity();

  Eigen::Quaterniond body_q_camera(body_T_camera_.rotation_data());
  ceres::CostFunction* projection_cf = ProjectionCostTerm::Create(
    kp_msmt.pixel_measurement, msmt_covariance, body_q_camera,
    body_T_camera_.translation(), camera_calibration_);

  // ceres_problem_.AddResidualBlock(projection_cf, NULL,
  // object_pose_.rotation_data(), object_pose_.translation_data(),
  // kps_[kp_msmt.kp_class_id]->data());
  auto& pose_node = local_pose_nodes_[Symbol(kp_msmt.measured_key).index()];
  ceres_problem_.AddResidualBlock(
    projection_cf, huber_loss, pose_node->pose().rotation_data(),
    pose_node->pose().translation_data(), kps_[kp_msmt.kp_class_id]->data());

  // // Add depth if available
  // if (params_.include_depth_constraints && kp_msmt.measured_depth > 0) {
  //   double depth_covariance =
  //     kp_msmt.measured_depth_sigma * kp_msmt.measured_depth_sigma;
  //   ceres::CostFunction* range_cf = RangeCostTerm::Create(
  //     kp_msmt.measured_depth, depth_covariance, body_T_camera_.rotation(),
  //     body_T_camera_.translation());

  //   ceres_problem_.AddResidualBlock(
  //     range_cf, cauchy_loss, cam_pose.rotation_data(),
  //     cam_pose.translation_data(), kps_[kp_msmt.kp_class_id]->data());
  // }

  solved_ = false;
  have_covariance_ = false;
}

void
StructureOptimizationProblem::solve()
{
  ceres::Solver::Options options;
  // options.linear_solver_type = ceres::DENSE_QR;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  // options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &ceres_problem_, &summary);

  // std::cout << summary.FullReport() << "\n";
  // std::cout << summary.BriefReport() << "\n";
  // for (size_t i = 0; i < kp_ptrs.size(); ++i)
  // {
  //   std::cout << "kp " << i << ": " << initial_kps[i].transpose() << " -> "
  //   << kp_ptrs[i]->transpose() << std::endl;
  // }

  // Eigen::Matrix3d R = math::quat2rot(object_pose_.rotation());

  // std::cout << "R = \n" << R << std::endl;
  // std::cout << "t = \n" << object_pose_.translation() << std::endl;

  solved_ = true;
}

// void
// StructureOptimizationProblem::addAllBlockPairs(
//   const std::vector<const double*>& to_add, CovarianceBlocks& blocks) const
// {
//   for (size_t i = 0; i < to_add.size(); ++i) {
//     for (size_t j = i + 1; j < to_add.size(); ++j) {
//       blocks.push_back(std::make_pair(to_add[i], to_add[j]));
//     }
//   }
// }

void
StructureOptimizationProblem::computeCovariances()
{
  ceres::Covariance::Options cov_options;

  // NOTE: with this false, the structure loss is artificially small even though we may not 
  // want the cauchy losses on the projections applied here. TODO modify structure cost fn so
  // this isn't the case
  cov_options.apply_loss_function = true;

  // cov_options.num_threads = 2; // can tweak this

  cov_options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE; // Eigen is SLOW

  // ceres::Covariance covariance(cov_options);

  covariance_ = boost::make_shared<ceres::Covariance>(cov_options);

  std::vector<const double*> blocks;
  for (auto& node_pair : local_pose_nodes_) {
    blocks.push_back(node_pair.second->pose().translation_data());
    blocks.push_back(node_pair.second->pose().rotation_data());
  }

  for (auto& kp : kps_) {
    blocks.push_back(kp->data());
  }

  blocks.push_back(object_pose_.rotation_data());
  blocks.push_back(object_pose_.translation_data());

  bool succeeded = covariance_->Compute(blocks, &ceres_problem_);

  if (!succeeded) {
    ROS_WARN_STREAM("Covariance computation failed");
    ROS_WARN_STREAM("Problem has " << ceres_problem_.NumParameters() << " parameters and "
         << ceres_problem_.NumResiduals() << " residuals.");
    have_covariance_ = false;
  } else {
    have_covariance_ = true;
  }
}

void
StructureOptimizationProblem::addCamera(SemanticKeyframe::Ptr keyframe,
                                        bool use_constant_camera_pose)
{
  if (keyframes_.find(keyframe->index()) == keyframes_.end()) {
    keyframes_.emplace(keyframe->index(), keyframe);
    local_pose_nodes_.emplace(keyframe->index(), util::allocate_aligned<SE3Node>(keyframe->key()));

    auto& node = local_pose_nodes_[keyframe->index()];
    node->pose() = keyframe->pose();

    ceres_problem_.AddParameterBlock(node->pose().rotation_data(), 4);
    ceres_problem_.SetParameterization(
      node->pose().rotation_data(), quaternion_parameterization_);

    ceres_problem_.AddParameterBlock(
      node->pose().translation_data(), 3);

    if (!use_constant_camera_pose) {
      ceres::CostFunction* pose_prior_cf =
        PosePriorCostTerm::Create(keyframe->pose(), keyframe->covariance());

      ceres_problem_.AddResidualBlock(
        pose_prior_cf, NULL, node->pose().rotation_data(),
        node->pose().translation_data());

    } else {

      ceres_problem_.SetParameterBlockConstant(
        node->pose().rotation_data());
      ceres_problem_.SetParameterBlockConstant(
        node->pose().translation_data());
    }

    solved_ = false;
    have_covariance_ = false;

    // Force the resulting object to be in front of this camera
    // ceres::CostFunction* front_cf = FrontOfCameraConstraint::Create(body_T_camera_.rotation(),
    //                                                                 body_T_camera_.translation());
    // ceres_problem_.AddResidualBlock(
    //   front_cf, NULL, camera_poses_[pose_index].rotation_data(),
    //                   camera_poses_[pose_index].translation_data(),
    //                   object_pose_.translation_data());
                      
  }
}

void
StructureOptimizationProblem::initializeKeypointPosition(
  size_t kp_id, const Eigen::Vector3d& p)
{
  *kps_[kp_id] = p;
}

void
StructureOptimizationProblem::initializePose(Pose3 pose)
{
  object_pose_ = pose;
}
