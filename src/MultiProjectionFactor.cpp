#include "semantic_slam/MultiProjectionFactor.h"

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/ProjectionFactor.h>

MultiProjectionFactor::MultiProjectionFactor(
  Vector3dNodePtr landmark_node,
  const Pose3& body_T_sensor,
  boost::shared_ptr<CameraCalibration> calibration,
  double reprojection_error_threshold,
  int tag)
  : CeresFactor(FactorType::DUMB_PROJECTION, tag)
  , I_T_C_(body_T_sensor)
  , calibration_(calibration)
  , reprojection_error_threshold_(reprojection_error_threshold)
  , in_graph_(false)
  , triangulation_good_(false)
{
    // Parameter block ordering:
    // Landmark position "pt", camera poses (q1 p1), (q2 p2), ...
    // [pt q1 p1 q2 p2 ... qn pn]
    // mutable_parameter_block_sizes()->push_back(3);
    nodes_.push_back(landmark_node);

    // parameter_blocks_.push_back(nodes[0]->vector().data());
    // parameter_blocks_.insert(parameter_blocks_.end(),
    //                          nodes_[0]->parameter_blocks().begin(),
    //                          nodes_[0]->parameter_blocks().end());
}

CeresFactor::Ptr
MultiProjectionFactor::clone() const
{
    auto fac = util::allocate_aligned<MultiProjectionFactor>(
      nullptr, I_T_C_, calibration_, reprojection_error_threshold_, tag_);

    for (size_t i = 0; i < nMeasurements(); ++i) {
        fac->addMeasurement(nullptr, msmts_[i], covariances_[i]);
    }

    return fac;
}

size_t
MultiProjectionFactor::nMeasurements() const
{
    return msmts_.size();
}

void
MultiProjectionFactor::triangulate(
  const aligned_vector<Pose3>& body_poses) const
{
    triangulation_good_ = false;
    triangulation_poses_.clear();

    if (nMeasurements() >= 2) {
        CameraSet cameras;
        for (size_t i = 0; i < msmts_.size(); ++i) {
            triangulation_poses_.push_back(body_poses[i].compose(I_T_C_));
            Camera camera(triangulation_poses_[i], calibration_);
            cameras.addCamera(camera);
        }

        // double cond;
        // TriangulationResult triangulation =
        // cameras.triangulateMeasurements(msmts_);

        TriangulationResult triangulation =
          cameras.triangulateMeasurementsApproximate(msmts_, 10);

        // TriangulationResult triangulation =
        // cameras.triangulateIterative(msmts_);

        if (triangulation.status == TriangulationStatus::SUCCESS &&
            triangulation.max_reprojection_error <=
              reprojection_error_threshold_) {
            landmark()->vector() = triangulation.point;
            triangulation_good_ = true;
        }
    }
}

void
MultiProjectionFactor::addToProblem(boost::shared_ptr<ceres::Problem> problem)
{
    // TODO add huber loss
    in_graph_ = true;

    aligned_vector<Pose3> body_poses;
    for (size_t i = 0; i < msmts_.size(); ++i) {
        body_poses.push_back(camera_node(i)->pose());
    }

    triangulate(body_poses);

    landmark()->addToProblem(problem);

    auto problem_it = std::find(problems_.begin(), problems_.end(), problem);
    if (problem_it == problems_.end()) {
        problems_.push_back(problem);
        internalAddToProblem(problem);
    } else {
        // ROS_WARN("Tried to add a factor to a problem it's already in.");
    }
}

void
MultiProjectionFactor::internalAddToProblem(
  boost::shared_ptr<ceres::Problem> problem)
{
    // Set up parameter block sizes and pointers
    mutable_parameter_block_sizes()->clear();
    parameter_blocks_.clear();

    // landmark node
    mutable_parameter_block_sizes()->push_back(3);
    parameter_blocks_.push_back(landmark()->vector().data());

    // camera nodes
    for (size_t i = 0; i < msmts_.size(); ++i) {
        mutable_parameter_block_sizes()->push_back(7);
        parameter_blocks_.push_back(camera_node(i)->pose().data());
    }

    set_num_residuals(2 * nMeasurements());

    ceres::ResidualBlockId residual_id =
      problem->AddResidualBlock(this, NULL, parameter_blocks_);
    residual_ids_[problem.get()] = residual_id;
    active_ = true;
}

void
MultiProjectionFactor::removeFromProblem(
  boost::shared_ptr<ceres::Problem> problem)
{
    auto problem_it = std::find(problems_.begin(), problems_.end(), problem);
    if (problem_it != problems_.end()) {
        problems_.erase(problem_it);
        internalRemoveFromProblem(problem);
    } else {
        // ROS_WARN("Tried to remove a factor from a problem it's not in");
    }

    landmark()->removeFromProblem(problem);
}

void
MultiProjectionFactor::internalRemoveFromProblem(
  boost::shared_ptr<ceres::Problem> problem)
{
    auto it = residual_ids_.find(problem.get());
    if (it != residual_ids_.end()) {
        problem->RemoveResidualBlock(it->second);
        residual_ids_.erase(it);
    }

    active_ = !residual_ids_.empty();
}

void
MultiProjectionFactor::addMeasurement(SE3NodePtr body_pose_node,
                                      const Eigen::Vector2d& pixel_coords,
                                      const Eigen::Matrix2d& msmt_covariance)
{
    // Compute the reprojection error...
    // if (in_graph_) {
    //     try {
    //         Camera camera(body_pose_node->pose().compose(I_T_C_),
    //         calibration_); Eigen::Vector2d zhat =
    //         camera.project(landmark()->vector()); double error =
    //         (pixel_coords - zhat).transpose() *
    //         msmt_covariance.llt().solve(pixel_coords - zhat);

    //         if (error > chi2inv99(2) || !std::isfinite(error)) return;

    //         // ROS_INFO_STREAM("[SmartProjectionFactor] Added measurement
    //         with reprojection error " << error);
    //         // std::cout << " zhat = " << zhat.transpose() << " ; msmt = " <<
    //         pixel_coords.transpose() << std::endl;
    //     } catch (CheiralityException& e) {
    //         return;
    //         // ROS_INFO_STREAM("[SmartProjectionFactor] Added measurement
    //         behind camera!");
    //     }
    // }

    // body_poses_.push_back(body_pose_node);
    msmts_.push_back(pixel_coords);
    covariances_.push_back(msmt_covariance);

    nodes_.push_back(body_pose_node);

    sqrt_informations_.push_back(Eigen::Matrix2d::Identity());
    Eigen::Matrix2d sqrtC = msmt_covariance.llt().matrixL();
    sqrtC.triangularView<Eigen::Lower>().solveInPlace(
      sqrt_informations_.back());

    // Ceres can't handle block sizes changing if we've already been added to
    // the Problem. So we need to remove and re-add ourself
    if (in_graph_) {
        for (auto& problem : problems_) {
            internalRemoveFromProblem(problem);
            internalAddToProblem(problem);
        }
    }
}

bool
MultiProjectionFactor::decideIfTriangulate(
  const aligned_vector<Pose3>& body_poses) const
{
    // Triangulate if:
    //  (1) the last triangulation failed,
    //  (2) camera poses are sufficiently different from last time,
    //  (3) or, the number of camera poses is different.
    if (!triangulation_good_ ||
        body_poses.size() != triangulation_poses_.size()) {
        return true;
    }

    // Compare each camera pose
    bool equal = true;
    for (size_t i = 0; i < body_poses.size(); ++i) {
        Pose3 camera_pose = body_poses[i].compose(I_T_C_);

        if (!camera_pose.equals(triangulation_poses_[i], 1e-5)) {
            equal = false;
            break;
        }
    }

    return !equal;
}

bool
MultiProjectionFactor::Evaluate(double const* const* parameters,
                                double* residuals_ptr,
                                double** jacobians) const
{
    // Collect parameters
    Eigen::Map<const Eigen::Vector3d> map_pt(parameters[0]);
    aligned_vector<Pose3> body_poses;
    for (size_t i = 0; i < nMeasurements(); ++i) {
        Eigen::Map<const Eigen::VectorXd> qp(parameters[1 + i], 7);

        body_poses.push_back(Pose3(qp));
    }

    Eigen::Map<Eigen::VectorXd> residuals(residuals_ptr, num_residuals());

    if (decideIfTriangulate(body_poses)) {
        triangulate(body_poses);
    }

    using JacobianType =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    // If the triangulation was good, compute & use the actual costs and
    // Jacobians. If it wasn't, zero out the residuals and jacobian, which is
    // equivalent to not including this factor in the estimation
    if (!triangulation_good_) {
        residuals.setZero();
        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<JacobianType> Dr_dl(
                  jacobians[0], num_residuals(), 3);
                Dr_dl.setZero();
            }

            for (size_t i = 0; i < nMeasurements(); ++i) {
                if (jacobians[i + 1]) {
                    Eigen::Map<JacobianType> Dr_dx(
                      jacobians[i + 1], num_residuals(), 7);
                    Dr_dx.setZero();
                }
            }
        }

        return true;
    }

    // Iterate through each measurement computing its residual & jacobian
    // If we need the Jacobians, compute them as we go...
    // Indices corresponding to this are 0 (pt), 1 + 2*i, and 2 + 2*i...
    if (jacobians) {
        for (size_t i = 0; i < nMeasurements(); ++i) {
            Eigen::MatrixXd Hpose_compose;
            Pose3 G_T_C =
              body_poses[i].compose(I_T_C_, Hpose_compose, boost::none);

            Camera cam(G_T_C, calibration_);

            Eigen::MatrixXd Hpose_project, Hpoint_project, Hpose;
            Eigen::Vector2d zhat;

            bool behind_camera = false;

            try {
                zhat = cam.project(map_pt, Hpose_project, Hpoint_project);
                Hpose = Hpose_project * Hpose_compose;
            } catch (CheiralityException& e) {
                // ignore this measurement for now
                // TODO can this ever even happen if triangulation_good_ ==
                // true??
                behind_camera = true;
                residuals.segment<2>(2 * i).setZero();
            }

            // std::cout << "camera pose: \n" << G_T_C << std::endl;
            // std::cout << "zhat = " << zhat.transpose() << ", msmt = " <<
            // msmts_[i].transpose() << std::endl;

            if (jacobians[0]) {
                Eigen::Map<JacobianType> Dr_dpt(
                  jacobians[0], num_residuals(), 3);

                if (!behind_camera) {
                    Dr_dpt.block<2, 3>(2 * i, 0) =
                      -sqrt_informations_[i] * Hpoint_project;
                } else {
                    Dr_dpt.block<2, 3>(2 * i, 0).setZero();
                }
            }

            if (jacobians[1 + i]) {
                Eigen::Map<JacobianType> Dr_dx(
                  jacobians[1 + i], num_residuals(), 7);

                Dr_dx.setZero();

                if (!behind_camera) {
                    Dr_dx.block<2, 7>(2 * i, 0) =
                      -sqrt_informations_[i] * Hpose;

                    // Dr_dx.block<2, 4>(2 * i, 0) =
                    //   -sqrt_informations_[i] * Hpose.block<2, 4>(0, 0);

                    // Dr_dx.block<2, 3>(2 * i, 4) =
                    //   -sqrt_informations_[i] * Hpose.block<2, 3>(0, 4);
                }
            }

            if (!behind_camera) {
                residuals.segment<2>(2 * i) =
                  sqrt_informations_[i] * (msmts_[i] - zhat);
            }
        }
    } else {
        // Just computing the residual
        for (size_t i = 0; i < nMeasurements(); ++i) {

            Pose3 G_T_C = body_poses[i].compose(I_T_C_);

            Camera cam(G_T_C, calibration_);

            try {
                Eigen::Vector2d zhat = cam.project(map_pt);

                residuals.segment<2>(2 * i) =
                  sqrt_informations_[i] * (msmts_[i] - zhat);
            } catch (CheiralityException& e) {
                residuals.segment<2>(2 * i).setZero();
            }
        }
    }

    return true;
}

void
MultiProjectionFactor::createGtsamFactors() const
{
    if (!landmark())
        return;

    for (size_t i = gtsam_factors_.size(); i < msmts_.size(); ++i) {

        if (!camera_node(i))
            return;

        // our calibration == nullptr corresponds to an already calibrated
        // camera, i.e. cx = cy = 0 and fx = fy = 1, which is what the default
        // gtsam calibration constructor provides
        auto gtsam_calib =
          calibration_ ? util::allocate_aligned<gtsam::Cal3DS2>(*calibration_)
                       : util::allocate_aligned<gtsam::Cal3DS2>();

        auto gtsam_noise =
          gtsam::noiseModel::Gaussian::Covariance(covariances_[i]);
        auto gtsam_fac =
          util::allocate_aligned<GtsamFactorType>(msmts_[i],
                                                  gtsam_noise,
                                                  camera_node(i)->key(),
                                                  landmark()->key(),
                                                  gtsam_calib,
                                                  gtsam::Pose3(I_T_C_));

        gtsam_factors_.push_back(gtsam_fac);
    }
}

void
MultiProjectionFactor::addToGtsamGraph(
  boost::shared_ptr<gtsam::NonlinearFactorGraph> graph) const
{
    // for (int i = 0; i < msmts_.size(); ++i) {
    //     if (body_poses_[i]->active()) {
    //         graph->push_back(gtsam_factors_[i]);
    //     }
    // }

    if (gtsam_factors_.size() != msmts_.size())
        createGtsamFactors();

    for (auto& f : gtsam_factors_)
        graph->push_back(f);
}
