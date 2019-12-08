#include "semantic_slam/MultiProjectionFactor.h"

#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

MultiProjectionFactor::MultiProjectionFactor(Vector3dNodePtr landmark_node, 
                                            const Pose3& body_T_sensor,
                                            boost::shared_ptr<CameraCalibration> calibration,
                                            double reprojection_error_threshold,
                                            int tag)
    : CeresFactor(FactorType::DUMB_PROJECTION, tag),
      landmark_node_(landmark_node),
      I_T_C_(body_T_sensor),
      calibration_(calibration),
      reprojection_error_threshold_(reprojection_error_threshold),
      in_graph_(false),
      problem_(nullptr),
      triangulation_good_(false)
{
    // Parameter block ordering:
    // Landmark position "pt", camera poses (q1 p1), (q2 p2), ...
    // [pt q1 p1 q2 p2 ... qn pn]
    mutable_parameter_block_sizes()->push_back(3); // add landmark parameter block
    parameter_blocks_.push_back(landmark_node_->vector().data());

    nodes_.push_back(landmark_node_);
}

size_t MultiProjectionFactor::nMeasurements() const
{
    return msmts_.size();
}

void MultiProjectionFactor::triangulate(const aligned_vector<Pose3>& body_poses) const
{
    triangulation_good_ = false;
    triangulation_poses_.clear();

    if (nMeasurements() >= 2) {
        CameraSet cameras;
        for (int i = 0; i < msmts_.size(); ++i) {
            triangulation_poses_.push_back(body_poses[i].compose(I_T_C_));
            Camera camera(triangulation_poses_[i], calibration_);
            cameras.addCamera(camera);
        }

        // double cond;
        // TriangulationResult triangulation = cameras.triangulateMeasurements(msmts_);

        TriangulationResult triangulation = cameras.triangulateMeasurementsApproximate(msmts_, 30);

        // TriangulationResult triangulation = cameras.triangulateIterative(msmts_);

        // check error
        // if (triangulation.status == TriangulationStatus::FAILURE) {
        //     ROS_INFO_STREAM("n frames = " << msmts_.size() << ", triangulation failed!!");
        // } else if (triangulation.status == TriangulationStatus::BEHIND_CAMERA) {
        //     ROS_INFO_STREAM("n frames = " << msmts_.size() << ", triangulation behind camera!!");
        // }

        // if (tri_iter.status != TriangulationStatus::SUCCESS) {
        //     ROS_INFO_STREAM("n frames = " << msmts_.size() << ", iterative failed!!");
        // }

        // if (triangulation.status == TriangulationStatus::SUCCESS && tri_iter.status == TriangulationStatus::SUCCESS) {
        //     double err = (triangulation.point - tri_iter.point).norm();
        //     ROS_INFO_STREAM("n frames = " << msmts_.size() << ", Approximation error = " << err);
        // }

        // std::cout << "Normal:  " << triangulation.point.transpose() << "; iter = " << tri_iter.point.transpose() << std::endl;

        // TODO check that it's ok

        // std::cout << "Triangulation result: " << pt.transpose() << std::endl;
        // std::cout << "  cond = " << cond << std::endl;

        if (triangulation.status == TriangulationStatus::SUCCESS
                    && triangulation.max_reprojection_error <= reprojection_error_threshold_) {
            landmark_node_->vector() = triangulation.point;
            triangulation_good_ = true;
        }
    } 
}

void MultiProjectionFactor::addToProblem(boost::shared_ptr<ceres::Problem> problem)
{
    // TODO add huber loss
    in_graph_ = true;
    problem_ = problem;

    if (triangulation_good_) {
        residual_id_ = problem->AddResidualBlock(this, NULL, parameter_blocks_);
        landmark_node_->addToProblem(problem);
        active_ = true;
    }
}

void MultiProjectionFactor::removeFromProblem(boost::shared_ptr<ceres::Problem> problem)
{
    problem->RemoveResidualBlock(residual_id_);

    landmark_node_->removeFromProblem(problem);

    active_ = false;
}


void MultiProjectionFactor::addMeasurement(SE3NodePtr body_pose_node,
                                        const Eigen::Vector2d& pixel_coords, 
                                        const Eigen::Matrix2d& msmt_covariance)
{
    // Compute the reprojection error...
    // if (in_graph_) {
    //     try {
    //         Camera camera(body_pose_node->pose().compose(I_T_C_), calibration_);
    //         Eigen::Vector2d zhat = camera.project(landmark_node_->vector());
    //         double error = (pixel_coords - zhat).transpose() * msmt_covariance.llt().solve(pixel_coords - zhat);

    //         if (error > chi2inv99(2) || !std::isfinite(error)) return;

    //         // ROS_INFO_STREAM("[SmartProjectionFactor] Added measurement with reprojection error " << error);
    //         // std::cout << " zhat = " << zhat.transpose() << " ; msmt = " << pixel_coords.transpose() << std::endl;
    //     } catch (CheiralityException& e) {
    //         return;
    //         // ROS_INFO_STREAM("[SmartProjectionFactor] Added measurement behind camera!");
    //     }
    // }

    body_poses_.push_back(body_pose_node);
    msmts_.push_back(pixel_coords);
    // covariances_.push_back(msmt_covariance);

    nodes_.push_back(body_pose_node);

    sqrt_informations_.push_back(Eigen::Matrix2d::Identity());
    Eigen::Matrix2d sqrtC = msmt_covariance.llt().matrixL();
    sqrtC.triangularView<Eigen::Lower>().solveInPlace(sqrt_informations_.back());

    // Ceres can't handle block sizes changing if we've already been added to the Problem.
    // So we need to remove and re-add ourself
    if (active_) {
        removeFromProblem(problem_);
    }

    mutable_parameter_block_sizes()->push_back(4); // q
    parameter_blocks_.push_back(body_pose_node->pose().rotation_data());
    mutable_parameter_block_sizes()->push_back(3); // p
    parameter_blocks_.push_back(body_pose_node->pose().translation_data());

    set_num_residuals(2 * nMeasurements());
    
    aligned_vector<Pose3> body_poses;
    for (auto& node : body_poses_) {
        body_poses.push_back(node->pose());
    }

    triangulate(body_poses);

    if (in_graph_ && triangulation_good_) {
        addToProblem(problem_);
    }

    // gtsam support
    auto gtsam_noise = gtsam::noiseModel::Gaussian::Covariance(msmt_covariance);
    auto gtsam_fac = util::allocate_aligned<GtsamFactorType>(
        pixel_coords,
        gtsam_noise,
        body_pose_node->key(),
        landmark_node_->key(),
        util::allocate_aligned<gtsam::Cal3DS2>(*calibration_),
        gtsam::Pose3(I_T_C_)
    );
    gtsam_factors_.push_back(gtsam_fac);
}

bool MultiProjectionFactor::decideIfTriangulate(const aligned_vector<Pose3>& body_poses) const
{
    // Triangulate if:
    //  (1) the last triangulation failed,
    //  (2) camera poses are sufficiently different from last time,
    //  (3) or, the number of camera poses is different.
    if (!triangulation_good_ || body_poses.size() != triangulation_poses_.size()) {
        return true;
    }

    // Compare each camera pose
    bool equal = true;
    for (int i = 0; i < body_poses.size(); ++i) {
        Pose3 camera_pose = body_poses[i].compose(I_T_C_);

        if (!camera_pose.equals(triangulation_poses_[i], 1e-5)) {
            equal = false;
            break;
        }
    }

    return !equal;
}

bool MultiProjectionFactor::Evaluate(double const* const* parameters, 
                                     double* residuals_ptr, 
                                     double** jacobians) const
{
    // Collect parameters
    Eigen::Map<const Eigen::Vector3d> map_pt(parameters[0]);
    aligned_vector<Pose3> body_poses;
    for (int i = 0; i < nMeasurements(); ++i) {
        Eigen::Map<const Eigen::Quaterniond> q(parameters[1 + 2*i]);
        Eigen::Map<const Eigen::Vector3d> p(parameters[2 + 2*i]);

        body_poses.push_back(Pose3(q,p));
    }

    Eigen::Map<Eigen::VectorXd> residuals(residuals_ptr, num_residuals());

    if (decideIfTriangulate(body_poses)) triangulate(body_poses);

    // Iterate through each measurement computing its residual & jacobian
    // If we need the Jacobians, compute them as we go...
    // Indices corresponding to this are 0 (pt), 1 + 2*i, and 2 + 2*i...
    using JacobianType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    if (jacobians) {
        for (int i = 0; i < nMeasurements(); ++i) {
            Eigen::MatrixXd Hpose_compose;
            Pose3 G_T_C = body_poses[i].compose(I_T_C_, Hpose_compose, boost::none);

            Camera cam(G_T_C, calibration_);

            Eigen::MatrixXd Hpose_project, Hpoint_project, Hpose;
            Eigen::Vector2d zhat;

            bool behind_camera = false;
            
            try {
                zhat = cam.project(map_pt, Hpose_project, Hpoint_project);
                Hpose = Hpose_project * Hpose_compose;
            } catch (CheiralityException& e) {
                // ignore this measurement for now
                behind_camera = true;
                residuals.segment<2>(2*i).setZero();
            }

            // std::cout << "camera pose: \n" << G_T_C << std::endl;
            // std::cout << "zhat = " << zhat.transpose() << ", msmt = " << msmts_[i].transpose() << std::endl;

            if (jacobians[0]) {
                Eigen::Map<JacobianType> Dr_dpt(jacobians[0], num_residuals(), 3);

                if (!behind_camera) {
                    Dr_dpt.block<2,3>(2*i,0) = -sqrt_informations_[i] * Hpoint_project;
                } else {
                    Dr_dpt.block<2,3>(2*i,0).setZero();
                }
            }

            if (jacobians[1+2*i]) {
                Eigen::Map<JacobianType> Dr_dq(jacobians[1+2*i], num_residuals(), 4);

                Dr_dq.setZero();

                if (!behind_camera) {
                    Dr_dq.block<2,4>(2*i,0) = -sqrt_informations_[i] * Hpose.block<2,4>(0,0);
                }
            }

            if (jacobians[2+2*i]) {
                Eigen::Map<JacobianType> Dr_dp(jacobians[2+2*i], num_residuals(), 3);

                Dr_dp.setZero();

                if (!behind_camera) {
                    Dr_dp.block<2,3>(2*i,0) = -sqrt_informations_[i] * Hpose.block<2,3>(0,4);
                }
            }

            if (!behind_camera) {
                residuals.segment<2>(2*i) = sqrt_informations_[i] * (msmts_[i] - zhat);
            }

        } 
    } else {
        // Just computing the residual
        for (int i = 0; i < nMeasurements(); ++i) {

            Pose3 G_T_C = body_poses[i].compose(I_T_C_);
            
            Camera cam(G_T_C, calibration_);

            try {
                Eigen::Vector2d zhat = cam.project(map_pt);

                residuals.segment<2>(2*i) = sqrt_informations_[i] * (msmts_[i] - zhat);
            } catch (CheiralityException& e) {
                residuals.segment<2>(2*i).setZero();
            }
        }
    }

    return true;
}

void
MultiProjectionFactor::addToGtsamGraph(boost::shared_ptr<gtsam::NonlinearFactorGraph> graph) const
{
    // for (int i = 0; i < msmts_.size(); ++i) {
    //     if (body_poses_[i]->active()) {
    //         graph->push_back(gtsam_factors_[i]);
    //     } 
    // }

    for (auto& f : gtsam_factors_) graph->push_back(f);
}
