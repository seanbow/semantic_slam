#include "semantic_slam/MultiProjectionFactor.h"

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
      active_(false),
      problem_(nullptr),
      triangulation_good_(false)
{
    // Parameter block ordering:
    // Landmark position "pt", camera poses (q1 p1), (q2 p2), ...
    // [pt q1 p1 q2 p2 ... qn pn]
    mutable_parameter_block_sizes()->push_back(3); // add landmark parameter block
    parameter_blocks_.push_back(landmark_node_->vector().data());
}

size_t MultiProjectionFactor::nMeasurements() const
{
    return msmts_.size();
}

void MultiProjectionFactor::triangulate()
{
    triangulation_good_ = false;

    if (nMeasurements() >= 2) {
        CameraSet cameras;
        for (int i = 0; i < msmts_.size(); ++i) {
            Camera camera(body_poses_[i]->pose().compose(I_T_C_), calibration_);
            cameras.addCamera(camera);
        }

        // double cond;
        TriangulationResult triangulation = cameras.triangulateMeasurements(msmts_);

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

    triangulate();

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
    body_poses_.push_back(body_pose_node);
    msmts_.push_back(pixel_coords);
    // covariances_.push_back(msmt_covariance);

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

    // triangulate();

    if (in_graph_) {
        addToProblem(problem_);
    }
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