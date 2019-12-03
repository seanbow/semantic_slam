#include "semantic_slam/CameraSet.h"

#include "semantic_slam/FactorGraph.h"
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/VectorNode.h"
#include "semantic_slam/CeresProjectionFactor.h"
#include "semantic_slam/Symbol.h"

TriangulationResult
CameraSet::triangulateMeasurementsApproximate(const aligned_vector<Eigen::Vector2d>& pixel_msmts,
                                                int n_cameras,
                                                boost::optional<double&> condition_number)
{
    if (pixel_msmts.size() != cameras_.size()) {
        throw std::runtime_error("[CameraSet::triangulateMeasurements] Error: number of measurements does not"
                                 " match number of cameras.");
    }

    TriangulationResult result;

    if (pixel_msmts.size() < 2) {
        result.status = TriangulationStatus::DEGENERATE;
        return result;
    }

    // Set up the least-squares system
    // Here we're making an approximation in the case of having a very large number of cameras
    // Select a subset of camera frames from which to triangulate
    int frame_skip = std::floor(pixel_msmts.size() / n_cameras + 0.5);
    frame_skip = std::max(frame_skip, 1);
    int n_frames = std::floor(pixel_msmts.size() / frame_skip);

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3 * n_frames, n_frames + 3);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(3 * n_frames);

    Eigen::Vector3d p_meas;
    p_meas(2) = 1.0;

    for (int i = 0; i < n_frames; ++i) {
        p_meas.head<2>() = cameras_[i * frame_skip].calibrate(pixel_msmts[i * frame_skip]);

        A.block<3,3>(3*i, 0) = Eigen::Matrix3d::Identity();
        A.block<3,1>(3*i, 3+i) = -(cameras_[i * frame_skip].pose().rotation() * p_meas);
        b.block<3,1>(3*i, 0) = cameras_[i * frame_skip].pose().translation();
    }

    // ROS_INFO_STREAM("Construction: " << TIME_TOC);
    // TIME_TIC;

    // Solve the system

    // If the condition number is requested we have to do SVD of A anyway so use it to
    // solve the system as well
    if (condition_number)
    {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        result.point = svd.solve(b).block<3, 1>(0, 0);
        *condition_number = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
    }
    else
    {
        // this function gets called a ~~lot~~
        // need to do this as efficiently as possible at the expense of accuracy
        result.point = A.colPivHouseholderQr().solve(b).block<3, 1>(0, 0);
        // result.point = (A.transpose() * A).ldlt().solve(A.transpose() * b).block<3,1>(0,0);
    }

    // ROS_INFO_STREAM("Solution: " << TIME_TOC);
    // ROS_INFO_STREAM("n cameras = " << cameras_.size());

    if (!result.point.allFinite()) {
        result.status = TriangulationStatus::FAILURE;
        return result;
    }

    // Check the quality of the triangulation by checking reprojection errors
    double max_error = 0;
    for (int i = 0; i < pixel_msmts.size(); ++i) {
        try {
            Eigen::Vector2d zhat = cameras_[i].project(result.point);
            double error = (zhat - pixel_msmts[i]).norm();
            max_error = std::max(max_error, error);
        } catch (CheiralityException& e) {
            result.status = TriangulationStatus::BEHIND_CAMERA;
            return result;
        }
    }

    // std::cout << "Max reproj error = " << max_error << std::endl;
    result.max_reprojection_error = max_error;

    result.status = TriangulationStatus::SUCCESS;

    return result;
}

TriangulationResult
CameraSet::triangulateMeasurements(const aligned_vector<Eigen::Vector2d>& pixel_msmts,
                                   boost::optional<double&> condition_number)
{
    if (pixel_msmts.size() != cameras_.size()) {
        throw std::runtime_error("[CameraSet::triangulateMeasurements] Error: number of measurements does not"
                                 " match number of cameras.");
    }

    TriangulationResult result;

    if (pixel_msmts.size() < 2) {
        result.status = TriangulationStatus::DEGENERATE;
        return result;
    }

    // Set up the least-squares system

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3 * pixel_msmts.size(), pixel_msmts.size() + 3);
    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(3 * pixel_msmts.size(), 1);

    Eigen::Vector3d p_meas;
    p_meas(2) = 1.0;

    for (int i = 0; i < pixel_msmts.size(); ++i) {
        p_meas.head<2>() = cameras_[i].calibrate(pixel_msmts[i]);

        A.block<3,3>(3*i, 0) = Eigen::Matrix3d::Identity();
        A.block<3,1>(3*i, 3+i) = -(cameras_[i].pose().rotation() * p_meas);
        b.block<3,1>(3*i, 0) = cameras_[i].pose().translation();
    }

    // ROS_INFO_STREAM("Construction: " << TIME_TOC);
    // TIME_TIC;

    // Solve the system

    // If the condition number is requested we have to do SVD of A anyway so use it to
    // solve the system as well
    if (condition_number)
    {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        result.point = svd.solve(b).block<3, 1>(0, 0);
        *condition_number = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
    }
    else
    {
        // this function gets called a ~~lot~~
        // need to do this as efficiently as possible at the expense of accuracy
        result.point = A.colPivHouseholderQr().solve(b).block<3, 1>(0, 0);
        // result.point = (A.transpose() * A).ldlt().solve(A.transpose() * b).block<3,1>(0,0);
    }

    // ROS_INFO_STREAM("Solution: " << TIME_TOC);
    // ROS_INFO_STREAM("n cameras = " << cameras_.size());

    if (!result.point.allFinite()) {
        result.status = TriangulationStatus::FAILURE;
        return result;
    }

    // Check the quality of the triangulation by checking reprojection errors
    double max_error = 0;
    for (int i = 0; i < pixel_msmts.size(); ++i) {
        try {
            Eigen::Vector2d zhat = cameras_[i].project(result.point);
            double error = (zhat - pixel_msmts[i]).norm();
            max_error = std::max(max_error, error);
        } catch (CheiralityException& e) {
            result.status = TriangulationStatus::BEHIND_CAMERA;
            return result;
        }
    }

    // std::cout << "Max reproj error = " << max_error << std::endl;
    result.max_reprojection_error = max_error;

    result.status = TriangulationStatus::SUCCESS;

    return result;
}


TriangulationResult
CameraSet::triangulateIterative(const aligned_vector<Eigen::Vector2d>& pixel_msmts)
{
    TriangulationResult result;

    if (pixel_msmts.size() < 2) {
        result.status = TriangulationStatus::DEGENERATE;
        return result;
    }

    // Need to initialize the point
    TriangulationResult linear_result = triangulateMeasurementsApproximate(pixel_msmts, 10);

    if (linear_result.status == TriangulationStatus::FAILURE) {
        return linear_result;
    }

    // Build a factor graph corresponding to the triangulation of the point
    FactorGraph graph;

    Vector3dNodePtr point = util::allocate_aligned<Vector3dNode>(symbol_shorthand::L(0));
    // point->vector() = cameras_[0].pose().translation();
    // point->vector() += cameras_[0].pose().transform_from(Eigen::Vector3d(0,0,20));
    point->vector() = linear_result.point;
    graph.addNode(point);

    for (int i = 0; i < pixel_msmts.size(); ++i) {
        auto node = util::allocate_aligned<SE3Node>(symbol_shorthand::X(i));
        node->pose() = cameras_[i].pose();

        Eigen::Vector2d px_noise(4, 4);

        auto factor = util::allocate_aligned<CeresProjectionFactor>(node, 
                                                                    point, 
                                                                    pixel_msmts[i], 
                                                                    px_noise.asDiagonal(), 
                                                                    cameras_[i].calibration(),
                                                                    Pose3(),
                                                                    false);

        graph.addNode(node);
        graph.setNodeConstant(node);

        graph.addFactor(factor);
    }

    bool succeeded = graph.solve(false);

    result.point = point->vector();

    // Check the quality of the triangulation by checking reprojection errors
    double max_error = 0;
    for (int i = 0; i < pixel_msmts.size(); ++i) {
        try {
            Eigen::Vector2d zhat = cameras_[i].project(result.point);
            double error = (zhat - pixel_msmts[i]).norm();
            max_error = std::max(max_error, error);
        } catch (CheiralityException& e) {
            result.status = TriangulationStatus::BEHIND_CAMERA;
            return result;
        }
    }

    // std::cout << "Max reproj error = " << max_error << std::endl;
    result.max_reprojection_error = max_error;
    result.status = succeeded ? TriangulationStatus::SUCCESS : TriangulationStatus::FAILURE;

    return result;
}