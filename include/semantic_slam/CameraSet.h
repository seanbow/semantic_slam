#pragma once

#include "semantic_slam/Common.h"
#include "semantic_slam/Camera.h"
#include "semantic_slam/pose_math.h"

enum class TriangulationStatus {
    SUCCESS,
    BEHIND_CAMERA,
    DEGENERATE,
    OUTLIER,
    FAILURE
};

struct TriangulationResult {
    Eigen::Vector3d point;
    TriangulationStatus status;
    double max_reprojection_error;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

class CameraSet {
public:
    CameraSet() { }

    void addCamera(const Camera& camera);

    aligned_vector<Camera>& cameras() { return cameras_; }
    const aligned_vector<Camera>& cameras() const { return cameras_; }

    TriangulationResult triangulateMeasurements(const aligned_vector<Eigen::Vector2d>& pixel_msmts,
                                            boost::optional<double&> condition_number=boost::none);

private:
    aligned_vector<Camera> cameras_;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

void CameraSet::addCamera(const Camera& camera)
{
    cameras_.push_back(camera);
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
        // throw std::runtime_error("[CameraSet::triangulateMeasurements] Error: not enough measurements to triangulate.");
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
        // QR should be stable enough
        result.point = A.colPivHouseholderQr().solve(b).block<3, 1>(0, 0);
    }

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
