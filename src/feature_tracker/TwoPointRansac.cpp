#include "semantic_slam/Common.h"

#include <boost/make_shared.hpp>
#include <chrono>

#include <opencv2/imgproc/imgproc.hpp>

#include "semantic_slam/feature_tracker/TwoPointRansac.h"

TwoPointRansac::TwoPointRansac(int n_hypotheses, double sqrt_samp_thresh)
  : n_hypotheses_(n_hypotheses)
  , calibrated_(false)
  , samp_thresh_(sqrt_samp_thresh * sqrt_samp_thresh)
{

    hypotheses_.reserve(n_hypotheses);

    for (int i = 0; i < n_hypotheses; i++) {
        hypotheses_.push_back(boost::make_shared<Hypothesis>());
    }
}

size_t
TwoPointRansac::computeInliers(const std::vector<cv::Point2f>& points_A,
                               const std::vector<cv::Point2f>& points_B,
                               const Eigen::Matrix3d& R,
                               Eigen::Matrix<bool, 1, Eigen::Dynamic>& inliers)
{
    size_t n_points = points_A.size();

    if (n_points < 5) {
        // impossible to compute any estimate with less than 2 correspondences.
        // with 2 points exactly it's impossible to do outlier rejection too (2
        // points will of course fit the model they define) so reject anything
        // less than 5 or so
        inliers = Eigen::Matrix<bool, 1, Eigen::Dynamic>::Zero(1, n_points);
        return 0;
    }

    // normalize & undistort points

    std::vector<cv::Point2f> pointsA_norm, pointsB_norm;

    cv::undistortPoints(points_A, pointsA_norm, camera_matrix_, dist_coeffs_);
    cv::undistortPoints(points_B, pointsB_norm, camera_matrix_, dist_coeffs_);

    // convert vector<point> to EIgen

    Eigen::MatrixXd eigenA(3, n_points);
    Eigen::MatrixXd eigenB(3, n_points);

    for (int i = 0; i < n_points; ++i) {
        eigenA.col(i) << pointsA_norm[i].x, pointsA_norm[i].y, 1;
        eigenB.col(i) << pointsB_norm[i].x, pointsB_norm[i].y, 1;
    }

    return computeInliersNormalized(eigenA, eigenB, R, inliers);
}

size_t
TwoPointRansac::computeInliersNormalized(
  const Eigen::MatrixXd& points_A,
  const Eigen::MatrixXd& points_B,
  const Eigen::Matrix3d& R,
  Eigen::Matrix<bool, 1, Eigen::Dynamic>& inliers)
{

    int n_points = points_A.cols();

    int winner = 0;
    int winning_inliers = 0;

    // double sqrt_samp_threshold = 6.25e-4;
    // double sqrt_samp_threshold = 5e-4;
    // double samp_threshold = 1*pow(sqrt_samp_threshold,2);

    for (int i = 0; i < n_hypotheses_; ++i) {
        selectRandomSet(n_points, i);

        solveTwoPoint(points_A, points_B, R, i);

        // std::cout << "Hypothesis..." << i << "total solutions ..." <<
        // hypothesis_storage_.at(i)->totalSolutions << "\n";

        for (int k = 0; k < n_points; ++k) {
            double err = computeSampsonError(
              points_A.col(k), points_B.col(k), R, hypotheses_[i]->solution);

            if (err < samp_thresh_) {
                hypotheses_[i]->inliers++;
            }

            // std::cout << "pt " << i << " samp error = " << err << std::endl;
            // std::cout << "(thresh = " << samp_thresh_ << ")" << std::endl;
        }

        if (hypotheses_[i]->inliers > winning_inliers) {
            winner = i;
            winning_inliers = hypotheses_[i]->inliers;
        }
    }

    inliers = Eigen::Matrix<bool, 1, Eigen::Dynamic>::Ones(1, n_points);

    for (int k = 0; k < n_points; ++k) {
        double err = computeSampsonError(
          points_A.col(k), points_B.col(k), R, hypotheses_[winner]->solution);

        if (pow(err, 1) > samp_thresh_) {
            inliers(0, k) = 0;
        }
    }

    return winning_inliers;
}

void
TwoPointRansac::setCameraCalibration(double fx,
                                     double fy,
                                     double s,
                                     double u0,
                                     double v0,
                                     double k1,
                                     double k2,
                                     double p1,
                                     double p2)
{
    camera_matrix_ = (cv::Mat_<double>(3, 3) << fx, 0, u0, 0, fy, v0, 0, 0, 1);

    dist_coeffs_ = (cv::Mat_<double>(1, 4) << k1, k2, p1, p2);

    calibrated_ = true;
}

double
TwoPointRansac::computeSampsonError(const Eigen::Vector3d& p1,
                                    const Eigen::Vector3d& p2,
                                    const Eigen::Matrix3d& R,
                                    const Eigen::Vector3d& trans)
{
    Eigen::Matrix3d E = skewsymm(trans) * R; // essential matrix

    Eigen::Vector3d Ex1 = E * p1;
    Eigen::Vector3d Ex2 = E.transpose() * p2;

    double err = std::pow(p2.transpose() * Ex1, 2);

    err /=
      Ex1(0) * Ex1(0) + Ex1(1) * Ex1(1) + Ex2(0) * Ex2(0) + Ex2(1) * Ex2(1);

    // Eigen::Vector3d Fx1;
    // Eigen::Vector3d Fx2;
    // Fx1 = E*p1;
    // Fx2 = E.transpose()*p2;
    // return (double) (pow(p2.transpose()*Fx1,2))/( (Fx1(0)*Fx1(0)) + (
    // Fx1(1)*Fx1(1) ) + (Fx2(1) * Fx2(1) ) + (Fx2(0)*Fx2(0)));

    return err;
}

void
TwoPointRansac::solveTwoPoint(const Eigen::MatrixXd& points_A,
                              const Eigen::MatrixXd& points_B,
                              const Eigen::Matrix3d& R,
                              int selection)
{

    auto& set = hypotheses_[selection]->set;

    Eigen::Vector3d C1_p_i = points_A.col(set(0));
    Eigen::Vector3d C1_p_j = points_A.col(set(1));

    Eigen::Vector3d C2_p_i = points_B.col(set(0));
    Eigen::Vector3d C2_p_j = points_B.col(set(1));

    Eigen::Matrix<double, 2, 3> M = Eigen::Matrix<double, 2, 3>::Zero();

    M.row(0) = (R * C1_p_i).transpose() * skewsymm(C2_p_i);
    M.row(1) = (R * C1_p_j).transpose() * skewsymm(C2_p_j);

    // std::cout << "Matrix M = " << std::endl << M << std::endl;

    // get null space of M via SVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeFullV);
    Eigen::Vector3d translation = svd.matrixV().topRightCorner<3, 1>();

    // std::cout << "Translation soln = " << translation.transpose() <<
    // std::endl;

    hypotheses_[selection]->solution = translation;
}

void
TwoPointRansac::selectRandomSet(int num_points, int hyp_index)
{
    hypotheses_[hyp_index]->set(0) = rand() % num_points;

    do {
        hypotheses_[hyp_index]->set(1) = rand() % num_points;
    } while (hypotheses_[hyp_index]->set(1) == hypotheses_[hyp_index]->set(0));
}
