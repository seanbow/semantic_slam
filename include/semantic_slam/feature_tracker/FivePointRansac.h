#ifndef RANSAC_H_
#define RANSAC_H_

#include "nister.h"
#include <boost/array.hpp>
#include <boost/shared_ptr.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <vector>
/* @brief RANSAC class. It is is initialized and preallocates all the space
 * needed for it to run. The space is pre-allocated since the ransac runs in an
 * iterative scheme and destroy.
 * @author Dimitrios Kottas MARS UMN
 * @author Sean Bowman MARS UMN
 */

class FivePointRansac
{
  public:
    FivePointRansac(int n_hypotheses, double sqrt_samp_thresh);

    size_t computeInliers(const std::vector<cv::Point2f>& points_A,
                          const std::vector<cv::Point2f>& points_B,
                          Eigen::Matrix<bool, 1, Eigen::Dynamic>& inliers);

    size_t computeInliersNormalized(
      const Eigen::MatrixXd& points_A_,
      const Eigen::MatrixXd& points_B_,
      Eigen::Matrix<bool, 1, Eigen::Dynamic>& inliers);

    /*
     * @brief selects a random set up to @param number_of_points_ for the
     * hypothesis
     * @param hypothesis_position_
     */
    void selectRandomSet(int number_of_points, int hypothesis_position);

    void setCameraCalibration(double fx,
                              double fy,
                              double s,
                              double u0,
                              double v0,
                              double k1,
                              double k2,
                              double p1,
                              double p2);

    struct Hypothesis
    {
        Eigen::Matrix<double, 3, 30>
          solutions; // essential matrices, 3x(3*totalSolutions)
        int totalSolutions;
        Eigen::Matrix<int, 10, 1> inliers;
        Eigen::Matrix<int, 5, 1> set;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    typedef boost::shared_ptr<Hypothesis> SharedHypothesis;

  private:
    size_t n_hypotheses_;

    bool calibrated_;

    /*
     * @brief Runs the five point algorithm using the points selected in
     * the hypothesis @param selection_
     */
    void solveFivePoint(const Eigen::MatrixXd& points_A,
                        const Eigen::MatrixXd& points_B,
                        int selection);

    double computeSampsonError(const Eigen::Vector3d& p1,
                               const Eigen::Vector3d& p2,
                               const Eigen::Matrix3d& essentialMatrix);

    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;

    double samp_thresh_;

    Eigen::Matrix<int, 1, 100> random_generator_;
    // int winner_1;
    // int winner_2;
    // int winner_inliers_
    std::vector<SharedHypothesis> hypotheses_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif
