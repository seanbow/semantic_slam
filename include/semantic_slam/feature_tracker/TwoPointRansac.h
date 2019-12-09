#ifndef TWOPOINTRANSAC_H_
#define TWOPOINTRANSAC_H_
#include "nister.h"
#include <boost/array.hpp>
#include <boost/shared_ptr.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>
#include <iostream>
#include <opencv/cv.h>
#include <ros/ros.h>
#include <vector>

/* @brief Two pointRANSAC class.
 * @author Sean Bowman
 */
class TwoPointRansac
{
  public:
    TwoPointRansac(int n_hypotheses, double sqrt_samp_thresh);

    size_t computeInliers(const std::vector<cv::Point2f>& points_A,
                          const std::vector<cv::Point2f>& points_B,
                          const Eigen::Matrix3d& R,
                          Eigen::Matrix<bool, 1, Eigen::Dynamic>& inliers);

    size_t computeInliersNormalized(
      const Eigen::MatrixXd& points_A_,
      const Eigen::MatrixXd& points_B_,
      const Eigen::Matrix3d& R,
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
        Eigen::Vector3d solution; // translation vector
        Eigen::Vector2i set;
        size_t inliers;

        Hypothesis()
          : inliers(0)
        {}

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
    void solveTwoPoint(const Eigen::MatrixXd& points_A,
                       const Eigen::MatrixXd& points_B,
                       const Eigen::Matrix3d& R,
                       int selection);

    double computeSampsonError(const Eigen::Vector3d& p1,
                               const Eigen::Vector3d& p2,
                               const Eigen::Matrix3d& R,
                               const Eigen::Vector3d& trans);

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
